import polars as pl
import json
import os
from tqdm import tqdm

# --- Configuration ---
INPUT_FILE = "../data/gold_chunks.parquet"
OUTPUT_DIR = "../data/training_sets"

# --- TUNING (THE FIX) ---
# Previous: 0.85 (Too strict, required "Jamie" + "Pull up", which triggered visual filter)
# New: 0.15 (Captures Questions "?" which score 0.15, and minor markers "100 percent")
JOE_CONFIDENCE = 0.15  
MIN_TEXT_LEN = 50      

def harvest_personas_v2():
    print(f"--- Loading Gold Chunks: {INPUT_FILE} ---")
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pl.read_parquet(INPUT_FILE)
    print(f"   > Total Chunks Available: {df.height}")

    # --- 1. HARVEST VIRTUAL JOE ---
    print("\n[Phase 1] Harvesting 'Virtual Joe' Training Data...")
    
    # We lower the confidence to capture Questions and Banter
    joe_df = df.filter(
        (pl.col("host_probability") >= JOE_CONFIDENCE) &
        (pl.col("has_visual_context") == False) &
        (pl.col("text_content").str.len_chars() > MIN_TEXT_LEN)
    )
    
    joe_count = joe_df.height
    print(f"   > Found {joe_count} Joe Rogan speech samples (Threshold: {JOE_CONFIDENCE}).")
    
    joe_output = f"{OUTPUT_DIR}/virtual_joe_corpus.jsonl"
    
    with open(joe_output, "w", encoding="utf-8") as f:
        for row in tqdm(joe_df.iter_rows(named=True), desc="   > Exporting Joe"):
            entry = {
                "text": row["text_content"],
                "meta": {
                    "chaos_score": row["episode_chaos"],
                    "topics": row["top_concepts"]
                }
            }
            f.write(json.dumps(entry) + "\n")

    # --- 2. HARVEST JAMIE (The Tool User) ---
    print("\n[Phase 2] Harvesting 'Jamie Agent' Training Data...")
    
    jamie_df = df.filter(pl.col("has_visual_context") == True)
    
    jamie_count = jamie_df.height
    print(f"   > Found {jamie_count} Multimodal/Visual trigger events.")
    
    jamie_output = f"{OUTPUT_DIR}/jamie_agent_corpus.jsonl"
    with open(jamie_output, "w", encoding="utf-8") as f:
        for row in tqdm(jamie_df.iter_rows(named=True), desc="   > Exporting Jamie"):
            entry = {
                "trigger_text": row["text_content"],
                "guest_context": row["guest_list"],
                "topics": row["top_concepts"]
            }
            f.write(json.dumps(entry) + "\n")

    print("\n[SUCCESS] Harvest Complete.")
    print(f"   > Joe Rogan Persona Data: {joe_output}")
    print(f"   > Jamie Agent Data:       {jamie_output}")

if __name__ == "__main__":
    harvest_personas_v2()