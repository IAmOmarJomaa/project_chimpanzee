"""
MODULE: Gold Layer (Semantic Chunking)
DESCRIPTION:
    Breaks massive episodes into "bite-sized" chunks for the Vector Database.
    1. Loads Gold DNA (Metadata-rich).
    2. Segments text respecting sentence boundaries (No mid-sentence cuts).
    3. Calculates "Host Probability" to distinguish Joe from Guests.
    4. Flushes to batched Parquet files to prevent RAM explosion.

ARCHITECTURE:
    - Logic: Sentence Aggregation Window
    - Output: Gold Chunks (Ready for Vectorization)
"""

import polars as pl
import spacy
from tqdm import tqdm
import yaml
from pathlib import Path
import os

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    exit(1)

INPUT_FILE = config["paths"]["gold"]
# We use a directory for chunks because there will be thousands of them
OUTPUT_DIR = "../data/gold_chunks_parts"
FINAL_OUTPUT = "../data/gold_chunks.parquet"

# --- TUNING ---
CHUNK_SIZE = 1000       
CHUNK_OVERLAP = 200     
MIN_CHUNK_LEN = 50      

# Load Spacy for boundary detection
print("Loading NLP Model...")
try:
    nlp = spacy.load(config["nlp"]["spacy_model"])
    nlp.add_pipe("sentencizer")
except:
    print(f"[ERROR] Model {config['nlp']['spacy_model']} not found.")
    exit(1)

# Markers for Host Detection logic
JOE_MARKERS = [
    "jamie", "pull that up", "it's entirely possible", "let me ask you this",
    "have you ever seen", "whoa", "jesus", "look at that", "100 percent",
    "extraordinary", "chimpanzee", "dmt", "sauna", "cold plunge", "elk"
]

def calculate_host_probability(text: str) -> float:
    """Heuristic to guess if the speaker is Joe Rogan."""
    text_lower = text.lower()
    score = 0.0
    for marker in JOE_MARKERS:
        if marker in text_lower: score += 0.25
    if "?" in text: score += 0.15 # Host asks the questions
    if "jamie" in text_lower: score += 0.4
    return min(score, 1.0)

def detect_visual_context(text: str) -> bool:
    visual_triggers = ["pull that up", "look at that", "see that", "on the screen"]
    return any(t in text.lower() for t in visual_triggers)

def create_semantic_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    The 'Senior Engineer' Splitter.
    Instead of cutting at char 1000, we aggregate sentences until we hit 1000.
    """
    if not text: return []
    
    doc = nlp(text[:500000]) # Safety limit for very long docs
    sentences = [sent.text.strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    # Sliding window over sentences
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_len = len(sent)
        
        # If adding this sentence exceeds chunk size, finalize the current chunk
        if current_len + sent_len > chunk_size and current_len >= MIN_CHUNK_LEN:
            chunk_text = " ".join(current_chunk)
            chunks.append({"text": chunk_text, "start_char": 0}) # Start char is approx
            
            # OVERLAP LOGIC: Keep the last few sentences for the next chunk
            # Calculate how many sentences fit in the overlap window
            overlap_buffer = []
            overlap_len = 0
            backtrack_idx = i - 1
            while backtrack_idx >= 0:
                prev_sent = sentences[backtrack_idx]
                if overlap_len + len(prev_sent) > overlap:
                    break
                overlap_buffer.insert(0, prev_sent)
                overlap_len += len(prev_sent)
                backtrack_idx -= 1
            
            current_chunk = overlap_buffer
            current_len = overlap_len
        
        current_chunk.append(sent)
        current_len += sent_len
        i += 1
    
    # Flush last chunk
    if current_chunk:
        chunks.append({"text": " ".join(current_chunk), "start_char": 0})
        
    return chunks

def execute_chunking():
    print(f"--- Loading Gold DNA Layer: {INPUT_FILE} ---")
    if not Path(INPUT_FILE).exists():
        print("[ERROR] Input file not found.")
        return

    df = pl.read_parquet(INPUT_FILE)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("\n[Phase 1] Semantic Chunking (Batched)...")
    
    current_batch = []
    batch_index = 0
    total_chunks = 0
    
    records = df.to_dicts()
    
    for row in tqdm(records, desc="   > Processing"):
        episode_id = row["episode_id"]
        # REFACTOR: Using the new semantic splitter
        raw_chunks = create_semantic_chunks(row["content_body"])
        
        for i, chunk in enumerate(raw_chunks):
            # Determine stage (Business vs Banter)
            # (Simplified for speed: we just assume Business start index)
            stage = "Business" 
            if "business_start_index" in row and row["business_start_index"] > 0:
                 # Roughly estimate if this chunk is before the business start
                 # Since we lost exact char indices in semantic split, we assume early chunks = banter
                 if i < 2: stage = "Feedforward"

            chunk_record = {
                "chunk_id": f"{episode_id}-{i:04d}",
                "episode_id": episode_id,
                "text_content": chunk["text"],
                "guest_list": row["nlp_guests"],
                "top_concepts": row["top_concepts"],
                "episode_chaos": row["chaos_score"],
                "stage": stage,
                "host_probability": calculate_host_probability(chunk["text"]),
                "has_visual_context": detect_visual_context(chunk["text"])
            }
            current_batch.append(chunk_record)
            
        # Flush to disk every 20k chunks
        if len(current_batch) >= 20000: 
            pl.DataFrame(current_batch).write_parquet(f"{OUTPUT_DIR}/part_{batch_index}.parquet")
            total_chunks += len(current_batch)
            current_batch = [] 
            batch_index += 1

    # Final Flush
    if current_batch:
        pl.DataFrame(current_batch).write_parquet(f"{OUTPUT_DIR}/part_{batch_index}.parquet")
        total_chunks += len(current_batch)

    print(f"\n[Phase 2] Merging Batches...")
    try:
        # Polars glob read
        full_df = pl.read_parquet(f"{OUTPUT_DIR}/*.parquet")
        
        # Ensure final output dir exists
        Path(FINAL_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        
        full_df.write_parquet(FINAL_OUTPUT, compression="zstd")
        print(f"SUCCESS. Merged {total_chunks} chunks into {FINAL_OUTPUT}")
    except Exception as e:
        print(f"Merge failed (RAM limit). Data saved in {OUTPUT_DIR}/. Error: {e}")

if __name__ == "__main__":
    execute_chunking()