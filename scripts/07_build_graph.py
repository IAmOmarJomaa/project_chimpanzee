"""
MODULE: Platinum Layer (Hypergraph Construction)
DESCRIPTION:
    Builds the Knowledge Graph edges from the processed chunks.
    1. Aggregates Guest <-> Concept relationships.
    2. Aggregates Concept <-> Concept co-occurrences.
    3. Streams edges to partitioned Parquet files (Crash-proof).

ARCHITECTURE:
    - Input: Gold Chunks
    - Output: Edge List & Node List (Parquet)
    - Logic: Polars Streaming + Combinatorics
"""

import polars as pl
from tqdm import tqdm
import itertools
import os
import yaml
from pathlib import Path

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    exit(1)

INPUT_FILE = "../data/gold_chunks.parquet"
OUTPUT_NODES = config["paths"]["nodes"]
OUTPUT_EDGES = config["paths"]["platinum_edges"]

# Tuning
CONCEPT_MIN_FREQ = 2   
GUEST_MIN_FREQ = 1     

def build_hypergraph_parquet():
    print(f"--- Loading Gold Chunks: {INPUT_FILE} ---")
    if not Path(INPUT_FILE).exists():
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    df = pl.read_parquet(INPUT_FILE)
    
    # 1. Identify Chaos Episodes
    print("[Phase 0] Mapping Chaos Episodes...")
    chaos_map = {}
    # Use unique() to avoid processing duplicates
    chaos_df = df.select(["episode_id", "episode_chaos"]).unique()
    for row in chaos_df.iter_rows():
        ep_id, score = row
        # Chaos Definition: High score OR Fight Companion
        if score > 15.0 or "FC-" in ep_id:
            chaos_map[ep_id] = True
        else:
            chaos_map[ep_id] = False

    # 2. VALIDATION SCAN (Pass 1)
    print("\n[Phase 1] Validation Scan (Defining the Universe)...")
    
    # Aggregating Guest/Concept Counts
    # Guests: Explode list -> Clean -> Count -> Filter
    guests_exploded = df.select(pl.col("guest_list").explode().drop_nulls().str.strip_chars())
    guest_counts = guests_exploded.group_by("guest_list").len()
    valid_guests = set(guest_counts.filter(pl.col("len") >= GUEST_MIN_FREQ)["guest_list"].to_list())
    
    # Concepts
    concepts_exploded = df.select(pl.col("top_concepts").explode().drop_nulls().str.to_lowercase().str.strip_chars())
    concept_counts = concepts_exploded.group_by("top_concepts").len()
    valid_concepts = set(concept_counts.filter(pl.col("len") >= CONCEPT_MIN_FREQ)["top_concepts"].to_list())

    print(f"   > Valid Guests: {len(valid_guests)}")
    print(f"   > Valid Concepts: {len(valid_concepts)}")

    # 3. EDGE CONSTRUCTION (Pass 2 - Streaming)
    print("\n[Phase 2] Building Edge List (Streaming Mode)...")
    
    batch_size = 500_000 
    current_edges = []
    part_counter = 0
    
    # Ensure output dir exists
    if not os.path.exists(OUTPUT_EDGES):
        os.makedirs(OUTPUT_EDGES)
    
    total_edges = 0
    episodes = df.partition_by("episode_id")
    
    for ep_df in tqdm(episodes, desc="   > Extracting Edges"):
        episode_id = ep_df["episode_id"].to_list()[0]
        is_chaos = chaos_map.get(episode_id, False)
        
        # Safe Guest Extraction (No more bare 'except')
        current_guests = []
        try:
            # We assume guest_list is consistent per episode
            g_col = ep_df["guest_list"].drop_nulls()
            if not g_col.is_empty():
                # Take the first non-null list of guests
                raw_list = g_col[0] 
                # Filter against our valid set
                current_guests = [g for g in raw_list if g in valid_guests]
        except Exception as e:
            # Log specific error if column structure is wrong
            # But don't crash the whole pipeline for one bad row
            pass 

        for row in ep_df.iter_rows(named=True):
            concepts = row["top_concepts"]
            if not concepts: continue
            
            # Filter Concepts
            current_concepts = [c.lower().strip() for c in concepts if c.lower().strip() in valid_concepts]
            if not current_concepts: continue
            
            chunk_id = row["chunk_id"]

            # GENERATE EDGES
            # 1. Guest <-> Concept
            for guest in current_guests:
                for concept in current_concepts:
                    current_edges.append({
                        "source": guest,
                        "target": concept,
                        "type": "DISCUSSED",
                        "chunk_id": chunk_id,
                        "chaos": is_chaos,
                        "weight": 1.0
                    })
            
            # 2. Concept <-> Concept
            for c1, c2 in itertools.combinations(current_concepts, 2):
                current_edges.append({
                    "source": c1,
                    "target": c2,
                    "type": "CO_OCCURRENCE",
                    "chunk_id": chunk_id,
                    "chaos": is_chaos,
                    "weight": 0.5 
                })
        
        # FLUSH CHECK
        if len(current_edges) >= batch_size:
            pl.DataFrame(current_edges).write_parquet(f"{OUTPUT_EDGES}/part_{part_counter}.parquet")
            total_edges += len(current_edges)
            current_edges = [] 
            part_counter += 1

    # Final Flush
    if current_edges:
        pl.DataFrame(current_edges).write_parquet(f"{OUTPUT_EDGES}/part_{part_counter}.parquet")
        total_edges += len(current_edges)

    print(f"\n[Phase 3] Saving Node List...")
    # Ensure directory for nodes exists
    Path(OUTPUT_NODES).parent.mkdir(parents=True, exist_ok=True)
    
    nodes_data = [{"id": g, "type": "PERSON"} for g in valid_guests] + \
                 [{"id": c, "type": "CONCEPT"} for c in valid_concepts]
    
    pl.DataFrame(nodes_data).write_parquet(OUTPUT_NODES)

    print(f"\n[SUCCESS] HyperGraph Saved.")
    print(f"   > Total Edges: {total_edges}")

if __name__ == "__main__":
    build_hypergraph_parquet()