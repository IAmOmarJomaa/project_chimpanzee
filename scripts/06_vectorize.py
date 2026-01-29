"""
MODULE: Platinum Layer (Vectorization)
DESCRIPTION:
    Transforms text chunks into mathematical vectors (embeddings).
    1. Loads 'all-MiniLM-L6-v2' (Speed/Quality balance).
    2. Encodes text in batches.
    3. Saves a hybrid Parquet file (Metadata + Vector Column).

ARCHITECTURE:
    - Input: Gold Chunks
    - Output: Platinum Vectors
    - Logic: Dense Vector Embedding
"""

import polars as pl
from sentence_transformers import SentenceTransformer
import torch
import yaml
from pathlib import Path

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    exit(1)

INPUT_FILE = "../data/gold_chunks.parquet" # Config doesn't have chunks explicit path yet, hardcoding safe here or add to config
OUTPUT_FILE = config["paths"]["platinum_vectors"]

MODEL_NAME = "all-MiniLM-L6-v2" 
BATCH_SIZE = 128  

def get_device():
    """Auto-detects the fastest available hardware."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps" 
    else:
        return "cpu"

def execute_vectorization():
    print(f"--- Loading Gold Chunks: {INPUT_FILE} ---")
    if not Path(INPUT_FILE).exists():
        print(f"[ERROR] File not found: {INPUT_FILE}")
        return

    df = pl.read_parquet(INPUT_FILE)
    print(f"   > Loaded {df.height} chunks.")
    
    device = get_device()
    print(f"--- Loading AI Model: {MODEL_NAME} on {device.upper()} ---")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return

    # Prepare Data
    text_data = df["text_content"].to_list()
    
    print(f"\n[Phase 1] Vectorizing {len(text_data)} chunks...")
    
    # Run Inference
    vectors = model.encode(
        text_data, 
        batch_size=BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Save Results
    print("\n[Phase 2] Merging Vectors with Metadata...")
    vector_df = df.with_columns(
        pl.Series("vector", vectors).alias("vector")
    )
    
    # Ensure directory exists
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving Platinum Layer to {OUTPUT_FILE}...")
    vector_df.write_parquet(OUTPUT_FILE, compression="zstd")
    print("\n[SUCCESS] The Brain is Built.")

if __name__ == "__main__":
    execute_vectorization()