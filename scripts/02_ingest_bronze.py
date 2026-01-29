"""
MODULE: Bronze Layer Ingestion
DESCRIPTION: 
    The entry point of the ELT pipeline.
    1. Reads raw .txt transcripts.
    2. Parses metadata (Episode ID, Guest Name) from messy filenames.
    3. Tokenizes text into sentences using NLP (Spacy) for downstream accuracy.
    4. Saves to 'Bronze' Parquet (Immutable Raw State).

ARCHITECTURE:
    - Input: Text Files (Unstructured)
    - Output: Parquet (Structured Columnar)
    - Logic: Regex Parsing + NLP Segmentation

MAINTAINER: [Your Name]
"""

import polars as pl
import re
import yaml
import spacy
from pathlib import Path
from tqdm import tqdm

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file not found. Ensure 'config/settings.yaml' exists.")
    exit(1)

SOURCE_DIRECTORY = config["paths"]["raw_data"]
OUTPUT_FILE = config["paths"]["bronze"]

# --- SETUP NLP ---
# DECISION: Switched from regex split to Spacy Sentencizer.
# Reason: Simple split('.') breaks on abbreviations (e.g., "U.S.A."), corrupting data.
print("Loading NLP Tokenizer...")
try:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
except Exception as e:
    print(f"[ERROR] Spacy model failed: {e}")
    print("Run: python -m spacy download en_core_web_sm")
    exit(1)

def parse_filename_metadata(filename: str) -> dict:
    """Extracts structured metadata from chaotic filenames."""
    name = filename.replace(".txt", "").strip()
    
    # 1. Detect Split Episodes
    part_suffix = ""
    part_match = re.search(r"\(Part\s*(\d+|One|Two|Three)\)", name, re.IGNORECASE)
    if part_match:
        p_val = part_match.group(1).lower()
        mapping = {"one": "1", "two": "2", "three": "3"}
        p_val = mapping.get(p_val, p_val)
        part_suffix = f"-{p_val}"
    
    # 2. Standard JRE
    jre_match = re.search(r"Experience\s*#(\d+)\s*-\s*(.*)", name, re.IGNORECASE)
    if jre_match:
        return {
            "id": f"JRE-{jre_match.group(1)}{part_suffix}",
            "show_type": "JRE",
            "guest_draft": jre_match.group(2).strip()
        }

    # 3. MMA Show
    mma_match = re.search(r"MMA Show\s*#(\d+)\s*(?:-\s*with\s*|-\s*|with\s*)(.*)", name, re.IGNORECASE)
    if mma_match:
        return {
            "id": f"MMA-{mma_match.group(1)}{part_suffix}",
            "show_type": "MMA",
            "guest_draft": mma_match.group(2).strip()
        }

    # 4. Fallback
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    return {
        "id": f"Special-{clean_name}",
        "show_type": "Special",
        "guest_draft": "Unknown"
    }

def execute_ingest():
    print(f"--- Starting Bronze Ingestion ---")
    path_obj = Path(SOURCE_DIRECTORY)
    files = list(path_obj.glob("*.txt"))
    
    if not files:
        print(f"[ERROR] No files found in {SOURCE_DIRECTORY}")
        return

    data_buffer = []
    print(f"   > Found {len(files)} files. Processing...")
    
    for p in tqdm(files, desc="   > Ingesting"):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
            
        meta = parse_filename_metadata(p.name)
        
        data_buffer.append({
            "episode_id": meta["id"],
            "original_filename": p.name,
            "show_type": meta["show_type"],
            "guest_draft": meta["guest_draft"], 
            "raw_text": text
        })
        
    df = pl.DataFrame(data_buffer)
    
    print("   > Tokenizing sentences (Spacy)...")
    df = df.with_columns(
        pl.col("raw_text").map_elements(
            lambda x: [s.text.strip() for s in nlp(x).sents],
            return_dtype=pl.List(pl.Utf8)
        ).alias("sentences")
    )
    
    print(f"--- Ingestion Complete ---")
    print(f"   > Total Rows: {df.height}")
    
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f"   > Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    execute_ingest()