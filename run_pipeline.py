import subprocess
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def run_step(script_path):
    print(f"\n[PIPELINE] Executing: {script_path}...")
    # Use sys.executable to ensure we stay in the (jre_lake) environment
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"!!! [CRITICAL FAILURE] {script_path} failed. Halting.")
        sys.exit(1)

def main():
    # FIXED: Added .py extensions and verified modular paths
    pipeline = [
        "ingestion_pipeline/core/etl/ingest_raw.py",
        "ingestion_pipeline/core/etl/clean_transcripts.py",
        "ingestion_pipeline/core/etl/extract_features.py",
        "ingestion_pipeline/core/etl/chunk_processor.py",
        "ingestion_pipeline/core/vectors/embedder.py",
        "ingestion_pipeline/core/graph/builder.py",
        "ingestion_pipeline/core/graph/neo4j_loader.py"
    ]

    print("=== PROJECT CHIMPANZEE: MEDALLION INGESTION START ===")
    
    for step in pipeline:
        if os.path.exists(step):
            run_step(step)
        else:
            print(f"[!] ERROR: Path not found: {step}")
            print("    Check if you ran the 'mv' commands to modularize your files.")
            sys.exit(1)

    print("\n=== SUCCESS: GRAPH AND VECTORS ARE SYNCED ===")

if __name__ == "__main__":
    main()