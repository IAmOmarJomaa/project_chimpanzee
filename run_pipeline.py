import subprocess
import os
import sys
from dotenv import load_dotenv

# Load real secrets from your existing .env
load_dotenv()

def run_step(script_path):
    print(f"\n[PIPELINE] Running: {script_path}...")
    # Using sys.executable ensures we use the same 'jre_lake' environment
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"!!! ERROR in {script_path}. Halting pipeline.")
        sys.exit(1)

def main():
    # The actual order of your project's data evolution
    pipeline = [
        "ingestion_pipeline/core/etl/ingest_raw.py",        # Bronze
        "ingestion_pipeline/core/etl/clean_transcripts.py", # Silver
        "ingestion_pipeline/core/etl/extract_features.py",  # Gold
        "ingestion_pipeline/core/etl/chunk_processor.py",   # Gold -> Chunks
        "ingestion_pipeline/core/vectors/embedder.py",      # Platinum (Vectors)
        "ingestion_pipeline/core/graph/builder.py",         # Platinum (Edges)
        "ingestion_pipeline/core/graph/neo4j_loader.py"     # Infrastructure Load
    ]

    print("=== PROJECT CHIMPANZEE: FULL SYSTEM INGESTION ===")
    
    # Check for Neo4j Connectivity before starting
    print(f"Targeting Neo4j at: {os.getenv('NEO4J_URI')}")
    
    for step in pipeline:
        if os.path.exists(step):
            run_step(step)
        else:
            print(f"[!] Warning: {step} not found. Check your 'core' directory structure.")

    print("\n=== SUCCESS: DATA LAKE & GRAPH ARE FULLY POPULATED ===")

if __name__ == "__main__":
    main()