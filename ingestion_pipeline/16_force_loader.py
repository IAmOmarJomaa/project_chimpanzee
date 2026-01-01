"""
MODULE: Platinum Layer (Graph Loader)
DESCRIPTION:
    The Bridge between Parquet files and the Neo4j Graph Database.
    1. Connects securely to Neo4j using environment variables.
    2. Batch loads Nodes (Person, Concept).
    3. Batch loads Edges (Discussed, Co-Occurrence) from partitioned files.
    4. Creates Database Indexes for O(1) lookup speeds.

ARCHITECTURE:
    - Input: Platinum Parquet Layers
    - Output: Neo4j Graph
    - Logic: Cypher UNWIND Batch Ingestion (High throughput)
"""

import sys
import polars as pl
from neo4j import GraphDatabase
import os
import yaml
from tqdm import tqdm
from dotenv import load_dotenv

# --- CONFIGURATION ---
# 1. Load Secrets
load_dotenv("../.env")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    print("[CRITICAL ERROR] NEO4J_PASSWORD is missing from environment variables.")
    print("Fix: Create a .env file with NEO4J_PASSWORD=...")
    sys.exit(1)

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

# 2. Load Paths
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    sys.exit(1)

NODES_FILE = config["paths"]["nodes"]
EDGES_FILE = config["paths"]["platinum_edges"]
BATCH_SIZE = 5000 

def get_driver():
    """Establishes a secure connection to the Graph Database."""
    print(f"--- Connecting to {NEO4J_URI} ---")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        return driver
    except Exception as e:
        print(f"\n[AUTH ERROR] Could not connect to Neo4j: {e}")
        sys.exit(1)

def main():
    driver = get_driver()
    
    # 1. LOAD NODES
    print(f"\n[Phase 1] Loading Nodes from {NODES_FILE}...")
    if not os.path.exists(NODES_FILE):
        print(f"[ERROR] Node file missing: {NODES_FILE}")
        sys.exit(1)
        
    df = pl.read_parquet(NODES_FILE)
    print(f"   > Found {df.height} nodes.")
    
    # Cypher Queries (Optimized for Batching)
    person_query = "UNWIND $batch AS row MERGE (:PERSON {id: row.id})"
    concept_query = "UNWIND $batch AS row MERGE (:CONCEPT {id: row.id})"

    persons = df.filter(pl.col("type") == "PERSON").select("id").to_dicts()
    concepts = df.filter(pl.col("type") == "CONCEPT").select("id").to_dicts()

    with driver.session() as session:
        # Create Indexes FIRST (Crucial for ingestion speed)
        print("   > Creating Indexes...")
        session.run("CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:PERSON) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:CONCEPT) REQUIRE c.id IS UNIQUE")
        
        for i in tqdm(range(0, len(persons), BATCH_SIZE), desc="   > Loading Persons"):
            session.run(person_query, batch=persons[i : i + BATCH_SIZE])
            
        for i in tqdm(range(0, len(concepts), BATCH_SIZE), desc="   > Loading Concepts"):
            session.run(concept_query, batch=concepts[i : i + BATCH_SIZE])

    # 2. LOAD EDGES
    print(f"\n[Phase 2] Loading Edges from {EDGES_FILE}...")
    if not os.path.exists(EDGES_FILE):
        print(f"[ERROR] Edge directory missing: {EDGES_FILE}")
        sys.exit(1)

    files = sorted([os.path.join(EDGES_FILE, f) for f in os.listdir(EDGES_FILE) if f.endswith(".parquet")])
    
    discussed_query = """
    UNWIND $batch AS row
    MATCH (p:PERSON {id: row.source})
    MATCH (c:CONCEPT {id: row.target})
    MERGE (p)-[r:DISCUSSED]->(c)
    SET r.chunk_id = row.chunk_id, r.chaos = row.chaos
    """
    
    occur_query = """
    UNWIND $batch AS row
    MATCH (c1:CONCEPT {id: row.source})
    MATCH (c2:CONCEPT {id: row.target})
    MERGE (c1)-[r:CO_OCCURRENCE]->(c2)
    SET r.chunk_id = row.chunk_id, r.chaos = row.chaos
    """

    with driver.session() as session:
        for p_file in tqdm(files, desc="   > Processing Partitions"):
            df = pl.read_parquet(p_file)
            
            # Split by type to use correct query
            discussed = df.filter(pl.col("type") == "DISCUSSED").to_dicts()
            co_occur = df.filter(pl.col("type") == "CO_OCCURRENCE").to_dicts()
            
            if discussed:
                for i in range(0, len(discussed), BATCH_SIZE):
                    session.run(discussed_query, batch=discussed[i : i + BATCH_SIZE])

            if co_occur:
                for i in range(0, len(co_occur), BATCH_SIZE):
                    session.run(occur_query, batch=co_occur[i : i + BATCH_SIZE])

    driver.close()
    print("\n[SUCCESS] GRAPH SYNC COMPLETE.")

if __name__ == "__main__":
    main()