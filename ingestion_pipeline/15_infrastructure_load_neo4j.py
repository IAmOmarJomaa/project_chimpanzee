import polars as pl
from neo4j import GraphDatabase
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv # <--- New Import

# --- LOAD SECRETS ---
# Load variables from .env file into os.environ
load_dotenv("../.env") 

# Now we fetch from OS environment, safe and professional
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # Will return None if missing

if not NEO4J_PASSWORD:
    print("[ERROR] NEO4J_PASSWORD is missing from .env file!")
    exit(1)

NEO4J_AUTH = (NEO4J_USER, NEO4J_PASSWORD)

NODES_FILE = "../data/hypergraph_nodes.parquet"
EDGES_FILE = "../data/hypergraph_edges.parquet"

# ... (Rest of your GraphLoader class remains exactly the same)