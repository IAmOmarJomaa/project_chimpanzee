from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import lancedb
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os
from typing import List, Dict
from dotenv import load_dotenv # <--- ADD THIS IMPORT

# Load environment variables from .env file
load_dotenv() # <--- ADD THIS CALL

# ... rest of your imports (FastAPI, lancedb, etc.) ...

from application.llm_engine import LLMEngine 
from application.agent.workflow import build_agent_graph 

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "chimpanzee") 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANCEDB_URI = os.path.join(BASE_DIR, "data", "lancedb_store")
TABLE_NAME = "chimpanzee_vectors"
MODEL_NAME = "all-MiniLM-L6-v2"

# --- GLOBAL STATE ---
state = {
    "neo4j_driver": None,
    "encoder": None,
    "vector_table": None,
    "llm": None,
    "agent_graph": None 
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- [STARTUP] System Init ---")
    
    # 1. Connect to Graph (Neo4j)
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
        state["neo4j_driver"] = driver
        print("   > [Graph] Online")
    except Exception as e:
        print(f"   > [Graph] Error: {e}")

    # 2. Connect to Vector DB
    if os.path.exists(LANCEDB_URI):
        try:
            db = lancedb.connect(LANCEDB_URI)
            state["vector_table"] = db.open_table(TABLE_NAME)
            print(f"   > [Vectors] Connected to '{TABLE_NAME}'")
        except Exception as e:
            print(f"   > [Vectors] Error: {e}")
    
    # 3. Load Components
    print("   > [AI Model] Loading Encoder...")
    state["encoder"] = SentenceTransformer(MODEL_NAME)
    
    print("   > [LLM] Connecting to Local Brain...")
    state["llm"] = LLMEngine()
    print("   > [LLM] Ready")
    
    # 4. BUILD THE AGENT GRAPH (Updated)
    print("   > [Agent] Compiling Cognitive Graph...")
    if state["neo4j_driver"] and state["vector_table"]:
        state["agent_graph"] = build_agent_graph(
            state["vector_table"], 
            state["encoder"], 
            state["llm"],
            state["neo4j_driver"] # <--- Now passing the driver!
        )
        print("   > [Agent] Online")
    else:
        print("   > [Agent] Skipped (Missing DB connection)")

    yield 
    
    if state["neo4j_driver"]:
        state["neo4j_driver"].close()

app = FastAPI(title="Project Chimpanzee API", lifespan=lifespan)

class ChatRequest(BaseModel):
    query: str
    history: List[Dict] = [] 

class GraphRequest(BaseModel):
    entity: str
    limit: int = 10

@app.get("/")
def health(): return {"status": "online"}

# --- RESTORED ENDPOINT ---
@app.post("/search/graph")
def graph_search(request: GraphRequest):
    driver = state["neo4j_driver"]
    if not driver:
        raise HTTPException(status_code=503, detail="Graph offline")

    query = """
    MATCH (source)-[r:DISCUSSED]-(target)
    WHERE toLower(source.id) CONTAINS toLower($entity)
    RETURN source.id as entity, type(r) as rel, target.id as connected_to, r.chunk_id as chunk
    LIMIT $limit
    """
    
    with driver.session() as session:
        records = session.run(query, entity=request.entity, limit=request.limit)
        results = [dict(r) for r in records]
            
    return {"results": results}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if not state["agent_graph"]:
        raise HTTPException(status_code=503, detail="Agent not ready")

    # FIX: Standardizing the entry state to match state.py
    initial_state = {
        "question": request.query,
        "context": [],
        "attempts": 0,
        "grade": "",
        "steps": [],
        "answer": "",
        "cypher_query": ""
    }
    
    result = state["agent_graph"].invoke(initial_state)
    
    return {
        "response": result.get("answer", "I couldn't find anything on that, man."),
        "sources": [c.get("chunk_id") for c in result.get("context", [])],
        "steps_taken": result.get("steps", [])
    }