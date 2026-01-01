# 🧠 Project Chimpanzee: Hybrid GraphRAG Agent (Local Llama 3.2)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Polars](https://img.shields.io/badge/Data-Polars-orange) ![Neo4j](https://img.shields.io/badge/Graph-Neo4j-green) ![Ollama](https://img.shields.io/badge/AI-Ollama_Llama3-purple)

## 1. Executive Summary
Project Chimpanzee is an **Autonomous Research Agent** that clones Joe Rogan's brain. Unlike standard RAG systems that rely solely on vector similarity (which misses "multi-hop" connections), Chimpanzee uses a **Hybrid GraphRAG** architecture.

It ingests thousands of unstructured transcripts, builds a **Knowledge Hypergraph** (Guest <-> Concept), and uses a **Self-Correcting State Machine** to answer questions.

**Key Engineering Achievements:**
* **Zero-Loss Ingestion:** Processed terabytes of text using a **Medallion Architecture** (Bronze -> Silver -> Gold).
* **Streaming Graph Build:** Constructed a graph of 26M+ edges using **Polars Streaming** ($O(1)$ RAM usage) to prevent memory overflows.
* **Reflective Agent:** Implemented a **LangGraph** cycle that detects hallucinations. If the agent finds poor data, it **rewrites its own query** and searches again.

## 2. System Architecture

### 🧠 The "Brain" (Inference Layer)
The agent is not a linear chain. It is a **State Machine** (`LangGraph`) running locally on Llama 3.2 (via Ollama).

```mermaid
graph TD
    Start([User Question]) --> CypherGen[Plan: Generate Cypher]
    CypherGen --> Retrieve[Action: Hybrid Search]
    Retrieve --> Grade{Grade Documents}
    
    Grade -- "Useful" --> Generate[Generate Answer]
    Grade -- "Not Useful" --> Check{Attempts < 3?}
    
    Check -- "Yes" --> Rewrite[Self-Correction: Rewrite Query]
    Rewrite --> Retrieve
    
    Check -- "No" --> Generate
    
    Generate --> End([Final Response])
'''

### 🏭 The "Factory" (Ingestion Pipeline)
A strict ETL pipeline transforms raw text into a queryable Graph + Vector store.

| Layer | Tech | Responsibility |
| :--- | :--- | :--- |
| **Bronze** | `Polars` | Raw ingestion & Metadata parsing. |
| **Silver** | `SpaCy` | NLP-based Ad removal & Sentence segmentation. |
| **Gold** | `TextBlob` | Feature extraction (Chaos Score, Complexity Index). |
| **Platinum** | `Neo4j` | Knowledge Graph (Entity Relationships). |
| **Vectors** | `LanceDB` | Semantic Embeddings (`all-MiniLM-L6-v2`). |

## 3. Engineering Challenges & Trade-offs

### Challenge 1: The "Split Brain" Problem
* **Issue:** Initial attempts using simple `text.split('.')` destroyed data (e.g., "U.S.A." became 3 sentences).
* **Solution:** Refactored the entire pipeline to use **Spacy's Statistical Sentencizer**. This slowed ingestion by 40% but increased Ad-Detection accuracy by 15%.

### Challenge 2: The "Memory Bomb"
* **Issue:** Generating 26 million edges for the graph crashed the server (32GB RAM).
* **Solution:** Implemented **Partitioned Streaming**. Instead of holding the graph in memory, the script flushes edges to Parquet files every 500,000 rows.
    * *Result:* Constant memory usage ($O(1)$) regardless of dataset size.

## 4. Setup & Usage

### Prerequisites
* **Ollama:** `ollama run llama3.2`
* **Neo4j:** Running on `localhost:7687`

### Installation
1.  **Clone & Config:**
    ```bash
    cp .env.example .env  # Set NEO4J_PASSWORD
    ```
2.  **Run the Pipeline (Optional - Data Provided):**
    ```bash
    python ingestion_pipeline/4_ingest_v3_final.py
    ```
3.  **Chat with the Brain:**
    ```bash
    python application/main.py
    ```

## 5. Repository Structure
```bash
├── ingestion_pipeline/    # ETL Scripts (1-18)
│   ├── 4_ingest.py        # Bronze Layer (Raw)
│   ├── 7_clean.py         # Silver Layer (Cleaning)
│   ├── 13_graph.py        # Platinum Layer (Graph Build)
│   └── 16_loader.py       # Neo4j Bulk Loader
├── application/           # The Agent
│   ├── agent/             
│   │   ├── nodes.py       # Functional Units (Search, Grade)
│   │   ├── state.py       # TypedDict Memory
│   │   └── workflow.py    # Graph Wiring
│   └── llm_engine.py      # Ollama Wrapper
└── config/                # Settings.yaml

