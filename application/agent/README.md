# 🧠 Chimpanzee Agent: Adaptive GraphRAG Engine

## 1. Architectural Thesis
The Chimpanzee Agent is not a linear chain. It is a **State Machine** built on `LangGraph`. It employs an **Adaptive RAG** pattern where the system critiques its own retrieved context before generating an answer.

### Core Design Principles:
1.  **Cyclic Reasoning:** If retrieved documents are irrelevant, the agent rewrites its own query and tries again.
2.  **Type Safety:** State is managed via `TypedDict` with reducer patterns for history tracking.
3.  **Guardrails:** Max-retry logic (`attempts >= 3`) prevents infinite loops and API cost spikes.

## 2. Complexity Analysis
The workflow executes a Directed Cyclic Graph (DCG).
* **Best Case:** $O(1)$ pass (Generate -> Retrieve -> Grade -> Answer).
* **Worst Case:** $O(N)$ where $N$ is `max_attempts`.
* **Latency:** Retrieval is optimized via LanceDB (ANN search) and Neo4j (Pointer traversal), minimizing the "Retrieve" node latency to <200ms.

## 3. Workflow Diagram (Mermaid)

```mermaid
graph TD
    Start([User Question]) --> CypherGen[Wait: Generate Cypher]
    CypherGen --> Retrieve[Action: Hybrid Search]
    Retrieve --> Grade{Grade Documents}
    
    Grade -- "Useful" --> Generate[Generate Answer]
    Grade -- "Not Useful" --> Check{Attempts < 3?}
    
    Check -- "Yes" --> Rewrite[Reasoning: Rewrite Query]
    Rewrite --> Retrieve
    
    Check -- "No" --> Generate
    
    Generate --> End([Final Response])