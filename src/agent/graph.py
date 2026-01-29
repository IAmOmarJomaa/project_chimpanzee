"""
MODULE: Agent Workflow (The Brain's Wiring)
DESCRIPTION:
    Constructs the LangGraph State Machine.
    Wires the nodes (Search, Grade, Rewrite) into a cyclic graph.

ARCHITECTURE:
    - Pattern: Adaptive RAG (Retrieval Augmented Generation)
    - Key Feature: Self-Correction Loop
      If the retrieved documents are irrelevant, the agent rewrites 
      the query and searches again (up to 3 times).
"""

from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import (
    cypher_generation_node, 
    retrieval_node, 
    grade_documents_node, 
    generate_node,
    query_rewrite_node
)

def build_agent_graph(vector_table, encoder, llm_engine, neo4j_driver):
    """
    Constructs the Adaptive GraphRAG Architecture.
    
    Args:
        vector_table: Connection to LanceDB table
        encoder: SentenceTransformer model
        llm_engine: Wrapper for the LLM API
        neo4j_driver: Connection to Graph DB
    """
    
    # 1. Initialize Graph
    workflow = StateGraph(AgentState)

    # 2. Add Nodes
    # We use lambdas to inject dependencies (Driver, Model, Table) into the nodes
    # This keeps the nodes.py pure and testable.
    workflow.add_node("cypher_gen", lambda state: cypher_generation_node(state, llm_engine, neo4j_driver))
    workflow.add_node("retrieve", lambda state: retrieval_node(state, vector_table, encoder, neo4j_driver))
    workflow.add_node("grade", lambda state: grade_documents_node(state, llm_engine))
    workflow.add_node("rewrite", lambda state: query_rewrite_node(state, llm_engine))
    workflow.add_node("generate", lambda state: generate_node(state, llm_engine))

    # 3. Define Main Flow (The "Happy Path")
    workflow.set_entry_point("cypher_gen")
    workflow.add_edge("cypher_gen", "retrieve")
    workflow.add_edge("retrieve", "grade")
    
    # 4. Conditional Logic (The "Brain")
    def decide_route(state):
        """
        Determines the next step based on document grading.
        """
        grade = state.get("grade")
        attempts = state.get("attempts", 0)
        
        # GUARDRAIL: Prevent infinite loops (max 3 retries)
        if attempts >= 3:
            print("--- [DECISION] Max attempts reached. Forcing generation. ---")
            return "generate"
            
        if grade == "useful":
            print("--- [DECISION] Context is good. Generating answer. ---")
            return "generate"
        else:
            print("--- [DECISION] Context is bad. Rewriting query. ---")
            return "rewrite"

    # Register the conditional edge
    workflow.add_conditional_edges(
        "grade",
        decide_route,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    
    # 5. Close the Loop
    # If we rewrite, we go back to retrieval
    workflow.add_edge("rewrite", "retrieve") 
    workflow.add_edge("generate", END)

    return workflow.compile()