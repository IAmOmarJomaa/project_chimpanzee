"""
MODULE: Agent Nodes
DESCRIPTION: 
    Defines the executable functions (nodes) for the GraphRAG State Machine.
    Each node accepts the current AgentState, performs a discrete operation 
    (Search, Grade, Generate), and returns a dictionary of state updates.

ARCHITECTURE:
    - Functional Core: Pure functions where possible.
    - Error Handling: Nodes include fallback logic to prevent graph execution halts.
    - Type Safety: Inputs/Outputs validated against AgentState schema.

MAINTAINER: [Your Name]
"""

from llm_engine import LLMEngine
from agent.state import AgentState

# --- HELPER: Run Cypher ---
def run_cypher(driver, query):
    try:
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    except Exception as e:
        print(f"   > [Cypher Error]: {e}")
        return []

# --- NODES ---

def cypher_generation_node(state: AgentState, llm_engine: LLMEngine):
    """
    Step 1: Translate User Question -> Neo4j Cypher Query
    """
    print("--- [AGENT] Thinking: Generating Cypher Query ---")
    question = state["question"]
    
    # --- SAFETY-FIRST PROMPT ---
    system_prompt = """
    You are an expert Neo4j Cypher developer.
    THE DATABASE SCHEMA:
    - Nodes: Generic entities (People, Topics).
    - Relationship: [:DISCUSSED]
    - Property: 'id' (holds the name).
    
    GOAL: Write a Cypher query to find connections.
    RULES:
    1. ALWAYS use case-insensitive check: toLower(n.id) CONTAINS toLower('...')
    2. ONLY return the properties: RETURN n.id, m.id
    3. Output ONLY the Cypher query.
    """
    
    # Using your custom LLM engine wrapper
    cypher_query = llm_engine.chat(question, [], [], system_override=system_prompt)
    
    # Cleanup markdown artifacts
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    
    print(f"   > Generated: {cypher_query}")
    return {"cypher_query": cypher_query, "steps": ["cypher_gen"]}

def retrieval_node(state: AgentState, vector_table, encoder, neo4j_driver):
    """
    Step 2: Hybrid Retrieval (Graph Logic + Vector Search)
    """
    print("--- [AGENT] Thinking: Retrieval (Hybrid) ---")
    question = state["question"]
    cypher_query = state.get("cypher_query", "")
    context = []
    
    # STRATEGY A: Graph Execution
    if "MATCH" in cypher_query:
        print("   > Attempting Graph Execution...")
        graph_results = run_cypher(neo4j_driver, cypher_query)
        if graph_results:
            print(f"   > Graph found {len(graph_results)} connections!")
            for r in graph_results:
                fact = f"Graph Connection: {r.get('n.id', 'Entity')} is discussed with {r.get('m.id', 'Entity')}"
                context.append({"text": fact, "chunk_id": "GRAPH_LOGIC", "chaos": 0})
    
    # STRATEGY B: Vector Fallback (Always run to supplement)
    if len(context) < 5:
        print("   > Running Vector Search to supplement...")
        try:
            query_vec = encoder.encode(question)
            results = vector_table.search(query_vec).limit(5).to_list()
            for r in results:
                context.append({
                    "text": r.get("text_content") or r.get("text") or "",
                    "chunk_id": r.get("chunk_id", "Unknown"),
                    "chaos": r.get("episode_chaos", 0)
                })
        except Exception as e:
            print(f"   > Vector Error: {e}")

    return {"context": context, "steps": ["retrieval"]}

def grade_documents_node(state: AgentState, llm_engine: LLMEngine):
    """Step 3: Grade Documents"""
    print("--- [AGENT] Thinking: Grading ---")
    context = state["context"]
    if not context: 
        return {"grade": "not_useful", "steps": ["grade_documents"]}
        
    prompt = f"Data: {str(context)[:500]}... Question: {state['question']}. Is this data relevant? Answer YES or NO."
    grade_response = llm_engine.chat(prompt, [], [], system_override="You are a grader. Answer only YES or NO.")
    grade = "useful" if "YES" in grade_response.upper() else "not_useful"
    print(f"   > Grade: {grade}")
    return {"grade": grade, "steps": ["grade_documents"]}

def generate_node(state: AgentState, llm_engine: LLMEngine):
    """Step 4: Generate Final Answer"""
    print("--- [AGENT] Thinking: Generation ---")
    response = llm_engine.chat(state["question"], state["context"], [])
    return {"answer": response, "steps": ["generate"]}

def query_rewrite_node(state: AgentState, llm_engine: LLMEngine) -> dict:
    """
    Step 3b: Reformulate Query (Self-Correction)
    """
    print("--- TRANSFORMATION: REWRITING QUERY ---")
    question = state["question"]
    attempts = state.get("attempts", 0)
    
    system_prompt = """You are a Search Optimization Expert.
    The original query failed to retrieve relevant documents.
    Goal: Transform the user's input into a more specific, keyword-rich search query.
    Output ONLY the new query.
    """
    
    try:
        better_question = llm_engine.chat(
            user_prompt=f"Original Query: {question}",
            context=[],
            chat_history=[],
            system_override=system_prompt
        )
        better_question = better_question.strip().replace('"', '')
        print(f"   > Optimized: {better_question}")
    except Exception as e:
        print(f"   [WARNING] Rewrite failed: {e}. Keeping original.")
        better_question = question

    return {
        "question": better_question, 
        "attempts": attempts + 1,
        "steps": ["rewrite_query"]
    }