"""
MODULE: Agent Nodes (Master Version)
DESCRIPTION: 
    Fixed variable mapping for LLMEngine and schema-aware Cypher generation.
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
        print(f"   > [Neo4j Error]: {e}")
        return []

# --- NODES ---

def cypher_generation_node(state: AgentState, llm_engine: LLMEngine, neo4j_driver):
    print("--- [AGENT] Thinking: Generating Cypher ---")
    
    # 1. Dynamic Schema Extraction
    schema_info = "Nodes: PERSON, CONCEPT. Relationships: DISCUSSED (Person-Concept), CO_OCCURRENCE (Concept-Concept)"
    try:
        with neo4j_driver.session() as s:
            labels = s.run("CALL db.labels()").value()
            rels = s.run("CALL db.relationshipTypes()").value()
            schema_info = f"Labels: {labels}, Relationships: {rels}. Property: 'id' holds names."
    except: pass

    system_prompt = f"""You are a Neo4j Cypher expert. 
    SCHEMA: {schema_info}
    RULES:
    1. Property 'id' is CASE SENSITIVE. Use toLower(n.id) CONTAINS toLower('...') for search.
    2. Output ONLY the raw Cypher query. No markdown. No chatter.
    3. If unsure, RETURN: MATCH (n:PERSON) RETURN n.id LIMIT 1
    """
    
    # FIX: Using 'user_query' to match llm_engine.py signature
    cypher_query = llm_engine.chat(
        user_query=state["question"], 
        context=[], 
        chat_history=[], 
        system_override=system_prompt
    )
    
    cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
    return {"cypher_query": cypher_query, "steps": ["cypher_gen"]}

def retrieval_node(state: AgentState, vector_table, encoder, neo4j_driver):
    print("--- [AGENT] Thinking: Retrieval ---")
    context = []
    
    # Strategy A: Graph Reasoning
    if "MATCH" in state.get("cypher_query", "").upper():
        results = run_cypher(neo4j_driver, state["cypher_query"])
        for r in results:
            # Generic parsing: flattening keys/values so LLM can read any return type
            fact = " | ".join([f"{k}: {v}" for k, v in r.items()])
            context.append({"text": f"Graph Fact: {fact}", "chunk_id": "GRAPH", "chaos": 0})
    
    # Strategy B: Vector Fallback
    try:
        query_vec = encoder.encode(state["question"]).tolist()
        vec_results = vector_table.search(query_vec).limit(5).to_list()
        for r in vec_results:
            context.append({
                "text": r.get("text_content") or r.get("text") or "",
                "chunk_id": r.get("chunk_id", "VEC"),
                "chaos": r.get("episode_chaos", 0)
            })
    except Exception as e:
        print(f"   > Vector Error: {e}")

    return {"context": context, "steps": ["retrieval"]}

def grade_documents_node(state: AgentState, llm_engine: LLMEngine):
    print("--- [AGENT] Thinking: Grading ---")
    if not state["context"]: return {"grade": "not_useful", "steps": ["grade"]}
    
    prompt = f"Data: {str(state['context'])[:500]}... Is this relevant to '{state['question']}'? YES/NO."
    # FIX: user_query mapping
    res = llm_engine.chat(user_query=prompt, context=[], chat_history=[], system_override="Answer only YES or NO.")
    grade = "useful" if "YES" in res.upper() else "not_useful"
    return {"grade": grade, "steps": ["grade"]}

def generate_node(state: AgentState, llm_engine: LLMEngine):
    print("--- [AGENT] Thinking: Generation ---")
    # FIX: user_query mapping
    response = llm_engine.chat(
        user_query=state["question"], 
        context=state["context"], 
        chat_history=[]
    )
    return {"answer": response, "steps": ["generate"]}

def query_rewrite_node(state: AgentState, llm_engine: LLMEngine):
    print("--- [AGENT] Action: Rewriting ---")
    # FIX: user_query mapping
    new_q = llm_engine.chat(
        user_query=f"Rewrite this query for better search: {state['question']}",
        context=[], chat_history=[],
        system_override="Output only the new string."
    )
    return {"question": new_q.strip(), "attempts": state.get("attempts", 0) + 1, "steps": ["rewrite"]}