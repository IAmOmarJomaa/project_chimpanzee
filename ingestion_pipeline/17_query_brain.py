import sys
from neo4j import GraphDatabase

# --- CONFIGURATION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "chimpanzee") 

def query_graph():
    print(f"--- Connecting to The Brain ---")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
        driver.verify_connectivity()
    except Exception as e:
        print(f"[ERROR] Could not connect: {e}")
        return

    session = driver.session()

    # 1. THE "KEVIN BACON" TEST (Shortest Path)
    # Let's find how Elon Musk is connected to someone random, like Alex Jones.
    print("\n[Test 1] Pathfinding: Elon Musk <-> Alex Jones")
    query_path = """
    MATCH (p1:PERSON {id: "Elon Musk"}), (p2:PERSON {id: "Alex Jones"})
    MATCH path = shortestPath((p1)-[:DISCUSSED*..4]-(p2))
    RETURN [n in nodes(path) | n.id] as connection_chain
    """
    result = session.run(query_path).single()
    if result:
        print(f"   > Connection: {' <-> '.join(result['connection_chain'])}")
    else:
        print("   > No direct path found (Try increasing hop count).")

    # 2. THE "TOPIC AUTHORITY" TEST
    # Who talks about "Aliens" the most?
    topic = "aliens"
    print(f"\n[Test 2] Who are the experts on '{topic}'?")
    query_topic = """
    MATCH (p:PERSON)-[r:DISCUSSED]->(c:CONCEPT {id: $topic})
    RETURN p.id as guest, count(r) as mentions
    ORDER BY mentions DESC
    LIMIT 5
    """
    results = session.run(query_topic, topic=topic)
    for record in results:
        print(f"   > {record['guest']}: {record['mentions']} mentions")

    # 3. THE "CHAOS" TEST
    # Find the most chaotic episode in the DB
    print("\n[Test 3] Identifying High-Chaos Event...")
    query_chaos = """
    MATCH (p:PERSON)-[r:DISCUSSED]->(c:CONCEPT)
    WHERE r.chaos = true
    RETURN r.chunk_id as chunk, p.id as guest, c.id as concept
    LIMIT 1
    """
    result = session.run(query_chaos).single()
    if result:
        print(f"   > Chaos Edge Found: {result['guest']} talking about {result['concept']} (Chunk {result['chunk']})")

    driver.close()
    print("\n[SUCCESS] The Brain is fully operational.")

if __name__ == "__main__":
    query_graph()