"""
MODULE: Project Setup Automator
DESCRIPTION: Generates infrastructure and dependency files programmatically.
"""
import os

def create_requirements():
    content = """polars\npyarrow\nspacy\ntextblob\nnetworkx\nneo4j\nlancedb\nsentence-transformers\nlanggraph\nollama\npython-dotenv\npyyaml\ntqdm\n"""
    with open("requirements.txt", "w") as f:
        f.write(content)
    print("[SUCCESS] Created requirements.txt")

def create_docker_compose():
    # This reads the NEO4J_PASSWORD variable from your environment if it exists, 
    # but hardcodes the 'chimpanzee' default for the container setup.
    content = """version: '3.8'
services:
  neo4j:
    image: neo4j:latest
    container_name: chimpanzee_graph
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=${NEO4J_USER:-neo4j}/${NEO4J_PASSWORD:-chimpanzee}
    volumes:
      - ./data/neo4j/data:/data
"""
    with open("docker-compose.yml", "w") as f:
        f.write(content)
    print("[SUCCESS] Created docker-compose.yml (configured for .env reading)")

if __name__ == "__main__":
    create_requirements()
    create_docker_compose()