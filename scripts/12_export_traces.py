import os
import pandas as pd
from langsmith import Client
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Fetch variables
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
PROJECT_NAME = os.getenv("LANGCHAIN_PROJECT", "default")

def export_langsmith_logs():
    print(f"--- ☁️ Connecting to LangSmith Cloud (Project: {PROJECT_NAME}) ---")
    
    try:
        client = Client()
        # Fetch runs from the last session
        runs = list(client.list_runs(
            project_name=PROJECT_NAME,
            is_root=True,
            limit=20
        ))
        
        if not runs:
            print("❌ No traces found. Check LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true in .env")
            return

        data = []
        for run in runs:
            latency = (run.end_time - run.start_time).total_seconds() if run.end_time else 0
            data.append({
                "Input": run.inputs.get("question") or run.inputs.get("input", "N/A"),
                "Output": run.outputs.get("answer", "N/A") if run.outputs else "Error",
                "Latency": round(latency, 2),
                "Status": run.status
            })
        
        df = pd.DataFrame(data)
        print(f"✅ Successfully audited {len(df)} traces from the cloud.")
        print("\n--- RECENT CLOUD LOGS ---")
        print(df.head().to_markdown(index=False))
        
    except Exception as e:
        print(f"💥 Connection Error: {e}")

if __name__ == "__main__":
    export_langsmith_logs()