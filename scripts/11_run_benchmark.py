import requests
import pandas as pd
import time
import json
import os
from tqdm import tqdm

# --- CONFIG ---
API_URL = "http://127.0.0.1:8001/chat"
DATASET_FILE = "../data/golden_dataset.json"
RESULTS_FILE = "../data/benchmark_results.csv"
def run_benchmark():
    print(f"--- 🚀 Starting Benchmark on {API_URL} ---")
    # 1. Load Golden Dataset
    if not os.path.exists(DATASET_FILE):
        print(f"❌ Error: {DATASET_FILE} not found. Create it first.")
        return
        
    with open(DATASET_FILE, "r") as f:
        test_suite = json.load(f)

    results = []
    print(f"   > Loaded {len(test_suite)} questions from Golden Dataset.")

    # 2. Run Tests
    # Note: Since the backend has LangChain tracing enabled, 
    # these requests will automatically be logged to LangSmith.
    for i, test in enumerate(tqdm(test_suite, desc="   > Testing")):
        start_time = time.time()
        
        try:
            payload = {"query": test["query"], "history": []}
            response = requests.post(API_URL, json=payload).json()
            
            latency = round(time.time() - start_time, 2)
            
            ai_answer = response.get("response", "Error")
            steps = response.get("steps_taken", [])
            sources = response.get("sources", [])
            rewrote = "rewrite" in steps
            
            results.append({
                "ID": i+1,
                "Category": test["category"],
                "Question": test["query"],
                "Latency (s)": latency,
                "Sources": len(sources),
                "Self_Corrected": rewrote,
                "Response": ai_answer[:100].replace("\n", " ") + "..."
            })
            
        except Exception as e:
            print(f"❌ Error on Q{i}: {e}")

    # 3. Save CSV (For README/Resume)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)
    
    # Stats
    correction_rate = (len(df[df['Self_Corrected']==True]) / len(df)) * 100
    
    print("\n=== 📊 Benchmark Complete ===")
    print(f"   > Avg Latency:     {df['Latency (s)'].mean():.2f}s")
    print(f"   > Correction Rate: {correction_rate:.1f}%")
    print(f"   > Results Saved:   {RESULTS_FILE}")
    print("   > LangSmith Traces: SENT (Check Cloud Dashboard)")

if __name__ == "__main__":
    run_benchmark()