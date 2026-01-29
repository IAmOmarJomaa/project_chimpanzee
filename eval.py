import requests
import pandas as pd
import time
from tqdm import tqdm

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8001/chat"
OUTPUT_FILE = "../data/benchmark_results.csv"

# --- THE "DEEP FAN" TEST SUITE ---
# 25 Prompts designed to break specific parts of the RAG pipeline.
TEST_SUITE = [
    # --- CATEGORY 1: VECTOR PRECISION ("The Deep Cuts") ---
    # Testing: Did the 'Gold Chunking' strategy preserve specific details?
    {"category": "Deep Cut", "query": "What specific story did Joey Diaz tell about the 'Susquehanna Weed'?"},
    {"category": "Deep Cut", "query": "What is the 'Stoned Ape Theory' and who explained it?"},
    {"category": "Deep Cut", "query": "What did Bob Lazar say about the propulsion system of the craft in specific technical terms?"},
    {"category": "Deep Cut", "query": "Describe the 'Dolce Diet' controversy regarding the weigh-ins."},
    {"category": "Deep Cut", "query": "What happened during the 'End of the World' podcast with Bill Burr?"},
    {"category": "Deep Cut", "query": "What did David Goggins say about 'taking souls'?"},

    # --- CATEGORY 2: GRAPH TRAVERSAL (Multi-Hop Reasoning) ---
    # Testing: Can the Platinum Layer connect two disparate nodes?
    {"category": "Graph Logic", "query": "Compare Graham Hancock's view on the Younger Dryas to Michael Shermer's view."},
    {"category": "Graph Logic", "query": "What connects Elon Musk and Jack Dorsey regarding censorship?"},
    {"category": "Graph Logic", "query": "How are Alex Jones and Tim Dillon connected in the context of conspiracy theories?"},
    {"category": "Graph Logic", "query": "Who are the guests that have discussed 'simulation theory' besides Elon Musk?"},
    {"category": "Graph Logic", "query": "What do Jordan Peterson and Bret Weinstein agree on regarding academia?"},

    # --- CATEGORY 3: META-COGNITION (Vague/Slang) ---
    # Testing: Does the 'Rewrite Node' trigger to fix these bad queries?
    {"category": "Ambiguous", "query": "Tell me about the elk meat thing."},
    {"category": "Ambiguous", "query": "What did he say about the chimpanzees ripping people apart?"},
    {"category": "Ambiguous", "query": "That time with the ufo guy."}, 
    {"category": "Ambiguous", "query": "Jamie pull that up about the bear."},
    
    # --- CATEGORY 4: ADVERSARIAL (Hallucination Traps) ---
    # Testing: Does the system reject fake premises?
    {"category": "Adversarial", "query": "What did Joe Rogan say about investing in 'CryptoZoo' with Logan Paul?"},
    {"category": "Adversarial", "query": "When did Joe interview Barack Obama and what did they eat?"},
    {"category": "Adversarial", "query": "Describe the episode where Joe fought a kangaroo."},
    {"category": "Adversarial", "query": "What did Elon Musk say about buying Spotify?"},

    # --- CATEGORY 5: CONCEPTUAL SYNTHESIS (The "Vibe" Check) ---
    # Testing: Can it summarize complex abstract topics?
    {"category": "Synthesis", "query": "Summarize the overarching argument against 'Cancel Culture' across all episodes."},
    {"category": "Synthesis", "query": "What is the consensus on 'Sauna use' for health?"},
    {"category": "Synthesis", "query": "Explain the recurring theme of 'discipline' in Jocko Willink's appearances."},
    {"category": "Synthesis", "query": "What are the most common arguments for and against Universal Basic Income (UBI) mentioned?"},
    {"category": "Synthesis", "query": "How has the discussion on 'Bitcoin' evolved from 2015 to now?"}
]

def run_benchmark():
    print(f"--- Starting 'Deep Fan' Benchmark on {API_URL} ---")
    results = []
    
    # 1. Check API Health
    try:
        requests.get("http://127.0.0.1:8001/")
        print("✅ API is Online.")
    except:
        print("❌ API is Offline. Run 'uvicorn src.api.main:app' first.")
        return

    # 2. Run Loop
    print(f"Testing {len(TEST_SUITE)} prompts...")
    
    for i, test in enumerate(tqdm(TEST_SUITE)):
        start_time = time.time()
        
        payload = {"query": test["query"], "history": []}
        try:
            response = requests.post(API_URL, json=payload).json()
            
            # Calculate Latency
            latency = round(time.time() - start_time, 2)
            
            # Extract Metrics
            ai_answer = response.get("response", "Error")
            sources = response.get("sources", [])
            steps = response.get("steps_taken", [])
            
            # Did it self-correct?
            rewrote_query = "rewrite" in steps
            
            results.append({
                "ID": i+1,
                "Category": test["category"],
                "Question": test["query"],
                "Latency (s)": latency,
                "Sources Used": len(sources),
                "Self-Corrected": rewrote_query,
                "Answer Preview": ai_answer[:100].replace("\n", " ") + "..."
            })
            
        except Exception as e:
            print(f"Error on '{test['query']}': {e}")

    # 3. Save & Report
    df = pd.DataFrame(results)
    
    # Calculate Stats
    avg_lat = df['Latency (s)'].mean()
    correction_rate = (len(df[df['Self-Corrected']==True]) / len(df)) * 100
    
    print("\n=== BENCHMARK REPORT ===")
    print(f"Total Tests:        {len(df)}")
    print(f"Avg Latency:        {avg_lat:.2f}s")
    print(f"Self-Correction:    {correction_rate:.1f}%")
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    print("\n--- SAMPLE RESULTS ---")
    print(df[["Category", "Latency (s)", "Self-Corrected"]].head(10).to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()