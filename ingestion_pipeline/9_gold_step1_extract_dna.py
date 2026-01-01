"""
MODULE: Gold Layer (Feature Extraction)
DESCRIPTION:
    Extracts the "DNA" of an episode.
    1. Sentiment Analysis (TextBlob).
    2. Structural Metrics (Chaos & Complexity Scores).
    3. Topic Modeling (Entity Extraction).
    4. "Jamie" Detection (Visual context triggers).

ARCHITECTURE:
    - Input: Silver Parquet (Cleaned Text)
    - Output: Gold DNA Parquet (Rich Metadata)
    - Logic: NLP Tokenization + Statistical Heuristics
"""

import polars as pl
from textblob import TextBlob
import numpy as np
import spacy
from tqdm import tqdm
import yaml
from pathlib import Path

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    exit(1)

INPUT_FILE = config["paths"]["silver"]
OUTPUT_FILE = config["paths"]["gold"]

# --- CONSTANTS ---
# Specific triggers for this analysis logic
VISUAL_TRIGGERS = [
    "pull that up", "pull it up", "look at that", "see that", "show me that",
    "google that", "look at this", "watch this", "on the screen", "jamie"
]

print("Loading NLP Model...")
try:
    nlp = spacy.load(config["nlp"]["spacy_model"])
    # We disable the 'ner' (Named Entity Recognition) pipe for the sentence splitting part 
    # but keep it for topic extraction to save memory if needed.
    # For this script, we need everything.
except:
    print(f"[ERROR] Model {config['nlp']['spacy_model']} not found.")
    exit(1)

def analyze_dna(text: str):
    """
    Extracts the 'Soul' of the episode: Chaos, Complexity, Vibe, and Structure.
    Uses Spacy for accurate tokenization instead of naive splitting.
    """
    if not text or len(text) < 100:
        return {
            "chaos_score": 0.0, "complexity_score": 0.0, 
            "sentiment": 0.0, "visual_event_count": 0,
            "top_concepts": [], "business_start_index": 0
        }
    
    # --- 1. LINGUISTIC ANALYSIS (Vibe) ---
    # Sample first 50k chars for sentiment to keep it fast
    blob = TextBlob(text[:50000]) 
    sentiment = blob.sentiment.polarity
    
    # --- 2. STRUCTURAL ANALYSIS (Chaos & Complexity) ---
    # REFACTOR: Using Spacy doc to get accurate sentence lengths
    doc = nlp(text[:100000]) # Limit to 100k chars for metrics to avoid RAM spikes
    
    sent_lens = [len(sent) for sent in doc.sents if len(sent) > 3]
    
    if not sent_lens:
        return {
            "chaos_score": 0.0, "complexity_score": 0.0, 
            "sentiment": round(sentiment, 2), "visual_event_count": 0,
            "top_concepts": [], "business_start_index": 0
        }
    
    # CHAOS: High Variance + Short Sentences = Chaos
    avg_len = np.mean(sent_lens)
    variance = np.std(sent_lens)
    # Protected division
    chaos_score = (variance / (avg_len + 1)) * 10
    
    # COMPLEXITY: Unique Lemmas / Total Words (Lexical Diversity)
    # We use lemmas (root words) to count true vocabulary size
    lemmas = [token.lemma_ for token in doc if token.is_alpha]
    vocab_richness = len(set(lemmas)) / len(lemmas) if lemmas else 0
    complexity_score = (vocab_richness * 100) + (avg_len * 0.5)

    # --- 3. JAMIE DETECTOR (Visual Context) ---
    visual_count = 0
    lower_text = text.lower()
    for trigger in VISUAL_TRIGGERS:
        visual_count += lower_text.count(trigger)

    # --- 4. TOPIC EXTRACTION (Graph Nodes) ---
    # Extract Entities (PERSON, ORG, GPE) from the first 15k chars
    # We reuse the 'doc' object if possible, or process a smaller chunk
    concept_doc = nlp(text[:15000])
    
    concepts = []
    for ent in concept_doc.ents:
        if ent.label_ in ["ORG", "GPE", "EVENT", "WORK_OF_ART", "PERSON"]:
            clean_ent = ent.text.strip().lower()
            # Basic stopwords filtering
            if len(clean_ent) > 3 and clean_ent not in ["youtube", "google", "instagram"]:
                concepts.append(clean_ent)
    
    # Top 8 frequent concepts
    from collections import Counter
    concept_counts = Counter(concepts)
    top_concepts = [c for c, _ in concept_counts.most_common(8)]

    # --- 5. STAGING DETECTOR (Business Start) ---
    # Find where the banter ends. 
    # Heuristic: First block of sentences with substantial length.
    business_idx = 0
    sentences = list(doc.sents)
    window_size = 10
    
    for i in range(0, len(sentences) - window_size, 5):
        window = sentences[i : i + window_size]
        # Check average token count in this window
        w_lens = [len(s) for s in window]
        if np.mean(w_lens) > 15: # Threshold for "Real Talk"
            business_idx = window[0].start_char
            break

    return {
        "chaos_score": round(chaos_score, 2),
        "complexity_score": round(complexity_score, 2),
        "sentiment": round(sentiment, 2),
        "visual_event_count": visual_count,
        "top_concepts": top_concepts,
        "business_start_index": business_idx
    }

def execute_gold_extraction():
    print(f"--- Loading Silver Layer: {INPUT_FILE} ---")
    if not Path(INPUT_FILE).exists():
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    df = pl.read_parquet(INPUT_FILE)
    
    print("\n[Phase 1] Extracting DNA (Style, Visuals, Topics)...")
    records = df.to_dicts()
    gold_results = []
    
    for row in tqdm(records, desc="   > Sequencing"):
        metrics = analyze_dna(row["content_body"])
        row.update(metrics)
        gold_results.append(row)
        
    gold_df = pl.DataFrame(gold_results)
    
    print(f"\n[Phase 2] Analysis Complete.")
    print("   > Top 3 'High Complexity':")
    print(gold_df.sort("complexity_score", descending=True).select(["episode_id", "complexity_score"]).head(3))

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    gold_df.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f"\nSaved Gold DNA to {OUTPUT_FILE}")

if __name__ == "__main__":
    execute_gold_extraction()