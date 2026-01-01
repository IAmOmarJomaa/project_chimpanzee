import polars as pl
from collections import Counter
import re

# --- Configuration ---
INPUT_FILE = "../data/bronze_lake_v3.parquet"
OUTPUT_FILE = "../data/silver_audit_v2.parquet" # Saving a V2 so we compare

# Tuning: A phrase must appear in X episodes to be considered an Ad Marker
PHRASE_EPISODE_THRESHOLD = 10 

def generate_ngrams(text: str, n=5) -> list:
    """
    Splits text into sliding window phrases of N words.
    Example: "brought to you by ting" -> ["brought to you by ting"]
    """
    # Normalize: lowercase, alpha-numeric only
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n:
        return []
    # Sliding window
    return [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]

def execute_refinement():
    print(f"--- Loading Bronze Lake: {INPUT_FILE} ---")
    df = pl.read_parquet(INPUT_FILE)
    
    print("\nStep 1: Mining Toxic Phrases (N-Grams)...")
    # We take a sample of 200 episodes to train the Ad Detector (Speed Optimization)
    # If a phrase is in 10 of 200 episodes, it's definitely an ad.
    sample_text = df.sample(n=200, seed=42)["raw_text"].to_list()
    
    phrase_counter = Counter()
    
    print(f"   > Scanning {len(sample_text)} episodes for repeated phrases...")
    for text in sample_text:
        # We only scan the first 3000 chars (Intro Zone) where ads live
        intro_zone = text[:3000]
        phrases = set(generate_ngrams(intro_zone, n=5)) # Use set to count 1 per episode
        phrase_counter.update(phrases)
        
    # Filter for Toxic Phrases
    toxic_phrases = {p for p, count in phrase_counter.items() if count >= PHRASE_EPISODE_THRESHOLD}
    
    print(f"   > DETECTED {len(toxic_phrases)} Toxic Ad Phrases (e.g., 'brought to you by').")
    if len(toxic_phrases) > 0:
        print(f"   > Sample: {list(toxic_phrases)[:5]}")

    print("\nStep 2: Applying Toxic Filter to All Episodes...")
    
    # Function to calculate density based on Phrases, not Sentences
    def check_ad_density(row_val):
        # row_val is struct: {raw_text}
        # We check the first 50 sentences (approx intro)
        text = row_val["raw_text"]
        sentences = text.split(".")[:50] # Check first 50 sentences
        
        ad_hits = 0
        if not sentences: return 0.0
        
        for sent in sentences:
            # Check if this sentence contains ANY toxic phrase
            norm_sent = sent.lower()
            for toxic in toxic_phrases:
                if toxic in norm_sent:
                    ad_hits += 1
                    break # Found one toxic phrase, sentence is dirty
        
        return (ad_hits / len(sentences)) * 100

    # Apply Logic
    df_audit = df.with_columns(
        pl.struct(["raw_text"])
        .map_elements(check_ad_density, return_dtype=pl.Float64)
        .alias("fuzzy_ad_density")
    )
    
    print("\n--- RESULTS ---")
    # Check episodes that had 0% before
    sample = df_audit.filter(pl.col("fuzzy_ad_density") > 10).head(5)
    
    print(f"Episodes with > 10% Intro Ads: {sample.height} (Sample below)")
    for row in sample.select(["episode_id", "fuzzy_ad_density"]).iter_rows():
        print(f"ID: {row[0]} | New Ad Density: {row[1]:.2f}%")

    print(f"Saving to {OUTPUT_FILE}...")
    df_audit.write_parquet(OUTPUT_FILE)

if __name__ == "__main__":
    execute_refinement()