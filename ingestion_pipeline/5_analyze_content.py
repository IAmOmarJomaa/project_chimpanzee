import polars as pl
import spacy
from collections import Counter
import re

# --- Configuration ---
INPUT_FILE = "../data/bronze_lake_v3.parquet"
OUTPUT_FILE = "../data/silver_audit.parquet"

# Load NLP Model (Small & Fast)
print("Loading NLP Model...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[ERROR] Model not found. Run: python -m spacy download en_core_web_sm")
    exit(1)

def extract_people_with_context(text: str, window=10) -> list:
    """
    Your Innovation: Extracts names AND the context around them.
    Only scans the first 3000 characters (Intro Zone) to save speed.
    """
    if not text: return []
    
    # Limit to Intro Zone
    intro_text = text[:3000] 
    
    # Spacy has a limit on text length, 3000 is safe
    doc = nlp(intro_text)
    
    candidates = []
    seen_names = set()
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Basic filter: ignore short names or duplicates
            if len(name) < 4 or name in seen_names:
                continue
                
            seen_names.add(name)
            
            # Grab context windows safely
            start = max(ent.start - window, 0)
            end = min(ent.end + window, len(doc))
            snippet = doc[start:end].text.replace("\n", " ").strip()
            candidates.append(f"{name} | Context: '...{snippet}...'")
            
    return candidates[:5] # Return top 5 detected people

def calculate_ad_metrics(sentences: list, global_ad_lines: set) -> tuple:
    """
    Returns (ad_density_percentage, is_heavy_ads_bool)
    """
    if not sentences or len(sentences) == 0: return (0.0, False)
    
    ad_count = 0
    # normalization logic must match the previous step
    for s in sentences:
        norm = "".join(c for c in s.lower() if c.isalnum())
        if norm in global_ad_lines:
            ad_count += 1
            
    density = (ad_count / len(sentences)) * 100
    return (round(density, 2), density > 25.0)

def execute_audit():
    print(f"--- Loading Bronze Lake: {INPUT_FILE} ---")
    df = pl.read_parquet(INPUT_FILE)
    print(f"Loaded {df.height} episodes.")

    # --- 1. Global Frequency Analysis (The Set Theory) ---
    print("\nStep 1/3: Calculating Sentence Frequency (finding Ads)...")
    
    # Flatten all sentences from all episodes
    print("   > Exploding sentences (this may take a moment)...")
    all_sentences = df.select(pl.col("sentences").explode()).drop_nulls()
    
    # Normalize
    print("   > Normalizing corpus...")
    raw_list = all_sentences["sentences"].to_list()
    # Normalize: lowercase, remove punctuation. Only keep sentences > 20 chars
    norm_list = ["".join(c for c in s.lower() if c.isalnum()) for s in raw_list if len(s) > 20]
    
    print(f"   > Counting repetitions across {len(norm_list)} sentences...")
    counts = Counter(norm_list)
    
    # Strict Threshold: If a line appears in > 15 episodes, it is an Ad.
    ad_lines = {line for line, count in counts.items() if count > 15}
    print(f"   > DETECTED {len(ad_lines)} unique Ad/Boilerplate lines.")

    # --- 2. NLP & Context Extraction ---
    print("\nStep 2/3: Extracting Guest Context & Ad Density...")
    
    # We use a pure python loop for the complex row logic (easier to debug than map_elements)
    # Then we construct a new dataframe to join back
    
    audit_results = []
    
    # Iterate over rows
    rows = df.select(["episode_id", "raw_text", "sentences"]).iter_rows(named=True)
    
    # We only process the first 10 for the "Sample" display, but process ALL for the file
    # Adding a progress bar
    for i, row in enumerate(rows):
        if i % 100 == 0: print(f"   Processing row {i}...", end="\r")
        
        candidates = extract_people_with_context(row["raw_text"])
        density, is_heavy = calculate_ad_metrics(row["sentences"], ad_lines)
        
        audit_results.append({
            "episode_id": row["episode_id"],
            "nlp_guest_candidates": candidates,
            "ad_density_pct": density,
            "flag_heavy_ads": is_heavy
        })
    
    print("\n") # Newline after progress

    # Convert results to DataFrame
    audit_df = pl.DataFrame(audit_results)

    # --- 3. Join & Save ---
    print("Step 3/3: Merging & Saving...")
    
    # Join the new audit data back to the main dataframe
    silver_df = df.join(audit_df, on="episode_id", how="left")
    
    print(f"--- Audit Complete ---")
    
    # Show Sample of what we found
    print("\n--- SAMPLE RESULTS ---")
    sample = silver_df.filter(pl.col("ad_density_pct") > 0).head(3)
    for row in sample.iter_rows(named=True):
        print(f"ID: {row['episode_id']}")
        print(f"Ad Density: {row['ad_density_pct']}%")
        print(f"NLP Candidates: {row['nlp_guest_candidates']}")
        print("-" * 30)

    print(f"Saving to {OUTPUT_FILE}...")
    silver_df.write_parquet(OUTPUT_FILE)

if __name__ == "__main__":
    execute_audit()