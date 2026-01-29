"""
MODULE: Silver Layer (Refinement)
DESCRIPTION:
    Takes the raw 'Bronze' data and performs "Hygiene" operations:
    1. Ad-Blocker: Identifies and removes sponsor reads using N-Gram analysis.
    2. Diarization: Extracts guest names using NLP.
    3. Segmentation: Re-segments sentences for cleaner reading.

ARCHITECTURE:
    - Input: Bronze Parquet (Raw)
    - Output: Silver Parquet (Cleaned)
    - Logic: Heuristic Ad-Detection + Config-based rules.
"""

import polars as pl
import spacy
from collections import Counter
import re
import yaml
from tqdm import tqdm
from pathlib import Path

# --- LOAD CONFIG ---
try:
    with open("../config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] Config file missing.")
    exit(1)

INPUT_FILE = config["paths"]["bronze"]
OUTPUT_FILE = config["paths"]["silver"]

# Load Rules from Config (No more magic strings!)
KNOWN_SPONSORS = set(config["ad_detection"]["known_sponsors"])
SALES_KEYWORDS = set(config["ad_detection"]["sales_keywords"])
NGRAM_SIZE = 6

# --- STATIC RULES ---
# These are linguistic rules, not data, so they can stay in code (or move to config if strict)
BOILERPLATE_INTRO = {
    "joe rogan podcast", "check it out", "the joe rogan experience", 
    "train by day", "joe rogan podcast by night", "all day", 
    "hello freak bitches", "welcome to the show"
}

SAFE_STARTERS = {
    "i ", "you ", "we ", "they ", "he ", "she ", "it ", "that ", "and ", 
    "but ", "so ", "yeah", "no ", "what", "how", "why", "when"
}

BLOCKLIST_ENTITIES = {
    "alpha", "alpha brain", "onnit", "squarespace", "cash app", "vpn", 
    "fleshlight", "ting", "dollar shave club", "youtube", "twitter", 
    "instagram", "facebook", "google", "alexa", "siri", "clip", "spotify",
    "zoom", "skype", "iphone", "android"
}

print("Loading NLP Model...")
try:
    nlp = spacy.load(config["nlp"]["spacy_model"])
    nlp.max_length = 3000000 
except:
    print(f"[ERROR] Model {config['nlp']['spacy_model']} not found.")
    exit(1)

def generate_ngrams(text: str, n=6) -> set:
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < n: return set()
    return {" ".join(words[i:i+n]) for i in range(len(words)-n+1)}

def is_phrase_safe(phrase: str) -> bool:
    if any(k in phrase for k in SALES_KEYWORDS): return False
    for starter in SAFE_STARTERS:
        if phrase.startswith(starter): return True
    return False

def extract_toxic_phrases(df: pl.DataFrame) -> set:
    print(f"\n[Phase 1] Mining Toxic Ad Phrases ({NGRAM_SIZE}-grams)...")
    # Sampling for speed
    raw_texts = df.sample(n=min(500, df.height), seed=42)["raw_text"].to_list()
    phrase_counter = Counter()
    
    print(f"   > Scanning sample of {len(raw_texts)} episodes...")
    for text in tqdm(raw_texts, desc="   > Learning Ads"):
        # Only scan the first 6000 chars (Intro Zone)
        intro_zone = text[:6000] 
        phrases = generate_ngrams(intro_zone, n=NGRAM_SIZE)
        phrase_counter.update(phrases)
        
    candidates = {p for p, count in phrase_counter.items() if count >= 5}
    
    toxic = set()
    for p in candidates:
        if not is_phrase_safe(p):
            toxic.add(p)
    
    print(f"   > SUCCESS: Identified {len(toxic)} toxic phrases.")
    return toxic

def clean_text_smart(text, toxic_phrases):
    """
    Uses SpaCy to split sentences intelligently, preventing data loss.
    """
    if not text: return "", 0, []
    
    doc = nlp(text)
    clean_sentences = []
    dropped_count = 0
    guests = []
    seen_names = set()
    
    for sent in doc.sents:
        s_text = sent.text.strip()
        if not s_text: continue
        
        # --- AD DETECTION ---
        norm_sent = s_text.lower()
        clean_norm = re.sub(r'[^\w\s]', '', norm_sent)
        
        is_ad = False
        
        # Rule 1: Boilerplate
        if clean_norm in BOILERPLATE_INTRO:
            is_ad = True

        # Rule 2: Sponsor + Sales Keyword
        if not is_ad:
            if any(s in norm_sent for s in KNOWN_SPONSORS) and any(k in norm_sent for k in SALES_KEYWORDS):
                is_ad = True
        
        # Rule 3: Toxic N-Grams
        if not is_ad and len(norm_sent) > 30:
            sent_ngrams = generate_ngrams(norm_sent, n=NGRAM_SIZE)
            if not sent_ngrams.isdisjoint(toxic_phrases):
                is_ad = True
                
        if is_ad:
            dropped_count += 1
            continue
            
        clean_sentences.append(s_text)
        
        # --- GUEST EXTRACTION ---
        # Scan first 10k chars for names
        if sent.start_char < 10000:
            for ent in sent.ents:
                if ent.label_ == "PERSON":
                    name = ent.text.strip()
                    if len(name) > 3 and name.lower() not in BLOCKLIST_ENTITIES and name not in seen_names:
                        guests.append(f"{name}")
                        seen_names.add(name)

    return " ".join(clean_sentences), dropped_count, guests[:5]

def execute_silver_run():
    print(f"--- Loading Bronze Lake: {INPUT_FILE} ---")
    if not Path(INPUT_FILE).exists():
        print(f"[ERROR] {INPUT_FILE} not found. Did you run the ingestion step?")
        return

    df = pl.read_parquet(INPUT_FILE)
    
    toxic_phrases = extract_toxic_phrases(df)
    
    print("\n[Phase 2] Smart Cleaning (SpaCy Segmentation)...")
    records = df.to_dicts()
    processed_data = []
    
    for row in tqdm(records, desc="   > Processing"):
        clean_body, drops, guests = clean_text_smart(row["raw_text"], toxic_phrases)
        
        row["content_body"] = clean_body
        row["ads_removed_count"] = drops
        row["nlp_guests"] = guests
        
        # Drop raw text to save space in Silver
        del row["raw_text"]
        if "sentences" in row: del row["sentences"]
        processed_data.append(row)
        
    silver_df = pl.DataFrame(processed_data)
    
    print(f"\n[Phase 3] Saving V3 Data...")
    print(f"   > Avg Ads Removed: {silver_df['ads_removed_count'].mean():.1f}")
    
    silver_df.write_parquet(OUTPUT_FILE, compression="zstd")
    print(f"SUCCESS. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    execute_silver_run()