import os
import re
from pathlib import Path
from collections import defaultdict

# --- Configuration ---
SOURCE_DIRECTORY = "../data/raw_transcripts"

def analyze_filenames():
    print(f"--- Analyzing Filename Chaos in {SOURCE_DIRECTORY} ---")
    
    path_obj = Path(SOURCE_DIRECTORY)
    files = list(path_obj.glob("*.txt"))
    
    if not files:
        print("[ERROR] No files found.")
        return

    # Buckets for Statistics
    stats = defaultdict(list)
    
    # Regex Patterns for Classification
    patterns = {
        "typo_experienced": re.compile(r"Experienced\s*#", re.IGNORECASE),
        "type_mma": re.compile(r"MMA Show", re.IGNORECASE),
        "type_fight_companion": re.compile(r"Fight Companion", re.IGNORECASE),
        "format_split_part": re.compile(r"\(Part\s*(?:\d+|One|Two|Three)\)", re.IGNORECASE),
        "format_missing_hyphen": re.compile(r"Experience\s*#\d+\s+[A-Za-z]", re.IGNORECASE), # #141 Name (no hyphen)
        "format_standard_jre": re.compile(r"Experience\s*#\d+\s*-", re.IGNORECASE),
        "format_no_number": re.compile(r"^((?!#\d).)*$", re.IGNORECASE) # Matches if NO "#123" exists
    }

    # Audit Loop
    for p in files:
        name = p.name
        matched = False
        
        # 1. Check for specific "Dirty" patterns first
        if patterns["typo_experienced"].search(name):
            stats["typo_experienced"].append(name)
        
        if patterns["format_split_part"].search(name):
            stats["format_split_part"].append(name)
            
        if patterns["format_missing_hyphen"].search(name):
            stats["format_missing_hyphen"].append(name)

        # 2. Check Show Type (Broad Categories)
        if patterns["type_mma"].search(name):
            stats["type_mma"].append(name)
            matched = True
        elif patterns["type_fight_companion"].search(name):
            stats["type_fight_companion"].append(name)
            matched = True
        elif patterns["format_standard_jre"].search(name):
            stats["type_standard_jre"].append(name)
            matched = True
        
        # 3. If it didn't match a standard type, is it a ghost?
        if not matched:
            # Check if it really has no number
            if patterns["format_no_number"].search(name):
                stats["type_special_no_number"].append(name)
            else:
                stats["uncategorized_chaos"].append(name)

    # --- THE REPORT ---
    print("\n=== THE CHAOS REPORT ===")
    print(f"Total Files: {len(files)}\n")
    
    print(f"--- TYPOS & FORMATTING ERRORS ---")
    print(f"1. 'Experienced' Typos:      {len(stats['typo_experienced'])}")
    print(f"2. Missing Hyphens (#123 Name): {len(stats['format_missing_hyphen'])}")
    print(f"3. Split Episodes (Part 1/2): {len(stats['format_split_part'])}")
    
    print(f"\n--- SHOW TYPES ---")
    print(f"4. Standard JRE Episodes:    {len(stats['type_standard_jre'])}")
    print(f"5. MMA Show Episodes:        {len(stats['type_mma'])}")
    print(f"6. Fight Companions:         {len(stats['type_fight_companion'])}")
    print(f"7. Specials (No Numbers):    {len(stats['type_special_no_number'])}")
    
    print(f"\n--- UNCATEGORIZED (The Weirdest Files) ---")
    print(f"Count: {len(stats['uncategorized_chaos'])}")
    if stats['uncategorized_chaos']:
        print("Sample Weird Files:")
        for f in stats['uncategorized_chaos'][:5]:
            print(f" - {f}")

    print(f"\n--- SAMPLE OF TYPOS (To Decide Fix) ---")
    if stats['typo_experienced']:
        print(f"Example 'Experienced': {stats['typo_experienced'][0]}")
    if stats['format_missing_hyphen']:
        print(f"Example Missing Hyphen: {stats['format_missing_hyphen'][0]}")

if __name__ == "__main__":
    analyze_filenames()