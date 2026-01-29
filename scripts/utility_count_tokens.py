import tiktoken
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIR = "../data/raw_transcripts"

ENCODING = "cl100k_base" 

def count_tokens():
    print(f"--- Counting Tokens in {SOURCE_DIR} ---")
    path_obj = Path(SOURCE_DIR)
    files = list(path_obj.glob("*.txt"))
    
    if not files:
        print("No files found.")
        return

    enc = tiktoken.get_encoding(ENCODING)
    total_tokens = 0
    total_files = 0
    
    for p in tqdm(files, desc="   > Tokenizing"):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            # Exact tokenization
            tokens = enc.encode(text)
            total_tokens += len(tokens)
            total_files += 1
        except Exception:
            pass

    print("\n=== THE FLEX METRICS ===")
    print(f"Total Files:  {total_files:,}")
    print(f"Total Tokens: {total_tokens:,}")
    print("========================")

if __name__ == "__main__":
    count_tokens()