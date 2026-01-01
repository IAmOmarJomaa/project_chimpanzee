import polars as pl

# --- Configuration ---
INPUT_FILE = "../data/silver_production_final.parquet"

def inspect_data():
    print(f"--- Inspecting Silver Layer: {INPUT_FILE} ---")
    df = pl.read_parquet(INPUT_FILE)
    
    # Configure Polars to show full strings and lists
    with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000, fmt_table_cell_list_len=10):
        
        # 1. Inspect Guest Extraction
        print("\n--- GUEST EXTRACTION SAMPLES (nlp_guests) ---")
        # Filter for rows where guests WERE found to see if they are accurate
        guest_samples = df.filter(pl.col("nlp_guests").list.len() > 0).sample(5)
        print(guest_samples.select(["episode_id", "nlp_guests"]))

        # 2. Inspect Cleaned Text (content_body)
        print("\n--- CLEANED TEXT SAMPLES (Start of Episode) ---")
        # Show the first 200 characters of the cleaned text to verify ad removal
        text_samples = df.sample(3)
        for row in text_samples.iter_rows(named=True):
            print(f"\nID: {row['episode_id']}")
            print(f"Clean Text Start: {row['content_body'][:300]}...")
            print("-" * 50)

if __name__ == "__main__":
    inspect_data()