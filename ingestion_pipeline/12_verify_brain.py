import polars as pl
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# --- Configuration ---
INPUT_FILE = "../data/platinum_vectors.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"

def cosine_similarity(query_vec, corpus_vecs):
    # Normalize query vector to unit length
    query_norm = query_vec / np.linalg.norm(query_vec)
    # Corpus vectors are already normalized by our previous script (normalize_embeddings=True)
    # So Dot Product == Cosine Similarity
    return np.dot(corpus_vecs, query_norm)

def interactive_search():
    print("--- Loading The Brain (Platinum Layer) ---")
    start = time.time()
    df = pl.read_parquet(INPUT_FILE)
    print(f"   > Database loaded in {time.time() - start:.2f}s")
    print(f"   > Total Memories: {df.height}")

    print("--- Loading Model ---")
    model = SentenceTransformer(MODEL_NAME)
    
    # Extract the vector column as a massive numpy matrix
    # This is fast: 400k x 384 matrix takes ~600MB RAM
    print("--- indexing vectors ---")
    vector_matrix = np.stack(df["vector"].to_numpy())
    
    print("\n[SYSTEM ONLINE]")
    print("Type 'exit' to quit.\n")
    
    while True:
        query = input("Ask Joe >> ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        # 1. Vectorize Query
        query_vec = model.encode(query)
        
        # 2. Search (Math)
        start_search = time.time()
        scores = cosine_similarity(query_vec, vector_matrix)
        
        # 3. Rank (Get Top 3)
        # argpartition is faster than sort for finding top k
        top_k_indices = np.argpartition(scores, -3)[-3:]
        # Sort top k by score descending
        top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])][::-1]
        
        search_time = time.time() - start_search
        print(f"\n--- Results (found in {search_time:.4f}s) ---")
        
        for idx in top_k_indices:
            row = df.row(idx, named=True)
            score = scores[idx]
            
            print(f"\n[Match: {score:.4f}] Episode: {row['episode_id']}")
            print(f"Guest: {row['guest_list']}")
            print(f"Chaos: {row['episode_chaos']:.1f} | Visual?: {row['has_visual_context']}")
            print(f"Text: \"{row['text_content']}...\"")
            print("-" * 50)
        print("\n")

if __name__ == "__main__":
    interactive_search()