import polars as pl
import networkx as nx

# --- Config ---
NODES_FILE = "../data/hypergraph_nodes.parquet"
EDGES_FILE = "../data/hypergraph_edges.parquet" # Folder or file
OUTPUT_GEPHI = "../data/sample_graph.gexf"

def export_sample_to_gephi():
    print("--- Reading Graph Data ---")
    
    # 1. Load Edges (Just the first partition to get a sample)
    # We assume the parquet folder contains part_0.parquet, etc.
    # We grab the first file we find to be fast.
    import glob
    edge_files = glob.glob(f"{EDGES_FILE}/*.parquet")
    if not edge_files:
        print("[ERROR] No edge files found.")
        return
        
    print(f"Sampling from: {edge_files[0]}")
    edges_df = pl.read_parquet(edge_files[0]).head(5000) # Get 5000 connections
    
    # 2. Build Mini-Graph
    G = nx.Graph()
    
    print("Building NetworkX Graph...")
    for row in edges_df.iter_rows(named=True):
        G.add_edge(row['source'], row['target'], weight=row['weight'])
        # Add a "type" attribute so we can color them in Gephi
        # (We don't have node types in the edge file, but we can guess or leave blank)
        
    # 3. Save
    print(f"Saving {OUTPUT_GEPHI}...")
    nx.write_gexf(G, OUTPUT_GEPHI)
    print("[SUCCESS] Open 'sample_graph.gexf' in Gephi to visualize your data.")

if __name__ == "__main__":
    export_sample_to_gephi()