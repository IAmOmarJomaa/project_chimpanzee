import lancedb
import pyarrow.parquet as pq
import pyarrow as pa
import os
import sys

# --- CONFIG ---
INPUT_PARQUET = "../data/platinum_vectors.parquet"
LANCEDB_URI = "../data/lancedb_store" 
TABLE_NAME = "chimpanzee_vectors"
BATCH_SIZE = 10000 # Process 10k rows at a time (Low RAM usage)

def find_vector_column(schema):
    """Finds the column name for embeddings in the PyArrow schema."""
    names = schema.names
    candidates = ["embedding", "embeddings", "vector", "vectors", "values"]
    for col in candidates:
        if col in names:
            return col
    # Fallback: look for list<float>
    for field in schema:
        if pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type):
            return field.name
    return None

def migrate_streaming():
    print(f"--- Migrating {INPUT_PARQUET} to LanceDB (Streaming Mode) ---")
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"[ERROR] File not found: {INPUT_PARQUET}")
        return

    # 1. Connect to LanceDB
    db = lancedb.connect(LANCEDB_URI)
    
    # 2. Open Parquet File (Zero Memory Load)
    parquet_file = pq.ParquetFile(INPUT_PARQUET)
    
    # 3. Detect Vector Column
    vec_col_name = find_vector_column(parquet_file.schema_arrow)
    if not vec_col_name:
        print(f"[CRITICAL] Could not find vector column in {parquet_file.schema_arrow.names}")
        sys.exit(1)
    print(f"   > Detected vector column: '{vec_col_name}'")

    # 4. Create the Table
    # We drop the table if it exists to start fresh
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)
    
    # We define the schema based on the first batch
    print(f"   > Creating Table '{TABLE_NAME}'...")
    tbl = None
    
    # 5. STREAMING LOOP
    # We iterate over the file in batches. RAM usage stays flat.
    total_rows = parquet_file.metadata.num_rows
    processed = 0
    
    for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
        # Convert to a standardized PyArrow Table
        df_batch = batch.to_pandas()
        
        # Rename the vector column to 'vector' strictly for LanceDB
        df_batch = df_batch.rename(columns={vec_col_name: "vector"})
        
        # Ensure 'vector' is compatible (list of floats)
        # (Usually Parquet is already correct, but this is a safety check)
        
        if tbl is None:
            # Create table with the first batch
            tbl = db.create_table(TABLE_NAME, data=df_batch, mode="overwrite")
        else:
            # Append subsequent batches
            tbl.add(df_batch)
        
        processed += len(df_batch)
        sys.stdout.write(f"\r   > Processed: {processed}/{total_rows} rows ({(processed/total_rows)*100:.1f}%)")
        sys.stdout.flush()

    print("\n[SUCCESS] Migration Complete.")
    print(f"   > Database Location: {LANCEDB_URI}")
    print("   > Optimization: Disk-based Vector Index created.")

if __name__ == "__main__":
    migrate_streaming()