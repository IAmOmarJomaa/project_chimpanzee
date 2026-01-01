import os
from pathlib import Path

# --- Configuration ---
SOURCE_DIRECTORY = "../data/raw_transcripts"
OUTPUT_REPORT = "filename_audit.txt"

def scan_raw_files():
    print(f"--- Starting Diagnostic Scan of {SOURCE_DIRECTORY} ---")
    
    path_obj = Path(SOURCE_DIRECTORY)
    files = list(path_obj.glob("*.txt"))
    
    if not files:
        print("[ERROR] No files found. Check your path.")
        return

    # Metrics
    total_files = len(files)
    empty_files = []
    read_errors = []
    filenames = []

    print(f"Scanning {total_files} files for damage and naming patterns...")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as report:
        report.write(f"=== DIAGNOSTIC REPORT ===\n")
        report.write(f"Total Files Scanned: {total_files}\n\n")

        # 1. Health Check Loop
        for p in files:
            # A. Check Size (Empty/Near Empty)
            size_bytes = p.stat().st_size
            if size_bytes < 500: # Less than 500 bytes is likely just a header or empty
                empty_files.append(f"{p.name} ({size_bytes} bytes)")
            
            # B. Check Readability (Corruption)
            try:
                # We just try to read the first 100 chars to ensure file isn't locked/corrupted
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    f.read(100)
            except Exception as e:
                read_errors.append(f"{p.name} - Error: {str(e)}")

            # C. Collect Name
            filenames.append(p.name)

        # 2. Write Health Issues
        report.write(f"=== HEALTH AUDIT ===\n")
        if empty_files:
            report.write(f"\n[CRITICAL] Found {len(empty_files)} Empty/Suspicious Files:\n")
            for f in empty_files:
                report.write(f" - {f}\n")
        else:
            report.write("\n[PASS] No empty files detected.\n")

        if read_errors:
            report.write(f"\n[CRITICAL] Found {len(read_errors)} Corrupted/Unreadable Files:\n")
            for f in read_errors:
                report.write(f" - {f}\n")
        else:
            report.write("\n[PASS] All files are readable.\n")

        # 3. Write All Filenames (For Regex Analysis)
        report.write(f"\n=== FILENAME DUMP (First 50 & Weird Patterns) ===\n")
        # We write ALL names to the file so we can analyze them, but organized
        report.write("\n--- ALL FILENAMES ---\n")
        for name in sorted(filenames):
            report.write(f"{name}\n")

    print(f"--- Scan Complete ---")
    print(f"Report saved to: {os.path.abspath(OUTPUT_REPORT)}")
    print(f"1. Open '{OUTPUT_REPORT}'")
    print(f"2. Copy the 'HEALTH AUDIT' section and the first few lines of 'ALL FILENAMES' and paste them here.")

if __name__ == "__main__":
    scan_raw_files()