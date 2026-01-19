#!/usr/bin/env python3
"""
Accumulate features from multiple runs
"""
import sys
import pandas as pd

def accumulate(existing_file: str, new_file: str, output_file: str):
    """Combine existing and new features, removing duplicates"""

    # Load existing
    existing = pd.read_csv(existing_file)
    print(f"Existing dataset: {len(existing)} rows")

    # Load new
    new = pd.read_csv(new_file)
    print(f"New batch: {len(new)} rows")

    # Combine and deduplicate
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=['url'], keep='last')

    print(f"Combined dataset: {len(combined)} rows (after deduplication)")

    # Save
    combined.to_csv(output_file, index=False)
    print(f"✅ Saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python accumulate_features.py <existing_file> <new_file> <output_file>")
        sys.exit(1)

    accumulate(sys.argv[1], sys.argv[2], sys.argv[3])
