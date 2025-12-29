#!/usr/bin/env python3
"""
URL Feature Extractor - Extract local URL features
==================================================
Extracts 39 URL structure features instantly (no network calls).
Called by daily_url_collector.sh
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.url_features import URLFeatureExtractor


def extract_url_features(input_file: str, output_file: str):
    """
    Extract URL features from CSV file.

    Args:
        input_file: CSV with 'url', 'label', 'source' columns
        output_file: Output CSV with URL features added

    Returns:
        Number of URLs processed
    """
    print(f"Extracting URL features...")
    print(f"  Input: {input_file}")

    # Read URLs
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} URLs")

    # Extract URL features
    extractor = URLFeatureExtractor()
    df_feats = extractor.transform_dataframe(df)

    # Preserve label and source columns
    if 'label' in df.columns:
        df_feats['label'] = df['label'].values
    if 'source' in df.columns:
        df_feats['source'] = df['source'].values

    # Save with features
    df_feats.to_csv(output_file, index=False)

    print(f"  ✅ Extracted {df_feats.shape[1]} features per URL")
    print(f"  Saved to: {output_file}")

    return len(df_feats)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 extract_url_features.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    count = extract_url_features(input_file, output_file)
    sys.exit(0 if count > 0 else 1)
