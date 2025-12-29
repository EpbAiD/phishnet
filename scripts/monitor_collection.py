#!/usr/bin/env python3
"""
Collection Monitoring Dashboard
================================
Real-time monitoring of continuous data collection.

Shows:
- Collection rate (URLs/hour)
- Data quality metrics
- Error rates
- Next auto-retrain estimate
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = "data/vm_collected"
CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint.json"

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def analyze_collection_data():
    """Analyze collected data and show statistics"""

    print("=" * 80)
    print("📊 PhishNet Data Collection Dashboard")
    print("=" * 80)

    # Load checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n✅ Last checkpoint: {checkpoint.get('timestamp', 'N/A')}")
        print(f"📦 Total URLs processed: {checkpoint.get('processed_count', 0):,}")
        print(f"🔄 URLs since last retrain: {checkpoint.get('processed_count', 0) - checkpoint.get('last_retrain', 0):,}")

    # Analyze CSV files
    whois_files = list(Path(OUTPUT_DIR).glob("whois_results_*.csv"))
    dns_files = list(Path(OUTPUT_DIR).glob("dns_results_*.csv"))

    print(f"\n📁 Data files:")
    print(f"  - WHOIS files: {len(whois_files)}")
    print(f"  - DNS files: {len(dns_files)}")

    if whois_files:
        # Load all WHOIS data
        whois_dfs = [pd.read_csv(f) for f in whois_files]
        whois_df = pd.concat(whois_dfs, ignore_index=True)

        print(f"\n🔍 WHOIS Data Quality:")
        print(f"  - Total records: {len(whois_df):,}")
        if 'label' in whois_df.columns:
            print(f"  - Phishing: {len(whois_df[whois_df['label']=='phishing']):,}")
            print(f"  - Legitimate: {len(whois_df[whois_df['label']=='legitimate']):,}")

        # Check for null values
        null_pct = (whois_df.isnull().sum().sum() / (len(whois_df) * len(whois_df.columns))) * 100
        print(f"  - Null values: {null_pct:.1f}%")

        # Collection rate
        if 'collected_at' in whois_df.columns:
            whois_df['collected_at'] = pd.to_datetime(whois_df['collected_at'])
            time_range = (whois_df['collected_at'].max() - whois_df['collected_at'].min()).total_seconds() / 3600
            if time_range > 0:
                rate = len(whois_df) / time_range
                print(f"\n⚡ Collection Rate: {rate:.1f} URLs/hour")

    if dns_files:
        dns_dfs = [pd.read_csv(f) for f in dns_files]
        dns_df = pd.concat(dns_dfs, ignore_index=True)

        print(f"\n🌐 DNS Data Quality:")
        print(f"  - Total records: {len(dns_df):,}")
        null_pct = (dns_df.isnull().sum().sum() / (len(dns_df) * len(dns_df.columns))) * 100
        print(f"  - Null values: {null_pct:.1f}%")

    # Estimate time to next retrain
    if checkpoint:
        remaining = 1000 - (checkpoint.get('processed_count', 0) - checkpoint.get('last_retrain', 0))
        if remaining > 0:
            print(f"\n🎯 Progress to Next Auto-Retrain:")
            print(f"  - URLs remaining: {remaining:,}")
            print(f"  - Progress: {((1000-remaining)/1000*100):.1f}%")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    analyze_collection_data()
