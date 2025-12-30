#!/usr/bin/env python3
"""
VM GCS Processor - Polls Cloud Storage for new URL batches and processes them.

This script runs continuously on the VM and:
1. Polls GCS queue folder for new batch files
2. Downloads batch and processes DNS/WHOIS features
3. Uploads results back to GCS incremental folder
4. Moves processed batch to GCS processed folder
"""

import os
import sys
import time
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configuration
GCS_BUCKET = "gs://phishnet-pipeline-data"
GCS_QUEUE = f"{GCS_BUCKET}/queue"
GCS_INCREMENTAL = f"{GCS_BUCKET}/incremental"
GCS_PROCESSED = f"{GCS_BUCKET}/processed"
POLL_INTERVAL = 300  # 5 minutes
WORK_DIR = Path("/tmp/phishnet_processing")

# Import feature extraction functions (assuming they exist in the repo)
sys.path.insert(0, str(Path(__file__).parent.parent))

def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def run_command(cmd, check=True):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {cmd}")
        log(f"Error: {e.stderr}")
        if check:
            raise
        return None

def list_queue_files():
    """List all batch files in GCS queue."""
    output = run_command(f"gcloud storage ls {GCS_QUEUE}/batch_*.csv 2>/dev/null", check=False)
    if not output:
        return []
    return [line.strip() for line in output.split('\n') if line.strip()]

def download_batch(gcs_path, local_path):
    """Download batch file from GCS."""
    log(f"Downloading {gcs_path}...")
    run_command(f"gcloud storage cp {gcs_path} {local_path}")
    log(f"Downloaded to {local_path}")

def upload_results(local_path, gcs_path):
    """Upload results to GCS."""
    log(f"Uploading {local_path} to {gcs_path}...")
    run_command(f"gcloud storage cp {local_path} {gcs_path}")
    log(f"Uploaded to {gcs_path}")

def move_to_processed(gcs_path):
    """Move batch file to processed folder."""
    filename = Path(gcs_path).name
    processed_path = f"{GCS_PROCESSED}/{filename}"
    log(f"Moving {gcs_path} to processed...")
    run_command(f"gcloud storage mv {gcs_path} {processed_path}")
    log(f"Moved to {processed_path}")

def extract_dns_features(batch_df):
    """Extract DNS features for URLs."""
    # TODO: Implement actual DNS feature extraction
    # For now, simulate processing
    log(f"Extracting DNS features for {len(batch_df)} URLs...")
    time.sleep(2)  # Simulate work

    # In production, this would call your actual DNS extraction code
    # dns_features = extract_dns_features_from_dataframe(batch_df)

    log("DNS features extracted")
    return batch_df  # Return with added DNS columns

def extract_whois_features(batch_df):
    """Extract WHOIS features for URLs."""
    # TODO: Implement actual WHOIS feature extraction
    # For now, simulate processing
    log(f"Extracting WHOIS features for {len(batch_df)} URLs...")
    time.sleep(2)  # Simulate work

    # In production, this would call your actual WHOIS extraction code
    # whois_features = extract_whois_features_from_dataframe(batch_df)

    log("WHOIS features extracted")
    return batch_df  # Return with added WHOIS columns

def process_batch(batch_file):
    """Process a single batch file."""
    try:
        filename = Path(batch_file).name
        log(f"Processing {filename}...")

        # Create work directory
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        # Download batch
        local_batch = WORK_DIR / filename
        download_batch(batch_file, local_batch)

        # Read batch
        df = pd.read_csv(local_batch)
        log(f"Loaded {len(df)} URLs from batch")

        # Extract features
        dns_df = extract_dns_features(df.copy())
        whois_df = extract_whois_features(df.copy())

        # Save results
        timestamp = filename.replace("batch_", "").replace(".csv", "")
        dns_result = WORK_DIR / f"dns_{timestamp}.csv"
        whois_result = WORK_DIR / f"whois_{timestamp}.csv"

        dns_df.to_csv(dns_result, index=False)
        whois_df.to_csv(whois_result, index=False)

        log(f"Saved DNS results: {len(dns_df)} rows")
        log(f"Saved WHOIS results: {len(whois_df)} rows")

        # Upload results to GCS
        upload_results(dns_result, f"{GCS_INCREMENTAL}/dns_{timestamp}.csv")
        upload_results(whois_result, f"{GCS_INCREMENTAL}/whois_{timestamp}.csv")

        # Move batch to processed
        move_to_processed(batch_file)

        # Cleanup local files
        local_batch.unlink(missing_ok=True)
        dns_result.unlink(missing_ok=True)
        whois_result.unlink(missing_ok=True)

        log(f"✅ Batch {filename} processed successfully!")
        return True

    except Exception as e:
        log(f"❌ Error processing {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main polling loop."""
    log("="*60)
    log("VM GCS Processor Started")
    log("="*60)
    log(f"Bucket: {GCS_BUCKET}")
    log(f"Queue: {GCS_QUEUE}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    log("="*60)

    while True:
        try:
            # List queue files
            queue_files = list_queue_files()

            if queue_files:
                log(f"Found {len(queue_files)} batch(es) to process")
                for batch_file in queue_files:
                    process_batch(batch_file)
            else:
                log("No batches in queue, waiting...")

            # Wait before next poll
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log("\nShutting down gracefully...")
            break
        except Exception as e:
            log(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()
