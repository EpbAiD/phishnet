#!/usr/bin/env python3
"""
VM GCS Processor - Polls Cloud Storage for new URL batches and processes them.

This script runs continuously on the VM and:
1. Polls GCS queue folder for new batch files
2. Downloads batch and processes DNS/WHOIS features
3. Uploads results back to GCS incremental folder
4. Moves processed batch to GCS processed folder
"""

import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from google.cloud import storage

# Configuration
BUCKET_NAME = "phishnet-pipeline-data"
QUEUE_FOLDER = "queue"
INCREMENTAL_FOLDER = "incremental"
PROCESSED_FOLDER = "processed"
POLL_INTERVAL = 300  # 5 minutes
WORK_DIR = Path("/tmp/phishnet_processing")

# Initialize GCS client (uses VM metadata service for auth)
storage_client = storage.Client()

# Import feature extraction functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.dns_ipwhois import extract_single_domain_features as extract_dns_single
from src.features.whois import extract_single_whois_features as extract_whois_single

def log(message):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def list_queue_files():
    """List all batch files in GCS queue."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=f"{QUEUE_FOLDER}/batch_")
    return [blob.name for blob in blobs if blob.name.endswith('.csv')]

def download_batch(blob_name, local_path):
    """Download batch file from GCS."""
    log(f"Downloading {blob_name}...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    log(f"Downloaded to {local_path}")

def upload_results(local_path, blob_name):
    """Upload results to GCS."""
    log(f"Uploading {local_path} to gs://{BUCKET_NAME}/{blob_name}...")
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    log(f"Uploaded to gs://{BUCKET_NAME}/{blob_name}")

def move_to_processed(blob_name):
    """Move batch file to processed folder."""
    filename = Path(blob_name).name
    processed_name = f"{PROCESSED_FOLDER}/{filename}"
    log(f"Moving {blob_name} to processed...")
    bucket = storage_client.bucket(BUCKET_NAME)
    source_blob = bucket.blob(blob_name)
    bucket.copy_blob(source_blob, bucket, processed_name)
    source_blob.delete()
    log(f"Moved to {processed_name}")

def extract_dns_features(batch_df):
    """Extract DNS features for URLs (DNS ONLY - no URL features merged)."""
    log(f"Extracting DNS features for {len(batch_df)} URLs...")

    dns_features = []
    for idx, row in batch_df.iterrows():
        url = row['url']
        label = row.get('label', 'unknown')

        try:
            features = extract_dns_single(url)
            # Keep ONLY DNS features + url + label (no URL features)
            features['url'] = url
            features['label'] = label
            dns_features.append(features)
        except Exception as e:
            log(f"Error extracting DNS for {url}: {e}")
            dns_features.append({'url': url, 'label': label, 'error': str(e)})

        # Log progress every 100 URLs
        if (idx + 1) % 100 == 0:
            log(f"  Processed {idx + 1}/{len(batch_df)} URLs")

    result_df = pd.DataFrame(dns_features)
    log(f"DNS features extracted: {len(result_df)} rows")
    return result_df

def extract_whois_features(batch_df):
    """Extract WHOIS features for URLs (WHOIS ONLY - no URL features merged)."""
    log(f"Extracting WHOIS features for {len(batch_df)} URLs...")

    whois_features = []
    for idx, row in batch_df.iterrows():
        url = row['url']
        label = row.get('label', 'unknown')

        try:
            features = extract_whois_single(url, live_lookup=True)
            # Keep ONLY WHOIS features + url + label (no URL features)
            features['url'] = url
            features['label'] = label
            whois_features.append(features)
        except Exception as e:
            log(f"Error extracting WHOIS for {url}: {e}")
            whois_features.append({'url': url, 'label': label, 'error': str(e)})

        # Log progress every 100 URLs
        if (idx + 1) % 100 == 0:
            log(f"  Processed {idx + 1}/{len(batch_df)} URLs")

    result_df = pd.DataFrame(whois_features)
    log(f"WHOIS features extracted: {len(result_df)} rows")
    return result_df

def process_batch(blob_name):
    """Process a single batch file."""
    try:
        filename = Path(blob_name).name
        log(f"Processing {filename}...")

        # Create work directory
        WORK_DIR.mkdir(parents=True, exist_ok=True)

        # Download batch
        local_batch = WORK_DIR / filename
        download_batch(blob_name, local_batch)

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
        upload_results(dns_result, f"{INCREMENTAL_FOLDER}/dns_{timestamp}.csv")
        upload_results(whois_result, f"{INCREMENTAL_FOLDER}/whois_{timestamp}.csv")

        # Move batch to processed
        move_to_processed(blob_name)

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
    log(f"Bucket: gs://{BUCKET_NAME}")
    log(f"Queue: gs://{BUCKET_NAME}/{QUEUE_FOLDER}")
    log(f"Poll interval: {POLL_INTERVAL}s")
    log("="*60)

    while True:
        try:
            # List queue files
            queue_blobs = list_queue_files()

            if queue_blobs:
                log(f"Found {len(queue_blobs)} batch(es) to process")
                for blob_name in queue_blobs:
                    process_batch(blob_name)
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
