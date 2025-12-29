#!/usr/bin/env python3
"""
VM Daily Processor - OPTIMIZED with Parallelization and Timeouts
============================================================================
Extract DNS and WHOIS features for queued URLs with:
- Parallel processing (10 URLs at a time)
- Per-URL timeout (30 seconds max)
- Progress logging every 100 URLs
- Automatic queue watching (continuous mode)
============================================================================
"""

import os
import sys
import time
import glob
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

# ============================================================================
# Configuration
# ============================================================================
QUEUE_DIR = "vm_data/url_queue"
OUTPUT_DIR = "vm_data/incremental"
READY_DIR = "vm_data/ready"
LOG_DIR = "logs"

# Performance tuning
MAX_WORKERS = 10  # Process 10 URLs in parallel
PER_URL_TIMEOUT = 30  # Max 30 seconds per URL (DNS + WHOIS combined)
CHECK_INTERVAL = 60  # Check for new batches every 60 seconds

# Create directories
os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(READY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
DATE = datetime.now().strftime('%Y%m%d')
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler(f'{LOG_DIR}/vm_processor_optimized_{DATE}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Single URL Processing with Timeout
# ============================================================================
def process_single_url(url_data):
    """
    Process a single URL (DNS + WHOIS extraction).
    Returns dict with features or None if failed/timeout.

    This function runs in a worker thread with timeout enforced by executor.
    """
    url = url_data['url']
    label = url_data.get('label', 'unknown')

    result = {
        'url': url,
        'label': label,
        'dns_features': None,
        'whois_features': None,
        'error': None
    }

    try:
        # DNS extraction (mode='single' = no delays, fast)
        dns_feats = extract_single_domain_features(url)
        result['dns_features'] = dns_feats

        # WHOIS extraction (mode='single' = no retries, fast)
        whois_feats = extract_single_whois_features(url, live_lookup=True)
        result['whois_features'] = whois_feats

    except Exception as e:
        result['error'] = str(e)

    return result


# ============================================================================
# Feature Validation
# ============================================================================
def validate_features(features, feature_type):
    """
    Validate extracted features to ensure they make sense.

    Returns: (is_valid, reason)
    """
    if not features or not isinstance(features, dict):
        return False, "Empty or invalid features"

    # Check for all None/null values
    non_null_count = sum(1 for v in features.values() if v is not None)
    if non_null_count == 0:
        return False, "All features are None"

    # Check for invalid numeric values
    for key, value in features.items():
        if isinstance(value, (int, float)):
            if pd.isna(value) or pd.isnull(value):
                continue  # NaN/null is acceptable
            if value == float('inf') or value == float('-inf'):
                return False, f"Infinite value in {key}"

    # Minimum feature count check
    if len(features) < 3:
        return False, f"Too few features ({len(features)})"

    # Sanitize string values to prevent CSV corruption
    for key, value in features.items():
        if isinstance(value, str):
            # Remove newlines and carriage returns that could corrupt CSV
            features[key] = value.replace('\n', ' ').replace('\r', ' ')

    return True, "OK"


# ============================================================================
# Incremental File Writer
# ============================================================================
def append_to_csv(row_dict, filepath, write_header=False):
    """
    Append a single row to CSV file immediately (incremental writing).

    Args:
        row_dict: Dictionary of features to write
        filepath: Output CSV file path
        write_header: Whether to write header (first row)
    """
    # Sanitize all string values to prevent CSV corruption from newlines
    sanitized_dict = {}
    for key, value in row_dict.items():
        if isinstance(value, str):
            # Remove newlines, carriage returns, and limit length
            sanitized_dict[key] = value.replace('\n', ' ').replace('\r', ' ')[:500]
        else:
            sanitized_dict[key] = value

    df_row = pd.DataFrame([sanitized_dict])

    if write_header or not os.path.exists(filepath):
        df_row.to_csv(filepath, index=False, mode='w')
    else:
        df_row.to_csv(filepath, index=False, mode='a', header=False)


# ============================================================================
# Batch Processing with Parallel Workers + Incremental Write
# ============================================================================
def process_batch_parallel(df_urls, batch_date):
    """
    Process batch of URLs in parallel with timeouts and incremental writing.

    Features are validated and written to disk IMMEDIATELY after extraction,
    providing real-time visibility into progress.

    Args:
        df_urls: DataFrame with 'url' and 'label' columns
        batch_date: Batch identifier (YYYYMMDD)

    Returns:
        (dns_file, whois_file) paths
    """
    logger.info(f"Processing batch: {batch_date}")
    logger.info(f"  Total URLs: {len(df_urls)}")
    logger.info(f"  Workers: {MAX_WORKERS}, Timeout: {PER_URL_TIMEOUT}s per URL")

    dns_output = f"{OUTPUT_DIR}/dns_{batch_date}.csv"
    whois_output = f"{OUTPUT_DIR}/whois_{batch_date}.csv"

    # Initialize output files with headers
    append_to_csv({'url': None, 'label': None, 'initialized': True}, dns_output, write_header=True)
    append_to_csv({'url': None, 'label': None, 'initialized': True}, whois_output, write_header=True)

    # Remove initialization row
    os.remove(dns_output)
    os.remove(whois_output)

    start_time = time.time()
    processed_count = 0
    dns_valid_count = 0
    whois_valid_count = 0

    # Convert DataFrame to list of dicts for parallel processing
    urls_data = df_urls.to_dict('records')

    # Process URLs in parallel with timeout
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(process_single_url, url_data): url_data['url']
            for url_data in urls_data
        }

        # Collect and write results as they complete (INCREMENTAL)
        for idx, future in enumerate(as_completed(future_to_url, timeout=None), 1):
            url = future_to_url[future]

            try:
                # Get result with per-URL timeout
                result = future.result(timeout=PER_URL_TIMEOUT)

                # Process DNS features
                if result['dns_features']:
                    dns_row = result['dns_features'].copy()
                    dns_row['url'] = result['url']
                    dns_row['label'] = result['label']

                    # Validate DNS features
                    is_valid, reason = validate_features(dns_row, 'DNS')
                    if is_valid:
                        # Write immediately to disk
                        append_to_csv(dns_row, dns_output, write_header=(idx == 1))
                        dns_valid_count += 1
                    else:
                        logger.warning(f"  ⚠️  Invalid DNS features for {url}: {reason}")
                        append_to_csv({'url': url, 'label': result['label'], 'error': f'INVALID_{reason}'},
                                     dns_output, write_header=(idx == 1))
                else:
                    append_to_csv({'url': result['url'], 'label': result['label'],
                                  'error': result.get('error', 'DNS_FAILED')},
                                 dns_output, write_header=(idx == 1))

                # Process WHOIS features
                if result['whois_features']:
                    whois_row = result['whois_features'].copy()
                    whois_row['url'] = result['url']
                    whois_row['label'] = result['label']

                    # Validate WHOIS features
                    is_valid, reason = validate_features(whois_row, 'WHOIS')
                    if is_valid:
                        # Write immediately to disk
                        append_to_csv(whois_row, whois_output, write_header=(idx == 1))
                        whois_valid_count += 1
                    else:
                        logger.warning(f"  ⚠️  Invalid WHOIS features for {url}: {reason}")
                        append_to_csv({'url': url, 'label': result['label'], 'error': f'INVALID_{reason}'},
                                     whois_output, write_header=(idx == 1))
                else:
                    append_to_csv({'url': result['url'], 'label': result['label'],
                                  'error': result.get('error', 'WHOIS_FAILED')},
                                 whois_output, write_header=(idx == 1))

                processed_count += 1

            except TimeoutError:
                logger.warning(f"  ⏱️  Timeout on {url}")
                append_to_csv({'url': url, 'label': df_urls[df_urls['url'] == url]['label'].iloc[0],
                              'error': 'TIMEOUT'}, dns_output, write_header=(idx == 1))
                append_to_csv({'url': url, 'label': df_urls[df_urls['url'] == url]['label'].iloc[0],
                              'error': 'TIMEOUT'}, whois_output, write_header=(idx == 1))
            except Exception as e:
                logger.error(f"  ❌ Error on {url}: {e}")
                append_to_csv({'url': url, 'label': df_urls[df_urls['url'] == url]['label'].iloc[0],
                              'error': str(e)}, dns_output, write_header=(idx == 1))
                append_to_csv({'url': url, 'label': df_urls[df_urls['url'] == url]['label'].iloc[0],
                              'error': str(e)}, whois_output, write_header=(idx == 1))

            # Progress logging every 100 URLs
            if idx % 100 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                remaining = (len(df_urls) - idx) / rate if rate > 0 else 0
                logger.info(f"  Progress: {idx}/{len(df_urls)} URLs ({rate:.1f} URLs/sec, {remaining/60:.1f} min remaining)")
                logger.info(f"    Valid - DNS: {dns_valid_count}, WHOIS: {whois_valid_count}")

                # Real-time file size check
                dns_size = os.path.getsize(dns_output) if os.path.exists(dns_output) else 0
                whois_size = os.path.getsize(whois_output) if os.path.exists(whois_output) else 0
                logger.info(f"    File sizes - DNS: {dns_size/1024:.1f}KB, WHOIS: {whois_size/1024:.1f}KB")

    elapsed = time.time() - start_time
    logger.info(f"  ✅ Batch complete in {elapsed/60:.1f} minutes")
    logger.info(f"     DNS output: {dns_output} ({processed_count} records, {dns_valid_count} valid)")
    logger.info(f"     WHOIS output: {whois_output} ({processed_count} records, {whois_valid_count} valid)")
    logger.info(f"     Average: {elapsed/len(df_urls):.2f}s per URL")

    return dns_output, whois_output


# ============================================================================
# Ready Signal
# ============================================================================
def create_ready_signal(batch_date, dns_file, whois_file, url_count):
    """Create .ready file to signal batch completion."""
    ready_file = f"{READY_DIR}/batch_{batch_date}.ready"

    with open(ready_file, 'w') as f:
        f.write(f"batch_date: {batch_date}\n")
        f.write(f"completed_at: {datetime.now().isoformat()}\n")
        f.write(f"dns_file: {dns_file}\n")
        f.write(f"whois_file: {whois_file}\n")
        f.write(f"url_count: {url_count}\n")

    logger.info(f"  ✅ Ready signal created: {ready_file}")


# ============================================================================
# Queue Processing
# ============================================================================
def process_queue():
    """Process all batches in queue (one iteration)"""
    queue_files = sorted(glob.glob(f"{QUEUE_DIR}/batch_*.csv"))

    if not queue_files:
        return 0

    logger.info(f"📦 Found {len(queue_files)} batches to process")
    processed_count = 0

    for queue_file in queue_files:
        batch_filename = os.path.basename(queue_file)
        batch_date = batch_filename.replace('batch_', '').replace('.csv', '')

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Processing: {batch_filename}")
        logger.info("=" * 80)

        # Check if already processed
        ready_file = f"{READY_DIR}/batch_{batch_date}.ready"
        if os.path.exists(ready_file):
            logger.info(f"⏭️  Already processed - skipping")
            continue

        # Load URLs
        try:
            df = pd.read_csv(queue_file)
            logger.info(f"  Loaded {len(df)} URLs from {queue_file}")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {queue_file}: {e}")
            continue

        # Process batch in parallel
        try:
            dns_file, whois_file = process_batch_parallel(df, batch_date)
        except Exception as e:
            logger.error(f"  ❌ Feature extraction failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Create ready signal
        create_ready_signal(batch_date, dns_file, whois_file, len(df))

        # Archive processed queue file
        archive_path = queue_file.replace("/url_queue/", "/url_queue/processed_")
        os.makedirs(os.path.dirname(archive_path), exist_ok=True)
        os.rename(queue_file, archive_path)
        logger.info(f"  ✅ Archived queue file: {archive_path}")

        processed_count += 1

    return processed_count


# ============================================================================
# Continuous Mode
# ============================================================================
def main():
    """Continuous processing loop - runs 24/7"""
    logger.info("=" * 80)
    logger.info("VM Daily Processor Started (OPTIMIZED - Parallel Mode)")
    logger.info("=" * 80)
    logger.info(f"Queue directory: {QUEUE_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Ready signals: {READY_DIR}")
    logger.info(f"Workers: {MAX_WORKERS}, Timeout: {PER_URL_TIMEOUT}s per URL")
    logger.info("")
    logger.info("Watching for new batches... (Ctrl+C to stop)")
    logger.info("")

    iteration = 0

    while True:
        iteration += 1

        try:
            processed = process_queue()

            if processed > 0:
                logger.info("")
                logger.info("=" * 80)
                logger.info(f"✅ Processed {processed} batches successfully")
                logger.info("=" * 80)
                logger.info("")
                logger.info("Waiting for new batches...")
            else:
                # Only log every 10 minutes when idle
                if iteration % 10 == 0:
                    logger.info(f"⏳ Still watching... (checked {iteration} times)")

        except Exception as e:
            logger.error(f"❌ Error processing queue: {e}")
            import traceback
            traceback.print_exc()

        # Sleep before next check
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupted by user - shutting down gracefully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
