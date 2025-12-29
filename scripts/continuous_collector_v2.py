#!/usr/bin/env python3
"""
Continuous Data Collector V2 - Production Grade
===============================================
Robust 24/7 data collection with:
- Intelligent parallelization (respects rate limits)
- Data quality validation (rejects zeros/nulls)
- Retry logic for failed API calls
- Automatic retraining triggers
- Crash recovery and checkpointing

Architecture:
- Worker pool for parallel WHOIS/DNS lookups
- Rate limiter to avoid bans
- Quality checks before saving
- Auto-retrain when threshold reached
"""

import os
import sys
import time
import logging
import asyncio
import aiohttp
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.whois import extract_single_whois_features
from src.features.dns_ipwhois import extract_single_domain_features
# URL features NOT extracted on VM - done locally on MacBook for speed

# ============================================
# Configuration
# ============================================
MAX_WORKERS = 5  # Parallel workers (conservative to avoid rate limits)
BATCH_SIZE = 20  # URLs per batch
RETRY_ATTEMPTS = 3  # Retry failed lookups
RETRY_DELAY = 5  # Seconds between retries

# Rate limiting (requests per second)
WHOIS_RPS = 0.5  # 1 request every 2 seconds (conservative)
DNS_RPS = 2.0    # 2 requests per second

# Data quality thresholds
MIN_VALID_FEATURES = 5  # Minimum non-zero features required
AUTO_RETRAIN_THRESHOLD = 1000  # Retrain after collecting 1000 new URLs

# Paths
OUTPUT_DIR = "data/vm_collected"
TRAINING_DATA_DIR = "data/processed"  # Main training data location
CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint.json"
QUALITY_LOG = f"{OUTPUT_DIR}/quality_metrics.csv"
WHOIS_OUTPUT = f"{OUTPUT_DIR}/whois_results.csv"
DNS_OUTPUT = f"{OUTPUT_DIR}/dns_results.csv"
# URL features extracted locally (not on VM) - see dataset_builder.py

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/continuous_collector_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================
# Rate Limiter
# ============================================
class RateLimiter:
    """Token bucket rate limiter"""
    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.min_interval = 1.0 / rate_per_second
        self.last_call = 0

    async def acquire(self):
        """Wait if needed to respect rate limit"""
        now = time.time()
        time_since_last = now - self.last_call

        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            await asyncio.sleep(sleep_time)

        self.last_call = time.time()


# ============================================
# Data Quality Validator
# ============================================
class DataQualityValidator:
    """Validates feature quality before saving"""

    def __init__(self):
        self.stats = defaultdict(int)

    def validate_features(self, features: dict, feature_type: str) -> bool:
        """
        Check if features are valid (not all zeros/nulls).

        Returns:
            True if features pass quality checks
        """
        if not features:
            self.stats[f"{feature_type}_empty"] += 1
            return False

        # Count non-null, non-zero values
        valid_count = sum(
            1 for k, v in features.items()
            if v is not None and v != 0 and v != '' and v != 'MISSING'
        )

        if valid_count < MIN_VALID_FEATURES:
            self.stats[f"{feature_type}_insufficient_features"] += 1
            logger.warning(f"{feature_type} features insufficient: only {valid_count} valid features")
            return False

        self.stats[f"{feature_type}_valid"] += 1
        return True

    def log_stats(self):
        """Log quality statistics"""
        logger.info("=" * 60)
        logger.info("Data Quality Statistics:")
        for key, count in self.stats.items():
            logger.info(f"  {key}: {count}")
        logger.info("=" * 60)


# ============================================
# URL Deduplication Checker
# ============================================
class URLDeduplicationChecker:
    """Check if URLs already exist in training/collected data"""

    def __init__(self):
        self.existing_urls = set()
        self.load_existing_urls()

    def load_existing_urls(self):
        """Load all existing URLs from training data and collected data"""
        logger.info("Loading existing URLs for deduplication...")

        # Load from main training data
        training_files = [
            f"{TRAINING_DATA_DIR}/url_features.csv",
            f"{TRAINING_DATA_DIR}/whois_results.csv"
        ]

        for filepath in training_files:
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if 'url' in df.columns:
                        urls = set(df['url'].dropna().tolist())
                        self.existing_urls.update(urls)
                        logger.info(f"Loaded {len(urls)} URLs from {filepath}")
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")

        # Load from VM collected data
        collected_files = [WHOIS_OUTPUT, DNS_OUTPUT]
        for filepath in collected_files:
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if 'url' in df.columns:
                        urls = set(df['url'].dropna().tolist())
                        self.existing_urls.update(urls)
                        logger.info(f"Loaded {len(urls)} URLs from {filepath}")
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")

        logger.info(f"Total existing URLs loaded: {len(self.existing_urls)}")

    def is_new_url(self, url: str) -> bool:
        """Check if URL is new (not in existing datasets)"""
        return url not in self.existing_urls

    def add_url(self, url: str):
        """Add URL to existing set (after successful collection)"""
        self.existing_urls.add(url)

    def get_stats(self) -> dict:
        """Get deduplication statistics"""
        return {
            "total_existing_urls": len(self.existing_urls)
        }


# ============================================
# Retry Wrapper
# ============================================
async def retry_with_backoff(func, *args, max_retries=RETRY_ATTEMPTS, **kwargs):
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum retry attempts

    Returns:
        Function result or None if all retries failed
    """
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
        except Exception as e:
            wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")

            if attempt < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                return None


# ============================================
# Feature Collection with Quality Checks
# ============================================
whois_limiter = RateLimiter(WHOIS_RPS)
dns_limiter = RateLimiter(DNS_RPS)
validator = DataQualityValidator()
dedup_checker = URLDeduplicationChecker()


async def collect_whois_with_quality(url: str, label: str, executor, processed_count: list) -> Optional[dict]:
    """Collect WHOIS with retry, quality validation, and incremental saving"""

    async def _fetch():
        await whois_limiter.acquire()
        logger.info(f"WHOIS: {url}")
        # Run blocking WHOIS call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, extract_single_whois_features, url, True)

    features = await retry_with_backoff(_fetch)

    if features and validator.validate_features(features, 'whois'):
        features['url'] = url
        features['label'] = label
        features['collected_at'] = datetime.now().isoformat()

        # Save immediately to CSV
        save_single_result(features, 'whois')

        # Mark URL as processed
        dedup_checker.add_url(url)
        processed_count[0] += 1

        return features

    return None


async def collect_dns_with_quality(url: str, label: str, executor, processed_count: list) -> Optional[dict]:
    """Collect DNS with retry, quality validation, and incremental saving"""

    async def _fetch():
        await dns_limiter.acquire()
        logger.info(f"DNS: {url}")
        # Run blocking DNS call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, extract_single_domain_features, url)

    features = await retry_with_backoff(_fetch)

    if features and validator.validate_features(features, 'dns'):
        features['url'] = url
        features['label'] = label
        features['collected_at'] = datetime.now().isoformat()

        # Save immediately to CSV
        save_single_result(features, 'dns')

        # Mark URL as processed (if not already marked by WHOIS)
        dedup_checker.add_url(url)

        return features

    return None


# ============================================
# URL Source (Queue File + PhishTank + Tranco)
# ============================================
def load_url_queue() -> List[Dict[str, str]]:
    """
    Load URLs from queue file (created during day/evening collection).

    Priority:
    1. Check for master_queue.csv (accumulated URLs)
    2. Check for individual queue files
    3. Fall back to empty list

    Returns:
        List of {url, label, source} dicts
    """
    queue_dir = os.path.join("data", "url_queue")
    master_queue = os.path.join(queue_dir, "master_queue.csv")

    if os.path.exists(master_queue):
        try:
            df = pd.read_csv(master_queue)
            logger.info(f"📋 Loaded {len(df)} URLs from master queue")

            # Convert to list of dicts
            urls = []
            for _, row in df.iterrows():
                urls.append({
                    "url": row['url'],
                    "label": str(row['label']) if 'label' in row else "unknown",
                    "source": row.get('source', 'queue')
                })

            return urls
        except Exception as e:
            logger.error(f"Error loading master queue: {e}")

    logger.info("No URL queue found")
    return []


def fetch_fresh_urls(limit_per_source: int = 500, use_queue: bool = True) -> List[Dict[str, str]]:
    """
    Fetch fresh URLs from multiple sources.

    Priority Order:
    1. URL queue (if use_queue=True and queue exists)
    2. PhishTank API
    3. Common legitimate domains

    Args:
        limit_per_source: Max URLs per source
        use_queue: Whether to check queue first

    Returns:
        List of {url, label} dicts
    """
    all_urls = []

    # 1. Check URL queue first (for overnight processing)
    if use_queue:
        queued_urls = load_url_queue()
        if len(queued_urls) > 0:
            logger.info(f"✅ Using {len(queued_urls)} queued URLs for processing")
            return queued_urls
        else:
            logger.info("No queued URLs found, fetching from APIs...")

    # 2. PhishTank (phishing)
    try:
        logger.info("Fetching phishing URLs from PhishTank...")
        response = requests.get("http://data.phishtank.com/data/online-valid.csv", timeout=30)
        df = pd.read_csv(pd.io.common.StringIO(response.text), on_bad_lines='skip')
        phishing_urls = [
            {"url": url, "label": "phishing"}
            for url in df['url'].head(limit_per_source).tolist()
        ]
        all_urls.extend(phishing_urls)
        logger.info(f"Fetched {len(phishing_urls)} phishing URLs")
    except Exception as e:
        logger.error(f"Error fetching PhishTank: {e}")

    # 3. Tranco (legitimate) - using top domains
    try:
        logger.info("Adding legitimate URLs from common sources...")
        # For now, use known legitimate domains
        # TODO: Download actual Tranco list
        legitimate_domains = [
            "google.com", "youtube.com", "facebook.com", "amazon.com",
            "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
            "reddit.com", "netflix.com", "microsoft.com", "apple.com",
            "github.com", "stackoverflow.com", "medium.com", "yahoo.com",
            # Add more...
        ]
        legitimate_urls = [
            {"url": f"https://www.{domain}/", "label": "legitimate"}
            for domain in legitimate_domains[:limit_per_source]
        ]
        all_urls.extend(legitimate_urls)
        logger.info(f"Added {len(legitimate_urls)} legitimate URLs")
    except Exception as e:
        logger.error(f"Error adding legitimate URLs: {e}")

    return all_urls


# ============================================
# Checkpoint System (Crash Recovery)
# ============================================
def load_checkpoint() -> dict:
    """Load progress checkpoint"""
    import json
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_count": 0, "last_retrain": 0}


def save_checkpoint(processed_count: int, last_retrain: int):
    """Save progress checkpoint"""
    import json
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            "processed_count": processed_count,
            "last_retrain": last_retrain,
            "timestamp": datetime.now().isoformat()
        }, f)


# ============================================
# Incremental Data Saving
# ============================================
def save_single_result(result: dict, result_type: str):
    """
    Save a single result immediately to CSV (incremental append).

    Args:
        result: Feature dictionary with 'url', 'label', 'collected_at', etc.
        result_type: 'whois' or 'dns'
    """
    if result_type == 'whois':
        output_file = WHOIS_OUTPUT
    elif result_type == 'dns':
        output_file = DNS_OUTPUT
    else:
        logger.error(f"Unknown result type: {result_type}")
        return

    try:
        # Convert to DataFrame
        df = pd.DataFrame([result])

        # Append to CSV (create with header if doesn't exist)
        file_exists = os.path.exists(output_file)
        df.to_csv(output_file, mode='a', header=not file_exists, index=False)

        logger.info(f"✅ Saved {result_type} for {result['url']}")

    except Exception as e:
        logger.error(f"Error saving {result_type} result: {e}")


# ============================================
# Auto-Retrain Trigger
# ============================================
def trigger_retrain():
    """Trigger model retraining"""
    logger.info("🚀 AUTO-RETRAIN TRIGGERED!")
    logger.info("=" * 80)

    try:
        # Run retraining script
        import subprocess
        result = subprocess.run(
            ["python3", "scripts/weekly_retrain.py"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info("✅ Retraining completed successfully!")
            logger.info(result.stdout)
            return True
        else:
            logger.error(f"❌ Retraining failed: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error during retraining: {e}")
        return False


# ============================================
# URL Queue Population (Day/Evening Collection)
# ============================================
def collect_urls_to_queue(limit_per_source: int = 1000) -> int:
    """
    Collect URLs from multiple APIs and save to queue for overnight VM processing.
    Can be run anytime during day/evening.

    Args:
        limit_per_source: How many URLs to fetch from each source

    Returns:
        Total number of URLs saved to queue
    """
    logger.info("=" * 80)
    logger.info("📋 URL Queue Collection Started")
    logger.info("=" * 80)

    queue_dir = os.path.join("data", "url_queue")
    os.makedirs(queue_dir, exist_ok=True)

    master_queue = os.path.join(queue_dir, "master_queue.csv")

    # Load existing queue to avoid duplicates
    existing_urls = set()
    if os.path.exists(master_queue):
        try:
            existing_df = pd.read_csv(master_queue)
            existing_urls = set(existing_df['url'].tolist())
            logger.info(f"📌 Found {len(existing_urls)} URLs already in queue")
        except Exception as e:
            logger.warning(f"Could not load existing queue: {e}")

    # Fetch URLs from all sources (same logic as fetch_fresh_urls but don't filter by processed)
    all_urls = []

    # 1. PhishTank (phishing)
    logger.info("\n1️⃣  Fetching from PhishTank...")
    try:
        response = requests.get("http://data.phishtank.com/data/online-valid.csv", timeout=30)
        # Use on_bad_lines='skip' to handle malformed lines
        df = pd.read_csv(pd.io.common.StringIO(response.text), on_bad_lines='skip')

        # Try different column names (PhishTank may use 'url' or 'phish_id')
        url_column = None
        for col in ['url', 'URL', 'phish_detail_url']:
            if col in df.columns:
                url_column = col
                break

        if url_column:
            phishtank_urls = df[url_column].head(limit_per_source).tolist()

            for url in phishtank_urls:
                if url not in existing_urls:
                    all_urls.append({
                        "url": url,
                        "label": "phishing",
                        "source": "phishtank"
                    })
            logger.info(f"   ✅ PhishTank: {len(phishtank_urls)} URLs ({len([u for u in all_urls if u['source']=='phishtank'])} new)")
        else:
            logger.warning(f"   ⚠️  PhishTank columns: {df.columns.tolist()}")
            logger.info(f"   ⚠️  Could not find URL column, skipping PhishTank")
    except Exception as e:
        logger.error(f"   ❌ Error fetching PhishTank: {e}")
        logger.info(f"   ⚠️  Continuing without PhishTank URLs...")

    # 2. Common Legitimate Domains (from existing code)
    logger.info("\n2️⃣  Adding common legitimate domains...")
    common_legit = [
        "https://www.google.com",
        "https://www.amazon.com",
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.reddit.com",
        "https://www.twitter.com",
        "https://www.facebook.com",
        "https://www.youtube.com",
        "https://www.linkedin.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.netflix.com",
        "https://www.spotify.com",
        "https://www.dropbox.com",
    ]

    for url in common_legit[:limit_per_source]:
        if url not in existing_urls:
            all_urls.append({
                "url": url,
                "label": "legitimate",
                "source": "common_legit"
            })
    logger.info(f"   ✅ Common Legit: {len(common_legit)} URLs")

    # Filter out duplicates within this collection
    unique_urls = []
    seen = existing_urls.copy()
    for url_info in all_urls:
        if url_info['url'] not in seen:
            unique_urls.append(url_info)
            seen.add(url_info['url'])

    logger.info(f"\n📊 Summary:")
    logger.info(f"   Total fetched: {len(all_urls)}")
    logger.info(f"   New unique: {len(unique_urls)}")
    logger.info(f"   Duplicates skipped: {len(all_urls) - len(unique_urls)}")

    if len(unique_urls) == 0:
        logger.info("✅ No new URLs to add to queue")
        return 0

    # Append to master queue
    new_df = pd.DataFrame(unique_urls)

    if os.path.exists(master_queue):
        # Append to existing
        new_df.to_csv(master_queue, mode='a', header=False, index=False)
        logger.info(f"\n✅ Appended {len(unique_urls)} URLs to existing queue")
    else:
        # Create new
        new_df.to_csv(master_queue, index=False)
        logger.info(f"\n✅ Created new queue with {len(unique_urls)} URLs")

    # Show queue stats
    final_df = pd.read_csv(master_queue)
    logger.info(f"\n📈 Queue Status:")
    logger.info(f"   Total URLs in queue: {len(final_df)}")
    logger.info(f"   Phishing: {len(final_df[final_df['label'] == 'phishing'])}")
    logger.info(f"   Legitimate: {len(final_df[final_df['label'] == 'legitimate'])}")
    logger.info(f"   Queue file: {master_queue}")
    logger.info("=" * 80)

    return len(unique_urls)


# ============================================
# Main Continuous Loop
# ============================================
async def continuous_collection_loop():
    """Main 24/7 collection loop"""

    logger.info("=" * 80)
    logger.info("🚀 Continuous Data Collector V2 Started")
    logger.info("=" * 80)

    checkpoint = load_checkpoint()
    processed_total = checkpoint["processed_count"]
    last_retrain = checkpoint["last_retrain"]

    # Create thread pool executor for blocking I/O
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # Use list to allow modification in async functions
    processed_count_tracker = [processed_total]

    while True:
        try:
            # Fetch fresh URLs
            all_urls = fetch_fresh_urls(limit_per_source=500)
            logger.info(f"Fetched {len(all_urls)} URLs from sources")

            # Filter out existing URLs (deduplication)
            new_urls = [
                url_info for url_info in all_urls
                if dedup_checker.is_new_url(url_info['url'])
            ]

            skipped_count = len(all_urls) - len(new_urls)
            if skipped_count > 0:
                logger.info(f"⏭️  Skipped {skipped_count} existing URLs (already in dataset)")

            logger.info(f"Processing {len(new_urls)} NEW URLs...")

            if not new_urls:
                logger.info("No new URLs to process. Waiting 1 hour before next fetch...")
                await asyncio.sleep(3600)  # Wait 1 hour
                continue

            # Process in batches
            for i in range(0, len(new_urls), BATCH_SIZE):
                batch = new_urls[i:i + BATCH_SIZE]
                logger.info(f"\n{'='*60}")
                logger.info(f"Batch {i//BATCH_SIZE + 1}: URLs {i} to {i+len(batch)}")
                logger.info(f"{'='*60}")

                # Collect WHOIS and DNS concurrently with quality checks
                # URL features will be extracted locally on MacBook (instant, no APIs)
                # Results are saved incrementally inside collect_* functions
                tasks = []
                for url_info in batch:
                    url = url_info['url']
                    label = url_info['label']
                    # WHOIS features (rate-limited, VM-based)
                    tasks.append(collect_whois_with_quality(url, label, executor, processed_count_tracker))
                    # DNS features (rate-limited, VM-based)
                    tasks.append(collect_dns_with_quality(url, label, executor, processed_count_tracker))

                # Wait for batch to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successful collections
                successful = sum(1 for r in results if r and not isinstance(r, Exception))
                logger.info(f"✅ Batch complete: {successful}/{len(results)} successful")

                # Save checkpoint after each batch
                save_checkpoint(processed_count_tracker[0], last_retrain)

                # Check if auto-retrain threshold reached
                if processed_count_tracker[0] - last_retrain >= AUTO_RETRAIN_THRESHOLD:
                    logger.info(f"\n🎯 Collected {processed_count_tracker[0] - last_retrain} new URLs since last retrain")
                    if trigger_retrain():
                        last_retrain = processed_count_tracker[0]
                        save_checkpoint(processed_count_tracker[0], last_retrain)

                # Log quality stats
                validator.log_stats()

                # Small delay between batches
                await asyncio.sleep(2)

            logger.info(f"\n✅ Cycle complete! Total processed: {processed_count_tracker[0]}")
            logger.info("Starting new cycle in 10 minutes...")
            await asyncio.sleep(600)  # 10 minute pause before next cycle

        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Collector stopped by user")
            save_checkpoint(processed_count_tracker[0], last_retrain)
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            logger.info("Retrying in 60 seconds...")
            await asyncio.sleep(60)


# ============================================
# Entry Point
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PhishNet VM Data Collector")
    parser.add_argument(
        "--mode",
        choices=["continuous", "queue"],
        default="continuous",
        help="Mode: 'continuous' for 24/7 overnight processing, 'queue' for daytime URL collection"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of URLs to fetch per source (only for queue mode)"
    )

    args = parser.parse_args()

    try:
        if args.mode == "queue":
            # Daytime/Evening: Collect URLs and save to queue
            logger.info("🌅 Running in QUEUE mode (URL collection)")
            count = collect_urls_to_queue(limit_per_source=args.limit)
            logger.info(f"\n✅ Queue collection complete! Added {count} new URLs")
            logger.info("💡 Next step: Run this script in 'continuous' mode on VM for overnight processing")

        else:
            # Overnight: Process URLs from queue (or fetch if queue empty)
            logger.info("🌙 Running in CONTINUOUS mode (overnight processing)")
            asyncio.run(continuous_collection_loop())

    except KeyboardInterrupt:
        logger.info("\n\nCollector stopped. Progress saved.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
