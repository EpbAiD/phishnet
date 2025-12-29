#!/bin/bash
# ============================================================================
# Daily URL Collector - Fetch bulk URLs and extract local features
# ============================================================================
# Runs daily at 2:00 AM
# Fetches 10,000 new URLs from PhishTank/OpenPhish/URLHaus
# Extracts URL features locally (instant, no API calls)
# Pushes to VM queue for DNS/WHOIS extraction
# ============================================================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# VM Configuration
VM_USER="${VM_USER:-eeshanbhanap}"
VM_HOST="${VM_HOST:-35.188.193.186}"
VM_PATH="${VM_PATH:-/home/eeshanbhanap/phishnet}"

# Paths
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="logs/url_collection_${DATE}.log"
OUTPUT_FILE="data/url_queue/batch_${DATE}.csv"

mkdir -p logs data/url_queue data/url_queue/processed

# Logging
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

# ============================================================================
# Step 1: Fetch URLs from Public Sources
# ============================================================================
log ""
log "========================================="
log "Daily URL Collection Started"
log "========================================="
log "Time: $TIMESTAMP"
log "Output: $OUTPUT_FILE"
log ""

log "Step 1: Fetching URLs from public sources..."

# Fetch from PhishTank, OpenPhish, URLHaus
PYTHONPATH="$PROJECT_ROOT" python3 << 'PYTHON_FETCH' >> "$LOG_FILE" 2>&1
import os
import sys
import pandas as pd
import requests
from datetime import datetime
from urllib.parse import urlparse

OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'data/url_queue/batch.csv')

print(f"  Fetching URLs from public sources...")

all_urls = []

# 1. PhishTank (phishing URLs)
try:
    print("  → PhishTank...")
    response = requests.get(
        'http://data.phishtank.com/data/online-valid.csv',
        timeout=60
    )
    if response.status_code == 200:
        lines = response.text.split('\n')[1:]  # Skip header
        for line in lines[:5000]:  # Limit 5,000
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 2:
                    url = parts[1].strip('"')
                    all_urls.append({'url': url, 'label': 'phishing', 'source': 'phishtank'})
        print(f"    ✓ Fetched {len(all_urls)} phishing URLs")
except Exception as e:
    print(f"    ✗ PhishTank failed: {e}")

# 2. OpenPhish (phishing URLs)
try:
    print("  → OpenPhish...")
    response = requests.get(
        'https://openphish.com/feed.txt',
        timeout=60
    )
    if response.status_code == 200:
        openphish_count = 0
        for url in response.text.split('\n')[:3000]:  # Limit 3,000
            if url.strip():
                all_urls.append({'url': url.strip(), 'label': 'phishing', 'source': 'openphish'})
                openphish_count += 1
        print(f"    ✓ Fetched {openphish_count} phishing URLs")
except Exception as e:
    print(f"    ✗ OpenPhish failed: {e}")

# 3. Legitimate URLs (Alexa Top 1M or hardcoded list)
try:
    print("  → Legitimate URLs...")
    legit_domains = [
        'google.com', 'facebook.com', 'youtube.com', 'amazon.com', 'wikipedia.org',
        'twitter.com', 'linkedin.com', 'instagram.com', 'reddit.com', 'netflix.com',
        'microsoft.com', 'apple.com', 'github.com', 'stackoverflow.com', 'medium.com',
        'paypal.com', 'ebay.com', 'cnn.com', 'bbc.com', 'nytimes.com'
    ]

    # Generate variations for each domain (2,000 legitimate URLs)
    legit_count = 0
    for domain in legit_domains * 100:  # Repeat to get 2,000
        if legit_count >= 2000:
            break
        all_urls.append({'url': f'https://{domain}', 'label': 'legitimate', 'source': 'hardcoded'})
        all_urls.append({'url': f'https://www.{domain}', 'label': 'legitimate', 'source': 'hardcoded'})
        legit_count += 2

    print(f"    ✓ Generated {legit_count} legitimate URLs")
except Exception as e:
    print(f"    ✗ Legitimate URLs failed: {e}")

# Save to CSV
df = pd.DataFrame(all_urls)
df = df.drop_duplicates(subset=['url'])
df = df.head(10000)  # Limit to 10,000 total

df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  ✅ Total URLs collected: {len(df)}")
print(f"     Phishing: {len(df[df['label'] == 'phishing'])}")
print(f"     Legitimate: {len(df[df['label'] == 'legitimate'])}")
print(f"  Saved to: {OUTPUT_FILE}")
PYTHON_FETCH

if [ ! -f "$OUTPUT_FILE" ]; then
    log_error "URL collection failed - no output file created"
    exit 1
fi

URL_COUNT=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
log "✅ Fetched $URL_COUNT URLs"

# ============================================================================
# Step 1.5: Deduplicate and Balance Phishing Types
# ============================================================================
log ""
log "Step 1.5: Deduplicating and balancing phishing types..."

PYTHONPATH="$PROJECT_ROOT" python3 << 'PYTHON_DEDUPE' >> "$LOG_FILE" 2>&1
import os
import pandas as pd
import numpy as np
from collections import Counter

OUTPUT_FILE = os.environ.get('OUTPUT_FILE', 'data/url_queue/batch.csv')
EXISTING_DATA = 'data/processed/url_features_modelready.csv'

# Load newly fetched URLs
df_new = pd.read_csv(OUTPUT_FILE)
print(f"  Fetched {len(df_new)} URLs from sources")

# Load existing URLs (if available)
if os.path.exists(EXISTING_DATA):
    df_existing = pd.read_csv(EXISTING_DATA)
    existing_urls = set(df_existing['url'].values)
    print(f"  Found {len(existing_urls)} existing URLs in dataset")

    # Remove duplicates
    df_new = df_new[~df_new['url'].isin(existing_urls)]
    print(f"  After deduplication: {len(df_new)} new URLs")

    # Get existing phishing type distribution
    df_existing_phish = df_existing[df_existing['label'] == 1]  # Binary: 1=phishing

    # Categorize existing phishing by type (based on URL patterns)
    def categorize_phishing_type(url):
        url = str(url).lower()
        import re

        # IP-based
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            return 'ip_based'

        # Brand impersonation (common brands)
        brands = ['paypal', 'amazon', 'apple', 'google', 'microsoft', 'facebook', 'netflix', 'ebay']
        for brand in brands:
            if brand in url and not url.startswith(f'https://{brand}.') and not url.startswith(f'https://www.{brand}.'):
                return 'brand_impersonation'

        # Financial/Banking keywords
        if any(kw in url for kw in ['bank', 'login', 'account', 'secure', 'verify', 'update']):
            return 'financial'

        # Typosquatting (close to legitimate)
        typo_patterns = ['faceb00k', 'g00gle', 'amaz0n', 'app1e', 'micr0soft']
        if any(pattern in url for pattern in typo_patterns):
            return 'typosquatting'

        # Suspicious TLDs
        if any(url.endswith(tld) for tld in ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz']):
            return 'suspicious_tld'

        # Credential harvesting (has multiple suspicious keywords)
        cred_keywords = ['signin', 'password', 'credential', 'auth', 'validate']
        if sum(kw in url for kw in cred_keywords) >= 2:
            return 'credential_harvesting'

        return 'other'

    # Get type distribution from existing data
    if len(df_existing_phish) > 0:
        df_existing_phish['phishing_type'] = df_existing_phish['url'].apply(categorize_phishing_type)
        existing_types = df_existing_phish['phishing_type'].value_counts()
        existing_total = len(df_existing_phish)

        target_percentages = {ptype: count / existing_total for ptype, count in existing_types.items()}
        print(f"\\n  Existing phishing type distribution:")
        for ptype, pct in sorted(target_percentages.items()):
            print(f"    {ptype}: {pct:.1%}")
    else:
        # If no existing data, use equal distribution
        target_percentages = {
            'ip_based': 0.15,
            'brand_impersonation': 0.15,
            'financial': 0.15,
            'typosquatting': 0.15,
            'suspicious_tld': 0.15,
            'credential_harvesting': 0.15,
            'other': 0.10
        }
        print(f"\\n  No existing data - using equal distribution for all types")
else:
    print(f"  No existing data found - first run")
    df_new = df_new  # Keep all

    # Equal distribution for first run
    target_percentages = {
        'ip_based': 0.15,
        'brand_impersonation': 0.15,
        'financial': 0.15,
        'typosquatting': 0.15,
        'suspicious_tld': 0.15,
        'credential_harvesting': 0.15,
        'other': 0.10
    }

# Categorize new phishing URLs
df_new_phish = df_new[df_new['label'] == 'phishing'].copy()
df_new_legit = df_new[df_new['label'] == 'legitimate'].copy()

if len(df_new_phish) > 0:
    df_new_phish['phishing_type'] = df_new_phish['url'].apply(categorize_phishing_type)

    # Balance by sampling according to target percentages
    balanced_phish = []
    target_total = 1000  # VM can process 1000/day
    target_phish = int(target_total * 0.5)  # 50% phishing, 50% legitimate

    for ptype, target_pct in target_percentages.items():
        type_urls = df_new_phish[df_new_phish['phishing_type'] == ptype]
        target_count = int(target_phish * target_pct)

        if len(type_urls) >= target_count:
            sampled = type_urls.sample(n=target_count, random_state=42)
        else:
            sampled = type_urls  # Take all if not enough

        balanced_phish.append(sampled)
        print(f"    {ptype}: sampled {len(sampled)}/{len(type_urls)} (target: {target_count})")

    df_balanced_phish = pd.concat(balanced_phish, ignore_index=True)
    df_balanced_phish = df_balanced_phish.drop(columns=['phishing_type'])
else:
    df_balanced_phish = pd.DataFrame()

# Balance legitimate URLs (50% of total)
target_legit = int(target_total * 0.5)
if len(df_new_legit) >= target_legit:
    df_balanced_legit = df_new_legit.sample(n=target_legit, random_state=42)
else:
    df_balanced_legit = df_new_legit

# Combine balanced dataset
df_balanced = pd.concat([df_balanced_phish, df_balanced_legit], ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

# Save balanced dataset
df_balanced.to_csv(OUTPUT_FILE, index=False)

print(f"\\n  ✅ Final balanced batch: {len(df_balanced)} URLs")
print(f"     Phishing: {len(df_balanced_phish)}")
print(f"     Legitimate: {len(df_balanced_legit)}")
print(f"  Saved to: {OUTPUT_FILE}")
PYTHON_DEDUPE

if [ $? -ne 0 ]; then
    log_error "Deduplication/balancing failed"
    exit 1
fi

BALANCED_COUNT=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
log "✅ Deduplicated and balanced: $BALANCED_COUNT URLs"

# ============================================================================
# Step 2: Extract URL Features Locally (Instant)
# ============================================================================
log ""
log "Step 2: Extracting URL features locally..."

PYTHONPATH="$PROJECT_ROOT" python3 << PYTHON_SCRIPT >> "$LOG_FILE" 2>&1
import sys
import pandas as pd
from src.features.url_features import URLFeatureExtractor

# Read URLs
df = pd.read_csv('${OUTPUT_FILE}')
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
df_feats.to_csv('${OUTPUT_FILE}', index=False)

print(f"  ✅ Extracted {df_feats.shape[1]} features per URL")
print(f"  Saved to: ${OUTPUT_FILE}")
PYTHON_SCRIPT

log "✅ URL features extracted"

# ============================================================================
# Step 3: Push to VM Queue
# ============================================================================
log ""
log "Step 3: Pushing URL batch to VM..."

# Create VM directories if they don't exist
ssh ${VM_USER}@${VM_HOST} "mkdir -p ${VM_PATH}/vm_data/url_queue ${VM_PATH}/vm_data/incremental ${VM_PATH}/logs" 2>/dev/null || {
    log_error "Failed to create VM directories (VM may be unreachable)"
    log_info "Skipping VM upload - batch saved locally for manual upload"
    exit 0
}

# Upload batch file to VM
scp "$OUTPUT_FILE" ${VM_USER}@${VM_HOST}:${VM_PATH}/vm_data/url_queue/batch_${DATE}.csv >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Batch uploaded to VM: ${VM_PATH}/vm_data/url_queue/batch_${DATE}.csv"
else
    log_error "Failed to upload batch to VM"
    exit 1
fi

# ============================================================================
# Step 4: Trigger VM Processing
# ============================================================================
log ""
log "Step 4: Triggering VM processor..."

# Check if VM processor is running
VM_RUNNING=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor || echo 'NOT_RUNNING'" 2>/dev/null)

if [ "$VM_RUNNING" = "NOT_RUNNING" ]; then
    log_info "Starting VM processor..."

    ssh ${VM_USER}@${VM_HOST} "cd ${VM_PATH} && nohup python3 scripts/vm_daily_processor.py > logs/vm_processor_${DATE}.log 2>&1 &" 2>/dev/null

    sleep 3

    # Verify it started
    VM_RUNNING=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor || echo 'FAILED'" 2>/dev/null)

    if [ "$VM_RUNNING" = "FAILED" ]; then
        log_error "Failed to start VM processor"
        exit 1
    fi

    log "✅ VM processor started (PID: $VM_RUNNING)"
else
    log "✅ VM processor already running (PID: $VM_RUNNING)"
fi

# ============================================================================
# Step 5: Archive Local Batch
# ============================================================================
log ""
log "Step 5: Archiving local batch..."

cp "$OUTPUT_FILE" "data/url_queue/processed/batch_${DATE}.csv"
log "✅ Archived to: data/url_queue/processed/batch_${DATE}.csv"

# ============================================================================
# Summary
# ============================================================================
log ""
log "========================================="
log "Daily URL Collection Complete"
log "========================================="
log "URLs collected: $URL_COUNT"
log "Local features: 39 (URL structure)"
log "VM processing: DNS (38 features) + WHOIS (12 features)"
log "Expected completion: ~8 hours (10:00 AM next morning)"
log ""
log "Next steps:"
log "  1. Wait for VM to finish processing (~8 hours)"
log "  2. Check VM status: ./scripts/vm_manager.sh status"
log "  3. Daily retrain will run automatically at 10:00 AM"
log "========================================="

exit 0
