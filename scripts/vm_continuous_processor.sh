#!/bin/bash
# ============================================================================
# VM Continuous Feature Processor
# ============================================================================
# Runs continuously on GCP VM in screen session
# Monitors for new URL batches and extracts DNS/WHOIS features
# Auto-transfers results back to local machine
#
# USAGE ON VM:
#   screen -S phishnet-processor
#   cd ~/phishnet
#   bash scripts/vm_continuous_processor.sh
#   # Detach: Ctrl+A then D
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VM_BASE="/home/eeshanbhanap/phishnet"
QUEUE_DIR="$VM_BASE/vm_data/url_queue"
OUTPUT_DIR="$VM_BASE/vm_data/incremental"
READY_DIR="$VM_BASE/vm_data/ready"
LOGS_DIR="$VM_BASE/logs"

# Local machine details (where to send results)
LOCAL_USER="eeshanbhanap"
LOCAL_HOST="YOUR_LOCAL_IP_OR_HOSTNAME"  # UPDATE THIS
LOCAL_PATH="/Users/eeshanbhanap/Desktop/PDF/data/vm_collected"

# Polling interval (seconds)
POLL_INTERVAL=60  # Check every 60 seconds

# Create directories
mkdir -p "$QUEUE_DIR" "$OUTPUT_DIR" "$READY_DIR" "$LOGS_DIR"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOGS_DIR/vm_processor.log"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOGS_DIR/vm_processor.log"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOGS_DIR/vm_processor.log"
}

# Process a single batch
process_batch() {
    local batch_file=$1
    local batch_name=$(basename "$batch_file" .csv)
    local date_tag=$(date +%Y%m%d_%H%M%S)
    
    log "========================================="
    log "Processing batch: $batch_name"
    log "========================================="
    
    # Count URLs
    local url_count=$(tail -n +2 "$batch_file" | wc -l | tr -d ' ')
    log "URLs in batch: $url_count"
    
    # Extract features using Python
    log "Extracting DNS and WHOIS features..."
    
    cd "$VM_BASE"
    PYTHONPATH="$VM_BASE" python3 << PYTHON_SCRIPT
import sys
import pandas as pd
from datetime import datetime
from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features
import tldextract

# Load batch
batch_file = "$batch_file"
df = pd.read_csv(batch_file)

print(f"Loaded {len(df)} URLs from batch")

# Extract features
dns_results = []
whois_results = []

for idx, row in df.iterrows():
    url = row['url']
    label = row.get('label', 0)
    
    # Extract domain
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        domain = f"{ext.domain}.{ext.suffix}"
    else:
        domain = url
    
    # Progress
    if (idx + 1) % 100 == 0:
        print(f"  Processed {idx + 1}/{len(df)} URLs...")
    
    # DNS features
    try:
        dns_feats = extract_single_domain_features(url)
        dns_feats['url'] = url
        dns_feats['label'] = label
        dns_feats['collected_at'] = datetime.now().isoformat()
        dns_results.append(dns_feats)
    except Exception as e:
        print(f"  DNS failed for {url}: {e}")
    
    # WHOIS features
    try:
        whois_feats = extract_single_whois_features(url)
        whois_feats['url'] = url
        whois_feats['label'] = label
        whois_feats['collected_at'] = datetime.now().isoformat()
        whois_results.append(whois_feats)
    except Exception as e:
        print(f"  WHOIS failed for {url}: {e}")

# Save results
dns_df = pd.DataFrame(dns_results)
whois_df = pd.DataFrame(whois_results)

dns_output = "$OUTPUT_DIR/dns_${date_tag}.csv"
whois_output = "$OUTPUT_DIR/whois_${date_tag}.csv"

dns_df.to_csv(dns_output, index=False)
whois_df.to_csv(whois_output, index=False)

print(f"\nExtraction complete:")
print(f"  DNS features: {len(dns_df)} URLs saved to {dns_output}")
print(f"  WHOIS features: {len(whois_df)} URLs saved to {whois_output}")
PYTHON_SCRIPT
    
    if [ $? -eq 0 ]; then
        log "✅ Feature extraction completed"
        
        # Create ready signal
        touch "$READY_DIR/${batch_name}.ready"
        log "✅ Created ready signal: ${batch_name}.ready"
        
        # Transfer results to local machine
        log "📤 Transferring results to local machine..."
        
        scp "$OUTPUT_DIR/dns_${date_tag}.csv" \
            "$OUTPUT_DIR/whois_${date_tag}.csv" \
            "$READY_DIR/${batch_name}.ready" \
            "${LOCAL_USER}@${LOCAL_HOST}:${LOCAL_PATH}/" 2>&1 | tee -a "$LOGS_DIR/vm_processor.log"
        
        if [ $? -eq 0 ]; then
            log "✅ Transfer successful"
            
            # Archive processed batch
            mkdir -p "$QUEUE_DIR/processed"
            mv "$batch_file" "$QUEUE_DIR/processed/"
            log "✅ Batch archived: $QUEUE_DIR/processed/$(basename $batch_file)"
        else
            log_error "Transfer failed - files remain in $OUTPUT_DIR"
        fi
    else
        log_error "Feature extraction failed"
    fi
    
    log "========================================="
}

# Main monitoring loop
log ""
log "🚀 VM Continuous Processor Started"
log "Monitoring: $QUEUE_DIR"
log "Poll interval: ${POLL_INTERVAL}s"
log "Press Ctrl+C to stop"
log ""

ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))
    
    # Check for new batches
    NEW_BATCHES=$(find "$QUEUE_DIR" -maxdepth 1 -name "batch_*.csv" -type f 2>/dev/null)
    
    if [ -n "$NEW_BATCHES" ]; then
        COUNT=$(echo "$NEW_BATCHES" | wc -l | tr -d ' ')
        log "📥 Found $COUNT new batch(es)"
        
        # Process each batch
        echo "$NEW_BATCHES" | while read -r batch_file; do
            process_batch "$batch_file"
        done
    else
        # No new batches - log every 10 iterations (10 minutes)
        if [ $((ITERATION % 10)) -eq 0 ]; then
            log_info "No new batches (checked $ITERATION times)"
        fi
    fi
    
    # Sleep before next check
    sleep "$POLL_INTERVAL"
done
