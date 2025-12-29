#!/bin/bash
# ============================================================================
# Daily URL Collection - Clean Orchestrator (NO Python code in this file)
# ============================================================================
# This script ONLY orchestrates - all logic is in Python scripts
# Runs daily at 2:00 AM
#
# Flow:
#   1. Fetch URLs from public sources (fetch_urls.py)
#   2. Extract local URL features (extract_url_features.py)
#   3. Upload to VM queue (scp)
#   4. Trigger VM processor (ssh)
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

# VM Configuration (using gcloud)
VM_NAME="dns-whois-fetch-25"
VM_ZONE="us-central1-c"
GCP_PROJECT="coms-452404"
GCP_ACCOUNT="eb3658@columbia.edu"
VM_PATH="/home/eeshanbhanap/phishnet"

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
python3 scripts/fetch_urls.py "$OUTPUT_FILE" 1000 >> "$LOG_FILE" 2>&1

if [ ! -f "$OUTPUT_FILE" ]; then
    log_error "URL fetch failed - no output file created"
    exit 1
fi

URL_COUNT=$(tail -n +2 "$OUTPUT_FILE" | wc -l | tr -d ' ')
log "✅ Fetched $URL_COUNT URLs"

# ============================================================================
# Step 2: Extract URL Features Locally (Instant)
# ============================================================================
log ""
log "Step 2: Extracting URL features locally..."

TEMP_FILE="${OUTPUT_FILE}.tmp"
python3 scripts/extract_url_features.py "$OUTPUT_FILE" "$TEMP_FILE" >> "$LOG_FILE" 2>&1
mv "$TEMP_FILE" "$OUTPUT_FILE"

log "✅ URL features extracted"

# ============================================================================
# Step 3: Push to VM Queue
# ============================================================================
log ""
log "Step 3: Pushing URL batch to VM..."

# Create VM directories if they don't exist
gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --command="sudo mkdir -p ${VM_PATH}/vm_data/url_queue ${VM_PATH}/vm_data/incremental ${VM_PATH}/logs && \
               sudo chmod 755 ${VM_PATH}/vm_data/url_queue ${VM_PATH}/vm_data/incremental ${VM_PATH}/logs" >> "$LOG_FILE" 2>&1 || {
    log_error "Failed to create VM directories (VM may be unreachable)"
    log_info "Skipping VM upload - batch saved locally for manual upload"
    exit 0
}

# Upload batch file to /tmp first, then move to target directory with sudo
gcloud compute scp "$OUTPUT_FILE" "${VM_NAME}:/tmp/batch_${DATE}.csv" \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" >> "$LOG_FILE" 2>&1

gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --command="sudo mv /tmp/batch_${DATE}.csv ${VM_PATH}/vm_data/url_queue/batch_${DATE}.csv && \
               sudo chown eeshanbhanap:eeshanbhanap ${VM_PATH}/vm_data/url_queue/batch_${DATE}.csv" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Batch uploaded to VM: ${VM_PATH}/vm_data/url_queue/batch_${DATE}.csv"
else
    log_error "Failed to upload batch to VM"
    exit 1
fi

# ============================================================================
# Step 4: VM Processor Should Be Running (Started by vm_start.sh)
# ============================================================================
log ""
log "Step 4: Checking VM processor status..."

# Check if VM processor is running
VM_RUNNING=$(gcloud compute ssh "$VM_NAME" --zone="$VM_ZONE" --project="$GCP_PROJECT" \
    --command="pgrep -f vm_daily_processor || echo 'NOT_RUNNING'" 2>/dev/null)

if [ "$VM_RUNNING" = "NOT_RUNNING" ]; then
    log_error "VM processor is NOT running!"
    log_info "Please start it manually: ./scripts/vm_start.sh"
    exit 1
else
    log "✅ VM processor is running (PID: $VM_RUNNING)"
    log "   Processor will automatically detect and process new batch"
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
log "  2. Check VM status: ./scripts/vm_status.sh"
log "  3. Sync results: ./scripts/vm_sync.sh"
log "  4. Daily retrain will run automatically at 10:00 AM"
log "========================================="

exit 0
