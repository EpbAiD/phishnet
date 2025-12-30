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

# GCS Configuration
GCS_BUCKET="gs://phishnet-pipeline-data"
GCS_QUEUE="${GCS_BUCKET}/queue"
GCS_INCREMENTAL="${GCS_BUCKET}/incremental"
GCS_PROCESSED="${GCS_BUCKET}/processed"
GCP_PROJECT="coms-452404"

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

URL_FEATURES_FILE="data/url_queue/url_features_${DATE}.csv"
python3 scripts/extract_url_features.py "$OUTPUT_FILE" "$URL_FEATURES_FILE" >> "$LOG_FILE" 2>&1

log "✅ URL features extracted and saved to: $URL_FEATURES_FILE"

# ============================================================================
# Step 3: Upload URL batch to Cloud Storage Queue (URL + Label ONLY)
# ============================================================================
log ""
log "Step 3: Uploading URL batch to Cloud Storage (url + label only)..."

# Upload ORIGINAL batch file (url, label, source) to GCS queue for VM processing
gcloud storage cp "$OUTPUT_FILE" "${GCS_QUEUE}/batch_${DATE}.csv" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Batch uploaded to GCS: ${GCS_QUEUE}/batch_${DATE}.csv"
else
    log_error "Failed to upload batch to Cloud Storage"
    exit 1
fi

# ============================================================================
# Step 3.5: Upload URL Features to GCS Incremental (for weekly merge)
# ============================================================================
log ""
log "Step 3.5: Uploading URL features to GCS incremental folder..."

gcloud storage cp "$URL_FEATURES_FILE" "${GCS_INCREMENTAL}/url_features_${DATE}.csv" >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ URL features uploaded to GCS: ${GCS_INCREMENTAL}/url_features_${DATE}.csv"
else
    log_error "Failed to upload URL features to Cloud Storage"
    exit 1
fi

# ============================================================================
# Step 4: Batch Ready for VM Processing
# ============================================================================
log ""
log "Step 4: Batch queued for processing..."

log "✅ Batch ready for VM processing"
log "   VM will automatically poll GCS and process new batches"
log "   Monitor progress with: gcloud storage ls ${GCS_INCREMENTAL}/"

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
log "  1. VM will poll GCS and process batch (~8 hours)"
log "  2. Check GCS for results: gcloud storage ls ${GCS_INCREMENTAL}/"
log "  3. GitHub Actions will merge results and retrain models"
log "  4. Monitor workflow at: https://github.com/EpbAiD/phishnet/actions"
log "========================================="

exit 0
