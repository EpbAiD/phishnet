#!/bin/bash
# ============================================================================
# Daily Model Retrain - Train models on main datasets
# ============================================================================
# Runs daily at 10:00 AM
# Trains URL, DNS, WHOIS models on the three main datasets
# Tests ensemble combinations and selects best tradeoff
# Saves best ensemble to production_metadata.json
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

# Paths
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
LOG_FILE="logs/daily_retrain_${DATE}.log"

mkdir -p logs models_backup/${DATE}

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

# VM Configuration (using gcloud)
VM_NAME="dns-whois-fetch-25"
VM_ZONE="us-central1-c"
GCP_PROJECT="coms-452404"
GCP_ACCOUNT="eb3658@columbia.edu"
VM_PATH="/home/eeshanbhanap/phishnet"

# ============================================================================
# Start
# ============================================================================
log ""
log "========================================="
log "Daily Model Retrain Started"
log "========================================="
log "Time: $TIMESTAMP"
log ""

# ============================================================================
# Step 1: Auto-Download Processed Features from VM
# ============================================================================
log "Step 1: Downloading processed features from VM..."

mkdir -p data/vm_processed

# Download all processed features from VM
gcloud compute scp "${VM_NAME}:${VM_PATH}/vm_data/incremental/*.csv" data/vm_processed/ \
    --zone="$VM_ZONE" --project="$GCP_PROJECT" --account="$GCP_ACCOUNT" 2>/dev/null || {
    log_info "No new files to download from VM (or VM unreachable)"
}

DOWNLOADED=$(ls -1 data/vm_processed/*.csv 2>/dev/null | wc -l | tr -d ' ')
log "✅ Downloaded $DOWNLOADED feature files from VM"

# ============================================================================
# Step 2: Merge New Data with Main Datasets
# ============================================================================
log ""
log "Step 2: Merging new data with main datasets..."

python3 scripts/merge_vm_data.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Data merged successfully"
else
    log_error "Data merge failed"
    exit 1
fi

# ============================================================================
# Step 3: Backup Existing Models
# ============================================================================
log ""
log "Step 3: Backing up existing models..."

if [ -d "models" ]; then
    cp models/*.pkl models_backup/${DATE}/ 2>/dev/null || true
    log "✅ Backed up existing models to models_backup/${DATE}/"
else
    log_info "No existing models to backup"
fi

# ============================================================================
# Step 2: Train Models (Parallel)
# ============================================================================
log ""
log "Step 2: Training models (this may take 30-60 minutes)..."

export PYTHONPATH="$PROJECT_ROOT"

# Train all 3 model types in parallel
log_info "Training URL models..."
python3 src/training/url_train.py >> "$LOG_FILE" 2>&1 &
URL_PID=$!

log_info "Training DNS models..."
python3 src/training/dns_train.py >> "$LOG_FILE" 2>&1 &
DNS_PID=$!

log_info "Training WHOIS models..."
python3 src/training/whois_train.py >> "$LOG_FILE" 2>&1 &
WHOIS_PID=$!

# Wait for all training to complete
log_info "Waiting for training to complete..."
wait $URL_PID
URL_STATUS=$?
wait $DNS_PID
DNS_STATUS=$?
wait $WHOIS_PID
WHOIS_STATUS=$?

# Check training status
if [ $URL_STATUS -eq 0 ] && [ $DNS_STATUS -eq 0 ] && [ $WHOIS_STATUS -eq 0 ]; then
    log "✅ All models trained successfully"
else
    log_error "Some models failed to train (URL=$URL_STATUS, DNS=$DNS_STATUS, WHOIS=$WHOIS_STATUS)"
    log_info "Check $LOG_FILE for details"
    exit 1
fi

# ============================================================================
# Step 3: Test Ensemble Combinations
# ============================================================================
log ""
log "Step 3: Testing ensemble combinations for optimal tradeoff..."

PYTHONPATH="$PROJECT_ROOT" python3 scripts/test_ensemble_combinations.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Ensemble testing complete"
else
    log_error "Ensemble testing failed"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "========================================="
log "✅ DAILY RETRAIN COMPLETED SUCCESSFULLY"
log "========================================="

echo ""
echo "📊 Model Performance:"
cat models/production_metadata.json 2>/dev/null | grep -E '"url"|"dns"|"whois"|roc_auc|f1|accuracy' | sed 's/^/   /' || true

echo ""
echo "📁 Output Files:"
echo "   - Models: models/url_*.pkl, models/dns_*.pkl, models/whois_*.pkl"
echo "   - Analysis: analysis/*_cv_results.csv"
echo "   - Ensemble: analysis/ensemble_combinations.csv"
echo "   - Metadata: models/production_metadata.json"
echo "   - Log: $LOG_FILE"

echo ""
log "Daily retrain finished at $(date +'%Y-%m-%d %H:%M:%S')"
log "Total runtime: $SECONDS seconds"

echo ""
echo "========================================="

exit 0
