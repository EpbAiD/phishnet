#!/bin/bash
# ============================================================================
# Full Pipeline Automation Script
# Phishing Detection System - Complete Workflow
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (EDIT THESE)
VM_USER="${VM_USER:-your_username}"
VM_HOST="${VM_HOST:-your_vm_ip}"
VM_PATH="${VM_PATH:-/home/username/phishing_collector}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Timestamp
DATE=$(date +%Y%m%d)
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Log file
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_${DATE}.log"

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1" | tee -a "$LOG_FILE"
}

# Banner
echo ""
echo "========================================="
echo "  PHISHING DETECTION - FULL PIPELINE"
echo "========================================="
echo "  Started: $TIMESTAMP"
echo "  Log: $LOG_FILE"
echo "========================================="
echo ""

cd "$PROJECT_ROOT"

# ============================================================================
# Phase 0: Pre-flight Checks
# ============================================================================
log "Phase 0: Running pre-flight checks..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found"
    exit 1
fi
log_info "✓ Python3 available"

# Check required files
REQUIRED_FILES=(
    "src/data_prep/dataset_builder.py"
    "src/training/url_train.py"
    "src/training/dns_train.py"
    "src/training/whois_train.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done
log_info "✓ All required files present"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT"
log_info "✓ PYTHONPATH set to $PROJECT_ROOT"

# ============================================================================
# Phase 1: Check for New VM Data
# ============================================================================
log ""
log "Phase 1: Checking VM for new data..."

# Check if VM is reachable
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$VM_USER@$VM_HOST" "echo 2>&1" &>/dev/null; then
    log_info "✓ VM connection successful"

    # Get VM data counts
    VM_DNS_COUNT=$(ssh "$VM_USER@$VM_HOST" "wc -l $VM_PATH/data/vm_collected/dns_results.csv 2>/dev/null | awk '{print \$1}'" || echo "0")
    VM_WHOIS_COUNT=$(ssh "$VM_USER@$VM_HOST" "wc -l $VM_PATH/data/vm_collected/whois_results.csv 2>/dev/null | awk '{print \$1}'" || echo "0")

    # Get local data counts
    LOCAL_DNS_COUNT=$(wc -l data/vm_collected/dns_results.csv 2>/dev/null | awk '{print $1}' || echo "0")
    LOCAL_WHOIS_COUNT=$(wc -l data/vm_collected/whois_results.csv 2>/dev/null | awk '{print $1}' || echo "0")

    # Calculate new records
    NEW_DNS=$((VM_DNS_COUNT - LOCAL_DNS_COUNT))
    NEW_WHOIS=$((VM_WHOIS_COUNT - LOCAL_WHOIS_COUNT))

    log_info "VM DNS records: $VM_DNS_COUNT (local: $LOCAL_DNS_COUNT) → +$NEW_DNS new"
    log_info "VM WHOIS records: $VM_WHOIS_COUNT (local: $LOCAL_WHOIS_COUNT) → +$NEW_WHOIS new"

    if [ "$NEW_DNS" -gt 100 ] || [ "$NEW_WHOIS" -gt 100 ]; then
        log "✅ Significant new data found on VM (DNS: +$NEW_DNS, WHOIS: +$NEW_WHOIS)"
        DOWNLOAD_VM_DATA=true
    else
        log_warning "Only $NEW_DNS/$NEW_WHOIS new records - may not be worth retraining"
        log_info "Proceeding anyway for demonstration..."
        DOWNLOAD_VM_DATA=true
    fi
else
    log_warning "VM not reachable - skipping VM data download"
    log_info "Will proceed with local data only"
    DOWNLOAD_VM_DATA=false
fi

# ============================================================================
# Phase 2: Download VM Data (if available)
# ============================================================================
if [ "$DOWNLOAD_VM_DATA" = true ]; then
    log ""
    log "Phase 2: Downloading VM data..."

    mkdir -p data/vm_collected

    # Backup existing data
    if [ -f "data/vm_collected/dns_results.csv" ]; then
        cp data/vm_collected/dns_results.csv data/vm_collected/dns_results_backup_${DATE}.csv
        log_info "Backed up existing DNS data"
    fi

    if [ -f "data/vm_collected/whois_results.csv" ]; then
        cp data/vm_collected/whois_results.csv data/vm_collected/whois_results_backup_${DATE}.csv
        log_info "Backed up existing WHOIS data"
    fi

    # Download from VM
    log_info "Downloading DNS data..."
    rsync -avz --progress "$VM_USER@$VM_HOST:$VM_PATH/data/vm_collected/dns_results.csv" data/vm_collected/ >> "$LOG_FILE" 2>&1

    log_info "Downloading WHOIS data..."
    rsync -avz --progress "$VM_USER@$VM_HOST:$VM_PATH/data/vm_collected/whois_results.csv" data/vm_collected/ >> "$LOG_FILE" 2>&1

    log "✅ VM data downloaded successfully"
else
    log "⏭️  Skipping VM data download"
fi

# ============================================================================
# Phase 3: Build Model-Ready Datasets
# ============================================================================
log ""
log "Phase 3: Building model-ready datasets..."

python3 src/data_prep/dataset_builder.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "✅ Model-ready datasets created"

    # Check dataset sizes
    URL_COUNT=$(wc -l data/processed/url_features_modelready.csv 2>/dev/null | awk '{print $1}' || echo "0")
    DNS_COUNT=$(wc -l data/processed/dns_features_modelready.csv 2>/dev/null | awk '{print $1}' || echo "0")
    WHOIS_COUNT=$(wc -l data/processed/whois_features_modelready.csv 2>/dev/null | awk '{print $1}' || echo "0")

    log_info "Dataset sizes: URL=$URL_COUNT, DNS=$DNS_COUNT, WHOIS=$WHOIS_COUNT"
else
    log_error "Failed to build model-ready datasets"
    exit 1
fi

# ============================================================================
# Phase 4: Validate Data Quality
# ============================================================================
log ""
log "Phase 4: Validating data quality..."

if [ -f "quick_validation_summary.py" ]; then
    python3 quick_validation_summary.py > /tmp/validation_${DATE}.txt 2>&1
    cat /tmp/validation_${DATE}.txt >> "$LOG_FILE"

    if grep -q "READY FOR TRAINING" /tmp/validation_${DATE}.txt; then
        log "✅ Data validation passed"
    else
        log_warning "Validation warnings detected - check $LOG_FILE"
        log_info "Proceeding with training anyway..."
    fi
else
    log_warning "Validation script not found - skipping validation"
fi

# ============================================================================
# Phase 5: Train Models
# ============================================================================
log ""
log "Phase 5: Training models (this may take 30-60 minutes)..."

# Create backup of old models
if [ -d "models" ]; then
    mkdir -p "models_backup/${DATE}"
    cp models/*.pkl "models_backup/${DATE}/" 2>/dev/null || true
    log_info "Backed up existing models to models_backup/${DATE}/"
fi

# Train URL models
log_info "Training URL models..."
python3 src/training/url_train.py >> "$LOG_FILE" 2>&1 &
URL_PID=$!

# Train DNS models
log_info "Training DNS models..."
python3 src/training/dns_train.py >> "$LOG_FILE" 2>&1 &
DNS_PID=$!

# Train WHOIS models
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
# Phase 6: Analyze Results and Select Best Ensemble
# ============================================================================
log ""
log "Phase 6: Analyzing results and selecting best ensemble..."

python3 << 'PYTHON_SCRIPT' >> "$LOG_FILE" 2>&1
import pandas as pd
import json
from datetime import datetime

try:
    # Load results
    url_results = pd.read_csv('analysis/url_cv_results.csv')
    dns_results = pd.read_csv('analysis/dns_cv_results.csv')
    whois_results = pd.read_csv('analysis/whois_cv_results.csv')

    # Get best models
    url_best = url_results.loc[url_results['roc_auc'].idxmax()]
    dns_best = dns_results.loc[dns_results['roc_auc'].idxmax()]
    whois_best = whois_results.loc[whois_results['roc_auc'].idxmax()]

    # Print results
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"\n🏆 Best URL Model: {url_best['model']} ({url_best['dataset_version']})")
    print(f"   ROC-AUC: {url_best['roc_auc']:.4f} ({url_best['roc_auc']*100:.2f}%)")
    print(f"   F1 Score: {url_best['phish_f1']:.4f} ({url_best['phish_f1']*100:.2f}%)")

    print(f"\n🏆 Best DNS Model: {dns_best['model']} ({dns_best['dataset_version']})")
    print(f"   ROC-AUC: {dns_best['roc_auc']:.4f} ({dns_best['roc_auc']*100:.2f}%)")
    print(f"   F1 Score: {dns_best['phish_f1']:.4f} ({dns_best['phish_f1']*100:.2f}%)")

    print(f"\n🏆 Best WHOIS Model: {whois_best['model']} ({whois_best['dataset_version']})")
    print(f"   ROC-AUC: {whois_best['roc_auc']:.4f} ({whois_best['roc_auc']*100:.2f}%)")
    print(f"   F1 Score: {whois_best['phish_f1']:.4f} ({whois_best['phish_f1']*100:.2f}%)")

    # Save metadata
    metadata = {
        'version': datetime.now().strftime('%Y%m%d'),
        'trained_at': datetime.now().isoformat(),
        'models': {
            'url': url_best['model'],
            'dns': dns_best['model'],
            'whois': whois_best['model']
        },
        'ensemble_weights': {
            'url': 0.60,
            'whois': 0.25,
            'dns': 0.15
        },
        'performance': {
            'url_roc_auc': float(url_best['roc_auc']),
            'url_f1': float(url_best['phish_f1']),
            'dns_roc_auc': float(dns_best['roc_auc']),
            'dns_f1': float(dns_best['phish_f1']),
            'whois_roc_auc': float(whois_best['roc_auc']),
            'whois_f1': float(whois_best['phish_f1'])
        }
    }

    with open('models/production_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Production metadata saved to models/production_metadata.json")
    print("="*80)

except Exception as e:
    print(f"❌ Error analyzing results: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    log "✅ Results analyzed and metadata saved"
else
    log_error "Failed to analyze results"
    exit 1
fi

# ============================================================================
# Phase 7: Summary and Next Steps
# ============================================================================
log ""
log "========================================="
log "✅ PIPELINE COMPLETED SUCCESSFULLY"
log "========================================="

echo ""
echo "📊 Summary:"
cat models/production_metadata.json | grep -E '"url"|"dns"|"whois"|roc_auc|f1' | sed 's/^/   /'

echo ""
echo "📁 Output Files:"
echo "   - Models: models/url_*.pkl, models/dns_*.pkl, models/whois_*.pkl"
echo "   - Analysis: analysis/*_cv_results.csv"
echo "   - Metadata: models/production_metadata.json"
echo "   - Log: $LOG_FILE"

echo ""
echo "🚀 Next Steps:"
echo "   1. Review model performance in analysis/*.csv"
echo "   2. Test models: python3 -c 'from src.api.predict_utils import predict_url_risk; print(predict_url_risk(\"https://google.com\"))'"
echo "   3. Deploy to production if satisfied with results"
echo "   4. Update API configuration with new model weights"

echo ""
log "Pipeline finished at $(date +'%Y-%m-%d %H:%M:%S')"
log "Total runtime: $SECONDS seconds"

echo ""
echo "========================================="

exit 0
