#!/bin/bash
# ============================================================================
# VM Data Syncer - Download processed features from VM
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VM_USER="${VM_USER:-eeshanbhanap}"
VM_HOST="${VM_HOST:-35.188.193.186}"
VM_PATH="${VM_PATH:-/home/eeshanbhanap/phishnet}"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log "Syncing processed data from VM..."
log ""

# Create local directories
mkdir -p "$PROJECT_ROOT/data/vm_processed"

# Sync incremental data from VM
log "Downloading processed features..."
scp ${VM_USER}@${VM_HOST}:${VM_PATH}/vm_data/incremental/*.csv "$PROJECT_ROOT/data/vm_processed/" 2>/dev/null || log_warning "No files to sync yet"

# Count files
FILE_COUNT=$(ls -1 "$PROJECT_ROOT/data/vm_processed"/*.csv 2>/dev/null | wc -l | tr -d ' ')
log "✅ Downloaded $FILE_COUNT batch files"

if [ "$FILE_COUNT" -gt 0 ]; then
    log ""
    log "Files downloaded to: $PROJECT_ROOT/data/vm_processed/"
    log "Next step: Run training script to merge and retrain models"
fi

exit 0
