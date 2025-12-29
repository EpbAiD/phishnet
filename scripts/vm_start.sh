#!/bin/bash
# ============================================================================
# VM Processor Starter - Start continuous processor in screen session
# ============================================================================
# Uploads VM processor script and starts it in a persistent screen session
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log "Starting VM processor..."
log ""

# ============================================================================
# Step 1: Upload VM processor script
# ============================================================================
log "Step 1: Uploading VM processor script..."

# Create directories on VM
ssh ${VM_USER}@${VM_HOST} "mkdir -p ${VM_PATH}/scripts ${VM_PATH}/vm_data/url_queue ${VM_PATH}/vm_data/incremental ${VM_PATH}/logs" 2>/dev/null

# Upload Python script
scp "$PROJECT_ROOT/scripts/vm_daily_processor.py" ${VM_USER}@${VM_HOST}:${VM_PATH}/scripts/

# Upload required source files
scp -r "$PROJECT_ROOT/src" ${VM_USER}@${VM_HOST}:${VM_PATH}/ 2>/dev/null || log_warning "Failed to upload src/ (may already exist)"

log "✅ Scripts uploaded"

# ============================================================================
# Step 2: Check if already running
# ============================================================================
log ""
log "Step 2: Checking if processor is already running..."

RUNNING_PID=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor" 2>/dev/null || echo "")

if [ -n "$RUNNING_PID" ]; then
    log_warning "VM processor is already running (PID: $RUNNING_PID)"
    log "To restart, first stop it: ./scripts/vm_stop.sh"
    exit 0
fi

# ============================================================================
# Step 3: Start in screen session
# ============================================================================
log ""
log "Step 3: Starting VM processor in screen session..."

# Start processor in detached screen session named 'phishnet'
ssh ${VM_USER}@${VM_HOST} "screen -dmS phishnet bash -c 'cd ${VM_PATH} && python3 scripts/vm_daily_processor.py >> logs/vm_processor_continuous.log 2>&1'"

sleep 2

# Verify it started
RUNNING_PID=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor" 2>/dev/null || echo "")

if [ -n "$RUNNING_PID" ]; then
    log "✅ VM processor started in screen session 'phishnet' (PID: $RUNNING_PID)"
else
    log_error "Failed to start VM processor"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
log ""
log "========================================="
log "VM Processor Started Successfully"
log "========================================="
log "Screen session: phishnet"
log "Process ID: $RUNNING_PID"
log "Log file: ${VM_PATH}/logs/vm_processor_continuous.log"
log ""
log "Useful commands:"
log "  View status:  ./scripts/vm_status.sh"
log "  View logs:    ssh ${VM_USER}@${VM_HOST} 'tail -f ${VM_PATH}/logs/vm_processor_continuous.log'"
log "  Attach screen: ssh ${VM_USER}@${VM_HOST} -t 'screen -r phishnet'"
log "  Stop processor: ./scripts/vm_stop.sh"
log "========================================="
log ""
log "The processor is now running continuously and will:"
log "  1. Watch for new batches in vm_data/url_queue/"
log "  2. Extract DNS + WHOIS features"
log "  3. Save results to vm_data/incremental/"
log "  4. Signal completion with .ready files"
log ""

exit 0
