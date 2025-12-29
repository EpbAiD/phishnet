#!/bin/bash
# ============================================================================
# VM Processor Status Checker
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
VM_USER="${VM_USER:-eeshanbhanap}"
VM_HOST="${VM_HOST:-35.188.193.186}"
VM_PATH="${VM_PATH:-/home/eeshanbhanap/phishnet}"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log "Checking VM processor status..."
log ""

# Check if process is running
RUNNING_PID=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor" 2>/dev/null || echo "")

if [ -n "$RUNNING_PID" ]; then
    log "✅ VM processor is ${GREEN}RUNNING${NC} (PID: $RUNNING_PID)"

    # Check queue status
    log ""
    log "Queue status:"
    PENDING=$(ssh ${VM_USER}@${VM_HOST} "ls -1 ${VM_PATH}/vm_data/url_queue/*.csv 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')
    COMPLETED=$(ssh ${VM_USER}@${VM_HOST} "ls -1 ${VM_PATH}/vm_data/incremental/*.csv 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ')

    log "  Pending batches: $PENDING"
    log "  Completed batches: $COMPLETED"

    # Show recent log lines
    log ""
    log "Recent activity (last 10 lines):"
    ssh ${VM_USER}@${VM_HOST} "tail -10 ${VM_PATH}/logs/vm_processor_continuous.log 2>/dev/null" || log_error "Could not read logs"

else
    log_error "VM processor is ${RED}NOT RUNNING${NC}"
    log ""
    log "To start: ./scripts/vm_start.sh"
fi

exit 0
