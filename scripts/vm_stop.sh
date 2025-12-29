#!/bin/bash
# ============================================================================
# VM Processor Stopper - Stop continuous processor
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Configuration
VM_USER="${VM_USER:-eeshanbhanap}"
VM_HOST="${VM_HOST:-35.188.193.186}"

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log "Stopping VM processor..."

# Stop processor
ssh ${VM_USER}@${VM_HOST} "pkill -f vm_daily_processor || true"

# Kill screen session
ssh ${VM_USER}@${VM_HOST} "screen -S phishnet -X quit" 2>/dev/null || true

sleep 1

# Verify stopped
RUNNING=$(ssh ${VM_USER}@${VM_HOST} "pgrep -f vm_daily_processor" 2>/dev/null || echo "")

if [ -z "$RUNNING" ]; then
    log "✅ VM processor stopped"
else
    log "${RED}[ERROR]${NC} Failed to stop VM processor (PID: $RUNNING)"
    exit 1
fi

exit 0
