#!/bin/bash
# =============================================================================
# GCP VM Manager for PhishNet Data Collection
# =============================================================================
# Quick commands to manage your GCP VM data collector
#
# Usage:
#   ./scripts/vm_manager.sh start      # Start collector on VM
#   ./scripts/vm_manager.sh stop       # Stop collector on VM
#   ./scripts/vm_manager.sh status     # Check collector status
#   ./scripts/vm_manager.sh logs       # View collector logs
#   ./scripts/vm_manager.sh sync       # Sync data from VM to local
#   ./scripts/vm_manager.sh deploy     # Deploy latest collector code to VM
# =============================================================================

# Configuration - Updated with your VM details
VM_NAME="dns-whois-fetch-25"
VM_ZONE="us-central1-c"
VM_USER="eeshanbhanap"  # VM username
VM_IP="35.188.193.186"
SSH_KEY="~/.ssh/google_compute_engine"  # GCP default key
PROJECT_DIR="/home/$VM_USER/phishnet"

# GCP Project and Account
GCP_PROJECT="coms-452404"  # Your COMS project
GCP_ACCOUNT="eb3658@columbia.edu"  # Your Columbia email

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# SSH helper - use gcloud with specific project and account
vm_exec() {
    gcloud compute ssh "$VM_NAME" \
        --zone="$VM_ZONE" \
        --project="$GCP_PROJECT" \
        --account="$GCP_ACCOUNT" \
        --command="$1"
}

# =============================================================================
# Commands
# =============================================================================

start_collector() {
    print_status "Starting continuous data collector V2 on GCP VM..."

    vm_exec "cd $PROJECT_DIR && \
             source venv/bin/activate && \
             nohup python3 scripts/continuous_collector_v2.py > logs/continuous_v2.log 2>&1 &"

    sleep 2
    status_collector
}

stop_collector() {
    print_status "Stopping data collector on GCP VM..."
    vm_exec "pkill -f continuous_collector_v2 || pkill -f gcp_vm_data_collector"
    print_status "Collector stopped"
}

status_collector() {
    print_status "Checking collector status..."

    RUNNING=$(vm_exec "pgrep -f continuous_collector_v2 || pgrep -f gcp_vm_data_collector")

    if [ -n "$RUNNING" ]; then
        print_status "Collector is ${GREEN}RUNNING${NC} (PID: $RUNNING)"

        # Get checkpoint data
        print_status "Progress:"
        vm_exec "cd $PROJECT_DIR && \
                 [ -f data/vm_collected/checkpoint.json ] && cat data/vm_collected/checkpoint.json || echo 'No checkpoint yet'"

        # Get file count
        FILES=$(vm_exec "ls -1 $PROJECT_DIR/data/vm_collected/*.csv 2>/dev/null | wc -l")
        print_status "CSV files collected: $FILES"

    else
        print_error "Collector is ${RED}NOT RUNNING${NC}"
        print_warning "Last 20 log lines:"
        vm_exec "tail -20 $PROJECT_DIR/logs/continuous_v2.log 2>/dev/null || tail -20 $PROJECT_DIR/logs/collector.log"
    fi
}

view_logs() {
    print_status "Tailing collector logs (Ctrl+C to exit)..."
    gcloud compute ssh "$VM_NAME" \
        --zone="$VM_ZONE" \
        --project="$GCP_PROJECT" \
        --account="$GCP_ACCOUNT" \
        -- "tail -f $PROJECT_DIR/logs/continuous_v2.log 2>/dev/null || tail -f $PROJECT_DIR/logs/collector.log"
}

sync_data() {
    print_status "Syncing data from VM to local machine..."

    LOCAL_DIR="$(pwd)/data/vm_collected"
    mkdir -p "$LOCAL_DIR"

    # Use gcloud compute scp for file transfer
    gcloud compute scp \
        --recurse \
        --zone="$VM_ZONE" \
        --project="$GCP_PROJECT" \
        --account="$GCP_ACCOUNT" \
        "$VM_NAME:$PROJECT_DIR/data/vm_collected/*" \
        "$LOCAL_DIR/" 2>/dev/null || print_warning "No files to sync yet"

    print_status "Sync complete!"
    FILE_COUNT=$(ls -1 "$LOCAL_DIR"/*.csv 2>/dev/null | wc -l)
    print_status "Files downloaded: $FILE_COUNT"
}

deploy_code() {
    print_status "Deploying latest collector code to VM..."

    # Stop collector first
    print_warning "Stopping collector..."
    stop_collector

    # Sync code using gcloud compute scp
    print_status "Uploading code..."
    gcloud compute scp \
        --recurse \
        --zone="$VM_ZONE" \
        --project="$GCP_PROJECT" \
        --account="$GCP_ACCOUNT" \
        "$(pwd)/scripts" \
        "$(pwd)/src" \
        "$(pwd)/requirements.txt" \
        "$VM_NAME:$PROJECT_DIR/"

    # Restart collector
    print_status "Restarting collector..."
    start_collector

    print_status "Deployment complete!"
}

# =============================================================================
# Main
# =============================================================================

case "$1" in
    start)
        start_collector
        ;;
    stop)
        stop_collector
        ;;
    status)
        status_collector
        ;;
    logs)
        view_logs
        ;;
    sync)
        sync_data
        ;;
    deploy)
        deploy_code
        ;;
    *)
        echo "PhishNet GCP VM Manager"
        echo ""
        echo "Usage: $0 {start|stop|status|logs|sync|deploy}"
        echo ""
        echo "Commands:"
        echo "  start   - Start data collector on VM"
        echo "  stop    - Stop data collector on VM"
        echo "  status  - Check if collector is running"
        echo "  logs    - Tail collector logs (live)"
        echo "  sync    - Sync collected data from VM to local"
        echo "  deploy  - Deploy latest code to VM and restart"
        echo ""
        echo "Before using, edit this script and update VM_NAME, VM_USER, VM_IP"
        exit 1
        ;;
esac
