# GCP VM Setup for Continuous Data Collection

## Overview

This guide helps you set up your existing GCP VM for 24/7 WHOIS/DNS data collection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         GCP VM                               │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  gcp_vm_data_collector.py (runs 24/7)                │   │
│  │  - Fetches Tranco + PhishTank URLs                   │   │
│  │  - Collects WHOIS/DNS features                       │   │
│  │  - Saves incrementally to CSV                        │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  data/vm_collected/*.csv                             │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             │ rsync (cron: every 6 hours)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Local Machine                            │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  weekly_retrain.py (cron: every Sunday 2am)          │   │
│  │  1. Download VM data via rsync                       │   │
│  │  2. Merge with existing training data                │   │
│  │  3. Retrain all models                               │   │
│  │  4. Validate and deploy                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Existing GCP VM instance
- SSH access configured
- Python 3.8+ installed on VM

## Step 1: Connect to Your GCP VM

```bash
# SSH into your GCP VM
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE

# Or use direct SSH if you have the IP
ssh -i ~/.ssh/your_key user@VM_IP_ADDRESS
```

## Step 2: Setup VM Environment

```bash
# Install dependencies on VM
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git

# Clone your project (if not already there)
git clone https://github.com/yourusername/phishnet.git
cd phishnet

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Step 3: Start Data Collector on VM

```bash
# Make script executable
chmod +x scripts/gcp_vm_data_collector.py

# Run in background (nohup)
nohup python3 scripts/gcp_vm_data_collector.py > logs/collector.log 2>&1 &

# Check it's running
ps aux | grep gcp_vm_data_collector

# Monitor logs
tail -f logs/collector.log
```

## Step 4: Setup Auto-Sync from VM to Local (Local Machine)

Create a script to sync data from VM:

```bash
# On your local machine
cat > ~/sync_vm_data.sh << 'EOF'
#!/bin/bash
# Sync collected data from GCP VM to local machine

VM_USER="your_username"
VM_IP="your_vm_ip"
VM_DATA_DIR="/home/$VM_USER/phishnet/data/vm_collected"
LOCAL_DATA_DIR="/Users/eeshanbhanap/Desktop/PDF/data/vm_collected"

echo "Syncing data from GCP VM..."
rsync -avz --progress \
  -e "ssh -i ~/.ssh/your_key" \
  $VM_USER@$VM_IP:$VM_DATA_DIR/ \
  $LOCAL_DATA_DIR/

echo "Sync complete! $(ls -lh $LOCAL_DATA_DIR/*.csv | wc -l) files downloaded"
EOF

chmod +x ~/sync_vm_data.sh
```

## Step 5: Setup Cron Jobs

### On GCP VM (restart collector if it crashes):

```bash
# Edit crontab
crontab -e

# Add these lines:
# Restart collector every day at midnight (if not running)
0 0 * * * cd /home/user/phishnet && pgrep -f gcp_vm_data_collector || nohup python3 scripts/gcp_vm_data_collector.py > logs/collector.log 2>&1 &
```

### On Local Machine (sync data + retrain):

```bash
# Edit crontab
crontab -e

# Add these lines:
# Sync VM data every 6 hours
0 */6 * * * ~/sync_vm_data.sh >> ~/logs/vm_sync.log 2>&1

# Retrain models every Sunday at 2am
0 2 * * 0 cd /Users/eeshanbhanap/Desktop/PDF && python3 scripts/weekly_retrain.py >> logs/retrain.log 2>&1
```

## Step 6: Monitor the Pipeline

### Check VM collector status:

```bash
# SSH into VM
gcloud compute ssh YOUR_VM_NAME

# Check if running
ps aux | grep gcp_vm_data_collector

# View logs
tail -f logs/collector.log

# Check collected data
ls -lh data/vm_collected/
wc -l data/vm_collected/*.csv
```

### Check local sync status:

```bash
# View sync logs
tail ~/logs/vm_sync.log

# Check downloaded data
ls -lh /Users/eeshanbhanap/Desktop/PDF/data/vm_collected/

# View retrain logs
tail /Users/eeshanbhanap/Desktop/PDF/logs/retrain.log
```

## Troubleshooting

### Collector stopped running:

```bash
# SSH into VM
# Check logs for errors
tail -100 logs/collector.log

# Manually restart
cd /home/user/phishnet
source venv/bin/activate
nohup python3 scripts/gcp_vm_data_collector.py > logs/collector.log 2>&1 &
```

### WHOIS/DNS lookups failing:

```bash
# Check rate limiting
# The collector uses 1 req/sec for WHOIS, 2 req/sec for DNS
# If still failing, increase SLEEP_BETWEEN_BATCHES in script
```

### VM running out of disk space:

```bash
# Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# Compress old CSV files
gzip data/vm_collected/*_$(date -d "30 days ago" +%Y%m)*.csv
```

## Performance Optimization

### Use Cloud SQL for WHOIS cache:

Instead of repeated lookups, cache WHOIS results in Cloud SQL:

```python
# TODO: Add Cloud SQL caching layer
# - Check if domain WHOIS already cached
# - If not, query and cache for 30 days
# - Reduces API load by 90%+
```

### Parallel processing with multiprocessing:

```python
# Modify collector to use multiprocessing
from multiprocessing import Pool

with Pool(processes=10) as pool:
    results = pool.map(extract_whois_features, urls)
```

## Cost Estimation

- **GCP VM e2-micro**: ~$5/month (free tier eligible)
- **Storage (100GB)**: ~$2/month
- **Egress (data transfer)**: ~$1/month

**Total**: ~$8/month for continuous data collection

## Next Steps

1. ✅ VM collector running 24/7
2. ✅ Auto-sync to local machine every 6 hours
3. ✅ Weekly model retraining on Sundays
4. 🔄 Monitor for first week
5. 📊 Track model performance improvements

## Quick Commands Cheat Sheet

```bash
# Start collector on VM
nohup python3 scripts/gcp_vm_data_collector.py > logs/collector.log 2>&1 &

# Check if running
ps aux | grep gcp_vm_data_collector

# Kill collector
pkill -f gcp_vm_data_collector

# Sync data manually (local)
~/sync_vm_data.sh

# Retrain manually (local)
cd /Users/eeshanbhanap/Desktop/PDF
python3 scripts/weekly_retrain.py

# View collected samples count
wc -l data/vm_collected/*.csv
```
