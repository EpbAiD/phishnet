# PhishNet Scripts

## Automated ML Pipeline Scripts

### Three-Stage Daily Workflow

#### Stage 1: URL Collection (Anytime - Day/Evening)

**[continuous_collector_v2.py](continuous_collector_v2.py) --mode queue**
- Run manually anytime during day or evening
- Fetches fresh URLs from multiple APIs (PhishTank, etc.)
- Saves to queue file for overnight processing
- No feature extraction yet (just URL collection)

```bash
# Local machine:
python3 scripts/continuous_collector_v2.py --mode queue --limit 1000
```

#### Stage 2: Feature Extraction (Evening/Overnight - GCP VM)

**[continuous_collector_v2.py](continuous_collector_v2.py) --mode continuous**
- Runs overnight on GCP VM
- Reads URLs from queue file (or fetches if queue empty)
- Extracts WHOIS + DNS features with rate limiting
- Smart retry: skips rate-limited URLs, retries after processing others
- Never saves zeros - only valid features
- Saves incrementally (crash-safe)

```bash
# On GCP VM (evening):
nohup python3 scripts/continuous_collector_v2.py --mode continuous > logs/collector.log 2>&1 &
```

#### Stage 3: Model Training (Morning - Local Machine)

**[mlops_pipeline.py](mlops_pipeline.py)**
- Run when you wake up in the morning
- Syncs features from VM
- Merges with existing training data
- Retrains models if conditions met (24h elapsed or >100 new features)
- Deploys only if new model outperforms current model
- Runs automatically every 6 hours in continuous mode

```bash
# Local machine (morning):
./scripts/vm_manager.sh sync          # Download features from VM
python3 scripts/mlops_pipeline.py     # Trigger training
```

### Legacy Scripts

**[gcp_vm_data_collector.py](gcp_vm_data_collector.py)**
- Legacy 24/7 collector (replaced by continuous_collector_v2.py)
- Use for reference only

### Weekly Retraining (Local)

**[weekly_retrain.py](weekly_retrain.py)**
- Runs every Sunday at 2am (cron)
- Downloads data from VM
- Merges with existing training data
- Retrains all models
- Validates and deploys

```bash
# Manual run:
python3 scripts/weekly_retrain.py

# Cron job (add to crontab -e):
0 2 * * 0 cd /Users/eeshanbhanap/Desktop/PDF && python3 scripts/weekly_retrain.py >> logs/retrain.log 2>&1
```

### VM Management (Local)

**[vm_manager.sh](vm_manager.sh)**
- Quick commands to manage GCP VM
- Start/stop collector
- View logs remotely
- Sync data
- Deploy code updates

```bash
# Usage:
./scripts/vm_manager.sh start     # Start collector
./scripts/vm_manager.sh stop      # Stop collector
./scripts/vm_manager.sh status    # Check status
./scripts/vm_manager.sh logs      # View logs
./scripts/vm_manager.sh sync      # Download data
./scripts/vm_manager.sh deploy    # Deploy code
```

## Quick Start

1. **Configure VM** - Edit `vm_manager.sh` with your VM details
2. **Deploy** - `./scripts/vm_manager.sh deploy`
3. **Setup Cron** - Add to crontab:
   ```bash
   # Sync data every 6 hours
   0 */6 * * * cd /Users/eeshanbhanap/Desktop/PDF && ./scripts/vm_manager.sh sync >> logs/sync.log 2>&1

   # Retrain every Sunday at 2am
   0 2 * * 0 cd /Users/eeshanbhanap/Desktop/PDF && python3 scripts/weekly_retrain.py >> logs/retrain.log 2>&1
   ```

## Full Guide

See [docs/QUICK_START_RETRAINING.md](../docs/QUICK_START_RETRAINING.md) for detailed setup instructions.
