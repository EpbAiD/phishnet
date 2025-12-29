# MLOps Pipeline - Quick Start

## 🚀 Get Started in 3 Steps

### Step 1: Test the Pipeline (Run Once)

```bash
cd /Users/eeshanbhanap/Desktop/PDF

# Run pipeline once to test
python3 scripts/mlops_pipeline.py --once
```

**Expected Output:**
```
================================================================================
🚀 MLOPS PIPELINE STARTED
Run #1
Current model: v0.0.0 (Accuracy: 0.0000)
================================================================================

STAGE 1: Data Collection & Sync
✅ Training triggered: 235 new URLs (>= 100)

STAGE 2: Data Processing & Feature Engineering
✅ Dataset ready: 32,360 total URLs

STAGE 3: Model Training
✅ Model training complete!

STAGE 4: Model Evaluation
📊 Model Performance:
  Accuracy: 0.9120
  FPR: 0.0280
✅ New model meets deployment criteria!

STAGE 5: Model Deployment
✅ Models deployed! Version: v0.0.1

✅ MLOPS PIPELINE COMPLETED SUCCESSFULLY
```

### Step 2: Run Continuously in Background

```bash
# Start in background
nohup python3 scripts/mlops_pipeline.py --continuous > logs/mlops_bg.log 2>&1 &

# Check it's running
ps aux | grep mlops_pipeline

# View logs
tail -f logs/mlops_bg.log
```

### Step 3: (Optional) Install as System Service

**macOS:**
```bash
# Create plist file
cat > ~/Library/LaunchAgents/com.phishnet.mlops.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.phishnet.mlops</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/Users/eeshanbhanap/Desktop/PDF/scripts/mlops_pipeline.py</string>
        <string>--continuous</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/eeshanbhanap/Desktop/PDF</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/eeshanbhanap/Desktop/PDF/logs/mlops_service.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/eeshanbhanap/Desktop/PDF/logs/mlops_service_error.log</string>
</dict>
</plist>
EOF

# Load service
launchctl load ~/Library/LaunchAgents/com.phishnet.mlops.plist

# Start service
launchctl start com.phishnet.mlops

# Check status
launchctl list | grep phishnet
```

**Linux:**
```bash
# Edit service file paths
sudo nano /etc/systemd/system/phishnet_mlops.service

# Enable and start
sudo systemctl enable phishnet_mlops
sudo systemctl start phishnet_mlops

# Check status
sudo systemctl status phishnet_mlops
```

---

## 📊 Monitor Progress

### Check Current Status

```bash
# View pipeline state
cat logs/pipeline_state.json

# Example output:
{
  "last_training_time": "2025-12-23T10:30:00",
  "current_model_version": "v0.0.12",
  "current_model_accuracy": 0.9120,
  "total_runs": 156,
  "successful_deployments": 12
}
```

### View Performance History

```bash
# See all model versions
cat logs/model_performance_history.json | jq '.[-5:]'

# Latest model metrics
cat logs/model_performance_history.json | jq '.[-1]'
```

### Monitor Live

```bash
# Watch logs in real-time
tail -f logs/mlops_pipeline_$(date +%Y%m%d).log

# Or background logs
tail -f logs/mlops_bg.log
```

---

## 🔧 Common Operations

### Force Retrain Now

```bash
# Manually trigger pipeline (ignores thresholds)
python3 scripts/mlops_pipeline.py --once
```

### Check What's Running

```bash
# Check VM collector
./scripts/vm_manager.sh status

# Check MLOps pipeline
ps aux | grep mlops_pipeline

# Check both
ps aux | grep -E "continuous_collector|mlops_pipeline"
```

### Stop Everything

```bash
# Stop MLOps pipeline
pkill -f mlops_pipeline

# Stop VM collector
./scripts/vm_manager.sh stop
```

### View All Logs

```bash
# MLOps pipeline
ls -lh logs/mlops_pipeline_*.log

# VM collector
ls -lh logs/continuous_v2.log

# Service logs
ls -lh logs/mlops_service*.log
```

---

## 📈 Expected Timeline

### Day 1 (Initial Setup)
```
Time  | Event
------|--------------------------------------------------------
00:00 | Start VM collector
00:10 | VM collecting URLs (rate: ~1.5 URLs/min)
06:00 | First MLOps pipeline check
      | → 235 new URLs collected
      | → Training triggered!
      | → Model v0.0.1 deployed (Accuracy: 0.91)
12:00 | Second pipeline check
      | → Only 90 new URLs (< 100 threshold)
      | → Training skipped
18:00 | Third pipeline check
      | → 110 new URLs since last training
      | → Training triggered!
      | → Model v0.0.2 deployed (Accuracy: 0.915)
```

### Week 1 Progress
```
Day | New URLs | Retrains | Latest Accuracy
----|----------|----------|----------------
1   | 235      | 2        | 0.915
2   | 180      | 1        | 0.918
3   | 190      | 1        | 0.922
4   | 175      | 1        | 0.925
5   | 185      | 1        | 0.928
6   | 180      | 1        | 0.930
7   | 190      | 1        | 0.933
```

---

## ⚙️ Configuration Tuning

### Aggressive Mode (Fast Iteration)
```python
# Edit scripts/mlops_pipeline.py
MIN_NEW_URLS = 50  # Lower threshold
MAX_TRAINING_INTERVAL_HOURS = 72  # 3 days
```

**Result:** Trains ~3x per week, faster improvement

### Conservative Mode (Stable)
```python
MIN_NEW_URLS = 200  # Higher threshold
MAX_TRAINING_INTERVAL_HOURS = 336  # 2 weeks
```

**Result:** Trains ~1x per week, more stable

### Production Mode (Balanced)
```python
MIN_NEW_URLS = 100
MAX_TRAINING_INTERVAL_HOURS = 168  # 1 week
```

**Result:** Trains ~2x per week (default, recommended)

---

## 🐛 Troubleshooting

### "No new data to train on"

```bash
# Check VM collector
./scripts/vm_manager.sh status

# If not running, start it
./scripts/vm_manager.sh start

# Force sync
./scripts/vm_manager.sh sync
```

### "Training failed"

```bash
# Check logs
grep "ERROR" logs/mlops_pipeline_*.log

# Common fixes:
# 1. Check data files exist
ls -lh data/processed/

# 2. Check models directory writable
ls -lh models/

# 3. Test training manually
python3 -c "
from src.training.url_train import train_all_url_models
train_all_url_models()
"
```

### "Model not deploying"

```bash
# Check evaluation metrics
cat logs/model_performance_history.json | jq '.[-1]'

# If accuracy too low or FPR too high:
# 1. Collect more data (wait longer)
# 2. Lower thresholds in PipelineConfig
# 3. Check training data quality
```

---

## 📦 Complete System Status

```bash
#!/bin/bash
# Save as scripts/system_status.sh

echo "=== PhishNet System Status ==="
echo ""

echo "1. VM Collector:"
./scripts/vm_manager.sh status
echo ""

echo "2. MLOps Pipeline:"
if pgrep -f mlops_pipeline > /dev/null; then
    echo "   ✅ RUNNING"
    cat logs/pipeline_state.json | jq -r '"   Model: \(.current_model_version) | Accuracy: \(.current_model_accuracy)"'
else
    echo "   ❌ NOT RUNNING"
fi
echo ""

echo "3. Latest Performance:"
cat logs/model_performance_history.json | jq -r '.[-1] | "   Accuracy: \(.accuracy) | FPR: \(.false_positive_rate) | Time: \(.timestamp)"'
echo ""

echo "4. Data Collection:"
wc -l data/vm_collected/*.csv | tail -1
echo ""

chmod +x scripts/system_status.sh
```

**Usage:**
```bash
./scripts/system_status.sh

# Output:
=== PhishNet System Status ===

1. VM Collector:
   ✅ RUNNING (PID: 1599961)
   Progress: 235 URLs processed

2. MLOps Pipeline:
   ✅ RUNNING
   Model: v0.0.12 | Accuracy: 0.9120

3. Latest Performance:
   Accuracy: 0.9120 | FPR: 0.0280 | Time: 2025-12-23T10:30:00

4. Data Collection:
   797 total
```

---

## Summary

✅ **Automated**: Runs 24/7 without intervention
✅ **Intelligent**: Trains only when beneficial
✅ **Safe**: Validates before deploying
✅ **Monitored**: Complete logging
✅ **Simple**: 3 commands to get started

**Your ML system is now fully automated!** 🎉
