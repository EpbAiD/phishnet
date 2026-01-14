# VM Processing Monitor Simplification

**Date**: 2026-01-14
**Status**: IMPLEMENTED ✅

---

## Problem Solved

The VM processing monitor workflow was taking **8+ hours** to complete, frequently timing out and failing, because it continuously monitored the VM waiting for processing to finish.

**User Requirement**: *"we just need verification that uploaded or fetched from gcs successfully... not monitor it continuously it takes hours which actions break and fail therefore only verification of successful upload and usage that is it"*

---

## Solution Implemented

### Before: Continuous Monitoring (PROBLEMATIC)

```yaml
- Monitor VM every 2 hours for up to 8 hours
- Check DNS/WHOIS row counts each iteration
- Wait for completion (1001 rows)
- Download and validate data
- Stop VM
```

**Problems**:
- ❌ Workflow timeout: 8 hours exceeds GitHub Actions limits
- ❌ Expensive: VM running + workflow running for 8 hours
- ❌ Fragile: Network issues during monitoring cause failures
- ❌ Unnecessary: We don't need real-time completion notification

### After: Startup Verification Only (FIXED)

```yaml
- Start VM if stopped
- Download ALL batches from GCS to VM
- Start processor in screen session
- Verify processor is running
- Exit workflow (VM continues processing in background)
```

**Benefits**:
- ✅ Workflow completes in ~2 minutes
- ✅ No timeouts
- ✅ VM processes in background
- ✅ Model retraining workflow verifies completion by checking GCS

---

## How It Works Now

### Daily Collection (9 AM EST)

```
1. Daily Data Pipeline runs
   ↓
2. Collects 1000 URLs, extracts URL features
   ↓
3. Uploads to GCS: batch_YYYYMMDD.csv
   ↓
4. Triggers VM Processing Monitor
   ↓
5. VM Monitor (completes in ~2 minutes):
   - Starts VM if stopped
   - Downloads ALL batches from GCS to VM
   - Starts processor in screen session
   - Verifies processor running
   - Exits (VM continues in background)
   ↓
6. VM processes batches in background (~8 hours each)
   - Extracts DNS features
   - Extracts WHOIS features
   - Uploads to GCS: incremental/dns_YYYYMMDD.csv
   - Uploads to GCS: incremental/whois_YYYYMMDD.csv
```

### Daily Model Retraining (9 AM EST)

```
1. Model Retraining runs
   ↓
2. Downloads incremental files from GCS
   ↓
3. If files exist → VM processing worked ✅
   If files missing → VM processing failed ❌
   ↓
4. Merges with existing datasets
   ↓
5. Checks if data grew 10%+
   ↓
6. If YES: Retrains models and commits
   If NO: Skips retraining (saves compute)
```

**Key Point**: Model retraining **implicitly verifies** VM processing worked by checking if incremental files exist in GCS.

---

## Changes Made

### File: `.github/workflows/vm_processing_monitor.yml`

**Renamed job**: `monitor-vm-processing` → `start-vm-processing`

**Removed**:
- Continuous monitoring loop (90-161 lines removed)
- Data validation job (entire job removed)
- Final statistics step (removed)

**Kept**:
- VM startup
- Batch download from GCS
- Processor startup in screen session

**Added**:
- Simple verification step that processor is running
- Informative output explaining processing continues in background

**Timeout**: Changed from 600 minutes (10 hours) to 30 minutes

### File: `.github/workflows/weekly_model_retrain.yml`

**Renamed**: `Weekly Model Retraining` → `Model Retraining`

**Changed schedule**:
```yaml
# Before (weekly on Sundays):
cron: '0 14 * * 0'

# After (daily for verification):
cron: '0 14 * * *'
```

**Added comment**:
```yaml
# TEMPORARY: Run daily at 9 AM EST (14:00 UTC) for one week to verify VM processing
# After verification (week of Jan 14-21, 2026), change back to weekly: '0 14 * * 0'
```

**Why daily temporarily**:
- Verify that 15-batch backlog is being processed correctly
- Confirm VM processing works end-to-end
- Daily retraining checks GCS for incremental files (verifies VM uploaded results)
- After one week of successful runs, revert to weekly schedule

---

## Verification

### Check VM is Processing

```bash
# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c

# Check screen session
screen -list
# Should show: phishnet (Detached)

# Attach to session
screen -r phishnet
# Should see processor logs

# Detach without stopping: Ctrl+A, then D
```

### Check GCS for Processed Results

```bash
# List incremental files (VM uploads here after processing)
gcloud storage ls gs://phishnet-pipeline-data/incremental/

# Expected output (example):
# gs://phishnet-pipeline-data/incremental/dns_20251230.csv
# gs://phishnet-pipeline-data/incremental/whois_20251230.csv
# gs://phishnet-pipeline-data/incremental/dns_20251231.csv
# ... (one pair per processed batch)
```

### Check Model Retraining Success

```bash
# Check latest retraining run
gh run list --workflow=weekly_model_retrain.yml --limit 5

# Expected output:
# ✅ Model Retraining: SUCCESS (if data found in GCS)
# ⏭️  Model Retraining: SUCCESS (skipped - no data growth)
```

**If SUCCESS with retraining**: VM processing worked! Incremental files were found in GCS.

**If FAILURE**: VM processing failed or files not uploaded to GCS yet.

---

## Monitoring Strategy

### Old Strategy (Problematic)
- Monitor VM continuously from GitHub Actions
- Wait 8 hours for completion
- Download and validate immediately
- Tightly coupled: collection → processing → validation

### New Strategy (Robust)
- Verify VM started processing
- Let VM work independently in background
- Model retraining checks GCS (decoupled verification)
- Loose coupling: collection → processing (async) → retraining (checks GCS)

**Benefits**:
1. **Fault tolerance**: If workflow fails, VM keeps processing
2. **Cost efficiency**: Workflow runs 2 min instead of 8 hours
3. **Simplicity**: Each workflow has one responsibility
4. **Scalability**: Can process multiple batches without workflow running

---

## Expected Timeline

### Day 1 (Jan 14, 2026)

```
09:00 AM - Daily collection runs, uploads batch_20260114.csv
09:03 AM - VM monitor starts VM, downloads ALL 15 batches, starts processor
09:05 AM - VM monitor completes ✅
09:05 AM - VM processes batch_20251230.csv in background (~8 hours)
05:00 PM - VM finishes batch_20251230, uploads to GCS
05:01 PM - VM starts batch_20251231.csv
```

### Day 2 (Jan 15, 2026)

```
01:00 AM - VM finishes batch_20251231, uploads to GCS
01:01 AM - VM starts batch_20260101.csv
...
09:00 AM - Daily collection runs, uploads batch_20260115.csv
09:00 AM - Model retraining runs:
           - Finds dns_20251230.csv in GCS ✅
           - Finds whois_20251230.csv in GCS ✅
           - Confirms VM is working!
           - Retrains models (if 10% growth threshold met)
09:03 AM - VM monitor adds batch_20260115.csv to queue
...
```

**By Jan 21, 2026**: All 15 backlog batches processed + 7 new daily batches = 22 batches total.

---

## When to Revert to Weekly Retraining

**After one week** (Jan 21, 2026), if model retraining shows consistent success:

```bash
# Edit workflow file
# Change line 7 from:
    - cron: '0 14 * * *'  # Daily at 9 AM EST

# Back to:
    - cron: '0 14 * * 0'  # Weekly on Sundays at 9 AM EST

# Commit and push
git add .github/workflows/weekly_model_retrain.yml
git commit -m "Revert to weekly retraining after VM verification complete"
git push
```

**Verification criteria**:
- ✅ At least 7 successful model retraining runs
- ✅ GCS shows incremental files for all dates
- ✅ VM processor state file shows progress
- ✅ No workflow timeouts or failures

---

## Troubleshooting

### If VM Monitor Fails

**Error**: "Processor not detected in screen session"

**Cause**: Processor failed to start or crashed

**Fix**:
```bash
# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c

# Check logs
tail -f ~/phishnet/logs/vm_processor_$(date +%Y%m%d).log

# Restart processor manually
cd ~/phishnet
screen -dmS phishnet python3 scripts/vm_daily_processor.py
```

### If Model Retraining Shows No Data

**Error**: "No incremental files found in GCS"

**Cause**: VM hasn't finished processing yet (takes ~8 hours per batch)

**Fix**: Wait longer or check VM status:
```bash
# Check what VM is processing
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c
screen -r phishnet

# Check GCS queue
gcloud storage ls gs://phishnet-pipeline-data/queue/

# Check GCS incremental (processed results)
gcloud storage ls gs://phishnet-pipeline-data/incremental/
```

### If Processing Takes Too Long

**Issue**: 15 batches = ~120 hours = 5 days

**Options**:

1. **Wait it out** (recommended): Let automation handle it gradually
   - Cost: ~$0.80/day (8 hours VM runtime)
   - Time: ~15 days (1 batch per run)

2. **Keep VM running**: Don't stop VM, process all batches continuously
   - Cost: ~$12 total (5 days × 24h × $0.10/h)
   - Time: 5 days
   - Command: Comment out stop-vm job in workflow

3. **Process in parallel**: Modify processor to handle multiple batches simultaneously
   - Requires code changes (not currently implemented)

---

## Related Documentation

- [Backlog Processing](BACKLOG_PROCESSING.md) - Automatic backlog handling
- [VM Processing Fix](VM_PROCESSING_FIX.md) - Root cause of original issue
- [Automation Schedule](AUTOMATION_SCHEDULE.md) - Complete automation schedule

**Key files**:
- `.github/workflows/vm_processing_monitor.yml` - Simplified startup verification
- `.github/workflows/weekly_model_retrain.yml` - Temporary daily schedule
- `scripts/vm_daily_processor.py` - Processor with state tracking

---

**Status**: Workflows simplified. VM processing verified by model retraining checking GCS. 🎉
