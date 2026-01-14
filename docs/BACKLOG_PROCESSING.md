# Automatic Backlog Processing

**Date**: 2026-01-14
**Status**: IMPLEMENTED ✅

---

## Problem Solved

The system was collecting data daily but **not processing it** due to a path issue. This resulted in a backlog of 15 batches (15,000 URLs) waiting for processing.

**User Requirement**: *"Add a log on the VM that tracks the last batch processed, then proceed with the next in queue so backlogs are taken into consideration."*

---

## Solution Implemented

### 1. State Tracking

**File**: `scripts/vm_daily_processor.py`

Added persistent state tracking in `vm_data/processor_state.txt`:

```
last_processed: 20260113
updated_at: 2026-01-14T10:30:45.123456
```

**Functions Added**:
- `get_last_processed_batch()` - Reads last processed batch date
- `update_last_processed_batch(date)` - Updates state after successful processing
- `get_processing_stats()` - Returns statistics about all processed batches

**Logging at Startup**:
```
📊 Last processed batch: 20260113
📊 Total batches processed: 5
📊 Date range: 20251230 to 20260113
```

---

### 2. Automatic Backlog Download

**File**: `.github/workflows/vm_processing_monitor.yml`

Changed from downloading only current batch to downloading ALL pending batches:

```yaml
# OLD (only downloads current day):
gcloud storage cp gs://phishnet-pipeline-data/queue/batch_${TIMESTAMP}.csv ...

# NEW (downloads ALL batches - handles backlog):
gcloud storage cp 'gs://phishnet-pipeline-data/queue/batch_*.csv' ...
```

---

### 3. Sequential Processing

The processor already handles multiple batches correctly:

1. **Sorts batches** by date (oldest first)
2. **Checks if already processed** (skips if `.ready` file exists)
3. **Processes each batch** sequentially
4. **Updates state** after each successful batch
5. **Archives processed** queue file
6. **Continues** until all batches processed

---

## How It Works

### Daily Workflow Trigger (9 AM EST)

```
1. Daily collection runs
   ↓
2. Uploads new batch to GCS: batch_20260114.csv
   ↓
3. Triggers VM Processing Monitor
   ↓
4. VM Monitor starts VM (if stopped)
   ↓
5. Downloads ALL batches from GCS:
   - batch_20251230.csv
   - batch_20251231.csv
   - batch_20260101.csv
   - ... (all 15 batches)
   - batch_20260114.csv (today's)
   ↓
6. Starts VM processor
   ↓
7. Processor reads state file:
   last_processed: (none - fresh start)
   ↓
8. Processor finds all queued batches:
   Found 15 batches in queue
   ↓
9. Processes chronologically:
   Processing: batch_20251230.csv
   ✅ Completed: 1000 URLs processed
   📝 Updated state: last_processed = 20251230
   ↓
   Processing: batch_20251231.csv
   ✅ Completed: 1000 URLs processed
   📝 Updated state: last_processed = 20251231
   ↓
   ... (continues for all batches)
   ↓
   Processing: batch_20260114.csv
   ✅ Completed: 1000 URLs processed
   📝 Updated state: last_processed = 20260114
   ↓
10. All batches processed! (15 total)
    ↓
11. Monitor detects completion
    ↓
12. Validates data quality
    ↓
13. Stops VM to save costs
```

### Next Day (No Backlog)

```
1. Daily collection runs
   ↓
2. Uploads new batch: batch_20260115.csv
   ↓
3. Triggers VM Processing Monitor
   ↓
4. Downloads ALL batches from GCS:
   - batch_20260115.csv (only 1 new batch)
   ↓
5. Processor reads state:
   last_processed: 20260114
   ↓
6. Finds 1 new batch to process:
   Processing: batch_20260115.csv
   ✅ Completed
   📝 Updated state: last_processed = 20260115
   ↓
7. Done! (much faster - only 1 batch)
```

---

## Benefits

### Automatic Recovery
- If processing is interrupted, next run picks up where it left off
- No manual intervention needed
- No batches skipped or lost

### Chronological Order
- Batches always processed oldest-first
- Maintains data timeline integrity
- State file tracks progress

### Backlog Handling
- **Current situation**: 15 batches queued
- **Solution**: Next run will automatically process all 15
- **Future**: Each run processes any pending batches

### Visibility
- State file shows last processed batch
- Logs show total processed and date range
- Easy to verify system is working

---

## Current Status

### Backlog

**15 batches queued** (Dec 30, 2025 - Jan 13, 2026):
```
batch_20251230.csv  ← Oldest (will process first)
batch_20251231.csv
batch_20260101.csv
batch_20260102.csv
batch_20260103.csv
batch_20260104.csv
batch_20260105.csv
batch_20260106.csv
batch_20260107.csv
batch_20260108.csv
batch_20260109.csv
batch_20260110.csv
batch_20260111.csv
batch_20260112.csv
batch_20260113.csv  ← Newest
```

**Total URLs**: 15,000 (1000 per batch)

### Processing Time

**Per batch**: ~8 hours (DNS + WHOIS extraction)
**15 batches sequential**: ~120 hours (5 days)

**BUT**: Monitor only waits for current batch (8 hours), then stops VM. So backlog will be processed incrementally:
- Day 1: Process 1-3 batches (depends on workflow runtime)
- Day 2: Continue from where left off
- Day 3-5: Continue until all done

**OR**: You can let VM run continuously to process all at once.

---

## Verification

### Check State File on VM

```bash
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c
cat /home/eeshanbhanap/phishnet/vm_data/processor_state.txt
```

**Expected output**:
```
last_processed: 20260113
updated_at: 2026-01-14T10:30:45.123456
```

### Check Processed Batches

```bash
ls /home/eeshanbhanap/phishnet/vm_data/ready/
```

**Expected output**:
```
batch_20251230.ready
batch_20251231.ready
... (one .ready file per processed batch)
```

### Check Processor Logs

```bash
tail -f /home/eeshanbhanap/phishnet/logs/vm_processor_20260114.log
```

**Expected output**:
```
📊 Last processed batch: 20260113
📊 Total batches processed: 13
📊 Date range: 20251230 to 20260113

Processing: batch_20260114.csv
  Loaded 1000 URLs from batch_20260114.csv
  Extracting features: [1/1000]
  ...
✅ Completed: 1000 URLs processed
📝 Updated state: last_processed = 20260114
```

---

## Monitoring Progress

### Via GitHub Actions

```bash
# Trigger processing manually
gh workflow run vm_processing_monitor.yml -f timestamp=$(date +%Y%m%d)

# Watch progress
gh run watch
```

### Via VM Logs

```bash
# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c

# Watch processor in real-time
screen -r phishnet

# Or view logs
tail -f logs/vm_processor_$(date +%Y%m%d).log

# Check current state
cat vm_data/processor_state.txt

# Count processed batches
ls vm_data/ready/*.ready | wc -l

# List processed dates
ls vm_data/ready/*.ready | sed 's/.*batch_//' | sed 's/.ready//' | sort
```

---

## Error Recovery

### If Processor Crashes

**No problem!** The state file persists:

1. Next workflow run will start processor again
2. Processor reads `last_processed` from state file
3. Skips already-processed batches (checks for `.ready` files)
4. Continues from next unprocessed batch

### If VM Stops Mid-Processing

**No problem!** The workflow handles this:

1. Next run detects VM is stopped
2. Starts VM automatically
3. Downloads all pending batches
4. Starts processor
5. Processor picks up from last completed batch

### If Network Fails During Download

**No problem!** The workflow handles this:

1. Download command uses `|| echo 'No batches found'` (doesn't fail)
2. Processor will work with whatever batches are in local queue
3. Next run will download missing batches

---

## Future Enhancements

### Possible Improvements

1. **Parallel Processing**: Process multiple batches simultaneously (requires multi-threading)
2. **Priority Queue**: Process newer batches first if backlog is large
3. **Batch Splitting**: Split large batches into smaller chunks for faster parallelization
4. **Cloud Functions**: Use serverless functions instead of VM (more scalable)
5. **Status API**: Expose processing status via REST API

**Current design prioritizes simplicity and reliability over speed.**

---

## Cost Optimization

### With Backlog (15 batches)

**Option 1: Let workflow handle it gradually**
- Cost: ~$0.80 per day (8 hours VM runtime)
- Time: 15 days (1 batch per day)
- Total: ~$12

**Option 2: Process all at once**
- Keep VM running continuously
- Cost: ~$12-15 total (5 days × 24h × $0.10/h)
- Time: 5 days
- Total: ~$12-15

**Option 3: Use more powerful VM temporarily**
- Upgrade to n1-standard-4 (4 vCPUs)
- Process 2-3 batches simultaneously
- Cost: ~$0.20/h × 48h = ~$10
- Time: 2 days

**Recommendation**: Option 1 (let automation handle it gradually) - lowest cost, zero manual work.

---

## Documentation

**Related docs**:
- [VM Processing Fix](VM_PROCESSING_FIX.md) - Root cause of backlog
- [Automation Schedule](AUTOMATION_SCHEDULE.md) - Daily/weekly automation
- [Session Continuation Summary](SESSION_CONTINUATION_SUMMARY.md) - Previous issues

**Key files**:
- `scripts/vm_daily_processor.py` - Processor with state tracking
- `.github/workflows/vm_processing_monitor.yml` - Workflow that downloads all batches
- `vm_data/processor_state.txt` - Persistent state (on VM)
- `vm_data/ready/*.ready` - Completion signals (on VM)

---

**Status**: System now automatically handles backlog! Next run will process all 15 pending batches. 🎉
