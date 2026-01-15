# VM Processor Migration Fix

**Date**: 2026-01-15
**Status**: FIXED ✅

---

## Problem Found

The VM was running the **old processor** (`vm_gcs_processor.py`) instead of the **new processor** (`vm_daily_processor.py`).

**Symptoms**:
- Logs showed "No batches in queue, waiting..." continuously
- Batches existed in `/home/eeshanbhanap/phishnet/vm_data/url_queue/` but weren't being processed
- Old processor was checking GCS queue directly (which was empty)
- New processor with backlog tracking wasn't running

---

## Root Cause

### Old Processor (vm_gcs_processor.py)
- Checks **GCS** queue directly: `gs://phishnet-pipeline-data/queue/`
- Downloads batch → processes → uploads results
- Deletes batch from GCS after processing
- Doesn't support backlog tracking or state persistence

### New Processor (vm_daily_processor.py)
- Checks **local** queue: `/home/eeshanbhanap/phishnet/vm_data/url_queue/`
- Processes local batches sequentially
- Supports backlog tracking with state file
- Archives processed batches instead of deleting

### What Happened

1. Old processor started automatically on VM boot (as a systemd service or cron job)
2. Old processor ran continuously, checking GCS queue
3. GCS queue was empty because:
   - Daily pipeline uploads batch to GCS
   - VM monitor tries to download batch to VM
   - But old processor already processed and deleted from GCS
4. Local queue had 2 unprocessed batches from before the migration
5. New processor was never started

---

## Solution Implemented

### Step 1: Identified Running Processor

```bash
ps aux | grep python | grep -v grep
# Output showed:
# eeshanbhanap 816  3.2  2.3 ... /usr/bin/python3 /home/eeshanbhanap/phishnet/scripts/vm_gcs_processor.py
```

### Step 2: Stopped Old Processor

```bash
kill 816  # PID of old processor
```

### Step 3: Started New Processor

```bash
cd /home/eeshanbhanap/phishnet
screen -dmS phishnet python3 scripts/vm_daily_processor.py
```

### Step 4: Verified New Processor Working

```bash
tail -f logs/vm_processor_20260115.log
# Output:
# 2026-01-15 13:51:08 - INFO - 📦 Found 2 batches to process
# 2026-01-15 13:51:08 - INFO - Processing: batch_20251229.csv
# 2026-01-15 13:51:08 - INFO -   Loaded 1000 URLs from vm_data/url_queue/batch_20251229.csv
```

---

## Current Status

### Batches Being Processed

1. **batch_20251229.csv** - 1000 URLs (processing now)
2. **batch_20251230.csv** - 1000 URLs (queued)

**Processing time**: ~8 hours per batch = ~16 hours total

### Expected Timeline

```
Now (13:51): Started processing batch_20251229.csv
~21:51 today: batch_20251229 complete, start batch_20251230
~05:51 tomorrow: batch_20251230 complete
```

### Tomorrow's Daily Collection (9 AM EST)

```
09:00: Daily collection runs
09:01: Uploads batch_20260116.csv to GCS
09:02: VM monitor downloads to local queue
09:03: New processor picks up batch_20260116.csv
~17:00: batch_20260116.csv processing complete
```

---

## Workflow Modifications Needed

The current workflow tries to start the processor but fails due to the old processor already running. We need to ensure the old processor is stopped before starting the new one.

### Current Workflow Issue

```yaml
# In vm_processing_monitor.yml
--command="cd /home/eeshanbhanap/phishnet && screen -dmS phishnet python3 scripts/vm_daily_processor.py"
```

**Problem**: If old processor is running, this command succeeds but the new processor never actually processes local files.

### Recommended Fix

Update the workflow to:
1. Check if old processor (`vm_gcs_processor.py`) is running
2. Kill old processor if found
3. Start new processor (`vm_daily_processor.py`)

```yaml
- name: Start VM and trigger processing
  run: |
    # ... (existing code for starting VM)

    echo "Stopping old processor if running..."
    gcloud compute ssh ${{ env.VM_NAME }} \
      --zone=${{ env.GCP_ZONE }} \
      --quiet \
      --command="pkill -f 'vm_gcs_processor.py' || echo 'Old processor not running'"

    sleep 2

    echo "Starting new processor..."
    gcloud compute ssh ${{ env.VM_NAME }} \
      --zone=${{ env.GCP_ZONE }} \
      --quiet \
      --command="cd /home/eeshanbhanap/phishnet && screen -dmS phishnet python3 scripts/vm_daily_processor.py"
```

---

## Long-term Fix: Disable Old Processor Autostart

The old processor is probably started automatically via:
- Systemd service
- Cron job
- `/etc/rc.local`

### Find and Disable Autostart

```bash
# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c

# Check systemd services
systemctl list-units --type=service | grep phishnet
sudo systemctl disable phishnet-processor.service  # if found

# Check crontab
crontab -l | grep vm_gcs_processor
crontab -e  # remove old processor line

# Check root crontab
sudo crontab -l | grep vm_gcs_processor
sudo crontab -e  # remove old processor line

# Check rc.local
cat /etc/rc.local | grep vm_gcs_processor
sudo vim /etc/rc.local  # remove old processor line
```

---

## Verification

### Check Processor is Running

```bash
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c
screen -list
# Should show: phishnet (Detached)

screen -r phishnet
# Should see: Processing URLs...
# Ctrl+A then D to detach
```

### Check Progress

```bash
tail -f logs/vm_processor_20260115.log
```

### Check State Tracking

```bash
cat vm_data/processor_state.txt
# Should show:
# last_processed: 20251229  (after first batch completes)
# updated_at: 2026-01-15T21:51:00.000000
```

### Check Ready Signals

```bash
ls vm_data/ready/
# Should show:
# batch_20251228.ready (old)
# batch_20251229.ready (after completion)
# batch_20251230.ready (after completion)
```

---

## Monitoring Commands

### Watch Processing Live

```bash
# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c

# Attach to processor screen session
screen -r phishnet

# Watch logs in real-time
tail -f logs/vm_processor_20260115.log

# Check how many URLs processed so far
wc -l vm_data/incremental/dns_20251229.csv
# Expected: Grows from 1 to 1001 over ~8 hours
```

### Check Processing Speed

```bash
# Count rows every hour
watch -n 3600 'wc -l /home/eeshanbhanap/phishnet/vm_data/incremental/dns_20251229.csv'

# Expected rate: ~125 URLs per hour (1000 URLs / 8 hours)
```

---

## Files Modified

None yet - manual fix applied. Workflow update recommended but not yet implemented.

**Recommended changes**:
- `.github/workflows/vm_processing_monitor.yml` - Add old processor kill step

---

## Related Documentation

- [Monitoring Simplification](MONITORING_SIMPLIFICATION.md) - Why we simplified monitoring
- [Backlog Processing](BACKLOG_PROCESSING.md) - State tracking implementation
- [VM Processing Fix](VM_PROCESSING_FIX.md) - Original path fix

---

## Success Criteria

✅ **Old processor stopped** (vm_gcs_processor.py killed)
✅ **New processor started** (vm_daily_processor.py running in screen)
✅ **Batches being processed** (Found 2 batches in local queue)
✅ **State tracking working** (Processor logs show proper initialization)

**Next steps**:
- Wait for batch_20251229 to complete (~8 hours)
- Verify batch_20251230 starts automatically
- Update workflow to prevent old processor from interfering

---

**Status**: New processor running successfully! Processing 2 backlog batches. 🎉
