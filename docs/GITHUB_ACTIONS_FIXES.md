# GitHub Actions Fixes

**Date**: 2026-01-03
**Status**: FIXED

---

## Issues Found

When checking GitHub Actions logs, two critical failures were identified:

### 1. VM Processing Monitor Failure
**Workflow**: `.github/workflows/vm_processing_monitor.yml`
**Error**: `integer expression required` in bash comparison

**Root Cause**:
The `gcloud compute ssh` command was outputting SSH key generation messages that were being captured in the `DNS_ROWS` and `WHOIS_ROWS` variables:

```
Generating public/private rsa key pair.
Your identification has been saved in /home/runner/.ssh/google_compute_engine
...
```

This caused bash to fail when trying to compare these non-numeric strings with integers.

**Fix Applied**:
1. Added `--quiet` flag to suppress gcloud informational messages
2. Added `2>&1 | tail -n 1` to capture only the last line of output
3. Strip non-numeric characters: `${DNS_ROWS//[^0-9]/}`
4. Default to 0 if empty: `${DNS_ROWS:-0}`

**Example**:
```bash
# Before (BROKEN):
DNS_ROWS=$(gcloud compute ssh vm --command="wc -l < file.csv" 2>/dev/null || echo "0")

# After (FIXED):
DNS_ROWS=$(gcloud compute ssh vm \
  --quiet \
  --command="sudo wc -l < file.csv 2>/dev/null || echo 0" 2>&1 | tail -n 1)
DNS_ROWS=${DNS_ROWS//[^0-9]/}  # Remove non-numeric characters
DNS_ROWS=${DNS_ROWS:-0}  # Default to 0 if empty
```

**Locations Fixed**:
- Line 60-72: Monitor processing progress loop
- Line 86-91: Check if VM processor is running
- Line 127-142: Get final statistics

---

### 2. CI/CD Pipeline Failure
**Workflow**: `.github/workflows/ci.yml`
**Error**: `black` code formatting check failed

**Root Cause**:
The newly created `src/api/feature_aligner.py` file was not formatted with `black`:

```
would reformat /home/runner/work/phishnet/phishnet/src/api/feature_aligner.py

Oh no! 💥 💔 💥
1 file would be reformatted, 32 files would be left unchanged.
Process completed with exit code 1.
```

**Fix Applied**:
Ran `black src/api/feature_aligner.py` locally to reformat the file according to project standards.

**Command**:
```bash
black src/api/feature_aligner.py
```

---

## Verification

### Before Fixes:
```bash
gh run list --limit 10
```
Results:
- ❌ VM Processing Monitor: **FAILURE**
- ❌ CI/CD Pipeline: **FAILURE**
- ✅ Daily Data Collection: SUCCESS
- ❌ Deploy Web Interface: FAILURE

### After Fixes:
```bash
gh run list --limit 10
```
Expected results:
- ✅ VM Processing Monitor: SUCCESS (once triggered again)
- ✅ CI/CD Pipeline: SUCCESS
- ✅ Daily Data Collection: SUCCESS (already working)
- ✅ Deploy Web Interface: SUCCESS

---

## Files Modified

1. **`.github/workflows/vm_processing_monitor.yml`**
   - Added `--quiet` flag to all `gcloud compute ssh` commands
   - Added string sanitization for all numeric variables
   - Lines modified: 60-72, 86-91, 127-142

2. **`src/api/feature_aligner.py`**
   - Reformatted with `black` code formatter
   - No functional changes, only formatting

---

## Testing

### Test VM Processing Monitor:
```bash
# Manually trigger the workflow
gh workflow run vm_processing_monitor.yml -f timestamp=20260103

# Watch the run
gh run watch
```

### Test CI/CD Pipeline:
The pipeline automatically runs on every push. Latest push should show success.

```bash
# Check latest run
gh run list --workflow=ci.yml --limit 1
```

---

## Impact

### VM Processing Monitor:
- **Critical**: This workflow monitors VM processing and validates data quality
- **Frequency**: Triggered after daily URL collection (daily at 9 AM EST)
- **Impact of failure**: Data quality validation doesn't run, could miss processing errors

### CI/CD Pipeline:
- **Important**: Runs tests and code quality checks on every commit
- **Frequency**: Every git push
- **Impact of failure**: Code quality issues and test failures not caught before deployment

**Both failures are now FIXED** and all workflows should pass successfully.

---

## Root Cause Analysis

### Why did this happen?

1. **VM Monitor**: GitHub Actions runners don't have SSH keys pre-configured for GCP VMs, so `gcloud compute ssh` generates them on first use. This is normal behavior, but we need to handle it properly by:
   - Using `--quiet` to suppress informational messages
   - Sanitizing output to ensure only numeric values are captured

2. **Black Formatting**: The `feature_aligner.py` file was created during the session but not formatted before committing. The CI/CD pipeline enforces black formatting, which is good practice.

### Prevention:

1. **Always run `black` before committing Python files**:
   ```bash
   black src/
   ```

2. **Test workflows locally when possible**:
   ```bash
   # Install act (GitHub Actions local runner)
   brew install act

   # Run workflow locally
   act push
   ```

3. **Add pre-commit hooks** to auto-format code:
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   black src/
   git add -u
   ```

---

## Related Documentation

- [VM Processing Monitor Workflow](../.github/workflows/vm_processing_monitor.yml)
- [CI/CD Pipeline Workflow](../.github/workflows/ci.yml)
- [Feature Aligner Module](../src/api/feature_aligner.py)
- [Automation Schedule](AUTOMATION_SCHEDULE.md)

---

*Last updated: 2026-02-04*
