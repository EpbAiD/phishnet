# PhishNet Automated Schedule

**Last Updated**: 2026-01-02
**Status**: ACTIVE

---

## Overview

PhishNet runs a fully automated pipeline for phishing URL detection with scheduled data collection and weekly model retraining.

---

## Daily Schedule (9 AM EST / 14:00 UTC)

### Daily Data Collection & Processing
**Workflow**: [`.github/workflows/daily_data_pipeline.yml`](../.github/workflows/daily_data_pipeline.yml)
**Cron**: `0 14 * * *` (Every day at 9 AM EST)
**Duration**: ~30 minutes

#### What Happens:
1. **Start VM** (if stopped)
   - GCP VM: `dns-whois-fetch-25`
   - Zone: `us-central1-c`

2. **Fetch 1000 URLs from Diverse Sources**:
   - **PhishTank** - Live phishing URLs
   - **OpenPhish** - Verified phishing feed
   - **URLhaus** - Malware/phishing URLs
   - **PhishStats** - Phishing score database
   - **Known Good** - 500 legitimate URLs for balance

3. **Extract URL Features** (39 features):
   - Domain characteristics
   - Path structure
   - Special characters
   - URL length metrics
   - Suspicious patterns

4. **Upload to Google Cloud Storage**:
   - Batch file: `gs://phishnet-pipeline-data/queue/batch_YYYYMMDD.csv`
   - URL features: `gs://phishnet-pipeline-data/incremental/url_features_YYYYMMDD.csv`

5. **Trigger VM Processing**:
   - VM polls GCS every 5 minutes
   - Extracts DNS features (32 features per URL)
   - Extracts WHOIS features (12 features per URL)
   - Uploads results to GCS incremental folder

---

## Weekly Schedule (Sundays 9 AM EST / 14:00 UTC)

### Weekly Model Retraining
**Workflow**: [`.github/workflows/weekly_model_retrain.yml`](../.github/workflows/weekly_model_retrain.yml)
**Cron**: `0 14 * * 0` (Every Sunday at 9 AM EST)
**Duration**: ~15-45 minutes (depending on data size)

#### What Happens:

**Step 1: Data Merge** (5-10 minutes)
- Download all incremental files from GCS (past week's data)
- Merge URL, DNS, WHOIS features
- Deduplicate by URL (keeps latest)
- Create main datasets in `data/processed/`

**Step 2: Growth Detection** (1 minute)
- Compare current dataset size vs previous training
- Calculate growth percentage for each feature type
- **Threshold**: 10% growth required to trigger retraining
- Save metadata for next comparison

**Step 3: Model Training** (30 minutes - only if growth detected)
- Train URL models (7 models × 3 variants = 21 models)
- Train DNS models (7 models)
- Train WHOIS models (7 models)
- **Total**: 21 models (reduced from 45 for efficiency)
- Save models to `models/` directory
- Save feature column order files for deployment

**Step 4: Validation** (2 minutes)
- Validate data quality (check for nulls, outliers)
- Verify model files created successfully
- Test predictions on sample URLs

**Step 5: Commit & Deploy** (2 minutes)
- Add trained models to git
- Add feature order files
- Add model-ready datasets
- Commit with informative message
- Push to GitHub main branch

**Step 6: Trigger Performance Monitor**
- Launch model performance monitoring workflow
- Compare new models vs previous version
- Log metrics and performance data

---

## Automation Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    DAILY (9 AM EST)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Fetch 1000 URLs (PhishTank + OpenPhish + URLhaus +     │
│     PhishStats + Legitimate)                                │
│                                                             │
│  2. Extract URL features (39 features)                      │
│                                                             │
│  3. Upload to GCS:                                          │
│     - queue/batch_YYYYMMDD.csv                              │
│     - incremental/url_features_YYYYMMDD.csv                 │
│                                                             │
│  4. VM Processing (continuous, every 5 min):                │
│     - Poll GCS for new batches                              │
│     - Extract DNS features (32)                             │
│     - Extract WHOIS features (12)                           │
│     - Upload to incremental/dns_features_*.csv              │
│     - Upload to incremental/whois_features_*.csv            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ (Accumulates daily data)
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 WEEKLY (Sunday 9 AM EST)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Download all incremental files from GCS                 │
│                                                             │
│  2. Merge with existing datasets                            │
│     - data/processed/url_features_modelready.csv            │
│     - data/processed/dns_features_modelready.csv            │
│     - data/processed/whois_features_modelready.csv          │
│                                                             │
│  3. Check Growth:                                           │
│     IF any dataset grew by >= 10%:                          │
│        ✅ Proceed to retraining                             │
│     ELSE:                                                   │
│        ⏭️  Skip retraining (save compute)                   │
│                                                             │
│  4. Train 21 Models:                                        │
│     - 7 URL models (CatBoost, XGB, LGBM, RF, LogReg,       │
│       MLP, SVM)                                             │
│     - 7 DNS models (same)                                   │
│     - 7 WHOIS models (same)                                 │
│                                                             │
│  5. Save Models + Feature Order Files:                      │
│     - models/{type}_{name}.pkl                              │
│     - models/{type}_{name}_feature_cols.pkl                 │
│                                                             │
│  6. Commit to GitHub:                                       │
│     - All models                                            │
│     - Feature order files                                   │
│     - Model-ready datasets                                  │
│     - Production metadata                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Expected Data Growth

| Timeframe | URLs Collected | Expected Rows with All 3 Features |
|-----------|----------------|-----------------------------------|
| Daily     | 1,000          | ~1,000 (VM processes all)         |
| Weekly    | 7,000          | ~7,000                            |
| Monthly   | 30,000         | ~30,000                           |

**Note**: Growth depends on VM uptime and processing speed. DNS/WHOIS features take ~8 hours to process 1000 URLs.

---

## Intelligent Retraining Logic

### Why Not Retrain Every Week?

Training 21 models is computationally expensive. We only retrain when there's significant new data:

```python
# In scripts/prepare_modelready.py
dns_growth = (current_dns_rows - prev_dns_rows) / prev_dns_rows
url_growth = (current_url_rows - prev_url_rows) / prev_url_rows
whois_growth = (current_whois_rows - prev_whois_rows) / prev_whois_rows

if dns_growth >= 0.10 or url_growth >= 0.10 or whois_growth >= 0.10:
    return True  # Trigger retraining
else:
    return False  # Skip retraining
```

### Growth Threshold: 10%

**Example**:
- Previous training: 1000 URLs
- Current dataset: 1100 URLs
- Growth: 10% ✅ **Triggers retraining**

**Example 2**:
- Previous training: 1000 URLs
- Current dataset: 1050 URLs
- Growth: 5% ⏭️ **Skips retraining**

This ensures:
- Models stay up-to-date with fresh data
- Compute resources not wasted on minimal updates
- GitHub Actions minutes used efficiently

---

## Manual Triggers

### Trigger Daily Collection Now
```bash
gh workflow run daily_data_pipeline.yml
```

### Trigger Weekly Retraining Now
```bash
gh workflow run weekly_model_retrain.yml
```

### Trigger with Custom Parameters
```bash
# Collect 5000 URLs instead of 1000
gh workflow run daily_data_pipeline.yml -f num_urls=5000

# Merge last 14 days of data instead of 7
gh workflow run weekly_model_retrain.yml -f days_to_merge=14
```

---

## Monitoring

### Check Workflow Status
```bash
# List recent workflow runs
gh run list --limit 10

# Watch live run
gh run watch

# View specific workflow runs
gh run list --workflow=daily_data_pipeline.yml --limit 5
gh run list --workflow=weekly_model_retrain.yml --limit 5
```

### Check GCS Data
```bash
# List incremental files
gcloud storage ls gs://phishnet-pipeline-data/incremental/

# Count rows in latest URL batch
DATE=$(date +%Y%m%d)
gcloud storage cat gs://phishnet-pipeline-data/incremental/url_features_${DATE}.csv | wc -l

# Check all incremental files
gcloud storage ls gs://phishnet-pipeline-data/incremental/*.csv
```

### Check VM Status
```bash
# VM status
gcloud compute instances describe dns-whois-fetch-25 \
  --zone=us-central1-c \
  --format="get(status)"

# Start VM manually
gcloud compute instances start dns-whois-fetch-25 --zone=us-central1-c

# Stop VM manually
gcloud compute instances stop dns-whois-fetch-25 --zone=us-central1-c

# SSH to VM
gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c
```

---

## Next Scheduled Runs

### Daily Collection
**Next Run**: Tomorrow at 9:00 AM EST (14:00 UTC)
**What to expect**: 1000 new URLs fetched and queued for VM processing

### Weekly Retraining
**Next Run**: Sunday, January 5, 2026 at 9:00 AM EST (14:00 UTC)
**What to expect**:
- If daily collection running: ~7000 new URLs merged
- Growth detection will trigger retraining (700% growth from current 20 URLs)
- All 21 models will be retrained and committed to GitHub

---

## Troubleshooting

### Daily Collection Failed
**Check**:
1. GitHub Actions logs: `gh run view --log`
2. GCP authentication: Verify `GCP_SA_KEY` secret is set
3. URL sources: Check if PhishTank/OpenPhish are accessible

**Fix**: Re-run manually after fixing issue

### VM Not Processing
**Check**:
1. VM status: `gcloud compute instances describe dns-whois-fetch-25 --zone=us-central1-c`
2. SSH to VM and check logs: `tail -f ~/phishnet/logs/vm_processor.log`
3. Verify GCS queue has files: `gcloud storage ls gs://phishnet-pipeline-data/queue/`

**Fix**: Restart VM processor or manually trigger

### Weekly Retraining Skipped
**Expected Behavior**: Retraining only happens when data grows by 10%+

**To force retraining**:
```bash
# Manually trigger with workflow_dispatch
gh workflow run weekly_model_retrain.yml
```

### Models Not Committed
**Check**:
1. GitHub Actions logs for commit step
2. Verify `contents: write` permission in workflow
3. Check git config in workflow (user.name, user.email)

---

## System Health Indicators

✅ **Healthy System**:
- Daily collection completes successfully
- GCS incremental folder grows daily
- VM processes batches within 24 hours
- Weekly retraining triggers when data grows
- Models committed to GitHub after training

⚠️ **Warning Signs**:
- Daily collection failing for 3+ days
- GCS incremental folder not growing
- VM stuck or processing very slowly
- Weekly retraining skipped 4+ weeks in a row
- Git commit failures in workflow

🔴 **Critical Issues**:
- All data collection stopped
- VM not accessible
- GCS bucket not accessible
- GitHub Actions disabled or quota exceeded

---

## Future Enhancements

Planned improvements to automation:

1. **Adaptive Scheduling**: Adjust collection frequency based on data needs
2. **Smart VM Management**: Auto-scale VM based on queue size
3. **Performance Tracking**: Log model metrics over time
4. **Alerting**: Slack/email notifications for failures
5. **A/B Testing**: Automatically test new model configurations
6. **Rollback**: Automatically revert if new models perform worse

---

*For production usage, see [PRODUCTION_USAGE.md](PRODUCTION_USAGE.md)*
*For system status, see [AUTOMATION_STATUS.md](AUTOMATION_STATUS.md)*
*For model details, see [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)*
