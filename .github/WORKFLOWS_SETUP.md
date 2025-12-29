# GitHub Actions Workflows Setup Guide

This document explains how to set up and configure all GitHub Actions workflows for the PhishNet phishing detection system.

## Overview

The system includes 4 main workflows:

1. **Daily Data Pipeline** - Automated URL collection and VM upload
2. **VM Processing Monitor** - Long-running VM processing with data quality validation
3. **Model Performance Monitoring** - Track model metrics and detect drift
4. **Web Deployment** - Deploy web interface and extension to GitHub Pages

## Prerequisites

### 1. GitHub Repository Settings

#### Enable GitHub Actions
1. Go to your repository on GitHub
2. Navigate to **Settings** → **Actions** → **General**
3. Under "Actions permissions", select **Allow all actions and reusable workflows**
4. Click **Save**

#### Enable GitHub Pages
1. Go to **Settings** → **Pages**
2. Under "Build and deployment":
   - Source: **GitHub Actions**
3. Click **Save**

### 2. Required GitHub Secrets

You need to add the following secrets to your repository:

#### Navigate to Secrets
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**

#### Required Secrets

##### `GCP_SA_KEY` - Google Cloud Service Account Key

This allows GitHub Actions to authenticate with GCP to manage your VM and download data.

**How to create:**

```bash
# 1. Create a service account
gcloud iam service-accounts create github-actions \
    --description="Service account for GitHub Actions" \
    --display-name="GitHub Actions"

# 2. Grant necessary permissions
PROJECT_ID="coms-452404"
SA_EMAIL="github-actions@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.instanceAdmin.v1"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.osLogin"

# 3. Create and download key
gcloud iam service-accounts keys create ~/gcp-key.json \
    --iam-account=$SA_EMAIL

# 4. Copy the key content
cat ~/gcp-key.json

# 5. Add to GitHub:
#    - Name: GCP_SA_KEY
#    - Value: Paste the entire JSON content
#    - Click "Add secret"

# 6. Securely delete the local key file
rm ~/gcp-key.json
```

**IMPORTANT**: Never commit this key to your repository!

#### Optional Secrets (for notifications)

##### `SLACK_WEBHOOK_URL` - Slack notifications (optional)
If you want Slack notifications for workflow failures:
1. Create a Slack webhook at https://api.slack.com/messaging/webhooks
2. Add as secret with name `SLACK_WEBHOOK_URL`

## Workflow Configuration

### 1. Daily Data Pipeline

**File**: `.github/workflows/daily_data_pipeline.yml`

**Trigger**: Daily at 2 AM UTC (configurable)

**What it does**:
- Fetches 1000 URLs from phishing sources
- Extracts URL features locally
- Starts GCP VM if stopped
- Uploads URLs to VM queue
- Verifies VM processor started
- Triggers VM processing monitor

**Configuration**:

```yaml
# Change schedule (cron syntax: minute hour day month weekday)
schedule:
  - cron: '0 2 * * *'  # Daily at 2 AM UTC

# To run at different times:
# - cron: '0 14 * * *'  # 2 PM UTC (10 AM EST)
# - cron: '0 6 * * *'   # 6 AM UTC (2 AM EST)
# - cron: '0 */6 * * *' # Every 6 hours
```

**Manual trigger**:
1. Go to **Actions** → **Daily Data Collection & Processing**
2. Click **Run workflow**
3. Optionally specify number of URLs (default: 1000)

### 2. VM Processing Monitor

**File**: `.github/workflows/vm_processing_monitor.yml`

**Trigger**: Automatically after Daily Data Pipeline, or manual

**What it does**:
- Monitors VM processing every 5 minutes
- Checks for completion (1001 rows in both CSV files)
- Validates data quality (95%+ success rates)
- Downloads validated data
- Merges with main dataset
- Retrains models
- Commits updated models to repository
- Stops VM to save costs

**Configuration**:

```yaml
# Maximum wait time (default: 8 hours)
max_wait_hours: '8'

# Data quality thresholds
DNS_THRESHOLD = 95.0      # 95% DNS success rate
WHOIS_THRESHOLD = 95.0    # 95% WHOIS success rate
```

**Important**: This workflow can run up to 10 hours (GitHub Actions timeout). If your processing takes longer, consider:
- Using a self-hosted runner
- Splitting into multiple jobs
- Increasing VM resources

### 3. Model Performance Monitoring

**File**: `.github/workflows/model_performance_monitor.yml`

**Trigger**: Automatically after model retraining completes

**What it does**:
- Evaluates all models on test set
- Calculates accuracy, precision, recall, F1, AUC
- Compares with previous performance
- Detects model drift (>5% degradation)
- Saves performance history
- Creates performance report

**Configuration**:

```yaml
# Performance thresholds
THRESHOLDS = {
    'accuracy': 0.90,    # 90% minimum accuracy
    'precision': 0.90,   # 90% minimum precision
    'recall': 0.85,      # 85% minimum recall
    'f1': 0.88           # 88% minimum F1 score
}

# Drift detection
DRIFT_THRESHOLD = 0.05   # 5% degradation triggers alert
```

### 4. Web Deployment

**File**: `.github/workflows/deploy_web.yml`

**Trigger**: On push to main branch (changes to web/, extension/, docs/)

**What it does**:
- Builds web interface
- Builds browser extension
- Deploys to GitHub Pages
- Makes extension downloadable

**Your website will be available at**:
```
https://[your-username].github.io/[repo-name]/
```

Extension download:
```
https://[your-username].github.io/[repo-name]/extension/
```

## Workflow Execution Order

```
1. Daily Data Pipeline (2 AM UTC)
   ↓
2. VM Processing Monitor (triggered automatically)
   ↓ (waits for VM to complete ~2-8 hours)
   ↓
3. Model Retrain (part of step 2)
   ↓
4. Model Performance Monitor (triggered after retraining)
   ↓
5. Web Deployment (if models updated on main branch)
```

## Monitoring Workflows

### View Workflow Runs
1. Go to **Actions** tab in your repository
2. See all workflow runs with status (success/failure/in progress)
3. Click on a run to see detailed logs

### View Artifacts
Workflows save artifacts (trained models, data, reports) that you can download:
1. Go to specific workflow run
2. Scroll to **Artifacts** section
3. Download zip files

**Available artifacts**:
- `validated-features-YYYYMMDD` - CSV files with features (30 day retention)
- `trained-models-YYYYMMDD` - Trained model files (90 day retention)
- `model-performance-report` - JSON with metrics

### Debugging Failed Workflows

**If a workflow fails**:
1. Click on the failed run
2. Click on the failed job
3. Expand the failed step to see error logs
4. Common issues:
   - **GCP authentication failed** → Check `GCP_SA_KEY` secret is correct
   - **VM not found** → Check VM name/zone in workflow env vars
   - **Timeout** → Increase `timeout-minutes` or split job
   - **Data quality failed** → Check VM processing logs

## Cost Optimization

### VM Auto-Stop
The workflow automatically stops the VM after processing to save costs:
- VM starts: ~$0.10/hour
- VM stopped: ~$0.01/hour (disk storage only)
- Savings: ~$2-3/day for 8-hour runs

### GitHub Actions Usage
- **Free tier**: 2,000 minutes/month for private repos
- **Expected usage**: ~30 minutes/day = ~900 minutes/month
- **Recommendation**: Should fit in free tier

## Customization

### Change VM Configuration

Edit workflow env vars:

```yaml
env:
  GCP_PROJECT_ID: coms-452404
  GCP_ZONE: us-central1-c
  VM_NAME: dns-whois-fetch-25
  GCP_ACCOUNT: eb3658@columbia.edu
```

### Add Email Notifications

Add this step to any workflow:

```yaml
- name: Send email notification
  if: failure()
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 465
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: Workflow ${{ github.workflow }} failed
    to: your-email@example.com
    from: GitHub Actions
    body: Workflow ${{ github.workflow }} failed in repository ${{ github.repository }}
```

### Disable Auto-VM-Stop

If you want to keep the VM running:

Comment out or remove the `stop-vm` job in `vm_processing_monitor.yml`:

```yaml
# stop-vm:
#   name: Stop VM to Save Costs
#   ...
```

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Use GitHub Secrets** for all sensitive data
3. **Limit service account permissions** to minimum required
4. **Rotate service account keys** every 90 days
5. **Review workflow logs** for exposed sensitive data
6. **Use `if: always()` carefully** - it runs even if previous steps failed

## Testing Workflows

### Test without triggering scheduled runs

1. Use `workflow_dispatch` to manually trigger
2. Test with smaller datasets first (set `num_urls: 100`)
3. Check VM processing with reduced URLs
4. Verify all steps complete successfully

### Local testing (limited)

Some checks can be run locally:

```bash
# Validate workflow YAML syntax
pip install yamllint
yamllint .github/workflows/*.yml

# Test Python scripts locally
python scripts/collect_urls_daily.sh
python scripts/test_full_pipeline.py
```

## Troubleshooting

### Problem: Workflow doesn't trigger on schedule

**Solution**:
- Check if repository is active (GitHub pauses workflows in inactive repos)
- Ensure workflow file is on default branch (main)
- Check `.github/workflows/` directory exists
- Verify cron syntax is correct

### Problem: GCP authentication fails

**Solution**:
```bash
# Verify service account has correct permissions
gcloud projects get-iam-policy coms-452404 \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:github-actions*"

# Check key is valid JSON
echo "$GCP_SA_KEY" | jq .
```

### Problem: VM processing timeout

**Solution**:
- Increase `max_wait_hours` input
- Increase `timeout-minutes` in workflow (max 360 for free tier)
- Consider using self-hosted runner (no timeout limit)
- Optimize VM processing speed

### Problem: Model performance below threshold

**Solution**:
- Check data quality validation results
- Review feature extraction success rates
- Investigate recent data changes
- Consider retraining with more data

## Support

For issues with workflows:
1. Check workflow run logs
2. Review this documentation
3. Check GitHub Actions documentation: https://docs.github.com/actions

---

**Author**: Eeshan Bhanap (eb3658@columbia.edu)
**Last Updated**: 2024-12-28
