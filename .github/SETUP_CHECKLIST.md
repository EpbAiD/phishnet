# Setup Checklist

Use this checklist to ensure everything is configured correctly before running workflows.

## ☐ Pre-Deployment Checklist

### GitHub Repository Configuration

- [ ] Repository pushed to GitHub
- [ ] Default branch is `main`
- [ ] Repository is private or public (your choice)
- [ ] Actions are enabled in repository settings

### GCP Service Account Setup

- [ ] GCP service account created (`github-actions`)
- [ ] Service account has `compute.instanceAdmin.v1` role
- [ ] Service account has `compute.osLogin` role (for SSH)
- [ ] Service account key downloaded as JSON
- [ ] JSON key added to GitHub Secrets as `GCP_SA_KEY`
- [ ] Local JSON key file deleted securely

### GitHub Secrets

Navigate to: `Settings` → `Secrets and variables` → `Actions`

- [ ] `GCP_SA_KEY` added (entire JSON content)
- [ ] Secret tested (run a simple workflow)

### GitHub Pages

Navigate to: `Settings` → `Pages`

- [ ] Source set to **GitHub Actions**
- [ ] Pages enabled and URL visible

### VM Configuration

- [ ] VM exists and is accessible
- [ ] VM name matches workflow env: `dns-whois-fetch-25`
- [ ] VM zone matches workflow env: `us-central1-c`
- [ ] VM has required software installed:
  - [ ] Python 3.11+
  - [ ] venv with dependencies
  - [ ] DNS/WHOIS processing scripts
  - [ ] Screen (for background processes)

### Repository Files

- [ ] All workflow files in `.github/workflows/`:
  - [ ] `daily_data_pipeline.yml`
  - [ ] `vm_processing_monitor.yml`
  - [ ] `model_performance_monitor.yml`
  - [ ] `deploy_web.yml`
  - [ ] `ci.yml`
- [ ] Documentation in `.github/`:
  - [ ] `README.md`
  - [ ] `QUICK_START.md`
  - [ ] `WORKFLOWS_SETUP.md`
  - [ ] `FEATURES.md`
  - [ ] `SETUP_CHECKLIST.md` (this file)
- [ ] `.mailmap` for git attribution
- [ ] Scripts are executable:
  - [ ] `scripts/collect_urls_daily.sh`
  - [ ] `scripts/daily_model_retrain.sh`
  - [ ] `scripts/vm_daily_processor_optimized.py`

## ☐ Pre-Push Verification

### Local Testing

- [ ] Scripts work locally:
  ```bash
  python3 scripts/fetch_urls.py
  python3 scripts/extract_url_features.py
  ```
- [ ] VM is accessible:
  ```bash
  gcloud compute ssh dns-whois-fetch-25 --zone=us-central1-c
  ```
- [ ] VM processor works:
  ```bash
  # On VM
  python3 scripts/vm_daily_processor_optimized.py
  ```

### Workflow YAML Validation

- [ ] YAML syntax is valid:
  ```bash
  pip install yamllint
  yamllint .github/workflows/*.yml
  ```
- [ ] No secrets hardcoded in workflows
- [ ] Git user config uses your name/email:
  ```yaml
  git config user.name "Eeshan Bhanap"
  git config user.email "eb3658@columbia.edu"
  ```

### Git Attribution

- [ ] `.mailmap` file exists
- [ ] All commits show correct author:
  ```bash
  git log --pretty=format:"%an <%ae>"
  ```

## ☐ Initial Deployment

### Push to GitHub

- [ ] All changes committed
- [ ] Pushed to main branch
- [ ] GitHub Actions tab shows workflows

### Manual Test Run

1. **Test Daily Data Pipeline**:
   - [ ] Go to Actions → Daily Data Collection & Processing
   - [ ] Click "Run workflow"
   - [ ] Set `num_urls` to 100 (small test)
   - [ ] Click "Run workflow"
   - [ ] Wait for completion (~10 min)
   - [ ] Check logs for errors

2. **Monitor VM Processing**:
   - [ ] VM Processing Monitor should auto-trigger
   - [ ] Check it's running (Status: "In progress")
   - [ ] Monitor logs every 5 minutes
   - [ ] Wait for completion (~1 hour for 100 URLs)

3. **Verify Data Quality**:
   - [ ] Data quality validation passed
   - [ ] Success rates shown in logs
   - [ ] Artifacts uploaded

4. **Check Model Retrain**:
   - [ ] Model retraining completed
   - [ ] New models committed to repo
   - [ ] Model performance monitor ran
   - [ ] Performance metrics displayed

5. **Verify Web Deployment**:
   - [ ] Go to Settings → Pages
   - [ ] Copy your GitHub Pages URL
   - [ ] Visit URL in browser
   - [ ] Landing page loads correctly
   - [ ] Extension directory accessible

### Post-Test Verification

- [ ] No errors in any workflow
- [ ] All jobs completed successfully
- [ ] VM stopped automatically
- [ ] Artifacts available for download
- [ ] Models updated in repository
- [ ] Website deployed and accessible

## ☐ Production Readiness

### Schedule Configuration

- [ ] Daily schedule set to preferred time
- [ ] Timezone confirmed (default: UTC)
- [ ] Workflow will run at correct local time

### Cost Verification

- [ ] GCP billing enabled
- [ ] Budget alerts configured
- [ ] VM auto-stop verified working
- [ ] Estimated costs acceptable (~$25-75/month)

### Monitoring Setup

- [ ] Bookmark Actions page:
  `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`
- [ ] Bookmark GitHub Pages:
  `https://YOUR_USERNAME.github.io/YOUR_REPO/`
- [ ] Set up notifications (optional):
  - [ ] Email notifications for workflow failures
  - [ ] Slack webhook configured

### Documentation

- [ ] Team members (if any) have access
- [ ] README updated with project status
- [ ] Documentation links shared
- [ ] Emergency contact info updated

## ☐ Ongoing Maintenance

### Weekly Checks

- [ ] Review workflow run history
- [ ] Check model performance trends
- [ ] Verify data quality metrics
- [ ] Monitor GCP costs

### Monthly Tasks

- [ ] Review and clean up old artifacts
- [ ] Update dependencies if needed
- [ ] Rotate GCP service account keys (every 90 days)
- [ ] Review and optimize costs

### When Issues Occur

- [ ] Check workflow logs first
- [ ] Review troubleshooting guide in WORKFLOWS_SETUP.md
- [ ] Check VM logs if processing fails
- [ ] Verify GCP quotas not exceeded
- [ ] Check GitHub Actions status page

## ✅ Completion

Once all items are checked:

1. **Workflows are ready for production**
2. **System will run automatically daily**
3. **All features are enabled**

### Expected Daily Flow

```
02:00 UTC - Workflow starts
02:30 UTC - VM processing begins
10:30 UTC - Processing complete (for 1000 URLs)
10:45 UTC - Models retrained
11:00 UTC - Performance monitoring done
11:05 UTC - Website updated
```

### Success Indicators

After first successful run, you should see:
- ✅ Green checkmarks in Actions tab
- ✅ New models in `models/` directory
- ✅ Artifacts available for download
- ✅ Website updated with latest data
- ✅ VM stopped (check GCP console)

---

## Need Help?

- **Quick Start**: See [QUICK_START.md](QUICK_START.md)
- **Full Guide**: See [WORKFLOWS_SETUP.md](WORKFLOWS_SETUP.md)
- **Features**: See [FEATURES.md](FEATURES.md)
- **Troubleshooting**: See WORKFLOWS_SETUP.md → Troubleshooting section

---

**Date Completed**: _______________

**Completed By**: Eeshan Bhanap

**Status**: ☐ Ready for Production
