# GitHub Actions Quick Start

## 5-Minute Setup

### Step 1: Create GCP Service Account Key (2 min)

```bash
# Run these commands in your terminal
gcloud iam service-accounts create github-actions \
    --project=coms-452404

gcloud projects add-iam-policy-binding coms-452404 \
    --member="serviceAccount:github-actions@coms-452404.iam.gserviceaccount.com" \
    --role="roles/compute.instanceAdmin.v1"

gcloud iam service-accounts keys create ~/gcp-key.json \
    --iam-account=github-actions@coms-452404.iam.gserviceaccount.com

# Copy the key content
cat ~/gcp-key.json
```

### Step 2: Add Secret to GitHub (1 min)

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/secrets/actions
2. Click **New repository secret**
3. Name: `GCP_SA_KEY`
4. Value: Paste entire JSON from previous step
5. Click **Add secret**

### Step 3: Enable GitHub Pages (1 min)

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/settings/pages
2. Source: **GitHub Actions**
3. Click **Save**

### Step 4: Test the Workflow (1 min)

1. Go to: https://github.com/YOUR_USERNAME/YOUR_REPO/actions
2. Click **Daily Data Collection & Processing**
3. Click **Run workflow** → **Run workflow**
4. Wait and watch it run!

## What Happens Next?

✅ **Every day at 2 AM UTC**, the system will:
1. Collect 1000 URLs (phishing + legitimate)
2. Start your GCP VM
3. Process all URLs for DNS/WHOIS features
4. Validate data quality
5. Retrain models with new data
6. Deploy updated models
7. Stop VM to save costs

## Monitoring

**Check status**: https://github.com/YOUR_USERNAME/YOUR_REPO/actions

**Download artifacts**:
- Trained models (90 days)
- Feature datasets (30 days)
- Performance reports

**View your website**: https://YOUR_USERNAME.github.io/YOUR_REPO/

## Daily Costs

- **GitHub Actions**: Free (within 2,000 min/month)
- **GCP VM**: ~$0.80-2.40/day (8 hours processing)
- **Total**: ~$25-75/month

## Need Help?

See [WORKFLOWS_SETUP.md](WORKFLOWS_SETUP.md) for detailed documentation.

---

**That's it! Your automated ML pipeline is now running.** 🎉
