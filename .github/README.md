# GitHub Actions Workflows

Automated CI/CD pipeline for PhishNet phishing detection system.

## 📋 Workflows Overview

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| [Daily Data Pipeline](workflows/daily_data_pipeline.yml) | Daily 2 AM UTC | ~30 min | Collect URLs, upload to VM |
| [VM Processing Monitor](workflows/vm_processing_monitor.yml) | Auto-triggered | 2-8 hours | Monitor VM, validate data, retrain models |
| [Model Performance Monitor](workflows/model_performance_monitor.yml) | After retraining | ~10 min | Evaluate models, detect drift |
| [Web Deployment](workflows/deploy_web.yml) | Push to main | ~5 min | Deploy website & extension |
| [CI/CD Pipeline](workflows/ci.yml) | Push/PR | ~15 min | Tests, linting, Docker build |

## 🚀 Quick Start

**New to GitHub Actions?** See [QUICK_START.md](QUICK_START.md)

**Full setup guide:** See [WORKFLOWS_SETUP.md](WORKFLOWS_SETUP.md)

## 🔑 Required Secrets

| Secret Name | Description | Required For |
|-------------|-------------|--------------|
| `GCP_SA_KEY` | Google Cloud service account key (JSON) | All VM workflows |

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Daily Data Pipeline                       │
│  • Fetch URLs from sources                                   │
│  • Extract URL features                                      │
│  • Start GCP VM                                              │
│  • Upload to VM queue                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │ triggers
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               VM Processing Monitor                          │
│  ┌──────────────────────────────────────────────┐           │
│  │  Monitor Loop (every 5 min, max 8 hours)     │           │
│  │  • Check DNS/WHOIS CSV row counts            │           │
│  │  • Wait for completion (1001 rows each)      │           │
│  └──────────────────────────────────────────────┘           │
│                                                               │
│  ┌──────────────────────────────────────────────┐           │
│  │  Data Quality Validation                     │           │
│  │  • Download CSV files from VM                │           │
│  │  • Check row counts match                    │           │
│  │  • Validate success rates (95%+)             │           │
│  │  • Upload as artifacts                       │           │
│  └──────────────────────────────────────────────┘           │
│                                                               │
│  ┌──────────────────────────────────────────────┐           │
│  │  Model Retraining                            │           │
│  │  • Merge VM data with main dataset           │           │
│  │  • Train URL/DNS/WHOIS/Ensemble models       │           │
│  │  • Save models to repository                 │           │
│  └──────────────────────────────────────────────┘           │
│                                                               │
│  ┌──────────────────────────────────────────────┐           │
│  │  VM Auto-Stop (cost optimization)            │           │
│  │  • Stop VM after processing                  │           │
│  │  • Save ~$2-3/day                            │           │
│  └──────────────────────────────────────────────┘           │
└──────────────────────┬──────────────────────────────────────┘
                       │ triggers
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Model Performance Monitoring                      │
│  • Evaluate on test set                                      │
│  • Calculate metrics (acc, prec, recall, F1, AUC)            │
│  • Compare with historical performance                       │
│  • Detect drift (>5% degradation)                            │
│  • Save performance history                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 Web Deployment                               │
│  • Build web interface                                       │
│  • Build browser extension                                   │
│  • Deploy to GitHub Pages                                    │
│  • Available at: username.github.io/repo                     │
└─────────────────────────────────────────────────────────────┘
```

## ⏱️ Execution Schedule

```
Daily Schedule (UTC):
02:00 - Daily Data Pipeline starts
02:30 - VM processing begins
10:30 - VM processing completes (estimated)
10:45 - Model retraining completes
11:00 - Performance monitoring completes
11:05 - Web deployment (if models updated)
```

## 📦 Artifacts

Workflows generate downloadable artifacts:

| Artifact | Retention | Size | Description |
|----------|-----------|------|-------------|
| `validated-features-YYYYMMDD` | 30 days | ~5-10 MB | DNS/WHOIS CSV files |
| `trained-models-YYYYMMDD` | 90 days | ~50-100 MB | Trained model files |
| `model-performance-report` | 30 days | <1 MB | JSON with metrics |

## 🎯 Success Criteria

| Check | Threshold | Action if Failed |
|-------|-----------|------------------|
| DNS success rate | ≥95% | Workflow fails, investigate |
| WHOIS success rate | ≥95% | Workflow fails, investigate |
| Model accuracy | ≥90% | Warning, but continues |
| Model drift | <5% degradation | Warning, flag for review |

## 💰 Cost Estimation

### GitHub Actions
- **Free tier**: 2,000 minutes/month
- **Expected usage**: ~900 minutes/month
- **Cost**: $0 (within free tier)

### GCP VM
- **Running**: $0.10-0.30/hour (depends on instance type)
- **Daily processing**: ~8 hours
- **Daily cost**: $0.80-2.40
- **Monthly cost**: $25-75

### Total: ~$25-75/month

**Optimization**: VM auto-stop saves ~$2-3/day compared to always-on.

## 🔧 Customization

### Change Schedule

Edit [daily_data_pipeline.yml](workflows/daily_data_pipeline.yml):

```yaml
schedule:
  - cron: '0 14 * * *'  # 2 PM UTC instead of 2 AM
```

### Change VM Configuration

Edit workflow env vars:

```yaml
env:
  VM_NAME: your-vm-name
  GCP_ZONE: your-zone
```

### Disable Auto-VM-Stop

Comment out `stop-vm` job in [vm_processing_monitor.yml](workflows/vm_processing_monitor.yml)

## 🐛 Troubleshooting

### Workflow not triggering on schedule?
- Repository must be active (GitHub disables inactive repos)
- Workflow must be on default branch (main)
- Check Actions are enabled in repository settings

### GCP authentication failing?
- Verify `GCP_SA_KEY` secret is set correctly
- Check service account has `compute.instanceAdmin.v1` role
- Ensure JSON is valid (not corrupted)

### VM processing timeout?
- Increase `max_wait_hours` input
- Increase VM resources (CPU/memory)
- Consider self-hosted runner for longer runs

### Data quality validation failing?
- Check VM processing logs
- Verify feature extraction scripts working
- Review success rate thresholds

## 📚 Documentation

- [Quick Start Guide](QUICK_START.md) - 5-minute setup
- [Full Setup Guide](WORKFLOWS_SETUP.md) - Detailed documentation
- [GitHub Actions Docs](https://docs.github.com/actions) - Official documentation

## 🔒 Security

- All secrets stored in GitHub Secrets (encrypted)
- GCP service account uses principle of least privilege
- No secrets in logs or artifacts
- Repository-scoped tokens (can't access other repos)

## 📈 Monitoring

**View workflow runs**: `https://github.com/YOUR_USERNAME/YOUR_REPO/actions`

**View website**: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

**Check VM status**:
```bash
gcloud compute instances describe dns-whois-fetch-25 --zone=us-central1-c
```

---

## Author

**Eeshan Bhanap**
Columbia University
eb3658@columbia.edu

## License

See repository LICENSE file.
