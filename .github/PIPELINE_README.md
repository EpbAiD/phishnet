# Unified Configurable Pipeline

## One Pipeline, One Config File

Change pipeline behavior by editing **ONE file**: `.github/pipeline_config.yml`

No need to edit workflow files!

---

## How It Works

### Configuration File: `.github/pipeline_config.yml`

```yaml
stages:
  data_collection:
    frequency: "daily"    # ← Change this

  model_training:
    frequency: "daily"    # ← Change this to "weekly" after validation

  deployment:
    frequency: "daily"    # ← Change this to "weekly" after validation
```

### Workflow: `.github/workflows/unified_pipeline.yml`

- Reads config file
- Decides what to run based on frequency
- Executes pipeline stages accordingly
- Auto-tracks validation progress

---

## Modes

### Validation Mode (Current)

**Goal**: Run 10 times to prove system works

**Config**:
```yaml
pipeline:
  mode: "validation"

stages:
  data_collection:
    frequency: "daily"
  model_training:
    frequency: "daily"      # Train every run
  deployment:
    frequency: "daily"      # Deploy every run

validation:
  runs_needed: 10
  runs_completed: 0         # Auto-increments
```

**Result**:
- 10 runs × 1000 URLs = 10,000 URLs total
- Models retrain on growing dataset each run
- Validation counter auto-updates after each run

---

### Production Mode (After Validation)

**Goal**: Cost-effective production operation

**Change config to**:
```yaml
pipeline:
  mode: "production"

stages:
  data_collection:
    frequency: "daily"      # Still daily
  model_training:
    frequency: "weekly"     # ← Changed!
  deployment:
    frequency: "weekly"     # ← Changed!
```

**Result**:
- Data collected daily (Mon-Sun)
- Features extracted daily
- Models trained weekly (Sunday)
- Deployment weekly (Sunday)

---

## Switching from Validation to Production

### Step 1: Wait for 10 Validation Runs

Monitor progress:
```bash
cat .github/pipeline_config.yml | grep runs_completed
```

When `runs_completed: 10`, you're ready!

### Step 2: Edit Config File

```bash
# Open config file
nano .github/pipeline_config.yml

# Change these lines:
mode: "validation"  →  mode: "production"
model_training: frequency: "daily"  →  frequency: "weekly"
deployment: frequency: "daily"  →  frequency: "weekly"

# Save and commit
git add .github/pipeline_config.yml
git commit -m "Switch to production mode - weekly training/deployment"
git push
```

### Step 3: Done!

Pipeline automatically adjusts:
- Monday-Saturday: Collect data only
- Sunday: Collect data + Train models + Deploy

---

## Manual Runs

Force run specific stages:

```bash
# Run everything
gh workflow run unified_pipeline.yml -f override_stage=all

# Run only data collection
gh workflow run unified_pipeline.yml -f override_stage=data

# Run only training
gh workflow run unified_pipeline.yml -f override_stage=training

# Run only deployment
gh workflow run unified_pipeline.yml -f override_stage=deploy

# Override number of URLs
gh workflow run unified_pipeline.yml -f override_stage=all -f num_urls=500
```

---

## Monitoring

### Check validation progress:
```bash
cat .github/pipeline_config.yml | grep runs_completed
```

### View recent runs:
```bash
gh run list --workflow=unified_pipeline.yml --limit 10
```

### Watch a run:
```bash
gh run watch <run-id>
```

---

## File Structure

```
.github/
├── pipeline_config.yml          ← EDIT THIS to change behavior
└── workflows/
    └── unified_pipeline.yml     ← Don't edit (reads config)
```

---

## Benefits

✅ **One source of truth**: All frequencies in one config file
✅ **No workflow editing**: Change behavior without touching YAML
✅ **Auto-tracking**: Validation progress auto-updates
✅ **Easy switch**: Validation → Production is 3 lines
✅ **Override capability**: Force run any stage manually
✅ **Git history**: See exactly when you switched modes

---

## Current Status

Check current configuration:
```bash
cat .github/pipeline_config.yml
```

Check validation progress:
```bash
grep -A 2 "validation:" .github/pipeline_config.yml
```

---

## After Validation Complete

You'll have:
- ✅ 10,000 URLs in master dataset
- ✅ 45 trained models
- ✅ Proven system reliability
- ✅ Ready for production (just edit config!)

Then showcase on LinkedIn! 🚀
