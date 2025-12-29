#!/usr/bin/env python3
"""
Test Full MLOps Pipeline End-to-End
====================================
Verifies:
1. VM Data Collection → Feature files exist
2. Feature Merger → Model-ready datasets created
3. Model Training → Models trained and saved
4. Model Deployment → Models loadable for prediction
5. Continuous Loop → Pipeline runs on schedule
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from datetime import datetime

print("="*80)
print("PHISHNET MLOPS PIPELINE VERIFICATION")
print("="*80)

# ============================================
# Step 1: Check VM Data Collection
# ============================================
print("\n[1/6] Checking VM Data Collection...")

vm_data_dir = Path("data/vm_collected")
if not vm_data_dir.exists():
    print("❌ VM data directory not found!")
    sys.exit(1)

# Check checkpoint
checkpoint_file = vm_data_dir / "checkpoint.json"
if checkpoint_file.exists():
    with open(checkpoint_file) as f:
        checkpoint = json.load(f)
    print(f"✅ Checkpoint found:")
    print(f"   - Processed URLs: {checkpoint.get('processed_count', 0)}")
    print(f"   - Last update: {checkpoint.get('timestamp', 'N/A')}")
else:
    print("⚠️  No checkpoint file found")

# Check collected features
dns_files = list(vm_data_dir.glob("dns_results*.csv"))
whois_files = list(vm_data_dir.glob("whois_results*.csv"))

print(f"✅ DNS feature files: {len(dns_files)}")
for f in dns_files:
    count = sum(1 for _ in open(f)) - 1  # -1 for header
    print(f"   - {f.name}: {count} records")

print(f"✅ WHOIS feature files: {len(whois_files)}")
for f in whois_files:
    count = sum(1 for _ in open(f)) - 1
    print(f"   - {f.name}: {count} records")

# ============================================
# Step 2: Check Feature Merger (Model-Ready Datasets)
# ============================================
print("\n[2/6] Checking Model-Ready Datasets...")

processed_dir = Path("data/processed")
datasets = {
    "URL Features": processed_dir / "url_features_modelready_imputed.csv",
    "DNS Features": processed_dir / "dns_features_modelready_imputed.csv",
    "WHOIS Features": processed_dir / "whois_features_modelready_imputed.csv"
}

for name, path in datasets.items():
    if path.exists():
        df = pd.read_csv(path)
        print(f"✅ {name}: {len(df)} rows, {len(df.columns)} columns")
        print(f"   - Labels: {df['label'].value_counts().to_dict()}")
    else:
        print(f"❌ {name} not found!")

# ============================================
# Step 3: Check Trained Models
# ============================================
print("\n[3/6] Checking Trained Models...")

models_dir = Path("models")
model_types = ["url", "dns", "whois"]
algorithms = ["catboost", "lgbm", "xgb", "rf"]

models_found = 0
for mtype in model_types:
    for algo in algorithms:
        model_file = models_dir / f"{mtype}_{algo}.pkl"
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024 * 1024)
            print(f"✅ {mtype}_{algo}.pkl ({size_mb:.2f} MB)")
            models_found += 1

print(f"\nTotal models found: {models_found}/12")

# ============================================
# Step 4: Check Model Deployment (Can Load Models)
# ============================================
print("\n[4/6] Checking Model Deployment...")

try:
    from src.api.model_loader import load_url_model, load_whois_model, load_dns_model

    url_model, url_cols, url_thresh = load_url_model()
    print(f"✅ URL Model loaded: {type(url_model).__name__}, {len(url_cols)} features")

    whois_model, whois_cols, whois_thresh = load_whois_model()
    print(f"✅ WHOIS Model loaded: {type(whois_model).__name__}, {len(whois_cols)} features")

    dns_model, dns_cols, dns_thresh = load_dns_model()
    print(f"✅ DNS Model loaded: {type(dns_model).__name__}, {len(dns_cols)} features")

except Exception as e:
    print(f"❌ Model loading failed: {e}")

# ============================================
# Step 5: Check Pipeline State
# ============================================
print("\n[5/6] Checking Pipeline State...")

state_file = Path("logs/pipeline_state.json")
if state_file.exists():
    with open(state_file) as f:
        state = json.load(f)

    print(f"✅ Pipeline state:")
    print(f"   - Last training: {state.get('last_training_time', 'Never')}")
    print(f"   - Model version: {state.get('current_model_version', 'N/A')}")
    print(f"   - Model accuracy: {state.get('current_model_accuracy', 0):.4f}")
    print(f"   - Total runs: {state.get('total_runs', 0)}")
    print(f"   - Successful deployments: {state.get('successful_deployments', 0)}")
    print(f"   - Failed deployments: {state.get('failed_deployments', 0)}")

    # Check if training needed
    if state.get('last_training_time'):
        last_train = datetime.fromisoformat(state['last_training_time'])
        hours_since = (datetime.now() - last_train).total_seconds() / 3600
        print(f"   - Hours since last training: {hours_since:.1f}")

        # Check new data
        last_data_size = state.get('last_training_data_size', 0)
        current_url_data = pd.read_csv("data/processed/url_features_modelready_imputed.csv")
        new_urls = len(current_url_data) - last_data_size
        print(f"   - New URLs since last training: {new_urls}")

        if new_urls >= 100:
            print("   ⚠️  >= 100 new URLs → Training should trigger")
        elif hours_since >= 168:
            print("   ⚠️  >= 1 week since training → Training should trigger")
        else:
            print(f"   ℹ️  Next training in {100 - new_urls} URLs or {168 - hours_since:.1f} hours")
else:
    print("⚠️  No pipeline state file found")

# ============================================
# Step 6: Test Prediction Pipeline
# ============================================
print("\n[6/6] Testing Prediction Pipeline...")

try:
    from src.api.predict_utils import predict_ensemble_risk

    test_urls = [
        "https://secure-paypal-verify.suspicious-domain.xyz/login",  # Likely phishing
        "https://google.com",  # Legit
    ]

    for url in test_urls:
        try:
            phish_prob, legit_prob, verdict, details = predict_ensemble_risk(url)
            print(f"✅ {url[:50]:50s} → {verdict:12s} (p={phish_prob:.4f})")
        except Exception as e:
            print(f"❌ Prediction failed for {url}: {e}")

except Exception as e:
    print(f"❌ Prediction pipeline not working: {e}")

# ============================================
# Summary
# ============================================
print("\n" + "="*80)
print("PIPELINE VERIFICATION COMPLETE")
print("="*80)
print("\nNext Steps:")
print("1. Ensure VM collector is running: ./scripts/vm_manager.sh status")
print("2. Ensure MLOps pipeline is running: ps aux | grep mlops_pipeline")
print("3. Check logs: tail -f logs/mlops_pipeline_*.log")
print("4. Monitor new data: watch -n 60 'wc -l data/vm_collected/*.csv'")
