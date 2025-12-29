#!/usr/bin/env python3
# ===============================================================
# compute_shap_importance.py
# ---------------------------------------------------------------
# ✅ Compute SHAP values on test data after model training
# ✅ Calculate global feature importance from SHAP
# ✅ Save top features for use during inference
# ✅ Run this ONCE after training each model
# ===============================================================

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
import shap
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def compute_shap_importance(
    model_path: str,
    test_data_path: str,
    output_path: str,
    model_type: str,
    top_n: int = 10,
    max_samples: int = 100
):
    """
    Compute SHAP values on test data and save global feature importance.

    Args:
        model_path: Path to trained model pickle file
        test_data_path: Path to test data CSV
        output_path: Path to save SHAP importance JSON
        model_type: 'url', 'dns', or 'whois'
        top_n: Number of top features to save
        max_samples: Max samples to use for SHAP computation (for speed)
    """
    print(f"\n{'='*80}")
    print(f"Computing SHAP Importance for {model_type.upper()} Model")
    print(f"{'='*80}\n")

    # Load model
    print(f"📂 Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        model = model_data['model']
        feature_cols = model_data['features']
        threshold = model_data.get('threshold', 0.5)
    else:
        model = model_data
        feature_cols = None
        threshold = 0.5

    print(f"✅ Model loaded: {type(model).__name__}")

    # Load test data
    print(f"\n📂 Loading test data from: {test_data_path}")
    df = pd.read_csv(test_data_path)

    # Get features and labels
    if 'label' in df.columns:
        X = df.drop(columns=['label'])
        y = df['label']
    elif 'is_phishing' in df.columns:
        X = df.drop(columns=['is_phishing'])
        y = df['is_phishing']
    else:
        raise ValueError("No label column found in test data")

    # Drop non-feature columns (url, bucket, etc.)
    non_feature_cols = ['url', 'bucket', 'Unnamed: 0']
    for col in non_feature_cols:
        if col in X.columns:
            print(f"   Dropping non-feature column: {col}")
            X = X.drop(columns=[col])

    # If model has feature_names_, use those directly
    if hasattr(model, 'feature_names_'):
        print(f"\n✅ Using model's feature_names_ attribute")
        feature_cols = list(model.feature_names_)
        print(f"   Model expects {len(feature_cols)} features")

    # If feature_cols specified, use only those
    if feature_cols is not None:
        missing_cols = set(feature_cols) - set(X.columns)
        if missing_cols:
            print(f"⚠️  Adding missing columns: {missing_cols}")
            for col in missing_cols:
                X[col] = 0

        # Reorder to match model's expected feature order
        print(f"   Aligning features to model's expected order...")
        X = X[feature_cols]

    # Convert object columns to numeric (CatBoost handles NaN natively - preserve them!)
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"   Converting categorical column to numeric: {col}")
            X[col] = pd.Categorical(X[col]).codes

    # DO NOT fill NaN - it's a meaningful feature in phishing detection!
    # NaN indicates missing WHOIS data, failed lookups, etc. which are red flags
    print(f"   Preserving NaN values (meaningful signal in phishing detection)")

    print(f"✅ Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Sample data if too large (SHAP computation can be slow)
    if X.shape[0] > max_samples:
        print(f"\n⚡ Sampling {max_samples} examples for faster SHAP computation")
        sample_idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
    else:
        X_sample = X
        y_sample = y

    # Compute SHAP values
    print(f"\n🔮 Computing SHAP values using TreeExplainer...")
    print(f"   This may take a few minutes for {X_sample.shape[0]} samples...")

    try:
        # Use TreeExplainer for tree-based models (fast)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample, check_additivity=False)

        # For binary classification, get positive class SHAP values
        if len(shap_values.shape) == 3:  # Multi-output
            shap_vals = shap_values.values[:, :, 1]
        else:
            shap_vals = shap_values.values

        print(f"✅ SHAP values computed: shape {shap_vals.shape}")

    except Exception as e:
        print(f"❌ TreeExplainer failed: {e}")
        print(f"   Falling back to KernelExplainer (slower)...")

        # Fallback to KernelExplainer
        background = shap.sample(X_sample, min(10, len(X_sample)))
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)

        # For binary classification
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]  # Positive class
        else:
            shap_vals = shap_values

        print(f"✅ SHAP values computed with KernelExplainer: shape {shap_vals.shape}")

    # Calculate global feature importance (mean absolute SHAP)
    print(f"\n📊 Calculating global feature importance...")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Get top N features
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]

    # Prepare output
    feature_importance = {
        "model_type": model_type,
        "model_path": model_path,
        "num_samples": int(X_sample.shape[0]),
        "num_features": int(X_sample.shape[1]),
        "top_n": top_n,
        "features": []
    }

    print(f"\n🏆 Top {top_n} Most Important Features (by mean |SHAP|):\n")
    print(f"{'Rank':<6} {'Feature':<40} {'Mean |SHAP|':<12}")
    print("-" * 60)

    for rank, idx in enumerate(top_indices, 1):
        feat_name = X.columns[idx]
        importance = float(mean_abs_shap[idx])

        feature_importance["features"].append({
            "rank": rank,
            "feature": feat_name,
            "mean_abs_shap": importance,
            "index": int(idx)
        })

        print(f"{rank:<6} {feat_name:<40} {importance:<12.6f}")

    # Save to JSON
    print(f"\n💾 Saving SHAP importance to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(feature_importance, f, indent=2)

    print(f"✅ SHAP importance saved successfully!")

    # Also save SHAP values for later analysis (optional)
    shap_values_path = output_path.replace('.json', '_values.npz')
    print(f"\n💾 Saving raw SHAP values to: {shap_values_path}")
    np.savez_compressed(
        shap_values_path,
        shap_values=shap_vals,
        feature_names=X.columns.tolist(),
        labels=y_sample.values
    )
    print(f"✅ SHAP values saved for later analysis!")

    print(f"\n{'='*80}")
    print(f"✅ SHAP Importance Computation Complete!")
    print(f"{'='*80}\n")

    return feature_importance


def compute_all_models():
    """Compute SHAP importance for all three models."""

    # Define paths (adjust based on your project structure)
    base_dir = Path(__file__).parent.parent.parent

    models = [
        {
            "model_type": "url",
            "model_path": base_dir / "models" / "url_model.pkl",
            "test_data_path": base_dir / "data" / "processed" / "url_features.csv",
            "output_path": base_dir / "models" / "shap_importance_url.json"
        },
        {
            "model_type": "whois",
            "model_path": base_dir / "models" / "whois_model.pkl",
            "test_data_path": base_dir / "data" / "processed" / "whois_features.csv",
            "output_path": base_dir / "models" / "shap_importance_whois.json"
        },
        # DNS commented out for now
        # {
        #     "model_type": "dns",
        #     "model_path": base_dir / "models" / "dns_model.pkl",
        #     "test_data_path": base_dir / "data" / "processed" / "dns_features.csv",
        #     "output_path": base_dir / "models" / "shap_importance_dns.json"
        # },
    ]

    results = {}

    for config in models:
        if not config["model_path"].exists():
            print(f"⚠️  Model not found: {config['model_path']}")
            print(f"   Skipping {config['model_type']} model...\n")
            continue

        if not config["test_data_path"].exists():
            print(f"⚠️  Test data not found: {config['test_data_path']}")
            print(f"   Skipping {config['model_type']} model...\n")
            continue

        try:
            result = compute_shap_importance(
                model_path=str(config["model_path"]),
                test_data_path=str(config["test_data_path"]),
                output_path=str(config["output_path"]),
                model_type=config["model_type"],
                top_n=10,
                max_samples=100
            )
            results[config["model_type"]] = result
        except Exception as e:
            print(f"❌ Failed to compute SHAP for {config['model_type']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SHAP importance from test data")
    parser.add_argument("--model", type=str, help="Model pickle file path")
    parser.add_argument("--test-data", type=str, help="Test data CSV path")
    parser.add_argument("--output", type=str, help="Output JSON path")
    parser.add_argument("--model-type", type=str, choices=["url", "dns", "whois"], help="Model type")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top features")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples for SHAP computation")
    parser.add_argument("--all", action="store_true", help="Compute for all models")

    args = parser.parse_args()

    if args.all:
        print("\n🚀 Computing SHAP importance for all models...\n")
        compute_all_models()
    elif args.model and args.test_data and args.output and args.model_type:
        compute_shap_importance(
            model_path=args.model,
            test_data_path=args.test_data,
            output_path=args.output,
            model_type=args.model_type,
            top_n=args.top_n,
            max_samples=args.max_samples
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python compute_shap_importance.py --all")
        print("  python compute_shap_importance.py --model models/url_model.pkl --test-data data/processed/url_features.csv --output models/shap_importance_url.json --model-type url")
