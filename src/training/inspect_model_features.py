#!/usr/bin/env python3
"""
Inspect pickled model to extract expected features.
This helps debug feature mismatch errors.
"""

import pickle
import sys
import os
from pathlib import Path

def inspect_model(model_path: str):
    """
    Load a pickled model and extract feature information.

    Args:
        model_path: Path to pickled model file
    """
    print(f"\n{'='*80}")
    print(f"Inspecting Model: {model_path}")
    print(f"{'='*80}\n")

    # Load model
    print(f"📂 Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"✅ Model loaded successfully\n")

    # Check if it's a dict or raw model
    if isinstance(model_data, dict):
        print("📦 Model is wrapped in a dictionary")
        print(f"   Dictionary keys: {list(model_data.keys())}\n")

        model = model_data.get('model')
        features = model_data.get('features')
        threshold = model_data.get('threshold')

        if features:
            print(f"✅ Feature list found in model metadata:")
            print(f"   Number of features: {len(features)}")
            print(f"\n📋 Feature list (in order):")
            for i, feat in enumerate(features, 1):
                print(f"   {i:3d}. {feat}")

            # Save to file
            output_path = model_path.replace('.pkl', '_features.txt')
            with open(output_path, 'w') as f:
                for feat in features:
                    f.write(f"{feat}\n")
            print(f"\n💾 Saved feature list to: {output_path}")

        if threshold:
            print(f"\n🎯 Classification threshold: {threshold}")

        if model:
            print(f"\n🤖 Model type: {type(model).__name__}")

            # Try to get features from model object
            if hasattr(model, 'feature_names_'):
                print(f"\n✅ Model has feature_names_ attribute:")
                print(f"   Number of features: {len(model.feature_names_)}")
                print(f"\n📋 Model feature names:")
                for i, feat in enumerate(model.feature_names_, 1):
                    print(f"   {i:3d}. {feat}")

            if hasattr(model, 'feature_importances_'):
                print(f"\n✅ Model has feature importance")
                print(f"   Shape: {model.feature_importances_.shape}")

            # CatBoost specific
            if hasattr(model, 'feature_names_'):
                print(f"\n✅ CatBoost feature names available")

            if hasattr(model, 'get_feature_importance'):
                print(f"✅ CatBoost get_feature_importance() method available")

    else:
        print("🤖 Model is a raw estimator (not wrapped)")
        model = model_data
        print(f"   Model type: {type(model).__name__}")

        # Try to get features from model
        if hasattr(model, 'feature_names_'):
            features = model.feature_names_
            print(f"\n✅ Model has feature_names_ attribute:")
            print(f"   Number of features: {len(features)}")
            print(f"\n📋 Model feature names:")
            for i, feat in enumerate(features, 1):
                print(f"   {i:3d}. {feat}")

            # Save to file
            output_path = model_path.replace('.pkl', '_features.txt')
            with open(output_path, 'w') as f:
                for feat in features:
                    f.write(f"{feat}\n")
            print(f"\n💾 Saved feature list to: {output_path}")
        else:
            print("\n⚠️  Model does not have feature_names_ attribute")

        if hasattr(model, 'feature_importances_'):
            print(f"\n✅ Model has feature importance")
            print(f"   Shape: {model.feature_importances_.shape}")

    print(f"\n{'='*80}")
    print(f"✅ Inspection complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect pickled model features")
    parser.add_argument("--model", type=str, required=True, help="Path to pickled model")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ Model file not found: {args.model}")
        sys.exit(1)

    inspect_model(args.model)
