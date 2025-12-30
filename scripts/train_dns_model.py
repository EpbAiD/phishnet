#!/usr/bin/env python3
"""
DNS Model Training Wrapper for GitHub Actions
=============================================
Trains DNS-based phishing detection models.

Usage:
  python3 scripts/train_dns_model.py [--subset model1,model2,...]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dns_train import train_all_dns_models


def main():
    parser = argparse.ArgumentParser(description='Train DNS models')
    parser.add_argument('--subset', type=str, help='Comma-separated list of model names to train')
    args = parser.parse_args()

    subset = args.subset.split(',') if args.subset else None

    print("=" * 60)
    print("DNS Model Training")
    print("=" * 60)

    if subset:
        print(f"Training subset: {subset}")
    else:
        print("Training all DNS models")

    print("=" * 60)
    print()

    train_all_dns_models(subset=subset)

    print()
    print("=" * 60)
    print("✅ DNS model training complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
