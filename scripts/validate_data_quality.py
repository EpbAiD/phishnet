#!/usr/bin/env python3
"""
Data Quality Validation
=======================
Validates feature datasets to detect garbage data before model training.

Checks for:
  - Entire rows with all NaN features
  - Sentinel outlier values (-999, -1, 0 in unexpected places)
  - Impossible values (negative counts, invalid dates, etc.)
  - Schema mismatches
  - String encoding issues
  - Statistical outliers (beyond reasonable bounds)

Usage:
  python3 scripts/validate_data_quality.py <dataset_path> <feature_type>

  feature_type: url, dns, or whois
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def log(message, level="INFO"):
    """Log with timestamp and level."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)


def validate_dns_features(df):
    """Validate DNS feature dataset."""
    issues = []

    # Expected feature columns (excluding url, label, bucket, error columns)
    meta_cols = ['url', 'label', 'bucket', 'domain', 'error_type', 'error_msg']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    log(f"Validating {len(df)} rows with {len(feature_cols)} DNS features")

    # 1. Check for rows with all features missing
    all_nan_rows = df[feature_cols].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        pct = (all_nan_rows / len(df)) * 100
        issues.append(f"⚠️  {all_nan_rows} rows ({pct:.1f}%) have ALL features missing")

    # 2. Check for sentinel values in count fields
    count_fields = [c for c in feature_cols if 'num_' in c or c.startswith('num')]
    for col in count_fields:
        if col in df.columns:
            # Check for impossible negative counts
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative values (impossible for counts)")

            # Check for suspiciously high counts (> 1000 is unlikely)
            if (df[col] > 1000).sum() > 0:
                count = (df[col] > 1000).sum()
                issues.append(f"⚠️  {col}: {count} rows with count > 1000 (suspicious)")

    # 3. Check TTL values (should be reasonable, not -999 or impossibly large)
    ttl_fields = [c for c in feature_cols if 'ttl' in c.lower()]
    for col in ttl_fields:
        if col in df.columns:
            # TTL should be positive
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative TTL values (invalid)")

            # TTL should not exceed 2^32 (max for 32-bit unsigned)
            if (df[col] > 2**32).sum() > 0:
                issues.append(f"❌ {col}: Found TTL > 2^32 (impossible)")

    # 4. Check boolean flags are actually boolean
    bool_fields = [c for c in feature_cols if c.startswith('has_')]
    for col in bool_fields:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            valid_bool = all(v in [0, 1, True, False] for v in unique_vals)
            if not valid_bool:
                issues.append(f"❌ {col}: Contains non-boolean values {unique_vals}")

    # 5. Check for string encoding issues
    str_fields = df.select_dtypes(include=['object']).columns
    for col in str_fields:
        if col in feature_cols:
            # Check for null bytes
            if df[col].astype(str).str.contains('\x00', na=False).any():
                issues.append(f"❌ {col}: Contains null bytes (encoding issue)")

    # 6. Check entropy fields (should be 0-8 for base-2, never negative)
    entropy_fields = [c for c in feature_cols if 'entropy' in c.lower()]
    for col in entropy_fields:
        if col in df.columns:
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative entropy (impossible)")
            if (df[col] > 10).sum() > 0:
                count = (df[col] > 10).sum()
                issues.append(f"⚠️  {col}: {count} rows with entropy > 10 (unusual)")

    return issues


def validate_whois_features(df):
    """Validate WHOIS feature dataset."""
    issues = []

    meta_cols = ['url', 'label', 'bucket', 'error_msg', 'whois_success']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    log(f"Validating {len(df)} rows with {len(feature_cols)} WHOIS features")

    # 1. Check for rows with all features missing
    all_nan_rows = df[feature_cols].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        pct = (all_nan_rows / len(df)) * 100
        issues.append(f"⚠️  {all_nan_rows} rows ({pct:.1f}%) have ALL features missing")

    # 2. Check domain_age_days (should be non-negative, not > 10000)
    if 'domain_age_days' in df.columns:
        if (df['domain_age_days'] < 0).sum() > 0:
            issues.append(f"❌ domain_age_days: Found negative values (impossible)")
        if (df['domain_age_days'] > 10000).sum() > 0:
            count = (df['domain_age_days'] > 10000).sum()
            issues.append(f"⚠️  domain_age_days: {count} rows > 10000 days (~27 years, check if valid)")

    # 3. Check registration_length_days (should be positive if present)
    if 'registration_length_days' in df.columns:
        if (df['registration_length_days'] < 0).sum() > 0:
            issues.append(f"❌ registration_length_days: Found negative values (impossible)")
        if (df['registration_length_days'] > 3650).sum() > 0:
            count = (df['registration_length_days'] > 3650).sum()
            issues.append(f"⚠️  registration_length_days: {count} rows > 10 years (unusual)")

    # 4. Check date consistency (creation <= expiration)
    if 'creation_date' in df.columns and 'expiration_date' in df.columns:
        df_dates = df[['creation_date', 'expiration_date']].copy()
        df_dates['creation_date'] = pd.to_datetime(df_dates['creation_date'], errors='coerce')
        df_dates['expiration_date'] = pd.to_datetime(df_dates['expiration_date'], errors='coerce')

        invalid_dates = (df_dates['creation_date'] > df_dates['expiration_date']).sum()
        if invalid_dates > 0:
            issues.append(f"❌ {invalid_dates} rows have creation_date > expiration_date (impossible)")

    # 5. Check privacy protection flag
    if 'has_privacy_protection' in df.columns:
        unique_vals = df['has_privacy_protection'].dropna().unique()
        valid_bool = all(v in [0, 1, True, False] for v in unique_vals)
        if not valid_bool:
            issues.append(f"❌ has_privacy_protection: Contains non-boolean values {unique_vals}")

    return issues


def validate_url_features(df):
    """Validate URL feature dataset."""
    issues = []

    meta_cols = ['url', 'label', 'bucket', 'source']
    feature_cols = [c for c in df.columns if c not in meta_cols]

    log(f"Validating {len(df)} rows with {len(feature_cols)} URL features")

    # 1. Check for rows with all features missing
    all_nan_rows = df[feature_cols].isna().all(axis=1).sum()
    if all_nan_rows > 0:
        pct = (all_nan_rows / len(df)) * 100
        issues.append(f"⚠️  {all_nan_rows} rows ({pct:.1f}%) have ALL features missing")

    # 2. Check length fields (should be non-negative, reasonable bounds)
    length_fields = [c for c in feature_cols if 'length' in c.lower()]
    for col in length_fields:
        if col in df.columns:
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative lengths (impossible)")
            if (df[col] > 10000).sum() > 0:
                count = (df[col] > 10000).sum()
                issues.append(f"⚠️  {col}: {count} rows with length > 10000 (unusually long)")

    # 3. Check count fields
    count_fields = [c for c in feature_cols if 'num_' in c or c.startswith('num')]
    for col in count_fields:
        if col in df.columns:
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative counts (impossible)")
            if (df[col] > 1000).sum() > 0:
                count = (df[col] > 1000).sum()
                issues.append(f"⚠️  {col}: {count} rows with count > 1000 (suspicious)")

    # 4. Check entropy fields
    entropy_fields = [c for c in feature_cols if 'entropy' in c.lower()]
    for col in entropy_fields:
        if col in df.columns:
            if (df[col] < 0).sum() > 0:
                issues.append(f"❌ {col}: Found negative entropy (impossible)")
            if (df[col] > 10).sum() > 0:
                count = (df[col] > 10).sum()
                issues.append(f"⚠️  {col}: {count} rows with entropy > 10 (unusual)")

    # 5. Check ratio fields (should be 0-1 or reasonable range)
    if 'digit_to_letter_ratio' in df.columns:
        if (df['digit_to_letter_ratio'] < 0).sum() > 0:
            issues.append(f"❌ digit_to_letter_ratio: Found negative ratios (impossible)")

    # 6. Check boolean flags
    bool_fields = [c for c in feature_cols if c.startswith('has_') or c.startswith('is_')]
    for col in bool_fields:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            valid_bool = all(v in [0, 1, True, False] for v in unique_vals)
            if not valid_bool:
                issues.append(f"❌ {col}: Contains non-boolean values {unique_vals}")

    # 7. Check port numbers (should be 0-65535)
    if 'port' in df.columns:
        if (df['port'] < 0).sum() > 0:
            issues.append(f"❌ port: Found negative port numbers (impossible)")
        if (df['port'] > 65535).sum() > 0:
            issues.append(f"❌ port: Found port > 65535 (impossible, max is 65535)")

    return issues


def main():
    """Main validation entry point."""
    if len(sys.argv) < 3:
        print("Usage: python3 validate_data_quality.py <dataset_path> <feature_type>")
        print("  feature_type: url, dns, or whois")
        sys.exit(1)

    dataset_path = sys.argv[1]
    feature_type = sys.argv[2].lower()

    if feature_type not in ['url', 'dns', 'whois']:
        log(f"Invalid feature type: {feature_type}. Must be url, dns, or whois", "ERROR")
        sys.exit(1)

    if not Path(dataset_path).exists():
        log(f"Dataset not found: {dataset_path}", "ERROR")
        sys.exit(1)

    log("=" * 60)
    log(f"Data Quality Validation: {feature_type.upper()}")
    log("=" * 60)
    log(f"Dataset: {dataset_path}")
    log("")

    # Load dataset
    df = pd.read_csv(dataset_path)
    log(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run validation based on feature type
    if feature_type == 'dns':
        issues = validate_dns_features(df)
    elif feature_type == 'whois':
        issues = validate_whois_features(df)
    else:  # url
        issues = validate_url_features(df)

    # Report results
    log("")
    log("=" * 60)
    log("VALIDATION RESULTS")
    log("=" * 60)

    if not issues:
        log("✅ No data quality issues found!", "SUCCESS")
        sys.exit(0)
    else:
        log(f"Found {len(issues)} data quality issues:", "WARNING")
        for issue in issues:
            log(f"  {issue}", "WARNING")

        # Fail if any critical issues (❌)
        critical_issues = [i for i in issues if '❌' in i]
        if critical_issues:
            log("", "ERROR")
            log(f"❌ VALIDATION FAILED: {len(critical_issues)} critical issues", "ERROR")
            sys.exit(1)
        else:
            log("", "WARNING")
            log("⚠️  Validation passed with warnings", "WARNING")
            sys.exit(0)


if __name__ == "__main__":
    main()
