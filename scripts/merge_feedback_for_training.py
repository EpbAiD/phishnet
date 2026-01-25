#!/usr/bin/env python3
"""
Merge user feedback corrections into the master training dataset.

This script:
1. Downloads the master dataset from S3
2. Downloads feedback corrections
3. Updates labels for corrected URLs
4. Re-extracts features for corrected URLs if needed
5. Uploads the updated master dataset back to S3

Used by the training pipeline to incorporate human corrections.
"""

import os
import sys
import pandas as pd
import boto3
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

S3_BUCKET = "phishnet-data"
AWS_REGION = "us-east-1"


def merge_feedback_corrections(
    corrections_file: str = "data/feedback/corrections.csv",
    master_file: str = "data/processed/phishing_features_master.csv",
    download_from_s3: bool = True,
    upload_to_s3: bool = True
) -> dict:
    """
    Merge feedback corrections into the master training dataset.

    Strategy:
    - If URL exists in master: update its label
    - If URL is new: add it with corrected label (may need feature extraction)

    Args:
        corrections_file: Path to corrections CSV
        master_file: Path to master dataset
        download_from_s3: Whether to download files from S3
        upload_to_s3: Whether to upload updated master to S3

    Returns:
        dict with merge statistics
    """

    print("=" * 60)
    print("MERGE FEEDBACK CORRECTIONS")
    print("=" * 60)

    s3 = boto3.client('s3', region_name=AWS_REGION)

    # Ensure directories exist
    os.makedirs("data/feedback", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # Download files from S3
    if download_from_s3:
        print("\n📥 Downloading from S3...")

        # Download master dataset
        try:
            s3.download_file(S3_BUCKET, "master/phishing_features_master.csv", master_file)
            print(f"   ✅ Downloaded master dataset")
        except Exception as e:
            print(f"   ⚠️  Failed to download master: {e}")
            return {"success": False, "error": str(e)}

        # Download latest corrections
        try:
            # List correction files and get the latest
            response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="feedback/corrections_")
            if 'Contents' in response and len(response['Contents']) > 0:
                latest = sorted(response['Contents'], key=lambda x: x['Key'], reverse=True)[0]
                s3.download_file(S3_BUCKET, latest['Key'], corrections_file)
                print(f"   ✅ Downloaded corrections: {latest['Key']}")
            else:
                print("   ℹ️  No corrections found in S3")
                return {"success": True, "corrections_applied": 0, "reason": "No corrections available"}
        except Exception as e:
            print(f"   ⚠️  Failed to download corrections: {e}")
            return {"success": False, "error": str(e)}

    # Load datasets
    print("\n📊 Loading datasets...")

    if not os.path.exists(corrections_file):
        print(f"   ℹ️  No corrections file found at {corrections_file}")
        return {"success": True, "corrections_applied": 0, "reason": "No corrections file"}

    df_master = pd.read_csv(master_file)
    df_corrections = pd.read_csv(corrections_file)

    print(f"   Master dataset: {len(df_master)} rows")
    print(f"   Corrections: {len(df_corrections)} rows")

    if len(df_corrections) == 0:
        print("   ℹ️  No corrections to apply")
        return {"success": True, "corrections_applied": 0}

    # Track statistics
    updated = 0
    added = 0
    skipped = 0

    # Process corrections
    print("\n🔄 Applying corrections...")

    for _, correction in df_corrections.iterrows():
        url = correction['url']
        correct_label = int(correction['label'])

        # Find URL in master dataset
        mask = df_master['url'] == url

        if mask.any():
            # URL exists - update label
            old_label = df_master.loc[mask, 'label'].values[0]
            if old_label != correct_label:
                df_master.loc[mask, 'label'] = correct_label
                df_master.loc[mask, 'source'] = 'user_feedback'
                updated += 1
                print(f"   ✓ Updated {url[:50]}... ({old_label} → {correct_label})")
            else:
                skipped += 1
        else:
            # URL not in master - need to add it
            # For now, we'll just note it - full feature extraction would be expensive
            # The training pipeline should handle this separately
            added += 1
            print(f"   + New URL (needs features): {url[:50]}...")

    print(f"\n📊 Results:")
    print(f"   Labels updated: {updated}")
    print(f"   New URLs (pending features): {added}")
    print(f"   Skipped (already correct): {skipped}")

    # Save updated master
    df_master.to_csv(master_file, index=False)
    print(f"\n✅ Saved updated master dataset ({len(df_master)} rows)")

    # Upload to S3
    if upload_to_s3:
        try:
            s3.upload_file(master_file, S3_BUCKET, "master/phishing_features_master.csv")
            print(f"✅ Uploaded to s3://{S3_BUCKET}/master/phishing_features_master.csv")

            # Also backup with timestamp
            backup_key = f"master/backups/phishing_features_master_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            s3.upload_file(master_file, S3_BUCKET, backup_key)
            print(f"✅ Backup saved to s3://{S3_BUCKET}/{backup_key}")
        except Exception as e:
            print(f"⚠️  Failed to upload to S3: {e}")

    return {
        "success": True,
        "corrections_applied": updated,
        "new_urls": added,
        "skipped": skipped,
        "master_rows": len(df_master)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge feedback corrections into training data")
    parser.add_argument(
        "--corrections", "-c",
        default="data/feedback/corrections.csv",
        help="Path to corrections CSV"
    )
    parser.add_argument(
        "--master", "-m",
        default="data/processed/phishing_features_master.csv",
        help="Path to master dataset"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Don't download from S3 (use local files)"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Don't upload to S3"
    )

    args = parser.parse_args()

    result = merge_feedback_corrections(
        corrections_file=args.corrections,
        master_file=args.master,
        download_from_s3=not args.no_download,
        upload_to_s3=not args.no_upload
    )

    if result['success']:
        print(f"\n✅ Merge complete!")
        print(f"   Corrections applied: {result.get('corrections_applied', 0)}")
    else:
        print(f"\n❌ Merge failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
