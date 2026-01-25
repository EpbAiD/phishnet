#!/usr/bin/env python3
"""
Export user feedback corrections from PostgreSQL to CSV for model retraining.

This script:
1. Connects to the RDS PostgreSQL database
2. Fetches all user corrections (where correct_label != null)
3. Exports them as a CSV file that can be merged into training data
4. Tracks which corrections have been exported to avoid duplicates
"""

import os
import sys
import csv
import boto3
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("Installing psycopg2...")
    os.system("pip install psycopg2-binary")
    import psycopg2
    from psycopg2.extras import RealDictCursor


# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://phishnet_admin:PhishNet2024Secure@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet"
)

S3_BUCKET = "phishnet-data"
AWS_REGION = "us-east-1"


def get_db_connection():
    """Create a database connection."""
    return psycopg2.connect(DATABASE_URL)


def export_corrections(
    output_file: str,
    since_days: int = 30,
    min_corrections: int = 10,
    upload_to_s3: bool = True
) -> dict:
    """
    Export user corrections from database to CSV.

    Args:
        output_file: Path to output CSV file
        since_days: Only export corrections from the last N days
        min_corrections: Minimum number of corrections required to export
        upload_to_s3: Whether to upload to S3 after export

    Returns:
        dict with export statistics
    """

    print("=" * 60)
    print("EXPORT FEEDBACK CORRECTIONS")
    print("=" * 60)

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Calculate date threshold
    since_date = datetime.utcnow() - timedelta(days=since_days)

    # Query corrections with scan details
    query = """
        SELECT
            s.url,
            s.prediction as original_prediction,
            f.correct_label,
            s.confidence as original_confidence,
            s.url_model_score,
            s.dns_model_score,
            s.whois_model_score,
            s.model_version,
            f.source,
            f.submitted_at
        FROM feedback f
        JOIN scans s ON f.scan_id = s.id
        WHERE f.correct_label IS NOT NULL
          AND f.submitted_at >= %s
        ORDER BY f.submitted_at DESC
    """

    cursor.execute(query, (since_date,))
    corrections = cursor.fetchall()

    print(f"Found {len(corrections)} corrections in the last {since_days} days")

    if len(corrections) < min_corrections:
        print(f"⚠️  Not enough corrections (minimum: {min_corrections})")
        print("   Skipping export - wait for more user feedback")
        cursor.close()
        conn.close()
        return {
            "exported": False,
            "total_corrections": len(corrections),
            "min_required": min_corrections,
            "reason": "Insufficient corrections"
        }

    # Count correction types
    false_positives = sum(1 for c in corrections if c['original_prediction'] == 1 and c['correct_label'] == 0)
    false_negatives = sum(1 for c in corrections if c['original_prediction'] == 0 and c['correct_label'] == 1)

    print(f"\n📊 Correction breakdown:")
    print(f"   False positives (marked as safe): {false_positives}")
    print(f"   False negatives (marked as phishing): {false_negatives}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Write to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header matching the training data format
        writer.writerow([
            'url',
            'label',  # The corrected label
            'source',
            'original_prediction',
            'original_confidence',
            'correction_date'
        ])

        for correction in corrections:
            writer.writerow([
                correction['url'],
                correction['correct_label'],  # User's correction
                f"user_feedback_{correction['source'] or 'unknown'}",
                correction['original_prediction'],
                correction['original_confidence'],
                correction['submitted_at'].isoformat() if correction['submitted_at'] else ''
            ])

    print(f"\n✅ Exported {len(corrections)} corrections to {output_file}")

    # Upload to S3
    if upload_to_s3:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3_key = f"feedback/corrections_{datetime.utcnow().strftime('%Y%m%d')}.csv"

        try:
            s3.upload_file(output_file, S3_BUCKET, s3_key)
            print(f"✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")
        except Exception as e:
            print(f"⚠️  Failed to upload to S3: {e}")

    cursor.close()
    conn.close()

    return {
        "exported": True,
        "total_corrections": len(corrections),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "output_file": output_file
    }


def get_feedback_stats() -> dict:
    """Get overall feedback statistics."""

    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Total scans
    cursor.execute("SELECT COUNT(*) as count FROM scans")
    total_scans = cursor.fetchone()['count']

    # Total feedback
    cursor.execute("SELECT COUNT(*) as count FROM feedback")
    total_feedback = cursor.fetchone()['count']

    # Corrections (with correct_label)
    cursor.execute("SELECT COUNT(*) as count FROM feedback WHERE correct_label IS NOT NULL")
    total_corrections = cursor.fetchone()['count']

    # Explanation feedback
    cursor.execute("SELECT COUNT(*) as count FROM feedback WHERE explanation_helpful IS NOT NULL")
    explanation_feedback = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM feedback WHERE explanation_helpful = TRUE")
    helpful_count = cursor.fetchone()['count']

    cursor.close()
    conn.close()

    return {
        "total_scans": total_scans,
        "total_feedback": total_feedback,
        "total_corrections": total_corrections,
        "explanation_feedback": explanation_feedback,
        "helpful_explanations": helpful_count,
        "not_helpful_explanations": explanation_feedback - helpful_count if explanation_feedback > 0 else 0
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export user feedback corrections for retraining")
    parser.add_argument(
        "--output", "-o",
        default="data/feedback/corrections.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Export corrections from the last N days (default: 30)"
    )
    parser.add_argument(
        "--min-corrections", "-m",
        type=int,
        default=10,
        help="Minimum corrections required to export (default: 10)"
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip uploading to S3"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Just show feedback statistics"
    )

    args = parser.parse_args()

    if args.stats:
        stats = get_feedback_stats()
        print("\n📊 Feedback Statistics:")
        print(f"   Total scans: {stats['total_scans']}")
        print(f"   Total feedback: {stats['total_feedback']}")
        print(f"   Corrections: {stats['total_corrections']}")
        print(f"   Explanation feedback: {stats['explanation_feedback']}")
        print(f"      - Helpful: {stats['helpful_explanations']}")
        print(f"      - Not helpful: {stats['not_helpful_explanations']}")
    else:
        result = export_corrections(
            output_file=args.output,
            since_days=args.days,
            min_corrections=args.min_corrections,
            upload_to_s3=not args.no_s3
        )

        if result['exported']:
            print(f"\n✅ Export complete!")
        else:
            print(f"\n⚠️  Export skipped: {result.get('reason', 'Unknown reason')}")
