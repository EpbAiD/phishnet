#!/usr/bin/env python3
"""
One-time backfill: stream the three S3 master CSVs into Postgres tables.

Runs on a GitHub Actions runner (has 16 GB RAM), NOT on the t3.micro daemon.
Uses psycopg2's COPY protocol which streams row-by-row instead of loading
the full CSV into pandas first. Idempotent for schema; TRUNCATE + reload
each table so re-runs converge to the S3 state.

Env vars:
  DATABASE_URL   Postgres connection string (falls back to hardcoded prod)
  S3_BUCKET      defaults to phishnet-data
"""

import io
import os
import sys
from pathlib import Path

import boto3
import psycopg2

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://phishnet_admin:PhishNet2024Secure@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet",
)
S3_BUCKET = os.getenv("S3_BUCKET", "phishnet-data")

# (table_name, s3_key, columns-in-CSV-order)
# Column order MUST match the CSV header exactly, otherwise COPY inserts
# values into the wrong columns. Header is read at runtime and validated.
MASTERS = [
    ("url_features",   "master/url_features_master.csv"),
    ("dns_features",   "master/dns_features_master.csv"),
    ("whois_features", "master/whois_features_master.csv"),
]

# Which columns can legitimately be empty in the CSV but must be NULL
# (not empty string) in the DB. TIMESTAMPTZ and numeric columns can't
# accept "" — Postgres rejects the row. This lets COPY treat "" as NULL.
NULL_STRING = "\\N"  # standard Postgres NULL sentinel for COPY

def stream_s3_csv(bucket: str, key: str):
    """Yield the CSV as a text stream. No local file, no full-buffer load."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    # Read the header separately so we can log + validate column list.
    header_bytes = b""
    while not header_bytes.endswith(b"\n"):
        chunk = body.read(1)
        if not chunk:
            break
        header_bytes += chunk
    header = header_bytes.decode("utf-8").rstrip("\n\r")
    columns = [c.strip() for c in header.split(",")]
    return header, columns, body

def transform_stream(body, columns):
    """
    Transform the raw CSV body into a stream Postgres COPY can ingest:
      - keep header
      - convert empty fields in TIMESTAMPTZ / numeric-ish columns to \\N
      - pass everything else through unchanged
    We do a lightweight per-line transformation instead of using csv module
    to avoid loading the whole file. Since our CSVs don't have embedded
    newlines within quoted fields for the timestamp column, a simple
    split(',') on trailing empty fields is enough.
    """
    # Any column whose CSV value is "" gets replaced with \N. Postgres COPY
    # with `NULL '\N'` will treat that as NULL. We do this only for columns
    # that Postgres would otherwise reject empty strings for (TIMESTAMPTZ,
    # INTEGER, DOUBLE PRECISION). We apply it universally — a "" TEXT
    # becomes NULL too, which is fine and matches pandas behavior on read_csv.
    buf = io.BytesIO()
    buf.write((",".join(columns) + "\n").encode("utf-8"))

    # Stream in chunks, but split at line boundaries. Use TextIOWrapper for
    # line-oriented reading without .read() blowing the buffer.
    import io as _io
    text = _io.TextIOWrapper(body, encoding="utf-8", newline="")
    for line in text:
        # Fields split on commas that are NOT inside quotes.
        # For robustness (some rows have quoted stringified lists), use csv reader:
        pass
    return buf

def copy_table_from_s3(cur, table: str, s3_key: str) -> tuple[int, list[str]]:
    """
    Stream one master CSV from S3 straight into `table` via COPY.
    Returns (row_count, column_list).
    """
    print(f"\n→ TRUNCATE {table}", flush=True)
    cur.execute(f"TRUNCATE TABLE {table}")

    # Download to a local temp file first. GH Actions runner has plenty of
    # disk; this is simpler than the streaming transform for CSVs with
    # quoted list-string fields (DNS asn_list = "['DE']" would confuse a
    # naive line splitter).
    import csv
    import tempfile

    s3 = boto3.client("s3")
    print(f"→ Downloading s3://{S3_BUCKET}/{s3_key}", flush=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as raw_f:
        raw_path = raw_f.name
    s3.download_file(S3_BUCKET, s3_key, raw_path)
    raw_size_mb = os.path.getsize(raw_path) / 1024 / 1024
    print(f"  ({raw_size_mb:.1f} MB downloaded to {raw_path})", flush=True)

    # Re-emit the CSV with empty fields replaced by \N. Uses csv.reader so
    # quoted fields (asn_list = "['DE']") stay intact. Also writes to a
    # temp file so we don't hold the whole thing in memory.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as cleaned_f:
        cleaned_path = cleaned_f.name
        with open(raw_path, "r", encoding="utf-8") as raw:
            reader = csv.reader(raw)
            header = next(reader)
            columns = [c.strip() for c in header]
            writer = csv.writer(cleaned_f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            row_count = 0
            for row in reader:
                cleaned = [(NULL_STRING if v == "" else v) for v in row]
                writer.writerow(cleaned)
                row_count += 1

    print(f"→ COPY {row_count} rows into {table} (columns: {len(columns)})", flush=True)
    with open(cleaned_path, "r", encoding="utf-8") as f:
        # Skip header, since we're using COPY with explicit column list.
        f.readline()
        cur.copy_expert(
            f"COPY {table} ({', '.join(columns)}) FROM STDIN "
            f"WITH (FORMAT csv, NULL '{NULL_STRING}')",
            f,
        )

    os.unlink(raw_path)
    os.unlink(cleaned_path)
    return row_count, columns


def main() -> int:
    schema_path = Path(__file__).parent / "db_schema.sql"
    if not schema_path.exists():
        print(f"❌ Schema file not found at {schema_path}", flush=True)
        return 1

    print(f"→ Connecting to Postgres", flush=True)
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            print(f"→ Applying schema from {schema_path.name}", flush=True)
            cur.execute(schema_path.read_text())

            totals = {}
            for table, s3_key in MASTERS:
                inserted, cols = copy_table_from_s3(cur, table, s3_key)
                # sanity: row count in table
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                (in_db,) = cur.fetchone()
                totals[table] = (inserted, in_db)
                print(f"  ✔ {table}: inserted {inserted}, in-DB {in_db}", flush=True)
                if inserted != in_db:
                    raise RuntimeError(
                        f"{table}: COPY reported {inserted} rows but SELECT COUNT(*) sees {in_db}"
                    )

            conn.commit()
            print("\n=== BACKFILL COMPLETE ===")
            for table, (i, d) in totals.items():
                print(f"  {table:20s}  {d:>8d} rows")
            print("Committed.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
