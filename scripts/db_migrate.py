#!/usr/bin/env python3
"""
Database migration helpers for the phishnet daemon.

Two functions callable from ec2_daemon.py on startup:
  1. ensure_schema(conn)              — apply db_schema.sql (idempotent)
  2. migrate_from_s3_if_empty(conn)   — one-shot: if a table is empty,
                                        stream the corresponding S3 master
                                        CSV into it via COPY. Skips any
                                        table that already has rows, so
                                        the daemon is safe to restart.

The migration streams via a temp file on disk, not into pandas — a
34 MB CSV can be COPY'd row-by-row in constant memory (~20 MB).
Safe on t3.micro even without swap.

Also runnable as a CLI for out-of-band invocation:
    python scripts/db_migrate.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import boto3
import psycopg2

# See earlier note: empty env var must fall through to default.
DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://phishnet_admin:PhishNet2024Secure"
    "@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet"
)
S3_BUCKET = os.getenv("S3_BUCKET", "phishnet-data")

# Mapping: (table_name, s3_key). Column order in CSV must match column
# order in the table; the schema was built to match the current masters.
MASTERS: tuple[tuple[str, str], ...] = (
    ("url_features",   "master/url_features_master.csv"),
    ("dns_features",   "master/dns_features_master.csv"),
    ("whois_features", "master/whois_features_master.csv"),
)

# Postgres COPY treats "" as an empty string, which is invalid for TIMESTAMPTZ
# and numeric columns. Replace empty CSV fields with this sentinel and tell
# COPY to treat the sentinel as NULL.
_NULL_SENTINEL = "\\N"


def _schema_path() -> Path:
    """Locate db_schema.sql relative to this file."""
    return Path(__file__).resolve().parent / "db_schema.sql"


def ensure_schema(conn) -> None:
    """Apply db_schema.sql. Idempotent — safe on every daemon boot."""
    sql = _schema_path().read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def _table_is_empty(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(f"SELECT EXISTS (SELECT 1 FROM {table} LIMIT 1)")
        (has_any,) = cur.fetchone()
    return not has_any


def _copy_csv_from_s3(conn, table: str, s3_key: str) -> int:
    """
    Download the S3 CSV to a temp file, rewrite empty fields as NULL
    sentinels via streaming csv.reader/writer (no pandas, no full-buffer
    load), then COPY into `table`. Returns row count.
    """
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as raw_f:
        raw_path = raw_f.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8", newline=""
    ) as clean_f:
        clean_path = clean_f.name

    try:
        print(f"[migrate] downloading s3://{S3_BUCKET}/{s3_key}", flush=True)
        s3.download_file(S3_BUCKET, s3_key, raw_path)
        print(
            f"[migrate]   ({os.path.getsize(raw_path) / 1024 / 1024:.1f} MB)",
            flush=True,
        )

        with open(raw_path, "r", encoding="utf-8", newline="") as raw, \
             open(clean_path, "w", encoding="utf-8", newline="") as clean:
            reader = csv.reader(raw)
            writer = csv.writer(clean, quoting=csv.QUOTE_MINIMAL)
            header = next(reader)
            columns = [c.strip() for c in header]
            writer.writerow(columns)
            row_count = 0
            for row in reader:
                writer.writerow(
                    _NULL_SENTINEL if v == "" else v for v in row
                )
                row_count += 1

        print(
            f"[migrate] COPY {row_count} rows into {table} "
            f"({len(columns)} cols)",
            flush=True,
        )
        with conn.cursor() as cur, open(clean_path, "r", encoding="utf-8") as f:
            f.readline()  # skip header — we pass column list explicitly
            cur.copy_expert(
                f"COPY {table} ({', '.join(columns)}) FROM STDIN "
                f"WITH (FORMAT csv, NULL '{_NULL_SENTINEL}')",
                f,
            )
        return row_count
    finally:
        for p in (raw_path, clean_path):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass


def migrate_from_s3_if_empty(conn) -> dict[str, int]:
    """
    For each master, if its table is empty, COPY the S3 CSV in.
    Never TRUNCATEs. Returns {table: rows_migrated} — 0 means skipped.
    """
    results: dict[str, int] = {}
    for table, s3_key in MASTERS:
        if not _table_is_empty(conn, table):
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                (n,) = cur.fetchone()
            print(f"[migrate] {table}: skip ({n} rows already present)", flush=True)
            results[table] = 0
            continue
        n = _copy_csv_from_s3(conn, table, s3_key)
        conn.commit()
        results[table] = n
    return results


def _open_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=15)


def main() -> int:
    conn = _open_conn()
    try:
        ensure_schema(conn)
        results = migrate_from_s3_if_empty(conn)
        print("\n[migrate] summary:")
        for t, n in results.items():
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {t}")
                (in_db,) = cur.fetchone()
            action = f"migrated {n}" if n else "skipped"
            print(f"  {t:20s} {action}, now {in_db} rows in DB")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
