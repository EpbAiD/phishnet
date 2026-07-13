#!/usr/bin/env python3
"""
Export the three feature tables from Postgres to CSV on disk.

Called from training workflows (daily_retrain.yml, unified_pipeline.yml)
after the runner IP has been whitelisted in the RDS security group. Writes
to the paths existing training/analysis scripts already read:
    data/processed/url_features_master.csv
    data/processed/dns_features_master.csv
    data/processed/whois_features_master.csv

Uses psycopg2 named cursors (server-side) with itersize so we stream rows
from Postgres in chunks instead of holding the whole table in memory. A
5 M row table exports in constant ~50 MB memory instead of ~2 GB.

Env vars:
  DATABASE_URL   Postgres connection string
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://phishnet_admin:PhishNet2024Secure"
    "@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet"
)

# Column lists are the ones training reads — pinned so a schema addition
# doesn't accidentally leak into training data. Same as db_upsert.py.
URL_COLS = [
    "url", "label", "source",
    "url_length", "hostname_length", "path_length",
    "num_subdomains", "num_dots", "num_special_chars", "num_digits",
    "num_uppercase_chars",
    "has_at_symbol", "has_double_slash_redirect", "has_dash_in_domain",
    "is_ip_address", "ip_category",
    "has_encoded_chars", "has_non_ascii_chars",
    "url_entropy", "hostname_entropy", "digit_to_letter_ratio",
    "domain_quality", "tld_length", "subdomain_entropy", "subdomain_length",
    "has_login_keyword", "has_suspicious_words", "has_brand_mismatch",
    "file_type", "is_file_download", "is_script_file", "is_shortened",
    "num_fragments", "num_query_params", "num_directories",
    "port", "is_risky_port", "protocol_mismatch", "is_unknown_port",
    "contains_hex_encoding", "starts_with_https_but_contains_http",
    "missing_hostname_flag", "collected_at",
]

DNS_COLS = [
    "domain",
    "has_A", "num_A", "has_AAAA", "num_AAAA",
    "has_MX", "num_MX", "has_NS", "num_NS",
    "has_TXT", "num_TXT", "has_CNAME", "cname_chain_length", "has_SOA",
    "ttl_min", "ttl_max", "ttl_mean", "ttl_var",
    "mx_priority_min", "mx_priority_max", "num_distinct_ips", "txt_entropy",
    "has_SPF", "has_DKIM", "has_DMARC", "has_wildcard_dns", "dnssec_enabled",
    "asn_list", "asn_org_list", "asn_country_list", "cidr_list", "error_type",
    "url", "label", "collected_at",
]

WHOIS_COLS = [
    "url", "registrar", "whois_server",
    "creation_date", "expiration_date", "updated_date",
    "domain_age_days", "registration_length_days",
    "status", "registrant_country",
    "has_privacy_protection", "whois_success", "error_msg",
    "label", "collected_at",
]

TABLES = (
    ("url_features",   URL_COLS,   "data/processed/url_features_master.csv"),
    ("dns_features",   DNS_COLS,   "data/processed/dns_features_master.csv"),
    ("whois_features", WHOIS_COLS, "data/processed/whois_features_master.csv"),
)

# Server-side cursor page size. 10K keeps memory bounded but avoids the
# per-fetch round-trip overhead of a tiny page.
_ITER_SIZE = 10_000


def export_table(conn, table: str, columns: list[str], out_path: str) -> int:
    """
    Stream SELECT <columns> FROM <table> straight into a CSV file, one
    server-side page at a time. Named cursors require autocommit=False,
    which is psycopg2's default.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    col_sql = ", ".join(columns)
    # Named cursor → server-side cursor → streaming fetch.
    with conn.cursor(name=f"export_{table}") as cur:
        cur.itersize = _ITER_SIZE
        cur.execute(f"SELECT {col_sql} FROM {table}")

        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(columns)
            row_count = 0
            for row in cur:
                # Convert Nones → empty string so downstream pandas.read_csv
                # doesn't get literal "None" strings. Everything else stays.
                writer.writerow("" if v is None else v for v in row)
                row_count += 1

    print(f"[export] {table}: {row_count} rows → {out_path}", flush=True)
    return row_count


def main() -> int:
    print(f"[export] connecting to Postgres", flush=True)
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
    try:
        totals = {}
        for table, columns, out_path in TABLES:
            totals[table] = export_table(conn, table, columns, out_path)
        print("\n[export] summary:")
        for t, n in totals.items():
            print(f"  {t:20s} {n:>8d} rows exported")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
