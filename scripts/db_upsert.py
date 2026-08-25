#!/usr/bin/env python3
"""
Postgres upsert helpers for the daemon's accumulation step.

Each function takes a pandas DataFrame with the batch's rows and INSERTs
them with ON CONFLICT DO UPDATE. Memory footprint is O(batch_size) — one
batch is ~3000 rows / ~5 MB, regardless of how big the tables have grown.

This replaces the "download master → concat → dedup → upload master"
loop in extract_vm_features_aws.py that OOMed t3.micro at ~500 MB masters.

The upsert uses psycopg2.extras.execute_values, which builds one multi-row
INSERT with ~1000 rows per statement. That's the fastest per-row throughput
psycopg2 offers without dropping into COPY.
"""

from __future__ import annotations

import math
import os
from typing import Iterable

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL") or (
    "postgresql://phishnet_admin:PhishNet2024Secure"
    "@phishnet-db.c83quikqw26n.us-east-1.rds.amazonaws.com:5432/phishnet"
)

# Batch size for each multi-row INSERT statement. 1000 is a well-known sweet
# spot for psycopg2's execute_values (larger batches don't win much and
# start straining the max_stack_depth on the server).
_INSERT_PAGE = 1000


def open_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=15)


def _rows_from_df(df: pd.DataFrame, columns: list[str]) -> Iterable[tuple]:
    """
    Yield tuples in the given column order. Convert pandas NaN → None so
    psycopg2 sends SQL NULL (INSERT would otherwise fail on numeric cols).
    """
    # Select only the columns we care about, in the exact order the SQL wants.
    subset = df[columns]
    # itertuples is ~3x faster than iterrows and it hands us plain values.
    for row in subset.itertuples(index=False, name=None):
        yield tuple(None if _is_null(v) else v for v in row)


def _is_null(v) -> bool:
    """True for pandas NaN, None, or empty string."""
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v == "":
        return True
    return False


def _upsert(
    conn,
    table: str,
    key_col: str,
    columns: list[str],
    df: pd.DataFrame,
) -> int:
    """
    Generic upsert. Uses ON CONFLICT (<key_col>) DO UPDATE for all non-key
    columns. `updated_at` is refreshed automatically via the schema DEFAULT
    plus an explicit SET below.
    """
    if df.empty:
        return 0

    non_key = [c for c in columns if c != key_col]
    col_list = ", ".join(columns)
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_key)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES %s "
        f"ON CONFLICT ({key_col}) DO UPDATE SET {update_set}, updated_at = NOW()"
    )

    # Drop rows where the PK would be NULL, dedupe by PK within the batch
    # (last write wins), AND drop rows where the PK is longer than the btree
    # index can hold. Four failure modes we're guarding against:
    #   1. Row with null PK → NotNullViolation, whole batch rolled back.
    #   2. Two rows with the same PK in ONE INSERT → 'ON CONFLICT DO UPDATE
    #      command cannot affect row a second time' → whole batch rolled back.
    #   3. PK longer than ~8 KB → ProgramLimitExceeded from btree page limit
    #      ('index row requires 13216 bytes, maximum size is 8191'). Once per
    #      ~thousand batches we get a garbage URL that's tens of KB long.
    #   4. Non-PK 'url' column in DNS table also has a btree index — same limit.
    # In practice ~1 row/batch has null PK, DNS batches often have same-domain
    # dupes (shared hosting), and oversized-URL rows come from adversarial
    # phishing kits with junk-padded querystrings.
    MAX_INDEXED_STR = 2000  # bytes. Standard URL length ceiling. Any real URL
                            # fits well under this; anything longer is garbage.
    key_idx = columns.index(key_col)
    url_idx = columns.index("url") if "url" in columns and key_col != "url" else None
    all_rows = list(_rows_from_df(df, columns))
    by_key: dict = {}
    null_dropped = 0
    oversized_dropped = 0
    for r in all_rows:
        k = r[key_idx]
        if k is None or k == "":
            null_dropped += 1
            continue
        if isinstance(k, str) and len(k.encode("utf-8")) > MAX_INDEXED_STR:
            oversized_dropped += 1
            continue
        # Also guard the non-PK url column (indexed on dns_features)
        if url_idx is not None:
            u = r[url_idx]
            if isinstance(u, str) and len(u.encode("utf-8")) > MAX_INDEXED_STR:
                oversized_dropped += 1
                continue
        by_key[k] = r  # last write wins
    rows = list(by_key.values())
    dupe_dropped = len(all_rows) - null_dropped - oversized_dropped - len(rows)
    if null_dropped:
        print(
            f"[db_upsert] {table}: dropped {null_dropped} rows with null/empty {key_col}",
            flush=True,
        )
    if oversized_dropped:
        print(
            f"[db_upsert] {table}: dropped {oversized_dropped} rows with url/{key_col} > {MAX_INDEXED_STR} bytes",
            flush=True,
        )
    if dupe_dropped:
        print(
            f"[db_upsert] {table}: deduped by {key_col} ({dupe_dropped} dupes, kept last)",
            flush=True,
        )
    if not rows:
        return 0

    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=_INSERT_PAGE)
    conn.commit()
    return len(rows)


# ---- feature-type-specific upserts ----
#
# Column lists are hard-coded here (not derived from the DataFrame) so a
# column-order bug or an accidental extra column can't slip through. If the
# schema changes, update these lists AND db_schema.sql together.

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


def _select_or_null(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Return `df[columns]` — but if any column is missing from df, add it as
    all-NULL. This makes the upsert robust to source CSV drift (e.g. an
    older batch that didn't yet have `collected_at`).
    """
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            out[c] = None
    return out[columns]


def upsert_url_features(conn, df: pd.DataFrame) -> int:
    return _upsert(conn, "url_features", "url", URL_COLS, _select_or_null(df, URL_COLS))


def upsert_dns_features(conn, df: pd.DataFrame) -> int:
    return _upsert(conn, "dns_features", "domain", DNS_COLS, _select_or_null(df, DNS_COLS))


def upsert_whois_features(conn, df: pd.DataFrame) -> int:
    return _upsert(conn, "whois_features", "url", WHOIS_COLS, _select_or_null(df, WHOIS_COLS))
