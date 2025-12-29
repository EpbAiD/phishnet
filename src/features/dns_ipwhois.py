# ===============================================================
# dns_ipwhois_features.py
# Unified DNS + IPWHOIS extraction (training + inference)
# ===============================================================
# ✅ Dual-mode support:
#    - "batch": Uses workers, delays, retries, checkpointing (for training)
#    - "single": Fast extraction without delays (for FastAPI inference)
# ✅ extract_single_domain_features() - clean API for FastAPI (no delays/workers)
# ✅ build_dns_features() - batch processing with caching & throttling
# ✅ IP and URL-aware (handles "192.168.1.1:8080", "https://...", "example.com")
# ✅ Caching, checkpointing, and latency logging for batch mode
# ===============================================================

import os
import re
import time
import random
import statistics
import ipaddress
from urllib.parse import urlparse
from collections import Counter
from math import log2
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union

import pandas as pd
import dns.resolver
from ipwhois import IPWhois

# ---------------- Configuration ----------------
BASE_DIR = "data"
RAW_DOMAIN_LIST = os.path.join(BASE_DIR, "processed/full_domain_list.csv")
DNS_FEATURE_FILE = os.path.join(BASE_DIR, "processed/dns_ipwhois_results.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE = os.path.join("logs", "lookup_times.csv")

os.makedirs(os.path.dirname(DNS_FEATURE_FILE), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

MAX_WORKERS = 5  # Only used in batch mode
PER_DOMAIN_DELAY = 2.0  # Only applied in batch mode
INTRA_QUERY_DELAY = 0.5  # Only applied in batch mode
CHECKPOINT_EVERY = 500  # Only used in batch mode

# ---------------- Resolver ----------------
resolver = dns.resolver.Resolver()
resolver.timeout = 3
resolver.lifetime = 5


# ---------------- Helper Functions ----------------
def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = [c / len(s) for c in counts.values()]
    return -sum(p * log2(p) for p in probs)


def _log_latency(
    source: str, op: str, ident: str, start_t: float, ok: bool, extra: Dict = None
):
    dur = time.time() - start_t
    row = {
        "source": source,
        "operation": op,
        "identifier": ident,
        "start_ts": start_t,
        "duration_s": round(dur, 6),
        "status": "ok" if ok else "fail",
    }
    if extra:
        row.update(extra)
    pd.DataFrame([row]).to_csv(
        LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False
    )


def safe_query(domain: str, rrtype: str, apply_delay: bool = True):
    """DNS query with optional delay (only for batch mode)"""
    try:
        ans = resolver.resolve(domain, rrtype)
        if apply_delay:
            time.sleep(INTRA_QUERY_DELAY)
        return ans
    except Exception:
        if apply_delay:
            time.sleep(INTRA_QUERY_DELAY)
        return None


def safe_ipwhois(ip: str) -> Dict[str, str]:
    """IPWHOIS lookup - no delays needed as it's rate-limited naturally"""
    try:
        info = IPWhois(ip).lookup_rdap(depth=1)
        return {
            "asn": info.get("asn", "NA"),
            "asn_org": info.get("asn_description", "NA"),
            "asn_country": info.get("asn_country_code", "NA"),
            "cidr_range": info.get("asn_cidr", "NA"),
        }
    except Exception:
        return {"asn": "NA", "asn_org": "NA", "asn_country": "NA", "cidr_range": "NA"}


def dns_record_template(domain: str) -> Dict:
    return {
        "domain": domain,
        "has_A": 0,
        "num_A": 0,
        "has_AAAA": 0,
        "num_AAAA": 0,
        "has_MX": 0,
        "num_MX": 0,
        "has_NS": 0,
        "num_NS": 0,
        "has_TXT": 0,
        "num_TXT": 0,
        "has_CNAME": 0,
        "cname_chain_length": 0,
        "has_SOA": 0,
        "ttl_min": -1,
        "ttl_max": -1,
        "ttl_mean": -1.0,
        "ttl_var": -1.0,
        "mx_priority_min": -1,
        "mx_priority_max": -1,
        "num_distinct_ips": 0,
        "txt_entropy": 0.0,
        "has_SPF": 0,
        "has_DKIM": 0,
        "has_DMARC": 0,
        "has_wildcard_dns": 0,
        "dnssec_enabled": 0,
        "asn_list": [],
        "asn_org_list": [],
        "asn_country_list": [],
        "cidr_list": [],
        "error_type": -1.0,
    }


# ---------------- Domain Normalization ----------------
def extract_domain_from_url(url: str) -> str:
    """Extract a clean hostname or IP from URL or raw string."""
    if not isinstance(url, str):
        return ""
    if re.match(r"^https?://", url):
        parsed = urlparse(url)
        host = parsed.hostname or url
    else:
        host = url.split("/")[0]
    # Strip port
    host = host.split(":")[0].strip().lower()
    return host


# ---------------- Core DNS + IPWHOIS Lookup ----------------
def extract_dns_ipwhois(domain: str, mode: str = "batch") -> Dict:
    """
    Extract DNS + IPWHOIS features for a given domain or IP.
    mode: "batch" includes throttling and delays; "single" runs fast for inference.

    ✅ FIXED: Independent try-except blocks for each query type
    """
    clean_host = extract_domain_from_url(domain)
    feats = dns_record_template(clean_host)
    ttl_values: List[int] = []
    ip_set = set()
    t0 = time.time()

    # Determine if delays should be applied
    apply_delay = mode == "batch"

    # --- If input is a direct IP, skip DNS lookups entirely ---
    try:
        ipaddress.ip_address(clean_host)
        info = safe_ipwhois(clean_host)
        feats["has_A"] = 1
        feats["num_A"] = 1
        feats["num_distinct_ips"] = 1
        feats["asn_list"] = [info["asn"]]
        feats["asn_org_list"] = [info["asn_org"]]
        feats["asn_country_list"] = [info["asn_country"]]
        feats["cidr_list"] = [info["cidr_range"]]
        _log_latency("ipwhois", "direct_ip", clean_host, t0, True)
        return feats
    except ValueError:
        # Not an IP; continue DNS route
        pass

    # --- A / AAAA (independent try-except) ---
    try:
        for rtype in ["A", "AAAA"]:
            ans = safe_query(clean_host, rtype, apply_delay)
            if ans:
                feats[f"has_{rtype}"] = 1
                feats[f"num_{rtype}"] = len(ans)
                try:
                    ips = [r.address for r in ans]
                    ip_set.update(ips)
                except Exception:
                    pass
                try:
                    if (
                        hasattr(ans, "rrset")
                        and ans.rrset
                        and hasattr(ans.rrset, "ttl")
                    ):
                        ttl_values.append(int(ans.rrset.ttl))
                    elif (
                        hasattr(ans, "response")
                        and ans.response
                        and ans.response.answer
                    ):
                        for rr in ans.response.answer:
                            if hasattr(rr, "ttl"):
                                ttl_values.append(int(rr.ttl))
                except Exception:
                    pass
    except Exception:
        pass  # Continue with other queries

    # --- IPWHOIS (independent try-except) ---
    try:
        feats["num_distinct_ips"] = len(ip_set)
        for ip in ip_set:
            info = safe_ipwhois(ip)
            feats["asn_list"].append(info["asn"])
            feats["asn_org_list"].append(info["asn_org"])
            feats["asn_country_list"].append(info["asn_country"])
            feats["cidr_list"].append(info["cidr_range"])
    except Exception:
        pass  # Continue with other queries

    # --- TTL stats (independent try-except) ---
    try:
        if ttl_values:
            feats["ttl_min"] = min(ttl_values)
            feats["ttl_max"] = max(ttl_values)
            feats["ttl_mean"] = float(statistics.mean(ttl_values))
            feats["ttl_var"] = (
                float(statistics.pvariance(ttl_values)) if len(ttl_values) > 1 else 0.0
            )
    except Exception:
        pass  # Continue with other queries

    # --- MX (independent try-except) ---
    try:
        ans = safe_query(clean_host, "MX", apply_delay)
        if ans:
            feats["has_MX"] = 1
            feats["num_MX"] = len(ans)
            try:
                prios = [int(r.preference) for r in ans]
                if prios:
                    feats["mx_priority_min"] = min(prios)
                    feats["mx_priority_max"] = max(prios)
            except Exception:
                pass
    except Exception:
        pass  # Continue with other queries

    # --- NS (independent try-except) ---
    try:
        ans = safe_query(clean_host, "NS", apply_delay)
        if ans:
            feats["has_NS"] = 1
            feats["num_NS"] = len(ans)
    except Exception:
        pass  # Continue with other queries

    # --- TXT (independent try-except) ---
    try:
        ans = safe_query(clean_host, "TXT", apply_delay)
        if ans:
            feats["has_TXT"] = 1
            feats["num_TXT"] = len(ans)
            txts = []
            try:
                for r in ans:
                    parts = getattr(r, "strings", None)
                    if parts:
                        txts.append("".join(p.decode("utf-8", "ignore") for p in parts))
                    else:
                        txts.append(r.to_text().strip('"'))
            except Exception:
                pass
            if txts:
                ent = [shannon_entropy(t) for t in txts]
                feats["txt_entropy"] = float(statistics.mean(ent))
                joined = " ".join(t.lower() for t in txts)
                feats["has_SPF"] = int("v=spf" in joined)
                feats["has_DKIM"] = int("dkim" in joined)
                feats["has_DMARC"] = int("dmarc" in joined)
    except Exception:
        pass  # Continue with other queries

    # --- CNAME (independent try-except) ---
    try:
        ans = safe_query(clean_host, "CNAME", apply_delay)
        if ans:
            feats["has_CNAME"] = 1
            feats["cname_chain_length"] = len(ans) if len(ans) > 0 else 1
    except Exception:
        pass  # Continue with other queries

    # --- SOA (independent try-except) ---
    try:
        ans = safe_query(clean_host, "SOA", apply_delay)
        if ans:
            feats["has_SOA"] = 1
    except Exception:
        pass  # Continue with other queries

    # --- Wildcard (independent try-except) ---
    try:
        rnd = f"rand{random.randint(100000,999999)}.{clean_host}"
        if safe_query(rnd, "A", apply_delay):
            feats["has_wildcard_dns"] = 1
    except Exception:
        pass  # Continue with other queries

    # --- DNSSEC (independent try-except) ---
    try:
        _ = resolver.resolve(clean_host, "DNSKEY")
        feats["dnssec_enabled"] = 1
        if apply_delay:
            time.sleep(INTRA_QUERY_DELAY)
    except Exception:
        if apply_delay:
            time.sleep(INTRA_QUERY_DELAY)

    # --- Log overall success ---
    try:
        _log_latency("dns_ipwhois", "lookup", clean_host, t0, True)
    except Exception:
        pass

    # Only apply per-domain delay in batch mode
    if mode == "batch":
        time.sleep(PER_DOMAIN_DELAY)

    return feats


# ---------------- Batch Builder ----------------
def build_dns_features(
    input_domains: Union[str, List[str]], live_lookup: bool = True
) -> pd.DataFrame:
    existing_df = (
        pd.read_csv(DNS_FEATURE_FILE)
        if os.path.exists(DNS_FEATURE_FILE)
        else pd.DataFrame()
    )
    have = set(existing_df["domain"].astype(str)) if not existing_df.empty else set()

    if isinstance(input_domains, str):
        if os.path.exists(input_domains):
            domains = pd.read_csv(input_domains)["domain"].astype(str).tolist()
        else:
            domains = [input_domains]
    else:
        domains = [str(d) for d in input_domains]

    domains = [extract_domain_from_url(d) for d in domains if isinstance(d, str)]
    missing = list(set(domains) - have)
    print(f"🔍 Cached: {len(have)} | Total: {len(domains)} | Missing: {len(missing)}")

    new_rows: List[Dict] = []
    if live_lookup and missing:
        print(
            f"⚙️ Fetching DNS/IPWHOIS for {len(missing)} domains (live_lookup={live_lookup}) ..."
        )
        if len(missing) == 1:
            new_rows = [extract_dns_ipwhois(missing[0], mode="single")]
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {
                    ex.submit(extract_dns_ipwhois, d, "batch"): d for d in missing
                }
                for i, f in enumerate(as_completed(futures), start=1):
                    new_rows.append(f.result())
                    if i % CHECKPOINT_EVERY == 0:
                        ck = os.path.join(CHECKPOINT_DIR, f"dns_part_{i}.csv")
                        pd.DataFrame(new_rows).to_csv(ck, index=False)
                        print(f"💾 Checkpoint → {ck}")

    if not existing_df.empty and new_rows:
        df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    elif existing_df.empty and new_rows:
        df = pd.DataFrame(new_rows)
    else:
        df = existing_df.copy()

    if not df.empty:
        df.to_csv(DNS_FEATURE_FILE, index=False)
        print(f"✅ Saved DNS+IPWHOIS → {DNS_FEATURE_FILE} (shape: {df.shape})")
    return df


# ---------------- Cache Alignment ----------------
def _align_to_cache_schema(row_df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    if not os.path.exists(cache_path):
        return row_df
    cache_cols = list(pd.read_csv(cache_path, nrows=1).columns)
    for col in cache_cols:
        if col not in row_df.columns:
            if col in {
                "ttl_min",
                "ttl_max",
                "mx_priority_min",
                "mx_priority_max",
                "num_A",
                "num_AAAA",
                "num_MX",
                "num_NS",
                "num_TXT",
                "cname_chain_length",
                "num_distinct_ips",
            }:
                row_df[col] = 0
            elif col in {"ttl_mean", "ttl_var", "txt_entropy", "error_type"}:
                row_df[col] = 0.0
            elif col.startswith("has_") or col in {"dnssec_enabled"}:
                row_df[col] = 0
            elif col in {"asn_list", "asn_org_list", "asn_country_list", "cidr_list"}:
                row_df[col] = [[]]
            elif col == "domain":
                row_df[col] = row_df.get("domain", pd.Series(["unknown"]))
            else:
                row_df[col] = pd.NA
    extra_cols = [c for c in row_df.columns if c not in cache_cols]
    ordered = cache_cols + extra_cols
    return row_df[ordered]


# ---------------- Single-domain Fetcher ----------------
def get_domain_features(domain: str, live_lookup: bool = False) -> pd.DataFrame:
    """
    Legacy function - returns DataFrame for backward compatibility.
    For FastAPI, use extract_single_domain_features() instead.
    """
    cache_path = DNS_FEATURE_FILE
    domain = extract_domain_from_url(domain) or domain

    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path, low_memory=False)
            hit = df_cache[df_cache["domain"].astype(str) == str(domain)]
            if not hit.empty:
                return hit.reset_index(drop=True).iloc[[0]]
        except Exception:
            pass

    if live_lookup:
        feats = extract_dns_ipwhois(domain, mode="single")
        row_df = pd.DataFrame([feats])
        row_df = _align_to_cache_schema(row_df, cache_path)
        if os.path.exists(cache_path):
            row_df.to_csv(cache_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(cache_path, index=False)
        return row_df.reset_index(drop=True).iloc[[0]]

    raise ValueError(f"Domain {domain} not found in cache and live_lookup=False")


# ---------------- UNIFIED API FOR FASTAPI INFERENCE ----------------
def extract_single_domain_features(url_or_domain: str) -> Dict:
    """
    Extract DNS/IPWHOIS features from a single URL or domain (for FastAPI inference).
    No delays, no retries, no workers - fast single extraction.

    Args:
        url_or_domain: URL or domain string (e.g., "https://example.com" or "example.com")

    Returns:
        Dictionary of DNS/IPWHOIS features ready for model prediction

    Example:
        >>> features = extract_single_domain_features("https://suspicious-site.tk")
        >>> # Use features dict directly or convert to values
        >>> # feature_values = [features[key] for key in feature_order]
    """
    return extract_dns_ipwhois(url_or_domain, mode="single")


# ---------------- Script Entry ----------------
if __name__ == "__main__":
    df = build_dns_features(RAW_DOMAIN_LIST, live_lookup=True)
    print(df.head())
