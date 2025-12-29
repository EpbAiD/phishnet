# ===============================================================
# whois_feature_builder.py
# Unified WHOIS feature extraction with caching (dataset + inference)
# ===============================================================
# ✅ Dual-mode support ("batch" vs "single")
#    - batch: uses retries, delays, workers for safe bulk processing
#    - single: no retries, no delays - fast for FastAPI inference
# ✅ Handles raw domains, URLs, and IPs gracefully
# ✅ Cache → live lookup → append → schema-align
# ✅ Returns DataFrame rows ready for model inference
# ✅ Consistent with url_features.py and dns_ipwhois.py
# ===============================================================

import os
import re
import time
import ipaddress
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Union
from urllib.parse import urlparse

import pandas as pd
import whois

# ---------------- Configuration ----------------
BASE_DIR = "data"
DOMAIN_LIST = os.path.join(BASE_DIR, "processed/full_domain_list.csv")
WHOIS_FEATURE_FILE = os.path.join(BASE_DIR, "processed/whois_results.csv")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOG_FILE = os.path.join("logs", "lookup_times.csv")

MAX_WORKERS = 4
PER_DOMAIN_DELAY = 5.0      # only used in batch mode
RETRIES = 2                 # only used in batch mode
CHECKPOINT_EVERY = 500

os.makedirs(os.path.dirname(WHOIS_FEATURE_FILE), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)


# ---------------- Helpers ----------------
def extract_domain_from_url(url: str) -> str:
    """Extract a clean hostname or IP from URL or raw string."""
    if not isinstance(url, str):
        return ""
    if re.match(r"^https?://", url):
        parsed = urlparse(url)
        host = parsed.hostname or url
    else:
        host = url.split("/")[0]
    # Strip port and lowercase
    host = host.split(":")[0].strip().lower()
    return host


def _log_latency(source: str, op: str, ident: str, start_t: float, ok: bool, extra: Dict = None):
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
    pd.DataFrame([row]).to_csv(LOG_FILE, mode="a", header=not os.path.exists(LOG_FILE), index=False)


def _whois_template(domain: str) -> Dict:
    return {
        "domain": domain,
        "registrar": None,
        "whois_server": None,
        "creation_date": None,
        "expiration_date": None,
        "updated_date": None,
        "domain_age_days": None,
        "registration_length_days": None,
        "status": None,
        "registrant_country": None,
        "has_privacy_protection": 0,
        "whois_success": 0,
        "error_msg": None,
    }


def _to_dt(x):
    """Normalize python-whois date field which may be list/datetime/str/None."""
    if x is None:
        return None
    if isinstance(x, list):
        x = x[0] if x else None
    if x is None:
        return None
    if isinstance(x, datetime):
        # Remove timezone info if present to ensure timezone-naive datetime
        if x.tzinfo is not None:
            return x.replace(tzinfo=None)
        return x
    try:
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.tz_localize(None) if getattr(dt, "tzinfo", None) else dt
    except Exception:
        return None


# ---------------- Core WHOIS Lookup ----------------
def extract_whois_features(domain: str, mode: str = "batch") -> Dict:
    """
    Extract WHOIS features for a single domain or IP.
    mode: 'batch' applies delay and retries for safe bulk processing
          'single' skips delays and retries for fast real-time inference
    """
    clean_domain = extract_domain_from_url(domain)
    feats = _whois_template(clean_domain)
    t0 = time.time()

    # --- Skip WHOIS for direct IP addresses ---
    try:
        ipaddress.ip_address(clean_domain)
        feats["error_msg"] = "WHOIS not applicable to IP address"
        _log_latency("whois", "skip_ip", clean_domain, t0, True)
        return feats
    except ValueError:
        pass

    # Determine retry count based on mode
    max_attempts = RETRIES if mode == "batch" else 1

    for attempt in range(max_attempts):
        try:
            w = whois.whois(clean_domain)

            feats["registrar"] = getattr(w, "registrar", None)
            feats["whois_server"] = getattr(w, "whois_server", None)

            creation_date = _to_dt(getattr(w, "creation_date", None))
            expiration_date = _to_dt(getattr(w, "expiration_date", None))
            updated_date = _to_dt(getattr(w, "updated_date", None))

            feats["creation_date"] = str(creation_date) if creation_date else None
            feats["expiration_date"] = str(expiration_date) if expiration_date else None
            feats["updated_date"] = str(updated_date) if updated_date else None

            now = datetime.utcnow()
            if creation_date:
                feats["domain_age_days"] = (now - creation_date).days
            if creation_date and expiration_date:
                feats["registration_length_days"] = (expiration_date - creation_date).days

            # status / country
            try:
                feats["status"] = str(getattr(w, "status", None))
            except Exception:
                feats["status"] = None
            feats["registrant_country"] = getattr(w, "country", None)

            # Privacy detection
            privacy_flags = ["privacy", "redacted", "protected", "whoisguard", "gdpr"]
            reg_name = str(getattr(w, "name", "")).lower()
            reg_org = str(getattr(w, "org", "")).lower()
            reg_email = str(getattr(w, "emails", "")).lower()
            if any(flag in reg_name for flag in privacy_flags) or \
               any(flag in reg_org for flag in privacy_flags) or \
               any(flag in reg_email for flag in privacy_flags):
                feats["has_privacy_protection"] = 1

            feats["whois_success"] = 1
            _log_latency("whois", "lookup", clean_domain, t0, True)
            break
        except Exception as e:
            feats["error_msg"] = str(e)
            if attempt == max_attempts - 1:
                _log_latency("whois", "lookup", clean_domain, t0, False)
        finally:
            if mode == "batch":
                time.sleep(PER_DOMAIN_DELAY)
    return feats


# ---------------- Cache Loader ----------------
def _load_existing_whois() -> pd.DataFrame:
    if os.path.exists(WHOIS_FEATURE_FILE):
        try:
            return pd.read_csv(WHOIS_FEATURE_FILE, low_memory=False)
        except Exception:
            pass
    return pd.DataFrame()


# ---------------- Schema Alignment ----------------
def _align_to_cache_schema(row_df: pd.DataFrame, cache_path: str) -> pd.DataFrame:
    """Align single-row DataFrame to existing WHOIS cache schema."""
    if not os.path.exists(cache_path):
        return row_df

    cache_cols = list(pd.read_csv(cache_path, nrows=1).columns)
    for col in cache_cols:
        if col not in row_df.columns:
            if col in {"domain_age_days", "registration_length_days"}:
                row_df[col] = pd.NA
            elif col in {"has_privacy_protection", "whois_success"}:
                row_df[col] = 0
            elif col in {"registrar", "whois_server", "creation_date", "expiration_date",
                         "updated_date", "status", "registrant_country", "error_msg"}:
                row_df[col] = None
            elif col == "domain":
                row_df[col] = row_df.get("domain", pd.Series(["unknown"]))
            else:
                row_df[col] = pd.NA

    extra_cols = [c for c in row_df.columns if c not in cache_cols]
    ordered = cache_cols + extra_cols
    return row_df[ordered]


# ---------------- Batch Builder ----------------
def build_whois_features(input_domains: Union[str, List[str]], live_lookup: bool = True) -> pd.DataFrame:
    existing_df = _load_existing_whois()
    have = set(existing_df["domain"].astype(str)) if not existing_df.empty else set()

    if isinstance(input_domains, str):
        if os.path.exists(input_domains):
            domains = pd.read_csv(input_domains)["domain"].astype(str).tolist()
        else:
            domains = [input_domains]
    else:
        domains = [str(d) for d in input_domains]

    # Normalize all domains first
    domains = [extract_domain_from_url(d) for d in domains if isinstance(d, str)]
    missing = list(set(domains) - have)
    print(f"🔍 Cached: {len(have)} | Total: {len(domains)} | Missing: {len(missing)}")

    new_rows: List[Dict] = []
    if live_lookup and missing:
        print(f"⚙️ Fetching WHOIS data for {len(missing)} domains (live_lookup={live_lookup}) ...")
        if len(missing) == 1:
            new_rows = [extract_whois_features(missing[0], mode="single")]
        else:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = {ex.submit(extract_whois_features, d, "batch"): d for d in missing}
                for i, f in enumerate(as_completed(futures), start=1):
                    new_rows.append(f.result())
                    if (i % CHECKPOINT_EVERY) == 0:
                        ck = os.path.join(CHECKPOINT_DIR, f"whois_part_{i}.csv")
                        pd.DataFrame(new_rows).to_csv(ck, index=False)
                        print(f"💾 Checkpoint → {ck}")

    if not existing_df.empty and new_rows:
        df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    elif existing_df.empty and new_rows:
        df = pd.DataFrame(new_rows)
    else:
        df = existing_df.copy()

    if not df.empty:
        df.to_csv(WHOIS_FEATURE_FILE, index=False)
        print(f"✅ WHOIS features saved → {WHOIS_FEATURE_FILE} (shape: {df.shape})")
    return df


# ---------------- Single Inference ----------------
def get_whois_features(domain: str, live_lookup: bool = False) -> pd.DataFrame:
    """Retrieve or compute WHOIS features for a single domain or IP."""
    cache_path = WHOIS_FEATURE_FILE
    domain = extract_domain_from_url(domain) or domain

    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path, low_memory=False)
            hit = df_cache[df_cache["domain"].astype(str) == str(domain)]
            if not hit.empty:
                print(f"✅ Found cached WHOIS record for {domain}")
                return hit.reset_index(drop=True).iloc[[0]]
        except Exception:
            pass

    if live_lookup:
        print(f"⚡ No cache for {domain}; performing live WHOIS lookup...")
        new_feat = extract_whois_features(domain, mode="single")
        row_df = pd.DataFrame([new_feat])
        row_df = _align_to_cache_schema(row_df, cache_path)

        if os.path.exists(cache_path):
            row_df.to_csv(cache_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(cache_path, index=False)

        return row_df.reset_index(drop=True).iloc[[0]]

    raise ValueError(f"Domain {domain} not found in cache and live_lookup=False")


# ---------------- Single Inference (FastAPI) ----------------
def extract_single_whois_features(url_or_domain: str, live_lookup: bool = True) -> Dict:
    """
    Extract WHOIS features from a single URL or domain (for FastAPI inference).
    No retries, no delays - fast single extraction.

    Args:
        url_or_domain: Single URL or domain string
        live_lookup: If True, performs live WHOIS lookup; if False, only checks cache

    Returns:
        Dictionary of WHOIS features ready for model prediction

    Example:
        >>> features = extract_single_whois_features("https://example.com")
        >>> # Returns dict with 12 WHOIS features
        >>> # Use features directly for model.predict([list(features.values())])
    """
    cache_path = WHOIS_FEATURE_FILE
    domain = extract_domain_from_url(url_or_domain) or url_or_domain

    # Try cache first
    if os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path, low_memory=False)
            hit = df_cache[df_cache["domain"].astype(str) == str(domain)]
            if not hit.empty:
                print(f"✅ Found cached WHOIS record for {domain}")
                row_dict = hit.iloc[0].to_dict()
                # Remove 'domain' key to return only feature values
                row_dict.pop('domain', None)
                return row_dict
        except Exception:
            pass

    if live_lookup:
        print(f"⚡ No cache for {domain}; performing live WHOIS lookup (single mode)...")
        # Use mode="single" for no retries, no delays
        new_feat = extract_whois_features(domain, mode="single")

        # Cache the result
        row_df = pd.DataFrame([new_feat])
        row_df = _align_to_cache_schema(row_df, cache_path)

        if os.path.exists(cache_path):
            row_df.to_csv(cache_path, mode="a", header=False, index=False)
        else:
            row_df.to_csv(cache_path, index=False)

        # Return as dict (without 'domain' key)
        result = new_feat.copy()
        result.pop('domain', None)
        return result

    raise ValueError(f"Domain {domain} not found in cache and live_lookup=False")


# ---------------- Entry ----------------
if __name__ == "__main__":
    # Example batch run
    df = build_whois_features(DOMAIN_LIST, live_lookup=True)
    print(df.head())

    # Example single inference
    # row = get_whois_features("https://104.0.237.199:9802", live_lookup=True)
    # print(row.to_dict(orient="records")[0])