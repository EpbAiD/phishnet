# ===============================================================
# feature_merger_modelready.py  🛡️ SAFE REVISION (Dual Version)
# ===============================================================
# 🆕 Changes:
#   ✔️ Safe skipping if DNS/WHOIS files do NOT exist yet
#   ✔️ Runs URL-only pipeline independently
#   ✔️ Merges DNS/WHOIS automatically when present later
#   ✔️ Saves feature order metadata for FastAPI inference
#
# 🏁 Outcome:
#   First run → ONLY URL dataset builds
#   Later, when VM finishes WHOIS/DNS → rerun → full feature merges!
#   Feature order saved → FastAPI uses exact same order
# ===============================================================

import os
import json
import pandas as pd
import tldextract
import ast

# ----------------------------- Paths -----------------------------
BASE_DIR = "data"

# Always use final balanced dataset
RAW_URL_PATH = os.path.join(BASE_DIR, "raw/final_urls_balanced.csv")

URL_FEATURE_PATH = os.path.join(BASE_DIR, "processed/url_features.csv")
DNS_FEATURE_PATH = os.path.join(BASE_DIR, "processed/dns_ipwhois_results.csv")
WHOIS_FEATURE_PATH = os.path.join(BASE_DIR, "processed/whois_results.csv")

# VM collected data paths (to be merged with processed data)
VM_DNS_PATH = os.path.join(BASE_DIR, "vm_collected/dns_results.csv")
VM_WHOIS_PATH = os.path.join(BASE_DIR, "vm_collected/whois_results.csv")
# URL features extracted locally from VM DNS/WHOIS URLs (instant, no APIs needed)

OUT_URL_MODELREADY = os.path.join(BASE_DIR, "processed/url_features_modelready.csv")
OUT_DNS_MODELREADY = os.path.join(BASE_DIR, "processed/dns_features_modelready.csv")
OUT_WHOIS_MODELREADY = os.path.join(BASE_DIR, "processed/whois_features_modelready.csv")

OUT_URL_MODELREADY_IMPUTED = os.path.join(
    BASE_DIR, "processed/url_features_modelready_imputed.csv"
)
OUT_DNS_MODELREADY_IMPUTED = os.path.join(
    BASE_DIR, "processed/dns_features_modelready_imputed.csv"
)
OUT_WHOIS_MODELREADY_IMPUTED = os.path.join(
    BASE_DIR, "processed/whois_features_modelready_imputed.csv"
)

# Feature metadata for FastAPI
FEATURE_METADATA = os.path.join(BASE_DIR, "processed/feature_metadata.json")

os.makedirs(os.path.join(BASE_DIR, "processed"), exist_ok=True)


# ----------------------------- Helpers -----------------------------
def safe_load(path, name):
    """Load a CSV if present, otherwise skip safely."""
    if os.path.exists(path):
        print(f"📄 Loading {name} → {path}")
        return pd.read_csv(path)
    else:
        print(f"⚠️ {name} NOT found → skipping ({path})")
        return None


def extract_domain(url: str):
    """Uniform domain extraction using tldextract."""
    try:
        ext = tldextract.extract(url)
        if not ext.domain or not ext.suffix:
            return None
        return f"{ext.domain}.{ext.suffix}"
    except Exception:
        return None


def flatten_listlike_columns(df: pd.DataFrame):
    """Convert list-like string columns into numeric counts."""
    for col in df.columns:
        if df[col].dtype == object:
            sample_val = str(df[col].dropna().iloc[0]) if df[col].notna().any() else ""
            if sample_val.startswith("[") and sample_val.endswith("]"):

                def parse_count(x):
                    try:
                        parsed = ast.literal_eval(x)
                        return len(parsed) if isinstance(parsed, list) else 0
                    except Exception:
                        return 0

                df[col] = df[col].apply(parse_count)
    return df


def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize string labels to binary 0/1 encoding.

    Converts:
        - "legitimate", "legit", "benign" → 0
        - "phishing", "phish", "malicious" → 1

    Args:
        df: DataFrame with 'label' or 'type' column

    Returns:
        DataFrame with normalized 'label' column (0/1)
    """
    df_norm = df.copy()

    # Determine which column has the labels
    label_col = None
    if "label" in df_norm.columns:
        label_col = "label"
    elif "type" in df_norm.columns:
        label_col = "type"
    else:
        print("⚠️  No 'label' or 'type' column found - skipping label normalization")
        return df_norm

    # Convert to lowercase string for comparison
    df_norm[label_col] = df_norm[label_col].astype(str).str.lower().str.strip()

    # Map to binary labels
    label_map = {
        "legitimate": 0,
        "legit": 0,
        "benign": 0,
        "0": 0,
        "phishing": 1,
        "phish": 1,
        "malicious": 1,
        "1": 1,
    }

    df_norm[label_col] = df_norm[label_col].map(label_map)

    # Rename to 'label' if it was 'type'
    if label_col == "type":
        df_norm = df_norm.rename(columns={"type": "label"})

    # Check for unmapped values
    if df_norm["label"].isna().any():
        unmapped_count = df_norm["label"].isna().sum()
        print(
            f"⚠️  Warning: {unmapped_count} rows with unmapped labels - dropping them"
        )
        df_norm = df_norm.dropna(subset=["label"])

    # Ensure label is integer
    df_norm["label"] = df_norm["label"].astype(int)

    print(f"✅ Labels normalized:")
    print(f"   Legitimate (0): {(df_norm['label'] == 0).sum()}")
    print(f"   Phishing (1): {(df_norm['label'] == 1).sum()}")

    return df_norm


def intelligent_impute(df: pd.DataFrame, exclude_cols=None):
    """Smart imputation preserving sentinel values."""
    exclude_cols = exclude_cols or []
    df_imp = df.copy()

    for col in df_imp.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df_imp[col]):
            df_imp[col] = df_imp[col].apply(lambda x: -999 if pd.isna(x) else x)
        else:
            df_imp[col] = df_imp[col].fillna("MISSING")

    return df_imp


def flatten_single_feature_dict(features: dict) -> dict:
    """
    Flatten list-like values in feature dict to counts (for FastAPI inference).
    This applies the SAME transformation as flatten_listlike_columns but for dicts.

    Args:
        features: Dictionary of extracted features (from URL/DNS/WHOIS extractors)

    Returns:
        Dictionary with list-like values converted to counts
    """
    flattened = {}
    for key, value in features.items():
        if isinstance(value, list):
            # Convert lists to count (same as dataset_builder for training)
            flattened[key] = len(value)
        else:
            flattened[key] = value
    return flattened


def impute_single_feature_dict(features: dict) -> dict:
    """
    Apply intelligent imputation to feature dict (for FastAPI inference).
    This applies the SAME transformation as intelligent_impute but for dicts.

    Args:
        features: Dictionary of features

    Returns:
        Dictionary with NaN/None values imputed
    """
    imputed = {}
    for key, value in features.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            # Numeric feature → use -999
            if key not in ["url", "type", "domain"]:
                imputed[key] = -999
            else:
                imputed[key] = "MISSING"
        elif isinstance(value, str):
            imputed[key] = value
        else:
            imputed[key] = value
    return imputed


def preprocess_features_for_inference(
    url_features: dict, dns_features: dict = None, whois_features: dict = None
) -> dict:
    """
    Preprocess raw feature dictionaries for FastAPI model inference.
    Applies the SAME transformations as training pipeline:
    1. Flatten list-like values to counts
    2. Impute missing values

    Args:
        url_features: Dictionary from extract_single_url_features()
        dns_features: Optional dict from extract_single_domain_features()
        whois_features: Optional dict from extract_single_whois_features()

    Returns:
        Dictionary of preprocessed features ready for model.predict()

    Example:
        >>> url_feats = extract_single_url_features("https://example.com")
        >>> dns_feats = extract_single_domain_features("https://example.com")
        >>> whois_feats = extract_single_whois_features("https://example.com")
        >>> all_features = preprocess_features_for_inference(url_feats, dns_feats, whois_feats)
        >>> feature_vector = list(all_features.values())
        >>> prediction = model.predict([feature_vector])
    """
    # Merge all features
    all_features = {**url_features}
    if dns_features:
        all_features.update(dns_features)
    if whois_features:
        all_features.update(whois_features)

    # Step 1: Flatten lists to counts
    all_features = flatten_single_feature_dict(all_features)

    # Step 2: Impute missing values
    all_features = impute_single_feature_dict(all_features)

    return all_features


def summarize_dataset(df: pd.DataFrame, label: str):
    """Print concise summary stats."""
    print(f"\n📊 ===== Summary for {label} =====")
    print(f"Total rows: {len(df):,}")
    print(f"Total columns: {len(df.columns):,}")
    feature_cols = [c for c in df.columns if c not in ["url", "type"]]
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Missing value ratio (avg): {df.isna().mean().mean():.3f}")
    print("Sample columns:", list(df.columns[:10]))
    print("🟢 Done.\n")


# ----------------------------- Core Merge -----------------------------
def _merge_and_prepare(
    feature_df: pd.DataFrame, url_df: pd.DataFrame, out_path_base: str, label: str
):
    """Merge, flatten, preserve missingness, save dual versions."""
    merged = url_df.merge(
        feature_df, on="domain", how="left", suffixes=("", "_feature")
    )

    # Clean up duplicate label columns from merge
    if "label_feature" in merged.columns:
        merged = merged.drop(columns=["label_feature"])
    if "url_feature" in merged.columns:
        merged = merged.drop(columns=["url_feature"])

    # Normalize labels to binary 0/1
    print(f"\n🔄 Normalizing labels for {label}...")
    merged = normalize_labels(merged)

    merged = flatten_listlike_columns(merged)

    if "domain" in merged.columns:
        merged = merged.drop(columns=["domain"])

    merged.to_csv(out_path_base, index=False)
    print(f"💾 Saved {label} model-ready → {out_path_base} (shape={merged.shape})")
    summarize_dataset(merged, label + " (NaN-preserved)")

    merged_imp = intelligent_impute(merged, exclude_cols=["url", "label"])
    out_path_imp = out_path_base.replace(".csv", "_imputed.csv")
    merged_imp.to_csv(out_path_imp, index=False)
    print(f"💾 Saved {label} (imputed) → {out_path_imp} (shape={merged_imp.shape})")
    summarize_dataset(merged_imp, label + " (Imputed)")

    return merged, merged_imp


# ----------------------------- Builders -----------------------------
def save_feature_metadata(url_df, dns_df=None, whois_df=None):
    """
    Save feature order metadata for FastAPI inference.
    This ensures FastAPI provides features in the exact same order as training.
    """
    # Get feature columns (exclude url, type, domain)
    url_features = [c for c in url_df.columns if c not in ["url", "type", "domain"]]
    dns_features = (
        [c for c in dns_df.columns if c not in ["url", "type", "domain"]]
        if dns_df is not None
        else []
    )
    whois_features = (
        [c for c in whois_df.columns if c not in ["url", "type", "domain"]]
        if whois_df is not None
        else []
    )

    metadata = {
        "feature_order": {
            "url": url_features,
            "dns": dns_features,
            "whois": whois_features,
            "all": url_features + dns_features + whois_features,
        },
        "feature_counts": {
            "url": len(url_features),
            "dns": len(dns_features),
            "whois": len(whois_features),
            "total": len(url_features) + len(dns_features) + len(whois_features),
        },
        "imputation_strategy": {"numeric": -999, "categorical": "MISSING"},
        "list_flattening": "Convert lists to counts (len(list))",
    }

    with open(FEATURE_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Saved feature metadata → {FEATURE_METADATA}")
    print(f"   URL features: {len(url_features)}")
    print(f"   DNS features: {len(dns_features)}")
    print(f"   WHOIS features: {len(whois_features)}")
    print(
        f"   TOTAL: {len(url_features) + len(dns_features) + len(whois_features)} features"
    )
    return metadata


def build_url_modelready():
    print("\n📂 Preparing URL-only features...")
    url_main = pd.read_csv(RAW_URL_PATH)
    url_feats = safe_load(URL_FEATURE_PATH, "URL Features")

    if url_feats is None:
        raise RuntimeError(
            "❌ URL features missing! Run extraction before model-ready step."
        )

    if "type" not in url_feats.columns and "type" in url_main.columns:
        url_feats = url_feats.merge(url_main[["url", "type"]], on="url", how="left")

    # EXTRACT URL features from VM collected DNS/WHOIS results (LOCAL extraction - instant!)
    # VM only collects DNS/WHOIS (slow, rate-limited). URL features extracted here (fast, no APIs)
    vm_dns_df = safe_load(VM_DNS_PATH, "VM DNS results")
    vm_whois_df = safe_load(VM_WHOIS_PATH, "VM WHOIS results")

    vm_urls = []
    if vm_dns_df is not None and len(vm_dns_df) > 0:
        if "url" in vm_dns_df.columns:
            vm_urls.extend(vm_dns_df[["url", "label"]].to_dict("records"))
    if vm_whois_df is not None and len(vm_whois_df) > 0:
        if "url" in vm_whois_df.columns:
            vm_urls.extend(vm_whois_df[["url", "label"]].to_dict("records"))

    if vm_urls:
        # Remove duplicates
        vm_urls_df = pd.DataFrame(vm_urls).drop_duplicates(subset=["url"])
        print(
            f"  📥 Extracting URL features for {len(vm_urls_df)} VM collected URLs (LOCAL extraction - instant)..."
        )

        # Extract URL features locally (instant, no APIs needed)
        from src.features.url_features import extract_single_url_features

        vm_url_features = []
        for _, row in vm_urls_df.iterrows():
            try:
                feats = extract_single_url_features(row["url"])
                feats["url"] = row["url"]
                feats["label"] = row["label"]
                vm_url_features.append(feats)
            except Exception as e:
                print(f"  ⚠️  Error extracting features for {row['url']}: {e}")

        if vm_url_features:
            vm_url_df = pd.DataFrame(vm_url_features)
            # Merge with original URL features (keep latest)
            url_feats = pd.concat([url_feats, vm_url_df], ignore_index=True)
            url_feats = url_feats.drop_duplicates(subset=["url"], keep="last")
            print(f"  ✅ Total URL features after merge: {len(url_feats)}")

    # Normalize labels to binary 0/1
    print("\n🔄 Normalizing labels...")
    url_feats = normalize_labels(url_feats)

    url_feats = flatten_listlike_columns(url_feats)

    url_feats.to_csv(OUT_URL_MODELREADY, index=False)
    print(f"💾 Saved URL model-ready → {OUT_URL_MODELREADY} (shape={url_feats.shape})")
    summarize_dataset(url_feats, "URL")

    url_feats_imp = intelligent_impute(url_feats, exclude_cols=["url", "label"])
    url_feats_imp.to_csv(OUT_URL_MODELREADY_IMPUTED, index=False)
    print(f"💾 Saved URL (imputed) → {OUT_URL_MODELREADY_IMPUTED}")
    return url_feats, url_feats_imp


def build_dns_modelready():
    print("\n📂 Preparing DNS/IPWHOIS model-ready...")
    dns_df = safe_load(DNS_FEATURE_PATH, "DNS/IPWHOIS features")
    if dns_df is None:
        print("⏭️ Skipping DNS model-ready (no data).")
        return None, None

    # MERGE VM collected DNS data if available
    vm_dns_df = safe_load(VM_DNS_PATH, "VM collected DNS features")
    if vm_dns_df is not None and len(vm_dns_df) > 0:
        print(f"  📥 Merging {len(vm_dns_df)} VM collected DNS rows...")
        # VM data has url/label/domain - keep only feature columns that match original
        # Drop url, label from VM data to avoid conflicts
        vm_dns_clean = vm_dns_df.drop(
            columns=["url", "label", "collected_at"], errors="ignore"
        )
        # Concatenate feature-only data
        dns_df = pd.concat([dns_df, vm_dns_clean], ignore_index=True)
        dns_df = dns_df.drop_duplicates(
            subset=["domain"], keep="last"
        )  # Keep latest version by domain
        print(f"  ✅ Total DNS rows after merge: {len(dns_df)}")

    url_df = pd.read_csv(RAW_URL_PATH)
    url_df["domain"] = url_df["url"].apply(extract_domain)
    # KEEP all URLs - treat missing domain as informative feature (missingness = likely phishing)
    url_df["domain"] = url_df["domain"].fillna("__NO_DOMAIN__")

    return _merge_and_prepare(dns_df, url_df, OUT_DNS_MODELREADY, "DNS")


def build_whois_modelready():
    print("\n📂 Preparing WHOIS model-ready...")
    whois_df = safe_load(WHOIS_FEATURE_PATH, "WHOIS features")
    if whois_df is None:
        print("⏭️ Skipping WHOIS model-ready (no data).")
        return None, None

    # MERGE VM collected WHOIS data if available
    vm_whois_df = safe_load(VM_WHOIS_PATH, "VM collected WHOIS features")
    if vm_whois_df is not None and len(vm_whois_df) > 0:
        print(f"  📥 Merging {len(vm_whois_df)} VM collected WHOIS rows...")
        # Extract domain from VM WHOIS url column first
        vm_whois_df["domain"] = vm_whois_df["url"].apply(extract_domain)
        # Drop url, label from VM data to avoid conflicts
        vm_whois_clean = vm_whois_df.drop(
            columns=["url", "label", "collected_at"], errors="ignore"
        )
        # Concatenate feature-only data
        whois_df = pd.concat([whois_df, vm_whois_clean], ignore_index=True)
        whois_df = whois_df.drop_duplicates(
            subset=["domain"], keep="last"
        )  # Keep latest version by domain
        print(f"  ✅ Total WHOIS rows after merge: {len(whois_df)}")

    url_df = pd.read_csv(RAW_URL_PATH)
    url_df["domain"] = url_df["url"].apply(extract_domain)
    # KEEP all URLs - treat missing domain as informative feature (missingness = likely phishing)
    url_df["domain"] = url_df["domain"].fillna("__NO_DOMAIN__")

    return _merge_and_prepare(whois_df, url_df, OUT_WHOIS_MODELREADY, "WHOIS")


# ----------------------------- Entry -----------------------------
if __name__ == "__main__":
    print("\n🚀 Building all model-ready datasets (safe mode)...")
    url_df, url_df_imp = build_url_modelready()
    dns_df, dns_df_imp = build_dns_modelready()
    whois_df, whois_df_imp = build_whois_modelready()

    # Save feature metadata for FastAPI
    print("\n📝 Saving feature metadata for FastAPI...")
    save_feature_metadata(
        url_df_imp if url_df_imp is not None else url_df,
        dns_df_imp if dns_df_imp is not None else dns_df,
        whois_df_imp if whois_df_imp is not None else whois_df,
    )

    print("\n🎯 Completed model-ready generation (safe mode).")
