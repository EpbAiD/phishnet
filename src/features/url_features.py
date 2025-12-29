# ===============================================================
# UPDATED URL FEATURE EXTRACTION (Corrected Behaviour)
# ===============================================================

import os
import re
import tldextract
import pandas as pd
from urllib.parse import urlparse, parse_qs
from collections import Counter
from math import log2
from typing import List, Union, Optional
from ipaddress import ip_address

# auto-create processed directory for outputs
os.makedirs("data/processed", exist_ok=True)


# ---------------------------------------------------------------
# 1️⃣  Core Feature Extractor
# ---------------------------------------------------------------
class URLFeatureExtractor:
    def __init__(self):
        self.suspicious_keywords = [
            "login",
            "secure",
            "account",
            "verify",
            "update",
            "password",
            "bank",
            "client",
            "mail",
            "outlook",
            "inbox",
            "admin",
            "billing",
            "free",
            "click",
            "bonus",
            "office",
            "paypal",
            "ebay",
        ]
        self.download_exts = [
            ".zip",
            ".rar",
            ".exe",
            ".tar",
            ".gz",
            ".iso",
            ".apk",
            ".sh",
            ".bat",
        ]
        self.script_exts = [".js", ".php", ".py", ".asp", ".jsp"]
        self.shortener_domains = [
            "bit.ly",
            "tinyurl.com",
            "is.gd",
            "t.co",
            "goo.gl",
            "ow.ly",
            "buff.ly",
        ]
        self.invalid_urls_ = []
        self.benign_subdomains = {"www", "m", "mobile", "en"}

    # -----------------------------------------
    def shannon_entropy(self, s):
        if not s:
            return 0
        prob = [n / len(s) for n in Counter(s).values()]
        return -sum(p * log2(p) for p in prob)

    # -----------------------------------------
    def ip_category(self, hostname):
        try:
            ip = ip_address(hostname)
            if ip.is_private:
                return "private"
            if ip.is_reserved:
                return "reserved"
            if ip.is_loopback:
                return "loopback"
            return "public"
        except Exception:
            return "none"

    # -----------------------------------------
    def detect_extension_type(self, path, query):
        q = query.lower()
        if any(ext in q for ext in self.download_exts):
            return "download"
        for ext in self.download_exts:
            if path.endswith(ext):
                return "download"
        for ext in self.script_exts:
            if path.endswith(ext):
                return "script"
        return "none"

    # -----------------------------------------
    def domain_quality(self, domain, suffix):
        if not domain:
            return "empty"
        if domain.isnumeric():
            return "numeric_only"
        if not suffix:
            return "no_tld"
        return "normal"

    # -----------------------------------------
    def _extract_base_features(self, url: str) -> dict:
        """Extract base features from a single URL (without engineered features)"""
        url = url.replace("[.]", ".")
        parsed = urlparse(url)
        ext = tldextract.extract(url)

        domain = ext.domain
        suffix = ext.suffix
        subdomain = ext.subdomain
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""

        features = {}

        # BASIC LENGTH / SYMBOLS
        features["url_length"] = len(url)
        features["hostname_length"] = len(hostname)
        features["path_length"] = len(path)

        subs = [
            s for s in subdomain.split(".") if s and s not in self.benign_subdomains
        ]
        features["num_subdomains"] = len(subs)
        features["num_dots"] = hostname.count(".") if hostname else 0
        features["num_special_chars"] = len(re.findall(r"[?#&%=+@/$!*]", url))
        features["num_digits"] = sum(c.isdigit() for c in url)
        features["num_uppercase_chars"] = sum(c.isupper() for c in url)

        # STRUCTURE
        features["has_at_symbol"] = "@" in url
        after_host = url.split(hostname, 1)[-1]
        features["has_double_slash_redirect"] = int("//" in after_host.strip("/"))
        features["has_dash_in_domain"] = "-" in domain
        features["is_ip_address"] = int(
            bool(re.match(r"\d{1,3}(\.\d{1,3}){3}$", hostname))
        )
        features["ip_category"] = self.ip_category(hostname)

        # ENCODING / OBFUSCATION
        features["has_encoded_chars"] = "%" in url
        features["has_non_ascii_chars"] = any(ord(c) > 127 for c in url)
        features["url_entropy"] = self.shannon_entropy(url)
        features["hostname_entropy"] = self.shannon_entropy(hostname)
        letters = sum(c.isalpha() for c in url)
        features["digit_to_letter_ratio"] = features["num_digits"] / (letters + 1e-6)

        # DOMAIN / TLD
        features["domain_quality"] = self.domain_quality(domain, suffix)
        features["tld_length"] = len(suffix)
        features["subdomain_entropy"] = (
            0 if not subs else self.shannon_entropy("".join(subs))
        )
        features["subdomain_length"] = sum(len(s) for s in subs)

        # KEYWORDS / BRAND
        URL_lower = url.lower()
        features["has_login_keyword"] = int(
            any(word in URL_lower for word in self.suspicious_keywords[:4])
        )
        features["has_suspicious_words"] = int(
            any(word in URL_lower for word in self.suspicious_keywords[4:])
        )
        features["has_brand_mismatch"] = int(
            domain not in URL_lower
            and any(word in URL_lower for word in ["paypal", "ebay", "amazon"])
        )

        # PAYLOAD
        features["file_type"] = self.detect_extension_type(path, query)
        features["is_file_download"] = int(features["file_type"] == "download")
        features["is_script_file"] = int(features["file_type"] == "script")

        # SHORTENER
        host_norm = hostname.replace("www.", "")
        features["is_shortened"] = int(host_norm in self.shortener_domains)

        # PATH + QUERY
        features["num_fragments"] = url.count("#")
        features["num_query_params"] = len(parse_qs(query))
        features["num_directories"] = path.count("/")

        # PORT
        port = parsed.port if parsed.port is not None else -1
        features["port"] = port
        risky_ports = [21, 22, 23, 25, 80, 8080, 8443, 3389, 5900, 53, 445]
        features["is_risky_port"] = int(port in risky_ports)
        features["protocol_mismatch"] = int(
            (parsed.scheme == "http" and port == 443)
            or (parsed.scheme == "https" and port == 80)
        )
        features["is_unknown_port"] = int(
            port not in [-1, 80, 443] and port not in risky_ports
        )

        return features

    # -----------------------------------------
    def _add_engineered_features(self, features: dict, url: str) -> dict:
        """Add engineered features to a feature dictionary"""
        # Engineered features that depend on base features
        features["contains_hex_encoding"] = int("%" in url)
        features["starts_with_https_but_contains_http"] = int(
            url.startswith("https") and "http://" in url[8:]
        )
        features["missing_hostname_flag"] = int(features["hostname_length"] == 0)
        return features

    # -----------------------------------------
    def extract_features(self, url: str, include_url: bool = False) -> dict:
        """
        Extract ALL features from a single URL (base + engineered).
        This method ensures consistency between training and inference.

        Args:
            url: The URL to extract features from
            include_url: Whether to include the original URL in the output

        Returns:
            Dictionary of all features ready for model inference
        """
        base_features = self._extract_base_features(url)
        all_features = self._add_engineered_features(base_features, url)

        if include_url:
            # Add URL at the beginning for consistency with batch processing
            all_features = {"url": url, **all_features}

        return all_features

    # -------------------------
    def transform_dataframe(self, df, url_column="url") -> pd.DataFrame:
        """
        Extract features from a DataFrame of URLs.
        Uses the same extract_features method as single URL processing for consistency.

        Args:
            df: DataFrame containing URLs
            url_column: Name of the column containing URLs

        Returns:
            DataFrame with original columns + all extracted features
        """
        feature_rows, valid_indices = [], []
        self.invalid_urls_ = []
        for idx, url in enumerate(df[url_column]):
            try:
                # Use extract_features which includes engineered features
                feature_rows.append(self.extract_features(url, include_url=False))
                valid_indices.append(idx)
            except Exception as e:
                self.invalid_urls_.append((url, str(e)))
        print(f"[INFO] Valid URLs: {len(valid_indices)}")
        print(f"[INFO] Invalid URLs: {len(self.invalid_urls_)}")
        feature_df = pd.DataFrame(feature_rows)
        return pd.concat(
            [df.iloc[valid_indices].reset_index(drop=True), feature_df], axis=1
        )


# ---------------------------------------------------------------
# 3️⃣ UNIFIED API FOR TRAINING & INFERENCE
# ---------------------------------------------------------------


def extract_features_from_dataset(input_csv, output_csv=None, url_column="url"):
    """
    Extract features from a CSV file (for training).
    Features are identical to those from extract_single_url_features.

    Args:
        input_csv: Path to input CSV file
        output_csv: Path to save output CSV (auto-generated if None)
        url_column: Name of column containing URLs

    Returns:
        DataFrame with all features
    """
    print(f"[INFO] Loading dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    extractor = URLFeatureExtractor()
    df_final = extractor.transform_dataframe(df, url_column=url_column)

    # if no output path is given, save automatically
    if not output_csv:
        output_csv = "data/processed/url_features.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_final.to_csv(output_csv, index=False)
    print(f"[INFO] Saved output: {output_csv}")
    return df_final


def extract_single_url_features(url: str) -> dict:
    """
    Extract features from a single URL (for FastAPI inference).
    Returns the SAME features as batch processing for consistency.

    Args:
        url: Single URL string

    Returns:
        Dictionary of features ready for model prediction

    Example:
        >>> extractor = URLFeatureExtractor()
        >>> features = extract_single_url_features("https://example.com/login")
        >>> # Use features directly for model.predict([list(features.values())])
    """
    extractor = URLFeatureExtractor()
    return extractor.extract_features(url, include_url=False)


def extract_features_for_urls(urls: Union[str, List[str]]) -> pd.DataFrame:
    """
    Extract features from one or more URLs, returning as DataFrame.
    Useful for debugging or batch prediction.

    Args:
        urls: Single URL string or list of URLs

    Returns:
        DataFrame with URL column + all features
    """
    if isinstance(urls, str):
        urls = [urls]
    extractor = URLFeatureExtractor()
    results = []
    for u in urls:
        feats = extractor.extract_features(u, include_url=False)
        feats["url"] = u  # Add URL at the front
        # Reorder to put url first
        results.append({"url": u, **feats})
    return pd.DataFrame(results)


# ----------------------------- RUNNER ---------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="URL Feature Extraction Utility")
    parser.add_argument(
        "--input", type=str, required=True, help="CSV path OR single URL"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save CSV results"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "single", "multi", "dataset"],
    )
    args = parser.parse_args()

    # SINGLE or MULTI URL
    if args.mode == "single":
        df = extract_features_for_urls(args.input)
        print(df)

    elif args.mode == "multi":
        urls = args.input.split(",")
        df = extract_features_for_urls(urls)
        print(df)

    else:  # DATASET
        extract_features_from_dataset(args.input, args.output)
