"""
Rate-Limited DNS/WHOIS Feature Fetcher
=======================================
Handles API rate limiting for DNS and WHOIS lookups with:
- Token bucket algorithm for rate limiting
- Exponential backoff retry logic
- Feature validation (checks for zeros/nulls/-999)
- Batch processing with progress tracking
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import deque
import tldextract

from src.features.dns_ipwhois import extract_single_domain_features
from src.features.whois import extract_single_whois_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        now = time.time()
        elapsed = now - self.last_update

        # Add new tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time needed to get tokens."""
        if self.tokens >= tokens:
            return 0.0
        return (tokens - self.tokens) / self.rate


class RateLimitedFetcher:
    """Fetches DNS/WHOIS features with rate limiting and retry logic."""

    def __init__(
        self,
        dns_rate: float = 2.0,  # 2 requests per second for DNS
        whois_rate: float = 1.0,  # 1 request per second for WHOIS
        max_retries: int = 3,
        initial_backoff: float = 1.0,
    ):
        """
        Args:
            dns_rate: DNS requests per second
            whois_rate: WHOIS requests per second
            max_retries: Maximum retry attempts
            initial_backoff: Initial backoff delay in seconds
        """
        self.dns_bucket = TokenBucket(rate=dns_rate, capacity=int(dns_rate * 2))
        self.whois_bucket = TokenBucket(rate=whois_rate, capacity=int(whois_rate * 2))
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

        # Stats
        self.dns_success = 0
        self.dns_failed = 0
        self.whois_success = 0
        self.whois_failed = 0

    def _validate_features(self, features: Dict[str, Any], feature_type: str) -> bool:
        """
        Validate that features are not all zeros/nulls/-999.

        Args:
            features: Feature dictionary
            feature_type: 'dns' or 'whois'

        Returns:
            True if features are valid (>30% non-default for DNS, >20% for WHOIS)
        """
        if not features:
            return False

        # Count non-default values
        total = len(features)
        non_default = 0

        for value in features.values():
            if value not in [0, -999, None, "", "MISSING", "N/A"]:
                non_default += 1

        validity_pct = non_default / total

        # Different thresholds for DNS vs WHOIS
        threshold = 0.30 if feature_type == "dns" else 0.20

        if validity_pct >= threshold:
            return True
        else:
            logger.warning(
                f"{feature_type.upper()} features only {validity_pct:.1%} valid "
                f"({non_default}/{total} non-default)"
            )
            return False

    def fetch_dns_features(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch DNS features with rate limiting and retry.

        Args:
            url: URL to extract domain from

        Returns:
            DNS features dict or None if failed
        """
        # Extract domain
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            domain = f"{ext.domain}.{ext.suffix}"
        else:
            logger.warning(f"Cannot extract domain from: {url}")
            self.dns_failed += 1
            return None

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            # Wait for token
            if not self.dns_bucket.consume():
                wait = self.dns_bucket.wait_time()
                logger.debug(f"DNS rate limit - waiting {wait:.2f}s")
                time.sleep(wait)
                self.dns_bucket.consume()

            try:
                features = extract_single_domain_features(url)

                # Validate features
                if self._validate_features(features, "dns"):
                    self.dns_success += 1
                    return features
                else:
                    logger.warning(f"DNS features invalid for {domain}")
                    if attempt < self.max_retries - 1:
                        backoff = self.initial_backoff * (2**attempt)
                        logger.info(f"Retrying DNS for {domain} after {backoff:.1f}s")
                        time.sleep(backoff)
                        continue
                    else:
                        self.dns_failed += 1
                        return None

            except Exception as e:
                logger.error(
                    f"DNS extraction error for {domain} (attempt {attempt+1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    backoff = self.initial_backoff * (2**attempt)
                    time.sleep(backoff)
                else:
                    self.dns_failed += 1
                    return None

        self.dns_failed += 1
        return None

    def fetch_whois_features(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch WHOIS features with rate limiting and retry.

        Args:
            url: URL to extract domain from

        Returns:
            WHOIS features dict or None if failed
        """
        # Extract domain
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            domain = f"{ext.domain}.{ext.suffix}"
        else:
            logger.warning(f"Cannot extract domain from: {url}")
            self.whois_failed += 1
            return None

        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            # Wait for token
            if not self.whois_bucket.consume():
                wait = self.whois_bucket.wait_time()
                logger.debug(f"WHOIS rate limit - waiting {wait:.2f}s")
                time.sleep(wait)
                self.whois_bucket.consume()

            try:
                features = extract_single_whois_features(url)

                # Validate features
                if self._validate_features(features, "whois"):
                    self.whois_success += 1
                    return features
                else:
                    logger.warning(f"WHOIS features invalid for {domain}")
                    if attempt < self.max_retries - 1:
                        backoff = self.initial_backoff * (2**attempt)
                        logger.info(f"Retrying WHOIS for {domain} after {backoff:.1f}s")
                        time.sleep(backoff)
                        continue
                    else:
                        self.whois_failed += 1
                        return None

            except Exception as e:
                logger.error(
                    f"WHOIS extraction error for {domain} (attempt {attempt+1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    backoff = self.initial_backoff * (2**attempt)
                    time.sleep(backoff)
                else:
                    self.whois_failed += 1
                    return None

        self.whois_failed += 1
        return None

    def fetch_all_features(self, url: str) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetch both DNS and WHOIS features.

        Args:
            url: URL to process

        Returns:
            Dict with 'dns' and 'whois' keys containing feature dicts or None
        """
        return {
            "dns": self.fetch_dns_features(url),
            "whois": self.fetch_whois_features(url),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get fetcher statistics."""
        dns_total = self.dns_success + self.dns_failed
        whois_total = self.whois_success + self.whois_failed

        return {
            "dns": {
                "success": self.dns_success,
                "failed": self.dns_failed,
                "total": dns_total,
                "success_rate": self.dns_success / dns_total if dns_total > 0 else 0.0,
            },
            "whois": {
                "success": self.whois_success,
                "failed": self.whois_failed,
                "total": whois_total,
                "success_rate": (
                    self.whois_success / whois_total if whois_total > 0 else 0.0
                ),
            },
        }

    def print_stats(self):
        """Print fetcher statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 60)
        print("RATE-LIMITED FETCHER STATISTICS")
        print("=" * 60)

        print(f"\nDNS Features:")
        print(f"  Success: {stats['dns']['success']}")
        print(f"  Failed:  {stats['dns']['failed']}")
        print(f"  Total:   {stats['dns']['total']}")
        print(f"  Success Rate: {stats['dns']['success_rate']:.1%}")

        print(f"\nWHOIS Features:")
        print(f"  Success: {stats['whois']['success']}")
        print(f"  Failed:  {stats['whois']['failed']}")
        print(f"  Total:   {stats['whois']['total']}")
        print(f"  Success Rate: {stats['whois']['success_rate']:.1%}")

        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test on sample URLs
    test_urls = [
        "https://google.com",
        "https://github.com",
        "https://stackoverflow.com",
        "https://stripe.io",
        "https://kubernetes.io",
    ]

    fetcher = RateLimitedFetcher(
        dns_rate=2.0, whois_rate=1.0, max_retries=3  # 2 req/s  # 1 req/s
    )

    print("Testing rate-limited fetcher on 5 sample URLs...\n")

    for i, url in enumerate(test_urls, 1):
        print(f"\n[{i}/{len(test_urls)}] Processing: {url}")
        start = time.time()

        results = fetcher.fetch_all_features(url)

        elapsed = time.time() - start
        print(f"  ⏱️  Time: {elapsed:.2f}s")
        print(f"  DNS: {'✅' if results['dns'] else '❌'}")
        print(f"  WHOIS: {'✅' if results['whois'] else '❌'}")

    fetcher.print_stats()
