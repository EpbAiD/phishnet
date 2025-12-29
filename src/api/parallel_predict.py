"""
Parallel Feature Extraction with Caching
=========================================
Extracts URL, WHOIS, and DNS features in parallel with timeout protection.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Tuple, Optional, Dict

from src.features.url_features import extract_single_url_features
from src.features.whois import extract_single_whois_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.api.cache import get_cache

logger = logging.getLogger(__name__)

# Global thread pool for async execution
_executor = None


def get_executor():
    """Get global thread pool executor"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="feature_")
    return _executor


async def extract_all_features_parallel(
    url: str,
    timeout_whois: float = 0.1,  # 100ms
    timeout_dns: float = 0.1,  # 100ms
    use_cache: bool = True,
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Extract all features in parallel with caching and timeout.

    Execution strategy:
    1. URL features: Always extracted (fast, <2ms)
    2. WHOIS features: Check cache → fetch if miss → timeout at 100ms
    3. DNS features: Check cache → fetch if miss → timeout at 100ms

    All three execute in parallel (not sequential).

    Args:
        url: URL to analyze
        timeout_whois: WHOIS extraction timeout (seconds)
        timeout_dns: DNS extraction timeout (seconds)
        use_cache: Enable feature caching

    Returns:
        (url_features, whois_features, dns_features)
        Any feature can be None if extraction fails/times out
    """
    loop = asyncio.get_event_loop()
    executor = get_executor()
    cache = get_cache(enabled=use_cache)

    async def extract_url():
        """
        Extract URL features (always fast, no timeout needed).
        """
        try:
            return await loop.run_in_executor(
                executor, extract_single_url_features, url
            )
        except Exception as e:
            logger.error(f"URL extraction failed for {url}: {e}")
            return None

    async def extract_whois_cached():
        """
        Extract WHOIS features with cache + timeout.
        """
        # Check cache first
        if use_cache:
            cached = cache.get_whois(url)
            if cached:
                logger.debug(f"✅ WHOIS cache hit: {url}")
                return cached

        # Cache miss - fetch with timeout
        try:
            logger.debug(f"⚡ Fetching WHOIS: {url}")
            features = await asyncio.wait_for(
                loop.run_in_executor(executor, extract_single_whois_features, url),
                timeout=timeout_whois,
            )

            # Cache successful result
            if features and use_cache:
                cache.set_whois(url, features)

            return features

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ WHOIS timeout ({timeout_whois}s): {url}")
            return None
        except Exception as e:
            logger.error(f"WHOIS extraction failed for {url}: {e}")
            return None

    async def extract_dns_cached():
        """
        Extract DNS features with cache + timeout.
        """
        # Check cache first
        if use_cache:
            cached = cache.get_dns(url)
            if cached:
                logger.debug(f"✅ DNS cache hit: {url}")
                return cached

        # Cache miss - fetch with timeout
        try:
            logger.debug(f"⚡ Fetching DNS: {url}")
            features = await asyncio.wait_for(
                loop.run_in_executor(executor, extract_single_domain_features, url),
                timeout=timeout_dns,
            )

            # Cache successful result
            if features and use_cache:
                cache.set_dns(url, features)

            return features

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ DNS timeout ({timeout_dns}s): {url}")
            return None
        except Exception as e:
            logger.error(f"DNS extraction failed for {url}: {e}")
            return None

    # Execute all three in parallel
    logger.info(f"🚀 Extracting features in parallel for: {url}")
    start = asyncio.get_event_loop().time()

    results = await asyncio.gather(
        extract_url(),
        extract_whois_cached(),
        extract_dns_cached(),
        return_exceptions=True,
    )

    elapsed = (asyncio.get_event_loop().time() - start) * 1000  # ms
    logger.info(f"✅ Feature extraction complete in {elapsed:.2f}ms")

    url_feats, whois_feats, dns_feats = results

    # Log feature availability
    features_available = []
    if url_feats:
        features_available.append("URL")
    if whois_feats:
        features_available.append("WHOIS")
    if dns_feats:
        features_available.append("DNS")

    logger.info(f"   Features available: {', '.join(features_available) or 'None'}")

    return url_feats, whois_feats, dns_feats


async def extract_features_batch(
    urls: list,
    timeout_whois: float = 0.1,
    timeout_dns: float = 0.1,
    use_cache: bool = True,
    max_concurrent: int = 10,
) -> list:
    """
    Extract features for multiple URLs in parallel.

    Args:
        urls: List of URLs to process
        timeout_whois: WHOIS timeout per URL
        timeout_dns: DNS timeout per URL
        use_cache: Enable caching
        max_concurrent: Max concurrent extractions

    Returns:
        List of (url_feats, whois_feats, dns_feats) tuples
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_with_limit(url):
        async with semaphore:
            return await extract_all_features_parallel(
                url, timeout_whois, timeout_dns, use_cache
            )

    logger.info(
        f"🚀 Batch extracting features for {len(urls)} URLs (max {max_concurrent} concurrent)"
    )
    results = await asyncio.gather(*[extract_with_limit(url) for url in urls])

    return results
