"""
Feature Caching Layer
=====================
Redis-based caching for WHOIS and DNS features.
TTL: 24 hours (domains rarely change)
"""

import json
import hashlib
import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Fallback to in-memory cache if Redis not available
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - using in-memory cache (not production-ready)")


class InMemoryCache:
    """Fallback in-memory cache (for development)"""

    def __init__(self):
        self.cache = {}

    def get(self, key: str) -> Optional[str]:
        entry = self.cache.get(key)
        if entry and entry["expires_at"] > datetime.now().timestamp():
            return entry["value"]
        return None

    def setex(self, key: str, ttl: int, value: str):
        self.cache[key] = {
            "value": value,
            "expires_at": datetime.now().timestamp() + ttl,
        }

    def dbsize(self) -> int:
        # Clean expired entries
        now = datetime.now().timestamp()
        self.cache = {k: v for k, v in self.cache.items() if v["expires_at"] > now}
        return len(self.cache)

    def info(self, section: str) -> Dict:
        return {"used_memory_human": f"{len(str(self.cache))} bytes"}


class FeatureCache:
    """
    Feature caching layer with Redis backend.
    Falls back to in-memory cache if Redis unavailable.
    """

    def __init__(self, host="localhost", port=6379, ttl=86400, enabled=True):
        """
        Initialize cache.

        Args:
            host: Redis host
            port: Redis port
            ttl: Time-to-live in seconds (default: 24 hours)
            enabled: Enable/disable caching
        """
        self.enabled = enabled
        self.ttl = ttl
        self.stats = {
            "whois_hits": 0,
            "whois_misses": 0,
            "dns_hits": 0,
            "dns_misses": 0,
        }

        if not enabled:
            logger.info("Feature caching is DISABLED")
            self.redis = None
            return

        if REDIS_AVAILABLE:
            try:
                self.redis = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_connect_timeout=1,
                )
                # Test connection
                self.redis.ping()
                logger.info(f"✅ Connected to Redis at {host}:{port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e} - using in-memory cache")
                self.redis = InMemoryCache()
        else:
            logger.warning("Redis not installed - using in-memory cache")
            self.redis = InMemoryCache()

    def _key(self, url: str, feature_type: str) -> str:
        """
        Generate cache key.

        Args:
            url: URL to cache
            feature_type: 'whois' or 'dns'

        Returns:
            Cache key string
        """
        # Use MD5 hash of URL to avoid key length issues
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"phishnet:{feature_type}:{url_hash}"

    def get_whois(self, url: str) -> Optional[Dict]:
        """
        Get cached WHOIS features.

        Args:
            url: URL to lookup

        Returns:
            Features dict or None if cache miss
        """
        if not self.enabled or not self.redis:
            return None

        try:
            key = self._key(url, "whois")
            cached = self.redis.get(key)

            if cached:
                self.stats["whois_hits"] += 1
                logger.debug(f"WHOIS cache HIT: {url}")
                return json.loads(cached)
            else:
                self.stats["whois_misses"] += 1
                logger.debug(f"WHOIS cache MISS: {url}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set_whois(self, url: str, features: Dict):
        """
        Cache WHOIS features.

        Args:
            url: URL to cache
            features: Features dict
        """
        if not self.enabled or not self.redis:
            return

        try:
            key = self._key(url, "whois")
            self.redis.setex(key, self.ttl, json.dumps(features))
            logger.debug(f"WHOIS cached: {url}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def get_dns(self, url: str) -> Optional[Dict]:
        """
        Get cached DNS features.

        Args:
            url: URL to lookup

        Returns:
            Features dict or None if cache miss
        """
        if not self.enabled or not self.redis:
            return None

        try:
            key = self._key(url, "dns")
            cached = self.redis.get(key)

            if cached:
                self.stats["dns_hits"] += 1
                logger.debug(f"DNS cache HIT: {url}")
                return json.loads(cached)
            else:
                self.stats["dns_misses"] += 1
                logger.debug(f"DNS cache MISS: {url}")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set_dns(self, url: str, features: Dict):
        """
        Cache DNS features.

        Args:
            url: URL to cache
            features: Features dict
        """
        if not self.enabled or not self.redis:
            return

        try:
            key = self._key(url, "dns")
            self.redis.setex(key, self.ttl, json.dumps(features))
            logger.debug(f"DNS cached: {url}")
        except Exception as e:
            logger.error(f"Cache set error: {e}")

    def get_hit_rate(self) -> Dict:
        """
        Calculate cache hit rates.

        Returns:
            Dict with hit rates and counts
        """
        whois_total = self.stats["whois_hits"] + self.stats["whois_misses"]
        dns_total = self.stats["dns_hits"] + self.stats["dns_misses"]

        return {
            "whois": {
                "hits": self.stats["whois_hits"],
                "misses": self.stats["whois_misses"],
                "total": whois_total,
                "hit_rate": (
                    self.stats["whois_hits"] / whois_total if whois_total > 0 else 0
                ),
            },
            "dns": {
                "hits": self.stats["dns_hits"],
                "misses": self.stats["dns_misses"],
                "total": dns_total,
                "hit_rate": self.stats["dns_hits"] / dns_total if dns_total > 0 else 0,
            },
        }

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Stats dict
        """
        if not self.enabled or not self.redis:
            return {"enabled": False}

        try:
            return {
                "enabled": True,
                "total_keys": self.redis.dbsize(),
                "memory_usage": self.redis.info("memory").get(
                    "used_memory_human", "N/A"
                ),
                "hit_rates": self.get_hit_rate(),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    def clear(self):
        """Clear all cached features"""
        if not self.enabled or not self.redis:
            return

        try:
            # Only clear phishnet keys
            for key in self.redis.scan_iter("phishnet:*"):
                self.redis.delete(key)
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")


# Global cache instance
_cache = None


def get_cache(enabled=True) -> FeatureCache:
    """
    Get global cache instance (singleton).

    Args:
        enabled: Enable caching

    Returns:
        FeatureCache instance
    """
    global _cache
    if _cache is None:
        _cache = FeatureCache(enabled=enabled)
    return _cache
