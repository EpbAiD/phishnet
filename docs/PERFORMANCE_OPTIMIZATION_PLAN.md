# ⚡ Performance Optimization Plan - Parallel Feature Extraction

## Problem Analysis

### Current Bottlenecks:

**From benchmark results:**
- URL extraction: **1.37ms** ⚡ (already fast)
- WHOIS lookup: **64ms** 🐌 (47x slower than URL)
- DNS lookup: **6084ms** 🐢 (4,443x slower than URL!)

**Total sequential time**: 1.37 + 64 + 6084 = **6,149ms**

### Root Causes:

1. **Serial Execution**: Features extracted one-by-one
   ```python
   # Current (SLOW):
   url_features = extract_url(url)      # 1.37ms
   whois_features = extract_whois(url)  # 64ms  (wait for URL)
   dns_features = extract_dns(url)      # 6084ms (wait for WHOIS)
   # Total: 6,149ms
   ```

2. **Live API Calls**: DNS/WHOIS hit external servers
   - No caching
   - Network latency dominates
   - Rate limiting causes delays

3. **No Parallelization**: All features extracted sequentially

## Solution: Parallel + Caching Architecture

### Strategy 1: Parallel Feature Extraction

Extract all features **simultaneously** using multiprocessing:

```python
# Optimized (FAST):
with ProcessPoolExecutor(max_workers=3) as executor:
    url_future = executor.submit(extract_url, url)       # Start immediately
    whois_future = executor.submit(extract_whois, url)   # Start in parallel
    dns_future = executor.submit(extract_dns, url)       # Start in parallel

    url_features = url_future.result()      # 1.37ms
    whois_features = whois_future.result()  # 64ms  (parallel)
    dns_features = dns_future.result()      # 6084ms (parallel)

# Total: max(1.37, 64, 6084) = 6084ms (same as slowest)
# Speedup: ~1.01x (not much better)
```

**Problem**: Still bottlenecked by DNS (6084ms)!

### Strategy 2: Aggressive Caching

**Redis Cache Layer:**
```
┌─────────────────────────────────────────────────────┐
│  Prediction Request: https://example.com            │
└─────────────────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌────────┐  ┌─────────┐  ┌─────────┐
    │  URL   │  │ WHOIS   │  │  DNS    │
    │ Extract│  │  Cache  │  │  Cache  │
    └────────┘  └─────────┘  └─────────┘
         │            │            │
         │       ┌────┴────┐  ┌────┴────┐
         │       │  Hit?   │  │  Hit?   │
         │       └────┬────┘  └────┬────┘
         │            │            │
         │       Yes  │  No   Yes  │  No
         │            ▼            ▼
         │       ┌─────────┐  ┌─────────┐
         │       │ Return  │  │ Fetch   │
         │       │ 2ms     │  │ 64ms    │
         │       └─────────┘  └─────────┘
         │            │            │
         └────────────┴────────────┘
                      │
            ┌─────────▼─────────┐
            │  Ensemble Model   │
            │  Prediction       │
            └───────────────────┘
```

**Cache Hit Rates (Expected):**
- WHOIS: **95%** (domains rarely change registration)
- DNS: **90%** (IP addresses stable for most domains)

**With Caching:**
- WHOIS: 64ms → **2ms** (32x faster)
- DNS: 6084ms → **50ms** (121x faster)
- **Total: 1.37 + 2 + 50 = 53ms** (116x improvement!)

### Strategy 3: Pre-computation for Training Data

During VM collection, pre-fetch ALL features:
```
VM Collector:
1. Fetch new URLs from PhishTank
2. Extract URL features (instant)
3. Extract WHOIS features (64ms, cached for 24h)
4. Extract DNS features (6084ms, cached for 24h)
5. Store in database

API Prediction:
1. Check cache (Redis)
2. If hit → 2ms response (95% of requests)
3. If miss → 64ms WHOIS + 6084ms DNS → update cache
```

### Strategy 4: Async + Parallel + Timeout

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

async def predict_with_timeout(url: str, timeout: int = 100):
    """
    Extract features in parallel with timeout.
    If DNS/WHOIS timeout, fallback to URL-only.
    """
    loop = asyncio.get_event_loop()

    # Start all extractions in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        url_future = loop.run_in_executor(executor, extract_url, url)
        whois_future = loop.run_in_executor(executor, extract_whois_cached, url)
        dns_future = loop.run_in_executor(executor, extract_dns_cached, url)

        # Wait for all with timeout
        try:
            url_feats, whois_feats, dns_feats = await asyncio.gather(
                url_future,
                asyncio.wait_for(whois_future, timeout=0.1),  # 100ms max
                asyncio.wait_for(dns_future, timeout=0.1),    # 100ms max
                return_exceptions=True
            )
        except asyncio.TimeoutError:
            # Fallback to URL-only if WHOIS/DNS timeout
            url_feats = await url_future
            whois_feats = None
            dns_feats = None

        # Make prediction with available features
        return ensemble_predict(url_feats, whois_feats, dns_feats)
```

**Result:**
- URL: 1.37ms (always completes)
- WHOIS: 2ms (if cached) or timeout at 100ms
- DNS: 50ms (if cached) or timeout at 100ms
- **Max latency: 100ms** (enforced timeout)
- **Typical latency: 50ms** (95% cache hit)

## Implementation Plan

### Phase 1: Add Redis Caching (Week 1)

**File**: `src/api/cache.py` (NEW)

```python
import redis
import json
import hashlib
from typing import Optional

class FeatureCache:
    def __init__(self, host='localhost', port=6379, ttl=86400):
        self.redis = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl  # 24 hours default

    def _key(self, url: str, feature_type: str) -> str:
        """Generate cache key"""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return f"phishnet:{feature_type}:{url_hash}"

    def get_whois(self, url: str) -> Optional[dict]:
        """Get cached WHOIS features"""
        key = self._key(url, 'whois')
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set_whois(self, url: str, features: dict):
        """Cache WHOIS features"""
        key = self._key(url, 'whois')
        self.redis.setex(key, self.ttl, json.dumps(features))

    def get_dns(self, url: str) -> Optional[dict]:
        """Get cached DNS features"""
        key = self._key(url, 'dns')
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None

    def set_dns(self, url: str, features: dict):
        """Cache DNS features"""
        key = self._key(url, 'dns')
        self.redis.setex(key, self.ttl, json.dumps(features))

    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "total_keys": self.redis.dbsize(),
            "memory_usage": self.redis.info('memory')['used_memory_human']
        }
```

### Phase 2: Parallel Feature Extraction (Week 1)

**File**: `src/api/parallel_predict.py` (NEW)

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional

from src.features.url_features import extract_single_url_features
from src.features.whois import extract_single_whois_features
from src.features.dns_ipwhois import extract_single_domain_features
from src.api.cache import FeatureCache

cache = FeatureCache()

async def extract_all_features_parallel(url: str, timeout: float = 0.1):
    """
    Extract all features in parallel with caching and timeout.

    Args:
        url: URL to analyze
        timeout: Max time per feature (seconds)

    Returns:
        (url_features, whois_features, dns_features)
    """
    loop = asyncio.get_event_loop()

    async def extract_url():
        """URL extraction (fast, no cache needed)"""
        return await loop.run_in_executor(None, extract_single_url_features, url)

    async def extract_whois_cached():
        """WHOIS extraction with cache"""
        # Check cache first
        cached = cache.get_whois(url)
        if cached:
            return cached

        # Cache miss - fetch and store
        try:
            features = await asyncio.wait_for(
                loop.run_in_executor(None, extract_single_whois_features, url),
                timeout=timeout
            )
            cache.set_whois(url, features)
            return features
        except asyncio.TimeoutError:
            return None

    async def extract_dns_cached():
        """DNS extraction with cache"""
        # Check cache first
        cached = cache.get_dns(url)
        if cached:
            return cached

        # Cache miss - fetch and store
        try:
            features = await asyncio.wait_for(
                loop.run_in_executor(None, extract_single_domain_features, url),
                timeout=timeout
            )
            cache.set_dns(url, features)
            return features
        except asyncio.TimeoutError:
            return None

    # Execute all in parallel
    url_feats, whois_feats, dns_feats = await asyncio.gather(
        extract_url(),
        extract_whois_cached(),
        extract_dns_cached(),
        return_exceptions=True
    )

    return url_feats, whois_feats, dns_feats
```

### Phase 3: Optimized Ensemble Prediction (Week 1)

**File**: `src/api/fast_predict.py` (NEW)

```python
import asyncio
import numpy as np
from typing import Tuple

from src.api.parallel_predict import extract_all_features_parallel
from src.api.model_loader import load_url_model, load_whois_model, load_dns_model

async def predict_fast(url: str, timeout: float = 0.1) -> Tuple[float, float, str, dict]:
    """
    Fast ensemble prediction with parallel feature extraction.

    Args:
        url: URL to analyze
        timeout: Feature extraction timeout (default: 100ms)

    Returns:
        (phish_prob, legit_prob, verdict, details)
    """
    # Extract features in parallel with timeout
    url_feats, whois_feats, dns_feats = await extract_all_features_parallel(url, timeout)

    # Load models
    url_model = load_url_model()
    whois_model = load_whois_model()
    dns_model = load_dns_model()

    # Weights (optimized)
    weights = {
        'url': 0.50,
        'whois': 0.30,
        'dns': 0.20
    }

    weighted_phish_prob = 0.0
    total_weight = 0.0

    # URL prediction (always available)
    if url_feats:
        url_prob = url_model.predict_proba([url_feats])[0][1]
        weighted_phish_prob += url_prob * weights['url']
        total_weight += weights['url']

    # WHOIS prediction (if available)
    if whois_feats:
        whois_prob = whois_model.predict_proba([whois_feats])[0][1]
        weighted_phish_prob += whois_prob * weights['whois']
        total_weight += weights['whois']

    # DNS prediction (if available)
    if dns_feats:
        dns_prob = dns_model.predict_proba([dns_feats])[0][1]
        weighted_phish_prob += dns_prob * weights['dns']
        total_weight += weights['dns']

    # Normalize
    phish_prob = weighted_phish_prob / total_weight if total_weight > 0 else 0.5
    legit_prob = 1 - phish_prob

    verdict = "phishing" if phish_prob > 0.5 else "legitimate"

    details = {
        "url_features": url_feats is not None,
        "whois_features": whois_feats is not None,
        "dns_features": dns_feats is not None,
        "weights_used": {k: v for k, v in weights.items() if eval(f"{k}_feats")}
    }

    return phish_prob, legit_prob, verdict, details
```

## Expected Performance

### Before Optimization:
```
Sequential execution:
- URL: 1.37ms
- WHOIS: 64ms (wait for URL)
- DNS: 6084ms (wait for WHOIS)
Total: 6,149ms
```

### After Optimization:
```
Parallel + Cached:
- URL: 1.37ms   }
- WHOIS: 2ms    } Execute in parallel
- DNS: 50ms     }
Total: max(1.37, 2, 50) = 50ms

Cache miss (5% of requests):
- URL: 1.37ms   }
- WHOIS: 64ms   } Execute in parallel
- DNS: timeout at 100ms
Total: 100ms (enforced timeout)

Average: (50ms × 0.95) + (100ms × 0.05) = 52.5ms
```

**Speedup: 6149ms → 52.5ms = 117x faster!**

## Technology Stack

1. **Redis**: In-memory cache (sub-millisecond access)
2. **AsyncIO**: Non-blocking I/O for parallel execution
3. **ThreadPoolExecutor**: Parallel feature extraction
4. **Timeout Protection**: Fallback to URL-only if slow

## Rollout Strategy

### Week 1: Infrastructure Setup
- Install Redis: `brew install redis` (Mac) or `apt-get install redis` (Linux)
- Implement caching layer
- Test cache hit rates

### Week 2: Parallel Implementation
- Implement async feature extraction
- Add timeout protection
- Integration testing

### Week 3: A/B Testing
- 10% traffic → Optimized pipeline
- 90% traffic → Current pipeline
- Monitor latency + accuracy

### Week 4: Full Deployment
- If metrics good → 100% traffic to optimized
- Rollback capability maintained

## Monitoring Metrics

**Latency**:
- P50, P95, P99 latency
- Cache hit rate (target: >90%)
- Timeout rate (target: <5%)

**Accuracy**:
- Overall accuracy (target: >99%)
- FPR (target: <1%)
- Feature availability rate

**Infrastructure**:
- Redis memory usage
- Cache eviction rate
- Connection pool utilization
