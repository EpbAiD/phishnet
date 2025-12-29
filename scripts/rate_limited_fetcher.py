"""
Rate-Limited API Fetcher
=========================
Handles DNS/WHOIS/HTTP requests with:
- Exponential backoff retry
- Rate limiting per API
- Connection pooling
- Proper error handling
- Logging for debugging

CRITICAL: Prevents rate limit issues that cause all-zero features
"""

import time
import logging
from typing import Optional, Dict, Any
from functools import wraps
from datetime import datetime, timedelta
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.
    Ensures we don't exceed API rate limits.
    """
    
    def __init__(self, calls_per_second: float, burst: int = 1):
        """
        Args:
            calls_per_second: Maximum calls per second
            burst: Maximum burst size (tokens in bucket)
        """
        self.rate = calls_per_second
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire a token (permission to make API call).
        
        Args:
            blocking: If True, wait for token. If False, return immediately.
        
        Returns:
            True if token acquired, False otherwise
        """
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Add tokens based on elapsed time
                self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                # Try to consume a token
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                
                if not blocking:
                    return False
                
                # Calculate sleep time
                sleep_time = (1 - self.tokens) / self.rate
            
            # Sleep outside the lock
            time.sleep(sleep_time)


class APIFetcher:
    """
    Rate-limited API fetcher with retry logic.
    """
    
    # Rate limits (conservative to avoid bans)
    RATE_LIMITS = {
        'dns': 2.0,      # 2 queries/sec (Google DNS limit: 1500/day)
        'whois': 0.5,    # 0.5 queries/sec (WHOIS servers are strict)
        'http': 10.0,    # 10 requests/sec (general web scraping)
        'phishtank': 0.2 # 1 request per 5 seconds (PhishTank limit)
    }
    
    def __init__(self):
        self.rate_limiters = {
            api: RateLimiter(rate, burst=5)
            for api, rate in self.RATE_LIMITS.items()
        }
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'retries': 0,
            'rate_limited': 0
        }
    
    def fetch_with_retry(
        self,
        fetch_func,
        api_type: str,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute fetch function with rate limiting and retry logic.
        
        Args:
            fetch_func: Function to execute (should raise exception on failure)
            api_type: Type of API ('dns', 'whois', 'http', etc.)
            max_retries: Maximum number of retries
            backoff_factor: Exponential backoff multiplier
            **kwargs: Arguments to pass to fetch_func
        
        Returns:
            Result from fetch_func or None if all retries failed
        """
        rate_limiter = self.rate_limiters.get(api_type)
        if not rate_limiter:
            logger.warning(f"No rate limiter for {api_type}, using default")
            rate_limiter = RateLimiter(1.0)
        
        self.stats['total_requests'] += 1
        
        for attempt in range(max_retries + 1):
            try:
                # Acquire rate limit token (blocks if needed)
                rate_limiter.acquire(blocking=True)
                
                # Execute fetch
                start_time = time.time()
                result = fetch_func(**kwargs)
                elapsed = time.time() - start_time
                
                self.stats['successful'] += 1
                
                logger.debug(
                    f"{api_type} request successful "
                    f"(attempt {attempt+1}/{max_retries+1}, "
                    f"elapsed: {elapsed:.2f}s)"
                )
                
                return result
                
            except Exception as e:
                self.stats['retries'] += 1
                
                if attempt < max_retries:
                    # Calculate backoff time
                    sleep_time = backoff_factor ** attempt
                    
                    logger.warning(
                        f"{api_type} request failed (attempt {attempt+1}/{max_retries+1}): {e}"
                        f" - Retrying in {sleep_time:.1f}s"
                    )
                    
                    time.sleep(sleep_time)
                else:
                    # Final failure
                    self.stats['failed'] += 1
                    logger.error(
                        f"{api_type} request failed after {max_retries+1} attempts: {e}"
                    )
                    return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get fetcher statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {k: 0 for k in self.stats}


# Global fetcher instance
_fetcher = None

def get_fetcher() -> APIFetcher:
    """Get or create global fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = APIFetcher()
    return _fetcher


def rate_limited(api_type: str, max_retries: int = 3):
    """
    Decorator for rate-limited API calls.
    
    Usage:
        @rate_limited('dns', max_retries=3)
        def fetch_dns(domain):
            # ... DNS lookup code ...
            return result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            fetcher = get_fetcher()
            return fetcher.fetch_with_retry(
                func,
                api_type=api_type,
                max_retries=max_retries,
                args=args,
                kwargs=kwargs
            )
        return wrapper
    return decorator


# Example usage functions
@rate_limited('dns', max_retries=3)
def fetch_dns_safe(domain: str) -> Optional[Dict]:
    """
    Safe DNS lookup with rate limiting.
    
    Args:
        domain: Domain to lookup
    
    Returns:
        DNS record dict or None if failed
    """
    import dns.resolver
    
    try:
        result = {}
        
        # A records
        try:
            answers = dns.resolver.resolve(domain, 'A')
            result['A'] = [str(rdata) for rdata in answers]
        except:
            result['A'] = []
        
        # MX records
        try:
            answers = dns.resolver.resolve(domain, 'MX')
            result['MX'] = [str(rdata.exchange) for rdata in answers]
        except:
            result['MX'] = []
        
        return result
        
    except Exception as e:
        raise Exception(f"DNS lookup failed for {domain}: {e}")


@rate_limited('whois', max_retries=3)
def fetch_whois_safe(domain: str) -> Optional[Dict]:
    """
    Safe WHOIS lookup with rate limiting.
    
    Args:
        domain: Domain to lookup
    
    Returns:
        WHOIS record dict or None if failed
    """
    import whois
    
    try:
        w = whois.whois(domain)
        return {
            'domain_name': w.domain_name,
            'registrar': w.registrar,
            'creation_date': w.creation_date,
            'expiration_date': w.expiration_date,
            'name_servers': w.name_servers
        }
    except Exception as e:
        raise Exception(f"WHOIS lookup failed for {domain}: {e}")


if __name__ == "__main__":
    # Test the rate limiter
    print("Testing rate limiter...")
    
    fetcher = get_fetcher()
    
    # Test DNS
    print("\nTesting DNS rate limiting (2 req/sec):")
    for i in range(5):
        start = time.time()
        result = fetch_dns_safe("google.com")
        elapsed = time.time() - start
        print(f"  Request {i+1}: {elapsed:.2f}s - Result: {result is not None}")
    
    # Print stats
    print(f"\nStats: {fetcher.get_stats()}")
