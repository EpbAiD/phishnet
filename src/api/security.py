# ===============================================================
# src/api/security.py
# API Security: Rate Limiting, Input Validation, Authentication
# ===============================================================
from fastapi import HTTPException, Security, status, Request
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from urllib.parse import urlparse
import re
import os
from typing import Optional
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ===============================================================
# Rate Limiting
# ===============================================================
limiter = Limiter(key_func=get_remote_address)


def get_rate_limit_string() -> str:
    """
    Get rate limit string from settings.
    Format: "{requests}/{window}second" or "{requests}/{window}minute"

    Example: "100/60second" or "100/1minute"
    """
    requests = settings.rate_limit_requests
    window = settings.rate_limit_window

    if window >= 60:
        return f"{requests}/{window//60}minute"
    return f"{requests}/{window}second"


# ===============================================================
# Input Validation
# ===============================================================
class URLValidator:
    """
    URL input validation and sanitization.
    Prevents common injection attacks and malformed URLs.
    """

    # Maximum URL length (prevent DoS via extremely long URLs)
    MAX_URL_LENGTH = 2048

    # Minimum URL length
    MIN_URL_LENGTH = 7  # http://a

    # Allowed URL schemes
    ALLOWED_SCHEMES = {"http", "https", "ftp"}

    # Blocked patterns (potential injection attempts)
    BLOCKED_PATTERNS = [
        r"<script",
        r"javascript:",
        r"data:",
        r"vbscript:",
        r"file://",
    ]

    @classmethod
    def validate(cls, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate URL input.

        Args:
            url: URL string to validate

        Returns:
            Tuple of (is_valid, error_message)

        Example:
            >>> is_valid, error = URLValidator.validate("http://example.com")
            >>> if not is_valid:
            >>>     raise HTTPException(status_code=400, detail=error)
        """
        # Check type
        if not isinstance(url, str):
            return False, "URL must be a string"

        # Check length
        if len(url) > cls.MAX_URL_LENGTH:
            logger.warning(
                "URL validation failed: too long",
                extra={"url_length": len(url), "max_length": cls.MAX_URL_LENGTH},
            )
            return (
                False,
                f"URL exceeds maximum length of {cls.MAX_URL_LENGTH} characters",
            )

        if len(url) < cls.MIN_URL_LENGTH:
            return False, f"URL too short (minimum {cls.MIN_URL_LENGTH} characters)"

        # Check for blocked patterns (injection attempts)
        url_lower = url.lower()
        for pattern in cls.BLOCKED_PATTERNS:
            if re.search(pattern, url_lower, re.IGNORECASE):
                logger.warning(
                    "URL validation failed: blocked pattern detected",
                    extra={"url": url, "pattern": pattern},
                )
                return False, f"URL contains blocked pattern: {pattern}"

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            logger.error("URL parsing failed", extra={"url": url, "error": str(e)})
            return False, f"Invalid URL format: {str(e)}"

        # Check scheme (if present)
        if parsed.scheme and parsed.scheme not in cls.ALLOWED_SCHEMES:
            return False, f"URL scheme '{parsed.scheme}' not allowed. Use http/https."

        # Must have either domain or path
        if not parsed.netloc and not parsed.path:
            return False, "URL must contain a domain or path"

        logger.info("URL validation passed", extra={"url": url})
        return True, None

    @classmethod
    def sanitize(cls, url: str) -> str:
        """
        Sanitize URL input (strip whitespace, normalize).

        Args:
            url: Raw URL input

        Returns:
            Sanitized URL string
        """
        # Strip whitespace
        url = url.strip()

        # Remove null bytes
        url = url.replace("\x00", "")

        # Normalize whitespace
        url = " ".join(url.split())

        return url


# ===============================================================
# API Key Authentication (Optional)
# ===============================================================
api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Args:
        api_key: API key from request header

    Returns:
        API key if valid

    Raises:
        HTTPException: If API key is missing or invalid

    Example:
        @app.get("/protected")
        async def protected_route(api_key: str = Depends(verify_api_key)):
            return {"message": "Access granted"}
    """
    if not settings.api_key_enabled:
        # API key authentication disabled
        return None

    if not api_key:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # In production, check against database/environment variable
    # For now, use environment variable
    valid_api_key = os.getenv("API_KEY")

    if not valid_api_key:
        logger.error("API_KEY not configured in environment")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation not configured",
        )

    if api_key != valid_api_key:
        logger.warning("Invalid API key attempt", extra={"api_key_prefix": api_key[:8]})
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key"
        )

    logger.info("API key validated successfully")
    return api_key


# ===============================================================
# Request Validation Helpers
# ===============================================================
def validate_url_request(url: str) -> str:
    """
    Validate and sanitize URL from API request.

    Args:
        url: Raw URL from request

    Returns:
        Sanitized URL

    Raises:
        HTTPException: If URL is invalid

    Example:
        @app.post("/predict")
        async def predict(url: str):
            url = validate_url_request(url)
            # Process validated URL...
    """
    # Sanitize input
    url = URLValidator.sanitize(url)

    # Validate
    is_valid, error_message = URLValidator.validate(url)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=error_message
        )

    return url


# ===============================================================
# Rate Limit Error Handler
# ===============================================================
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors.

    Returns:
        HTTPException with 429 status code
    """
    logger.warning(
        "Rate limit exceeded",
        extra={"client_ip": get_remote_address(request), "path": request.url.path},
    )

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Rate limit exceeded: {exc.detail}",
    )
