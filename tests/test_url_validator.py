# ===============================================================
# tests/test_url_validator.py
# Unit tests for URL validation and security
# ===============================================================
import pytest
from src.api.security import URLValidator


class TestURLValidator:
    """Test suite for URL validation."""

    def test_valid_http_url(self):
        """Test valid HTTP URL."""
        is_valid, error = URLValidator.validate("http://example.com")
        assert is_valid is True
        assert error is None

    def test_valid_https_url(self):
        """Test valid HTTPS URL."""
        is_valid, error = URLValidator.validate("https://example.com")
        assert is_valid is True
        assert error is None

    def test_valid_url_with_path(self):
        """Test valid URL with path."""
        is_valid, error = URLValidator.validate("https://example.com/path/to/page")
        assert is_valid is True
        assert error is None

    def test_valid_url_with_query(self):
        """Test valid URL with query parameters."""
        is_valid, error = URLValidator.validate("https://example.com?key=value&foo=bar")
        assert is_valid is True
        assert error is None

    def test_url_too_long(self):
        """Test URL exceeding maximum length."""
        long_url = "http://example.com/" + "a" * 3000
        is_valid, error = URLValidator.validate(long_url)
        assert is_valid is False
        assert "exceeds maximum length" in error

    def test_url_too_short(self):
        """Test URL below minimum length."""
        is_valid, error = URLValidator.validate("http")
        assert is_valid is False
        assert "too short" in error.lower()

    def test_blocked_pattern_script(self):
        """Test URL with blocked <script pattern."""
        is_valid, error = URLValidator.validate("http://example.com/<script>alert(1)</script>")
        assert is_valid is False
        assert "blocked pattern" in error.lower()

    def test_blocked_pattern_javascript(self):
        """Test URL with javascript: scheme."""
        is_valid, error = URLValidator.validate("javascript:alert(1)")
        assert is_valid is False
        assert "blocked pattern" in error.lower()

    def test_blocked_pattern_data(self):
        """Test URL with data: scheme."""
        is_valid, error = URLValidator.validate("data:text/html,<script>alert(1)</script>")
        assert is_valid is False
        assert "blocked pattern" in error.lower()

    def test_invalid_scheme(self):
        """Test URL with invalid scheme."""
        is_valid, error = URLValidator.validate("ftp://example.com")
        # FTP is in allowed schemes, so this should pass
        # Change test to use unsupported scheme
        is_valid, error = URLValidator.validate("gopher://example.com")
        assert is_valid is False
        assert "not allowed" in error

    def test_url_without_scheme(self):
        """Test URL without scheme (should be valid, will be normalized later)."""
        is_valid, error = URLValidator.validate("example.com")
        assert is_valid is True
        assert error is None

    def test_sanitize_whitespace(self):
        """Test URL sanitization removes whitespace."""
        sanitized = URLValidator.sanitize("  https://example.com  ")
        assert sanitized == "https://example.com"

    def test_sanitize_null_bytes(self):
        """Test URL sanitization removes null bytes."""
        sanitized = URLValidator.sanitize("https://example.com\x00/path")
        assert "\x00" not in sanitized

    def test_non_string_input(self):
        """Test validation rejects non-string input."""
        is_valid, error = URLValidator.validate(12345)
        assert is_valid is False
        assert "must be a string" in error


class TestURLValidatorEdgeCases:
    """Test edge cases and security scenarios."""

    def test_empty_string(self):
        """Test empty string URL."""
        is_valid, error = URLValidator.validate("")
        assert is_valid is False

    def test_only_whitespace(self):
        """Test URL with only whitespace."""
        is_valid, error = URLValidator.validate("   ")
        assert is_valid is False

    def test_url_with_unicode(self):
        """Test URL with unicode characters."""
        is_valid, error = URLValidator.validate("https://例え.jp")
        assert is_valid is True

    def test_localhost_url(self):
        """Test localhost URL."""
        is_valid, error = URLValidator.validate("http://localhost:8000")
        assert is_valid is True

    def test_ip_address_url(self):
        """Test IP address URL."""
        is_valid, error = URLValidator.validate("http://192.168.1.1")
        assert is_valid is True

    def test_url_with_port(self):
        """Test URL with port number."""
        is_valid, error = URLValidator.validate("https://example.com:8443")
        assert is_valid is True

    def test_url_with_fragment(self):
        """Test URL with fragment."""
        is_valid, error = URLValidator.validate("https://example.com#section")
        assert is_valid is True

    def test_url_with_credentials(self):
        """Test URL with credentials (should be valid but flagged in logs)."""
        is_valid, error = URLValidator.validate("https://user:pass@example.com")
        assert is_valid is True
