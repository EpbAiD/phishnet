# ===============================================================
# tests/test_config.py
# Unit tests for configuration management
# ===============================================================
import pytest
from pathlib import Path
from src.config import Settings


class TestSettings:
    """Test suite for Settings configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        settings = Settings()

        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.model_version == "v1.0.0"
        assert settings.log_level == "INFO"

    def test_model_paths(self):
        """Test model path properties."""
        settings = Settings()

        assert isinstance(settings.url_model_path, Path)
        assert str(settings.url_model_path).endswith("url_catboost.pkl")
        assert str(settings.whois_model_path).endswith("whois_catboost.pkl")

    def test_cors_origins_list_wildcard(self):
        """Test CORS origins parsing with wildcard."""
        settings = Settings(cors_origins="*")
        assert settings.cors_origins_list == ["*"]

    def test_cors_origins_list_multiple(self):
        """Test CORS origins parsing with multiple origins."""
        settings = Settings(cors_origins="http://localhost:3000,http://example.com")
        assert len(settings.cors_origins_list) == 2
        assert "http://localhost:3000" in settings.cors_origins_list
        assert "http://example.com" in settings.cors_origins_list

    def test_ensemble_weights(self):
        """Test ensemble weight configuration."""
        settings = Settings()

        assert settings.ensemble_url_weight == 0.6
        assert settings.ensemble_whois_weight == 0.4
        assert settings.ensemble_dns_weight == 0.0

        # Weights should sum to 1.0 (excluding DNS)
        total = settings.ensemble_url_weight + settings.ensemble_whois_weight
        assert abs(total - 1.0) < 0.01

    def test_custom_values(self):
        """Test setting custom configuration values."""
        settings = Settings(
            api_port=9000,
            log_level="DEBUG",
            model_version="v2.0.0"
        )

        assert settings.api_port == 9000
        assert settings.log_level == "DEBUG"
        assert settings.model_version == "v2.0.0"
