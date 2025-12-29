# ===============================================================
# src/config.py
# Centralized Configuration Management
# ===============================================================
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application configuration using environment variables.
    Supports .env file loading via pydantic-settings.
    """

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="Number of worker processes")
    api_reload: bool = Field(default=False, description="Enable auto-reload")

    # CORS Settings
    cors_origins: str = Field(
        default="*", description="Allowed CORS origins (comma-separated)"
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    # Rate Limiting
    rate_limit_requests: int = Field(default=100, description="Max requests per window")
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Model Configuration
    model_dir: Path = Field(default=Path("models"), description="Model directory")
    model_version: str = Field(default="v1.0.0", description="Model version")
    url_model_name: str = Field(default="url_catboost", description="URL model name")
    whois_model_name: str = Field(
        default="whois_catboost", description="WHOIS model name"
    )
    dns_model_name: str = Field(default="dns_catboost", description="DNS model name")

    @property
    def url_model_path(self) -> Path:
        """Full path to URL model file."""
        return self.model_dir / f"{self.url_model_name}.pkl"

    @property
    def whois_model_path(self) -> Path:
        """Full path to WHOIS model file."""
        return self.model_dir / f"{self.whois_model_name}.pkl"

    @property
    def dns_model_path(self) -> Path:
        """Full path to DNS model file."""
        return self.model_dir / f"{self.dns_model_name}.pkl"

    # Ensemble Weights
    ensemble_url_weight: float = Field(
        default=0.6, description="URL model weight in ensemble"
    )
    ensemble_whois_weight: float = Field(
        default=0.4, description="WHOIS model weight in ensemble"
    )
    ensemble_dns_weight: float = Field(
        default=0.0, description="DNS model weight in ensemble"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log format: json or text")
    log_file: Path = Field(default=Path("logs/app.log"), description="Log file path")

    # Feature Extraction Timeouts
    whois_timeout: int = Field(default=10, description="WHOIS lookup timeout (seconds)")
    dns_timeout: int = Field(default=5, description="DNS lookup timeout (seconds)")

    # Data Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    processed_data_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )

    # Cache Configuration
    enable_model_cache: bool = Field(default=True, description="Enable model caching")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Prometheus metrics port")

    # Security
    api_key_enabled: bool = Field(
        default=False, description="Enable API key authentication"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# ===============================================================
# Helper functions
# ===============================================================
def get_settings() -> Settings:
    """Get global settings instance."""
    return settings


def is_production() -> bool:
    """Check if running in production mode."""
    return not settings.api_reload and settings.log_level in ["WARNING", "ERROR"]


def is_development() -> bool:
    """Check if running in development mode."""
    return settings.api_reload or settings.log_level == "DEBUG"
