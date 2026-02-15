# ===============================================================
# src/utils/logger.py
# Structured Logging Configuration
# ===============================================================
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.
    Adds timestamp, service name, and correlation ID.
    """

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add timestamp in ISO format
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"

        # Add service name
        log_record["service"] = "phishing-detection-api"

        # Add log level
        log_record["level"] = record.levelname

        # Add module and function info
        log_record["module"] = record.module
        log_record["function"] = record.funcName

        # Add line number for debugging
        log_record["line"] = record.lineno


def setup_logger(
    name: str, log_level: str = "INFO", log_format: str = "json", log_file: Path = None
) -> logging.Logger:
    """
    Set up structured logger with JSON formatting.

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("json" or "text")
        log_file: Optional file path for logging

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, log_level="INFO", log_format="json")
        >>> logger.info("API request received", extra={"url": "example.com", "latency_ms": 45.2})
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if log_format == "json":
        # JSON formatter for production
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(service)s %(module)s %(function)s %(message)s"
        )
    else:
        # Text formatter for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance with default configuration from settings.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    from src.config import settings

    return setup_logger(
        name=name,
        log_level=settings.log_level,
        log_format=settings.log_format,
        log_file=settings.log_file if settings.log_file else None,
    )


# ===============================================================
# Request ID Middleware for Correlation
# ===============================================================
import uuid
from contextvars import ContextVar
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Context variable for request ID
request_id_var: ContextVar[str] = ContextVar("request_id", default=None)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation ID to all requests.
    Useful for tracing requests across logs.
    """

    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request_id_var.set(request_id)

        # Add to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


def get_request_id() -> str:
    """Get current request ID from context."""
    return request_id_var.get() or "no-request-id"
