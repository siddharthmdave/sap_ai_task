# app/core/logging.py
"""
Structured Logging Configuration Module.

Configures structlog for JSON-formatted, structured logging suitable for
log aggregation systems (ELK, Datadog, CloudWatch, etc.).

Features:
    - JSON output in production, human-readable in development
    - Automatic request-id injection via context variables
    - Log level filtering from settings
    - Correlation ID support for distributed tracing

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("order_processed", order_id="1001", amount_usd=200.0)
...
"""

from __future__ import annotations

import logging
import logging.config
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

# — Context Variables ——————————————————————————————————————

# Stores the current request correlation ID for log injection
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_tenant_id_var: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)

def set_request_id(request_id: str) -> None:
    """Set the current request ID in context (called by middleware)."""
    _request_id_var.set(request_id)

def set_tenant_id(tenant_id: str) -> None:
    """Set the current tenant ID in context (called by middleware)."""
    _tenant_id_var.set(tenant_id)

def get_request_id() -> Optional[str]:
    """Retrieve the current request ID from context."""
    return _request_id_var.get()

# — Custom Processors ——————————————————————————————————————

def inject_context_vars(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Structlog processor that injects context variables into every log record.

    Adds request_id and tenant_id from ContextVar storage so that all log
    lines within a request share the same correlation identifiers.
    """
    request_id = _request_id_var.get()
    tenant_id = _tenant_id_var.get()

    if request_id:
        event_dict["request_id"] = request_id
    if tenant_id:
        event_dict["tenant_id"] = tenant_id

    return event_dict

def add_app_info(
    logger: WrappedLogger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Structlog processor that adds static application metadata to every log record.

    Injects service name and version for log aggregation filtering.
    """
    from app.core.config import settings
    event_dict["service"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["env"] = settings.APP_ENV.value
    return event_dict

# — Configuration ——————————————————————————————————————

def configure_logging(log_level: str = "INFO", is_development: bool = True) -> None:
    """
    Configure structlog and stdlib logging for the application.

    In development: uses ConsoleRenderer for human-readable colored output.
    In production: uses JSONRenderer for machine-parseable structured logs.

    Args:
        log_level: Minimum log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        is_development: When True, enables pretty console output.
    """
    # Shared processors applied to every log record
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        inject_context_vars,
        add_app_info,
    ]

    if is_development:
        # Human-readable output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for production log aggregation
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy_logger in ("uvicorn.access", "sqlalchemy.engine", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Factory function to obtain a named structured logger.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        A structlog BoundLogger instance with context injection.

    Example:
        logger = get_logger(__name__)
        logger.info("etl_complete", rows_processed=500, duration_ms=1200)
    """
    return structlog.get_logger(name)