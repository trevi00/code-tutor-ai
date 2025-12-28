"""Structured logging configuration using structlog"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import Processor

from code_tutor.shared.config import get_settings


def configure_logging() -> None:
    """Configure structured logging for the application"""
    settings = get_settings()

    # Determine log level based on environment
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors for all environments
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if settings.ENVIRONMENT == "development":
        # Development: pretty console output
        processors: list[Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # Production: JSON output for log aggregation
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance"""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin to add logging capability to classes"""

    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger bound to class name"""
        return get_logger(self.__class__.__name__)


def log_context(**kwargs: Any) -> None:
    """Add context variables for structured logging"""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_log_context() -> None:
    """Clear all context variables"""
    structlog.contextvars.clear_contextvars()
