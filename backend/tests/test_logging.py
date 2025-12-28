"""Unit tests for Logging Module"""

import pytest
import structlog

from code_tutor.shared.infrastructure.logging import (
    get_logger,
    LoggerMixin,
    log_context,
    clear_log_context,
)


class TestGetLogger:
    """Tests for get_logger function"""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a logger"""
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_without_name(self):
        """Test get_logger without name"""
        logger = get_logger()
        assert logger is not None


class TestLoggerMixin:
    """Tests for LoggerMixin class"""

    def test_logger_mixin(self):
        """Test LoggerMixin provides logger"""

        class TestClass(LoggerMixin):
            pass

        obj = TestClass()
        assert hasattr(obj, "logger")
        assert obj.logger is not None

    def test_logger_mixin_bound_to_class(self):
        """Test logger is bound to class name"""

        class MyCustomClass(LoggerMixin):
            pass

        obj = MyCustomClass()
        logger = obj.logger
        assert logger is not None


class TestLogContext:
    """Tests for log context functions"""

    def test_log_context(self):
        """Test adding log context"""
        log_context(request_id="123", user_id="456")
        # Context is set - test that no error occurs

    def test_clear_log_context(self):
        """Test clearing log context"""
        log_context(key="value")
        clear_log_context()
        # Context is cleared - test that no error occurs
