"""Unit tests for Dependency Injection Container"""

import pytest

from code_tutor.shared.container import Container
from code_tutor.shared.config import Settings


class TestContainer:
    """Tests for DI Container"""

    def test_container_instantiation(self):
        """Test container can be instantiated"""
        container = Container()
        assert container is not None

    def test_container_config_provider(self):
        """Test config provider returns settings"""
        container = Container()
        config = container.config()
        assert config is not None
        assert isinstance(config, Settings)

    def test_container_has_db_session_factory(self):
        """Test container has db_session_factory provider"""
        container = Container()
        assert hasattr(container, "db_session_factory")

    def test_container_has_redis_client(self):
        """Test container has redis_client provider"""
        container = Container()
        assert hasattr(container, "redis_client")

    def test_container_wiring_config(self):
        """Test container has wiring configuration"""
        assert Container.wiring_config is not None
