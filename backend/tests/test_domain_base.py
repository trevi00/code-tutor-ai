"""Unit tests for Domain Base Classes"""

import pytest
from uuid import uuid4
from datetime import datetime
from dataclasses import dataclass

from code_tutor.shared.domain.base import Entity, AggregateRoot, ValueObject
from code_tutor.shared.domain.events import DomainEvent, IntegrationEvent


class TestEntity:
    """Tests for Entity base class"""

    def test_entity_creation(self):
        """Test entity creation with auto-generated ID"""
        entity = Entity()
        assert entity.id is not None
        assert entity.created_at is not None
        assert entity.updated_at is not None

    def test_entity_with_id(self):
        """Test entity creation with specified ID"""
        custom_id = uuid4()
        entity = Entity(id=custom_id)
        assert entity.id == custom_id

    def test_entity_equality(self):
        """Test entity equality based on ID"""
        id1 = uuid4()
        entity1 = Entity(id=id1)
        entity2 = Entity(id=id1)
        entity3 = Entity()

        assert entity1 == entity2
        assert entity1 != entity3


class TestAggregateRoot:
    """Tests for AggregateRoot base class"""

    def test_aggregate_root_creation(self):
        """Test aggregate root creation"""
        agg = AggregateRoot()
        assert agg.id is not None

    def test_aggregate_root_domain_events(self):
        """Test adding and clearing domain events"""

        @dataclass(frozen=True)
        class TestEvent(DomainEvent):
            aggregate_id: str = ""

        agg = AggregateRoot()
        assert len(agg.domain_events) == 0

        event = TestEvent(aggregate_id=str(agg.id))
        agg.add_domain_event(event)
        assert len(agg.domain_events) == 1

        cleared = agg.clear_domain_events()
        assert len(cleared) == 1  # Returns cleared events
        assert len(agg.domain_events) == 0  # Now empty

    def test_aggregate_root_version(self):
        """Test aggregate root version tracking"""
        agg = AggregateRoot()
        assert agg.version == 0

        agg.increment_version()
        assert agg.version == 1


class TestDomainEvent:
    """Tests for DomainEvent base class"""

    def test_domain_event_creation(self):
        """Test domain event creation"""

        @dataclass(frozen=True)
        class TestEvent(DomainEvent):
            data: str = ""

        event = TestEvent(data="test")
        assert event.event_id is not None
        assert event.occurred_at is not None
        assert event.event_type == "TestEvent"

    def test_integration_event_creation(self):
        """Test integration event creation"""

        @dataclass(frozen=True)
        class TestIntegrationEvent(IntegrationEvent):
            message: str = ""

        event = TestIntegrationEvent(
            message="test",
            source_context="TestContext",
        )
        assert event.event_id is not None
        assert event.source_context == "TestContext"
        assert event.event_type == "TestIntegrationEvent"
