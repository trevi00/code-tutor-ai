"""Tests for Collaboration Module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from code_tutor.collaboration.domain.value_objects import (
    SessionStatus,
    OperationType,
    CursorPosition,
    SelectionRange,
    CodeOperation,
    PARTICIPANT_COLORS,
    get_participant_color,
)
from code_tutor.collaboration.domain.entities import (
    Participant,
    CodeChange,
    CollaborationSession,
    utc_now,
)
from code_tutor.collaboration.infrastructure.connection_manager import (
    Connection,
    ConnectionManager,
    MessageType,
    create_message,
    create_error_message,
)


# ============= Value Objects Tests =============

class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_session_status_values(self):
        """Test all session status values exist."""
        assert SessionStatus.WAITING.value == "waiting"
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.CLOSED.value == "closed"

    def test_session_status_from_string(self):
        """Test creating status from string."""
        assert SessionStatus("waiting") == SessionStatus.WAITING
        assert SessionStatus("active") == SessionStatus.ACTIVE
        assert SessionStatus("closed") == SessionStatus.CLOSED


class TestOperationType:
    """Tests for OperationType enum."""

    def test_operation_type_values(self):
        """Test all operation type values exist."""
        assert OperationType.INSERT.value == "insert"
        assert OperationType.DELETE.value == "delete"
        assert OperationType.REPLACE.value == "replace"

    def test_operation_type_from_string(self):
        """Test creating operation type from string."""
        assert OperationType("insert") == OperationType.INSERT
        assert OperationType("delete") == OperationType.DELETE
        assert OperationType("replace") == OperationType.REPLACE


class TestCursorPosition:
    """Tests for CursorPosition value object."""

    def test_cursor_position_creation(self):
        """Test creating cursor position."""
        cursor = CursorPosition(line=10, column=5)
        assert cursor.line == 10
        assert cursor.column == 5

    def test_cursor_position_to_dict(self):
        """Test cursor position serialization."""
        cursor = CursorPosition(line=10, column=5)
        data = cursor.to_dict()
        assert data == {"line": 10, "column": 5}

    def test_cursor_position_from_dict(self):
        """Test cursor position deserialization."""
        data = {"line": 15, "column": 20}
        cursor = CursorPosition.from_dict(data)
        assert cursor.line == 15
        assert cursor.column == 20

    def test_cursor_position_from_dict_defaults(self):
        """Test cursor position deserialization with defaults."""
        cursor = CursorPosition.from_dict({})
        assert cursor.line == 1
        assert cursor.column == 1

    def test_cursor_position_frozen(self):
        """Test that cursor position is immutable."""
        cursor = CursorPosition(line=1, column=1)
        with pytest.raises(AttributeError):
            cursor.line = 2


class TestSelectionRange:
    """Tests for SelectionRange value object."""

    def test_selection_range_creation(self):
        """Test creating selection range."""
        selection = SelectionRange(
            start_line=1, start_column=5,
            end_line=3, end_column=10
        )
        assert selection.start_line == 1
        assert selection.start_column == 5
        assert selection.end_line == 3
        assert selection.end_column == 10

    def test_selection_range_to_dict(self):
        """Test selection range serialization."""
        selection = SelectionRange(
            start_line=1, start_column=5,
            end_line=3, end_column=10
        )
        data = selection.to_dict()
        assert data == {
            "start_line": 1,
            "start_column": 5,
            "end_line": 3,
            "end_column": 10,
        }

    def test_selection_range_from_dict(self):
        """Test selection range deserialization."""
        data = {
            "start_line": 2,
            "start_column": 1,
            "end_line": 5,
            "end_column": 15,
        }
        selection = SelectionRange.from_dict(data)
        assert selection.start_line == 2
        assert selection.start_column == 1
        assert selection.end_line == 5
        assert selection.end_column == 15

    def test_selection_range_from_dict_defaults(self):
        """Test selection range deserialization with defaults."""
        selection = SelectionRange.from_dict({})
        assert selection.start_line == 1
        assert selection.start_column == 1
        assert selection.end_line == 1
        assert selection.end_column == 1


class TestCodeOperation:
    """Tests for CodeOperation value object."""

    def test_code_operation_insert(self):
        """Test creating insert operation."""
        op = CodeOperation(
            operation_type=OperationType.INSERT,
            position=10,
            content="hello",
            length=0,
        )
        assert op.operation_type == OperationType.INSERT
        assert op.position == 10
        assert op.content == "hello"
        assert op.length == 0

    def test_code_operation_delete(self):
        """Test creating delete operation."""
        op = CodeOperation(
            operation_type=OperationType.DELETE,
            position=5,
            content="",
            length=10,
        )
        assert op.operation_type == OperationType.DELETE
        assert op.position == 5
        assert op.length == 10

    def test_code_operation_replace(self):
        """Test creating replace operation."""
        op = CodeOperation(
            operation_type=OperationType.REPLACE,
            position=0,
            content="new text",
            length=8,
        )
        assert op.operation_type == OperationType.REPLACE
        assert op.content == "new text"
        assert op.length == 8

    def test_code_operation_to_dict(self):
        """Test code operation serialization."""
        op = CodeOperation(
            operation_type=OperationType.INSERT,
            position=10,
            content="test",
            length=0,
        )
        data = op.to_dict()
        assert data == {
            "type": "insert",
            "position": 10,
            "content": "test",
            "length": 0,
        }

    def test_code_operation_from_dict(self):
        """Test code operation deserialization."""
        data = {
            "type": "delete",
            "position": 5,
            "content": "",
            "length": 3,
        }
        op = CodeOperation.from_dict(data)
        assert op.operation_type == OperationType.DELETE
        assert op.position == 5
        assert op.content == ""
        assert op.length == 3

    def test_code_operation_from_dict_defaults(self):
        """Test code operation deserialization with defaults."""
        op = CodeOperation.from_dict({})
        assert op.operation_type == OperationType.INSERT
        assert op.position == 0
        assert op.content == ""
        assert op.length == 0


class TestGetParticipantColor:
    """Tests for get_participant_color function."""

    def test_get_participant_color_within_range(self):
        """Test getting colors within range."""
        for i in range(len(PARTICIPANT_COLORS)):
            color = get_participant_color(i)
            assert color == PARTICIPANT_COLORS[i]

    def test_get_participant_color_wraps_around(self):
        """Test that color selection wraps around."""
        num_colors = len(PARTICIPANT_COLORS)
        assert get_participant_color(0) == get_participant_color(num_colors)
        assert get_participant_color(1) == get_participant_color(num_colors + 1)

    def test_get_participant_color_returns_valid_hex(self):
        """Test that colors are valid hex codes."""
        for i in range(10):
            color = get_participant_color(i)
            assert color.startswith("#")
            assert len(color) == 7


# ============= Entities Tests =============

class TestParticipant:
    """Tests for Participant entity."""

    def test_participant_creation(self):
        """Test creating a participant."""
        participant_id = uuid4()
        user_id = uuid4()
        session_id = uuid4()
        participant = Participant(
            id=participant_id,
            user_id=user_id,
            session_id=session_id,
            username="testuser",
            color="#FF6B6B",
        )
        assert participant.id == participant_id
        assert participant.user_id == user_id
        assert participant.session_id == session_id
        assert participant.username == "testuser"
        assert participant.color == "#FF6B6B"
        assert participant.is_active is True
        assert participant.cursor_position is None
        assert participant.selection_range is None

    def test_participant_update_cursor(self):
        """Test updating cursor position."""
        participant = Participant(
            id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            username="testuser",
        )
        cursor = CursorPosition(line=5, column=10)
        participant.update_cursor(cursor)
        assert participant.cursor_position == cursor
        assert participant.cursor_position.line == 5
        assert participant.cursor_position.column == 10

    def test_participant_update_selection(self):
        """Test updating selection range."""
        participant = Participant(
            id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            username="testuser",
        )
        selection = SelectionRange(
            start_line=1, start_column=1,
            end_line=3, end_column=5
        )
        participant.update_selection(selection)
        assert participant.selection_range == selection

    def test_participant_update_selection_none(self):
        """Test clearing selection."""
        participant = Participant(
            id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            username="testuser",
        )
        selection = SelectionRange(
            start_line=1, start_column=1,
            end_line=3, end_column=5
        )
        participant.update_selection(selection)
        participant.update_selection(None)
        assert participant.selection_range is None

    def test_participant_deactivate(self):
        """Test deactivating a participant."""
        participant = Participant(
            id=uuid4(),
            user_id=uuid4(),
            session_id=uuid4(),
            username="testuser",
        )
        assert participant.is_active is True
        participant.deactivate()
        assert participant.is_active is False

    def test_participant_to_dict(self):
        """Test participant serialization."""
        participant_id = uuid4()
        user_id = uuid4()
        participant = Participant(
            id=participant_id,
            user_id=user_id,
            session_id=uuid4(),
            username="testuser",
            color="#4ECDC4",
            cursor_position=CursorPosition(line=5, column=10),
        )
        data = participant.to_dict()
        assert data["id"] == str(participant_id)
        assert data["user_id"] == str(user_id)
        assert data["username"] == "testuser"
        assert data["color"] == "#4ECDC4"
        assert data["is_active"] is True
        assert data["cursor_position"] == {"line": 5, "column": 10}
        assert data["selection_range"] is None


class TestCodeChange:
    """Tests for CodeChange entity."""

    def test_code_change_creation(self):
        """Test creating a code change record."""
        change_id = uuid4()
        session_id = uuid4()
        user_id = uuid4()
        operation = CodeOperation(
            operation_type=OperationType.INSERT,
            position=0,
            content="hello",
            length=0,
        )
        change = CodeChange(
            id=change_id,
            session_id=session_id,
            user_id=user_id,
            operation=operation,
            version=1,
        )
        assert change.id == change_id
        assert change.session_id == session_id
        assert change.user_id == user_id
        assert change.operation == operation
        assert change.version == 1

    def test_code_change_to_dict(self):
        """Test code change serialization."""
        change_id = uuid4()
        session_id = uuid4()
        user_id = uuid4()
        operation = CodeOperation(
            operation_type=OperationType.INSERT,
            position=0,
            content="test",
            length=0,
        )
        change = CodeChange(
            id=change_id,
            session_id=session_id,
            user_id=user_id,
            operation=operation,
            version=2,
        )
        data = change.to_dict()
        assert data["id"] == str(change_id)
        assert data["session_id"] == str(session_id)
        assert data["user_id"] == str(user_id)
        assert data["version"] == 2
        assert data["operation"]["type"] == "insert"


class TestCollaborationSession:
    """Tests for CollaborationSession aggregate root."""

    def test_session_creation(self):
        """Test creating a session."""
        session_id = uuid4()
        host_id = uuid4()
        session = CollaborationSession(
            id=session_id,
            problem_id=None,
            host_id=host_id,
            title="Test Session",
        )
        assert session.id == session_id
        assert session.host_id == host_id
        assert session.title == "Test Session"
        assert session.status == SessionStatus.WAITING
        assert session.code_content == ""
        assert session.language == "python"
        assert session.version == 0
        assert len(session.participants) == 0
        assert session.max_participants == 5

    def test_session_create_factory(self):
        """Test session factory method."""
        host_id = uuid4()
        problem_id = uuid4()
        session = CollaborationSession.create(
            host_id=host_id,
            title="My Session",
            problem_id=problem_id,
            initial_code="print('hello')",
            language="python",
        )
        assert session.id is not None
        assert session.host_id == host_id
        assert session.problem_id == problem_id
        assert session.title == "My Session"
        assert session.code_content == "print('hello')"
        assert session.language == "python"

    def test_session_add_participant(self):
        """Test adding a participant."""
        host_id = uuid4()
        user_id = uuid4()
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=host_id,
            title="Test",
        )
        participant = session.add_participant(user_id, "testuser")
        assert participant.user_id == user_id
        assert participant.username == "testuser"
        assert participant.is_active is True
        assert participant.color is not None
        assert len(session.participants) == 1
        assert session.status == SessionStatus.ACTIVE

    def test_session_add_participant_reactivate(self):
        """Test reactivating an existing participant."""
        host_id = uuid4()
        user_id = uuid4()
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=host_id,
            title="Test",
        )
        # Add first time
        session.add_participant(user_id, "testuser")
        # Deactivate
        session.participants[0].deactivate()
        assert session.participants[0].is_active is False
        # Re-add (reactivate)
        participant = session.add_participant(user_id, "testuser")
        assert participant.is_active is True
        assert len(session.participants) == 1

    def test_session_add_participant_full(self):
        """Test adding participant to full session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            max_participants=2,
        )
        session.add_participant(uuid4(), "user1")
        session.add_participant(uuid4(), "user2")
        with pytest.raises(ValueError, match="Session is full"):
            session.add_participant(uuid4(), "user3")

    def test_session_remove_participant(self):
        """Test removing a participant."""
        host_id = uuid4()
        user_id = uuid4()
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=host_id,
            title="Test",
        )
        session.add_participant(user_id, "testuser")
        assert len(session.active_participants) == 1
        session.remove_participant(user_id)
        assert len(session.active_participants) == 0
        # Session should close when no active participants
        assert session.status == SessionStatus.CLOSED

    def test_session_get_participant_by_user_id(self):
        """Test getting participant by user ID."""
        user_id = uuid4()
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        session.add_participant(user_id, "testuser")
        participant = session.get_participant_by_user_id(user_id)
        assert participant is not None
        assert participant.user_id == user_id

    def test_session_get_participant_by_user_id_not_found(self):
        """Test getting non-existent participant."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        result = session.get_participant_by_user_id(uuid4())
        assert result is None

    def test_session_active_participants(self):
        """Test getting active participants."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        user1 = uuid4()
        user2 = uuid4()
        session.add_participant(user1, "user1")
        session.add_participant(user2, "user2")
        assert len(session.active_participants) == 2
        session.remove_participant(user1)
        assert len(session.active_participants) == 1
        assert session.active_participants[0].user_id == user2

    def test_session_apply_operation_insert(self):
        """Test applying insert operation."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            code_content="Hello World",
        )
        user_id = uuid4()
        session.add_participant(user_id, "testuser")
        operation = CodeOperation(
            operation_type=OperationType.INSERT,
            position=5,
            content=" Beautiful",
            length=0,
        )
        version = session.apply_operation(operation, user_id)
        assert version == 1
        assert session.code_content == "Hello Beautiful World"

    def test_session_apply_operation_delete(self):
        """Test applying delete operation."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            code_content="Hello World",
        )
        user_id = uuid4()
        operation = CodeOperation(
            operation_type=OperationType.DELETE,
            position=5,
            content="",
            length=6,  # Delete " World"
        )
        version = session.apply_operation(operation, user_id)
        assert version == 1
        assert session.code_content == "Hello"

    def test_session_apply_operation_replace(self):
        """Test applying replace operation."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            code_content="Hello World",
        )
        user_id = uuid4()
        operation = CodeOperation(
            operation_type=OperationType.REPLACE,
            position=6,
            content="Python",
            length=5,  # Replace "World"
        )
        version = session.apply_operation(operation, user_id)
        assert version == 1
        assert session.code_content == "Hello Python"

    def test_session_update_participant_cursor(self):
        """Test updating participant cursor position."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        user_id = uuid4()
        session.add_participant(user_id, "testuser")
        cursor = CursorPosition(line=10, column=5)
        session.update_participant_cursor(user_id, cursor)
        participant = session.get_participant_by_user_id(user_id)
        assert participant.cursor_position == cursor

    def test_session_update_cursor_by_line_column(self):
        """Test updating cursor by line and column."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        user_id = uuid4()
        session.add_participant(user_id, "testuser")
        session.update_cursor(user_id, 5, 10)
        participant = session.get_participant_by_user_id(user_id)
        assert participant.cursor_position.line == 5
        assert participant.cursor_position.column == 10

    def test_session_close(self):
        """Test closing a session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
        )
        user_id = uuid4()
        session.add_participant(user_id, "testuser")
        session.close()
        assert session.status == SessionStatus.CLOSED
        assert len(session.active_participants) == 0

    def test_session_is_host(self):
        """Test is_host check."""
        host_id = uuid4()
        other_id = uuid4()
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=host_id,
            title="Test",
        )
        assert session.is_host(host_id) is True
        assert session.is_host(other_id) is False

    def test_session_can_join_open(self):
        """Test can_join for open session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            max_participants=5,
        )
        assert session.can_join(uuid4()) is True

    def test_session_can_join_closed(self):
        """Test can_join for closed session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            status=SessionStatus.CLOSED,
        )
        assert session.can_join(uuid4()) is False

    def test_session_can_join_full(self):
        """Test can_join for full session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=None,
            host_id=uuid4(),
            title="Test",
            max_participants=2,
        )
        session.add_participant(uuid4(), "user1")
        session.add_participant(uuid4(), "user2")
        # New user cannot join
        assert session.can_join(uuid4()) is False
        # Existing participant can rejoin
        existing_user = session.participants[0].user_id
        assert session.can_join(existing_user) is True

    def test_session_to_dict(self):
        """Test session serialization."""
        session_id = uuid4()
        host_id = uuid4()
        session = CollaborationSession(
            id=session_id,
            problem_id=None,
            host_id=host_id,
            title="Test Session",
            code_content="print('hi')",
            language="python",
        )
        user_id = uuid4()
        session.add_participant(user_id, "testuser")
        data = session.to_dict()
        assert data["id"] == str(session_id)
        assert data["host_id"] == str(host_id)
        assert data["title"] == "Test Session"
        assert data["status"] == "active"
        assert data["code_content"] == "print('hi')"
        assert data["language"] == "python"
        assert data["version"] == 0
        assert len(data["participants"]) == 1


# ============= Connection Manager Tests =============

class TestConnection:
    """Tests for Connection dataclass."""

    @pytest.mark.asyncio
    async def test_connection_send_json_success(self):
        """Test successful JSON send."""
        mock_websocket = AsyncMock()
        connection = Connection(
            websocket=mock_websocket,
            user_id=uuid4(),
            username="testuser",
            session_id=uuid4(),
        )
        result = await connection.send_json({"type": "test"})
        assert result is True
        mock_websocket.send_json.assert_called_once_with({"type": "test"})

    @pytest.mark.asyncio
    async def test_connection_send_json_failure(self):
        """Test failed JSON send."""
        mock_websocket = AsyncMock()
        mock_websocket.send_json.side_effect = Exception("Connection closed")
        connection = Connection(
            websocket=mock_websocket,
            user_id=uuid4(),
            username="testuser",
            session_id=uuid4(),
        )
        result = await connection.send_json({"type": "test"})
        assert result is False


class TestConnectionManager:
    """Tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test connecting a websocket."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        session_id = uuid4()
        user_id = uuid4()
        connection = await manager.connect(
            mock_websocket, session_id, user_id, "testuser"
        )
        assert connection.user_id == user_id
        assert connection.session_id == session_id
        assert connection.username == "testuser"
        mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_replaces_existing(self):
        """Test that new connection replaces existing for same user."""
        manager = ConnectionManager()
        session_id = uuid4()
        user_id = uuid4()
        # First connection
        mock_ws1 = AsyncMock()
        await manager.connect(mock_ws1, session_id, user_id, "testuser")
        assert manager.get_connection_count(session_id) == 1
        # Second connection for same user
        mock_ws2 = AsyncMock()
        await manager.connect(mock_ws2, session_id, user_id, "testuser")
        assert manager.get_connection_count(session_id) == 1

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnecting a websocket."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        session_id = uuid4()
        connection = await manager.connect(
            mock_websocket, session_id, uuid4(), "testuser"
        )
        assert manager.get_connection_count(session_id) == 1
        await manager.disconnect(connection)
        assert manager.get_connection_count(session_id) == 0

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_empty_session(self):
        """Test that disconnecting last user removes session from dict."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        session_id = uuid4()
        connection = await manager.connect(
            mock_websocket, session_id, uuid4(), "testuser"
        )
        await manager.disconnect(connection)
        assert session_id not in manager._connections

    @pytest.mark.asyncio
    async def test_broadcast_to_session(self):
        """Test broadcasting to all connections in session."""
        manager = ConnectionManager()
        session_id = uuid4()
        # Add two connections
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        await manager.connect(mock_ws1, session_id, uuid4(), "user1")
        await manager.connect(mock_ws2, session_id, uuid4(), "user2")
        message = {"type": "test", "data": "hello"}
        await manager.broadcast_to_session(session_id, message)
        mock_ws1.send_json.assert_called_once_with(message)
        mock_ws2.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_session_exclude_user(self):
        """Test broadcasting excluding a specific user."""
        manager = ConnectionManager()
        session_id = uuid4()
        user1_id = uuid4()
        user2_id = uuid4()
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()
        await manager.connect(mock_ws1, session_id, user1_id, "user1")
        await manager.connect(mock_ws2, session_id, user2_id, "user2")
        message = {"type": "test"}
        await manager.broadcast_to_session(session_id, message, exclude_user_id=user1_id)
        mock_ws1.send_json.assert_not_called()
        mock_ws2.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_user(self):
        """Test sending to specific user."""
        manager = ConnectionManager()
        session_id = uuid4()
        user_id = uuid4()
        mock_ws = AsyncMock()
        await manager.connect(mock_ws, session_id, user_id, "testuser")
        message = {"type": "private"}
        result = await manager.send_to_user(session_id, user_id, message)
        assert result is True
        mock_ws.send_json.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_send_to_user_not_found(self):
        """Test sending to non-existent user."""
        manager = ConnectionManager()
        session_id = uuid4()
        result = await manager.send_to_user(session_id, uuid4(), {"type": "test"})
        assert result is False

    def test_get_session_users(self):
        """Test getting user IDs in session."""
        manager = ConnectionManager()
        session_id = uuid4()
        user1 = uuid4()
        user2 = uuid4()
        # Manually add connections to test sync method
        mock_ws = MagicMock()
        manager._connections[session_id] = [
            Connection(mock_ws, user1, "user1", session_id),
            Connection(mock_ws, user2, "user2", session_id),
        ]
        users = manager.get_session_users(session_id)
        assert len(users) == 2
        assert user1 in users
        assert user2 in users

    def test_get_session_users_empty(self):
        """Test getting users from non-existent session."""
        manager = ConnectionManager()
        users = manager.get_session_users(uuid4())
        assert users == []

    def test_get_connection_count(self):
        """Test getting connection count."""
        manager = ConnectionManager()
        session_id = uuid4()
        mock_ws = MagicMock()
        manager._connections[session_id] = [
            Connection(mock_ws, uuid4(), "user1", session_id),
            Connection(mock_ws, uuid4(), "user2", session_id),
        ]
        assert manager.get_connection_count(session_id) == 2

    def test_get_connection_count_empty(self):
        """Test getting count for non-existent session."""
        manager = ConnectionManager()
        assert manager.get_connection_count(uuid4()) == 0

    def test_is_user_connected(self):
        """Test checking if user is connected."""
        manager = ConnectionManager()
        session_id = uuid4()
        user_id = uuid4()
        mock_ws = MagicMock()
        manager._connections[session_id] = [
            Connection(mock_ws, user_id, "testuser", session_id)
        ]
        assert manager.is_user_connected(session_id, user_id) is True
        assert manager.is_user_connected(session_id, uuid4()) is False


class TestMessageType:
    """Tests for MessageType constants."""

    def test_client_message_types(self):
        """Test client to server message types."""
        assert MessageType.JOIN == "join"
        assert MessageType.LEAVE == "leave"
        assert MessageType.CODE_CHANGE == "code_change"
        assert MessageType.CURSOR_MOVE == "cursor_move"
        assert MessageType.SELECTION_CHANGE == "selection_change"
        assert MessageType.CHAT == "chat"

    def test_server_message_types(self):
        """Test server to client message types."""
        assert MessageType.SESSION_STATE == "session_state"
        assert MessageType.USER_JOINED == "user_joined"
        assert MessageType.USER_LEFT == "user_left"
        assert MessageType.CODE_UPDATE == "code_update"
        assert MessageType.CURSOR_UPDATE == "cursor_update"
        assert MessageType.SELECTION_UPDATE == "selection_update"
        assert MessageType.CHAT_MESSAGE == "chat_message"
        assert MessageType.ERROR == "error"
        assert MessageType.SYNC == "sync"


class TestCreateMessage:
    """Tests for create_message function."""

    def test_create_message(self):
        """Test creating a message."""
        message = create_message(MessageType.CODE_UPDATE, {"content": "test"})
        assert message["type"] == "code_update"
        assert message["data"] == {"content": "test"}
        assert "timestamp" in message

    def test_create_message_timestamp_format(self):
        """Test that timestamp is ISO format."""
        message = create_message(MessageType.ERROR, {"error": "test"})
        timestamp = message["timestamp"]
        # Should be parseable as ISO format
        assert "T" in timestamp


class TestCreateErrorMessage:
    """Tests for create_error_message function."""

    def test_create_error_message(self):
        """Test creating an error message."""
        message = create_error_message("Something went wrong")
        assert message["type"] == MessageType.ERROR
        assert message["data"]["error"] == "Something went wrong"
        assert message["data"]["code"] == "error"

    def test_create_error_message_with_code(self):
        """Test creating error message with custom code."""
        message = create_error_message("Not found", code="not_found")
        assert message["data"]["code"] == "not_found"


# ============= Helper Function Tests =============

class TestUtcNow:
    """Tests for utc_now helper function."""

    def test_utc_now_is_aware(self):
        """Test that utc_now returns timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test that utc_now returns current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after


# ============= Integration Tests =============

class TestCollaborationIntegration:
    """Integration tests for collaboration module."""

    def test_full_session_workflow(self):
        """Test complete session workflow."""
        host_id = uuid4()
        user1_id = uuid4()
        user2_id = uuid4()

        # Create session
        session = CollaborationSession.create(
            host_id=host_id,
            title="Pair Programming",
            initial_code="def main():\n    pass",
            language="python",
        )

        # Host joins
        session.add_participant(host_id, "host")
        assert session.status == SessionStatus.ACTIVE
        assert len(session.active_participants) == 1

        # User1 joins
        session.add_participant(user1_id, "user1")
        assert len(session.active_participants) == 2

        # User2 joins
        session.add_participant(user2_id, "user2")
        assert len(session.active_participants) == 3

        # User1 makes a code change
        operation = CodeOperation(
            operation_type=OperationType.REPLACE,
            position=14,
            content="print('hello')",
            length=4,
        )
        version = session.apply_operation(operation, user1_id)
        assert version == 1
        assert "print('hello')" in session.code_content

        # User1 updates cursor
        session.update_cursor(user1_id, 2, 15)
        participant = session.get_participant_by_user_id(user1_id)
        assert participant.cursor_position.line == 2
        assert participant.cursor_position.column == 15

        # User2 leaves
        session.remove_participant(user2_id)
        assert len(session.active_participants) == 2

        # Host closes session
        session.close()
        assert session.status == SessionStatus.CLOSED
        assert len(session.active_participants) == 0

    def test_operation_sequence(self):
        """Test sequence of operations on code."""
        session = CollaborationSession.create(
            host_id=uuid4(),
            title="Test",
            initial_code="",
        )
        user_id = uuid4()
        session.add_participant(user_id, "user")

        # Insert "Hello"
        op1 = CodeOperation(OperationType.INSERT, 0, "Hello", 0)
        session.apply_operation(op1, user_id)
        assert session.code_content == "Hello"

        # Insert " World" at position 5
        op2 = CodeOperation(OperationType.INSERT, 5, " World", 0)
        session.apply_operation(op2, user_id)
        assert session.code_content == "Hello World"

        # Delete " World"
        op3 = CodeOperation(OperationType.DELETE, 5, "", 6)
        session.apply_operation(op3, user_id)
        assert session.code_content == "Hello"

        # Replace "Hello" with "Hi"
        op4 = CodeOperation(OperationType.REPLACE, 0, "Hi", 5)
        session.apply_operation(op4, user_id)
        assert session.code_content == "Hi"

        assert session.version == 4
