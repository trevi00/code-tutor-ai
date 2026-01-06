"""WebSocket connection manager for collaboration sessions."""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)

from fastapi import WebSocket

from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Connection:
    """Represents a WebSocket connection."""

    websocket: WebSocket
    user_id: UUID
    username: str
    session_id: UUID
    connected_at: datetime = field(default_factory=datetime.utcnow)

    async def send_json(self, data: dict[str, Any]) -> bool:
        """Send JSON data to this connection."""
        try:
            await self.websocket.send_json(data)
            return True
        except Exception as e:
            logger.warning(
                "send_failed",
                user_id=str(self.user_id),
                error=str(e),
            )
            return False


class ConnectionManager:
    """Manages WebSocket connections for collaboration sessions."""

    def __init__(self):
        # session_id -> list of connections
        self._connections: dict[UUID, list[Connection]] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: WebSocket,
        session_id: UUID,
        user_id: UUID,
        username: str,
    ) -> Connection:
        """Accept a WebSocket connection and add to session."""
        await websocket.accept()

        connection = Connection(
            websocket=websocket,
            user_id=user_id,
            username=username,
            session_id=session_id,
        )

        async with self._lock:
            if session_id not in self._connections:
                self._connections[session_id] = []

            # Remove any existing connection for this user in this session
            self._connections[session_id] = [
                c for c in self._connections[session_id] if c.user_id != user_id
            ]
            self._connections[session_id].append(connection)

        logger.info(
            "websocket_connected",
            session_id=str(session_id),
            user_id=str(user_id),
            username=username,
        )

        return connection

    async def disconnect(self, connection: Connection) -> None:
        """Remove a connection from session."""
        async with self._lock:
            session_id = connection.session_id
            if session_id in self._connections:
                self._connections[session_id] = [
                    c for c in self._connections[session_id] if c != connection
                ]
                if not self._connections[session_id]:
                    del self._connections[session_id]

        logger.info(
            "websocket_disconnected",
            session_id=str(connection.session_id),
            user_id=str(connection.user_id),
        )

    async def broadcast_to_session(
        self,
        session_id: UUID,
        message: dict[str, Any],
        exclude_user_id: UUID | None = None,
    ) -> None:
        """Broadcast a message to all connections in a session."""
        async with self._lock:
            connections = self._connections.get(session_id, [])

        tasks = []
        for conn in connections:
            if exclude_user_id and conn.user_id == exclude_user_id:
                continue
            tasks.append(conn.send_json(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_to_user(
        self,
        session_id: UUID,
        user_id: UUID,
        message: dict[str, Any],
    ) -> bool:
        """Send a message to a specific user in a session."""
        async with self._lock:
            connections = self._connections.get(session_id, [])

        for conn in connections:
            if conn.user_id == user_id:
                return await conn.send_json(message)
        return False

    def get_session_users(self, session_id: UUID) -> list[UUID]:
        """Get list of connected user IDs in a session."""
        connections = self._connections.get(session_id, [])
        return [c.user_id for c in connections]

    def get_connection_count(self, session_id: UUID) -> int:
        """Get number of connections in a session."""
        return len(self._connections.get(session_id, []))

    def is_user_connected(self, session_id: UUID, user_id: UUID) -> bool:
        """Check if a user is connected to a session."""
        connections = self._connections.get(session_id, [])
        return any(c.user_id == user_id for c in connections)


# Global connection manager instance
connection_manager = ConnectionManager()


# Message types
class MessageType:
    """WebSocket message types."""

    # Client -> Server
    JOIN = "join"
    LEAVE = "leave"
    CODE_CHANGE = "code_change"
    CURSOR_MOVE = "cursor_move"
    SELECTION_CHANGE = "selection_change"
    CHAT = "chat"

    # Server -> Client
    SESSION_STATE = "session_state"
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    CODE_UPDATE = "code_update"
    CURSOR_UPDATE = "cursor_update"
    SELECTION_UPDATE = "selection_update"
    CHAT_MESSAGE = "chat_message"
    ERROR = "error"
    SYNC = "sync"


def create_message(msg_type: str, data: dict[str, Any]) -> dict[str, Any]:
    """Create a WebSocket message."""
    return {
        "type": msg_type,
        "data": data,
        "timestamp": utc_now().isoformat(),
    }


def create_error_message(error: str, code: str = "error") -> dict[str, Any]:
    """Create an error message."""
    return create_message(MessageType.ERROR, {"error": error, "code": code})
