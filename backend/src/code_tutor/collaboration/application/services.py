"""Collaboration service for managing real-time coding sessions."""

import random
from datetime import UTC, datetime
from uuid import UUID, uuid4

from code_tutor.collaboration.application.dto import (
    CodeChangeRequest,
    CreateSessionRequest,
    CursorUpdateRequest,
    ParticipantResponse,
    SelectionUpdateRequest,
    SessionDetailResponse,
    SessionResponse,
)
from code_tutor.collaboration.domain.entities import (
    CodeChange,
    CollaborationSession,
)
from code_tutor.collaboration.domain.repository import CollaborationRepository
from code_tutor.collaboration.domain.value_objects import (
    CodeOperation,
    OperationType,
    SessionStatus,
)
from code_tutor.collaboration.infrastructure.connection_manager import (
    Connection,
    MessageType,
    connection_manager,
    create_message,
)
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)


# Predefined colors for participants
PARTICIPANT_COLORS = [
    "#4ECDC4",  # Teal
    "#FF6B6B",  # Red
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
    "#BB8FCE",  # Purple
    "#85C1E9",  # Light Blue
]


class CollaborationService:
    """Service for managing collaboration sessions."""

    def __init__(self, repository: CollaborationRepository):
        self.repository = repository

    def _get_random_color(self, existing_colors: list[str]) -> str:
        """Get a random color not already in use."""
        available = [c for c in PARTICIPANT_COLORS if c not in existing_colors]
        if not available:
            available = PARTICIPANT_COLORS
        return random.choice(available)

    def _session_to_response(self, session: CollaborationSession) -> SessionResponse:
        """Convert session entity to response DTO."""
        return SessionResponse(
            id=session.id,
            problem_id=session.problem_id,
            host_id=session.host_id,
            title=session.title,
            status=session.status.value,
            language=session.language,
            version=session.version,
            participant_count=len([p for p in session.participants if p.is_active]),
            max_participants=session.max_participants,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    def _session_to_detail_response(
        self, session: CollaborationSession
    ) -> SessionDetailResponse:
        """Convert session entity to detailed response DTO."""
        participants = []
        for p in session.participants:
            participants.append(
                ParticipantResponse(
                    id=p.id,
                    user_id=p.user_id,
                    username=p.username,
                    cursor_line=p.cursor_position.line if p.cursor_position else None,
                    cursor_column=p.cursor_position.column
                    if p.cursor_position
                    else None,
                    selection_start_line=p.selection_range.start.line
                    if p.selection_range
                    else None,
                    selection_start_column=p.selection_range.start.column
                    if p.selection_range
                    else None,
                    selection_end_line=p.selection_range.end.line
                    if p.selection_range
                    else None,
                    selection_end_column=p.selection_range.end.column
                    if p.selection_range
                    else None,
                    is_active=p.is_active,
                    color=p.color,
                    joined_at=p.joined_at,
                )
            )

        return SessionDetailResponse(
            id=session.id,
            problem_id=session.problem_id,
            host_id=session.host_id,
            title=session.title,
            status=session.status.value,
            language=session.language,
            version=session.version,
            participant_count=len([p for p in session.participants if p.is_active]),
            max_participants=session.max_participants,
            created_at=session.created_at,
            updated_at=session.updated_at,
            code_content=session.code_content,
            participants=participants,
        )

    async def create_session(
        self,
        request: CreateSessionRequest,
        user_id: UUID,
        username: str,
    ) -> SessionDetailResponse:
        """Create a new collaboration session."""
        session = CollaborationSession(
            id=uuid4(),
            problem_id=request.problem_id,
            host_id=user_id,
            title=request.title,
            language=request.language,
            max_participants=request.max_participants,
        )

        # Add host as first participant
        session.add_participant(user_id, username)

        saved = await self.repository.save(session)

        logger.info(
            "session_created",
            session_id=str(saved.id),
            host_id=str(user_id),
            title=request.title,
        )

        return self._session_to_detail_response(saved)

    async def get_session(self, session_id: UUID) -> SessionDetailResponse | None:
        """Get session details."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return None
        return self._session_to_detail_response(session)

    async def join_session(
        self,
        session_id: UUID,
        user_id: UUID,
        username: str,
    ) -> SessionDetailResponse | None:
        """Join an existing collaboration session."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return None

        if session.status == SessionStatus.CLOSED:
            raise ValueError("Session is closed")

        active_count = len([p for p in session.participants if p.is_active])
        if active_count >= session.max_participants:
            raise ValueError("Session is full")

        # Add or reactivate participant
        session.add_participant(user_id, username)

        # Update status to active if was waiting
        if session.status == SessionStatus.WAITING:
            session.status = SessionStatus.ACTIVE

        saved = await self.repository.save(session)

        logger.info(
            "user_joined_session",
            session_id=str(session_id),
            user_id=str(user_id),
            username=username,
        )

        return self._session_to_detail_response(saved)

    async def leave_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Leave a collaboration session."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return False

        participant = session.get_participant(user_id)
        if not participant:
            return False

        participant.is_active = False

        # If host leaves and no active participants, close session
        if user_id == session.host_id:
            active_participants = [p for p in session.participants if p.is_active]
            if not active_participants:
                session.status = SessionStatus.CLOSED

        await self.repository.save(session)

        logger.info(
            "user_left_session",
            session_id=str(session_id),
            user_id=str(user_id),
        )

        return True

    async def apply_code_change(
        self,
        session_id: UUID,
        user_id: UUID,
        request: CodeChangeRequest,
    ) -> int | None:
        """Apply a code change to the session."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return None

        if session.status != SessionStatus.ACTIVE:
            raise ValueError("Session is not active")

        # Verify user is participant
        participant = session.get_participant(user_id)
        if not participant or not participant.is_active:
            raise ValueError("User is not an active participant")

        # Create and apply operation
        operation = CodeOperation(
            operation_type=OperationType(request.operation_type),
            position=request.position,
            content=request.content,
            length=request.length,
        )

        new_version = session.apply_operation(operation, user_id)

        # Save session and code change
        await self.repository.save(session)

        change = CodeChange(
            id=uuid4(),
            session_id=session_id,
            user_id=user_id,
            operation=operation,
            version=new_version,
            timestamp=utc_now(),
        )
        await self.repository.save_code_change(change)

        logger.debug(
            "code_change_applied",
            session_id=str(session_id),
            user_id=str(user_id),
            version=new_version,
        )

        return new_version

    async def update_cursor(
        self,
        session_id: UUID,
        user_id: UUID,
        request: CursorUpdateRequest,
    ) -> bool:
        """Update user's cursor position."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return False

        participant = session.get_participant(user_id)
        if not participant or not participant.is_active:
            return False

        session.update_cursor(user_id, request.line, request.column)
        await self.repository.save(session)
        return True

    async def update_selection(
        self,
        session_id: UUID,
        user_id: UUID,
        request: SelectionUpdateRequest,
    ) -> bool:
        """Update user's selection range."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return False

        participant = session.get_participant(user_id)
        if not participant or not participant.is_active:
            return False

        session.update_selection(
            user_id,
            request.start_line,
            request.start_column,
            request.end_line,
            request.end_column,
        )
        await self.repository.save(session)
        return True

    async def get_user_sessions(
        self,
        user_id: UUID,
        active_only: bool = True,
    ) -> list[SessionResponse]:
        """Get all sessions for a user."""
        sessions = await self.repository.get_user_sessions(user_id, active_only)
        return [self._session_to_response(s) for s in sessions]

    async def get_active_sessions(self, limit: int = 10) -> list[SessionResponse]:
        """Get active public sessions."""
        sessions = await self.repository.get_active_sessions(limit)
        return [self._session_to_response(s) for s in sessions]

    async def close_session(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Close a collaboration session (host only)."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return False

        if session.host_id != user_id:
            raise ValueError("Only the host can close the session")

        session.status = SessionStatus.CLOSED
        await self.repository.save(session)

        logger.info(
            "session_closed",
            session_id=str(session_id),
            host_id=str(user_id),
        )

        return True

    # === WebSocket handlers ===

    async def handle_websocket_join(
        self,
        connection: Connection,
    ) -> None:
        """Handle WebSocket join - broadcast to other participants."""
        session = await self.repository.get_by_id(connection.session_id)
        if not session:
            return

        participant = session.get_participant(connection.user_id)
        if not participant:
            return

        # Send session state to the joining user
        state_message = create_message(
            MessageType.SESSION_STATE,
            {
                "session_id": str(session.id),
                "code_content": session.code_content,
                "language": session.language,
                "version": session.version,
                "participants": [
                    {
                        "user_id": str(p.user_id),
                        "username": p.username,
                        "color": p.color,
                        "cursor": p.cursor_position.to_dict()
                        if p.cursor_position
                        else None,
                        "selection": p.selection_range.to_dict()
                        if p.selection_range
                        else None,
                        "is_active": p.is_active,
                    }
                    for p in session.participants
                ],
            },
        )
        await connection.send_json(state_message)

        # Broadcast join to others
        join_message = create_message(
            MessageType.USER_JOINED,
            {
                "user_id": str(connection.user_id),
                "username": connection.username,
                "color": participant.color,
            },
        )
        await connection_manager.broadcast_to_session(
            connection.session_id,
            join_message,
            exclude_user_id=connection.user_id,
        )

    async def handle_websocket_leave(
        self,
        connection: Connection,
    ) -> None:
        """Handle WebSocket disconnect - broadcast to other participants."""
        leave_message = create_message(
            MessageType.USER_LEFT,
            {
                "user_id": str(connection.user_id),
                "username": connection.username,
            },
        )
        await connection_manager.broadcast_to_session(
            connection.session_id,
            leave_message,
            exclude_user_id=connection.user_id,
        )

    async def handle_code_change_broadcast(
        self,
        session_id: UUID,
        user_id: UUID,
        username: str,
        request: CodeChangeRequest,
        version: int,
    ) -> None:
        """Broadcast code change to other participants."""
        message = create_message(
            MessageType.CODE_UPDATE,
            {
                "user_id": str(user_id),
                "username": username,
                "operation_type": request.operation_type,
                "position": request.position,
                "content": request.content,
                "length": request.length,
                "version": version,
            },
        )
        await connection_manager.broadcast_to_session(
            session_id,
            message,
            exclude_user_id=user_id,
        )

    async def handle_cursor_broadcast(
        self,
        session_id: UUID,
        user_id: UUID,
        username: str,
        request: CursorUpdateRequest,
    ) -> None:
        """Broadcast cursor update to other participants."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return

        participant = session.get_participant(user_id)
        if not participant:
            return

        message = create_message(
            MessageType.CURSOR_UPDATE,
            {
                "user_id": str(user_id),
                "username": username,
                "line": request.line,
                "column": request.column,
                "color": participant.color,
            },
        )
        await connection_manager.broadcast_to_session(
            session_id,
            message,
            exclude_user_id=user_id,
        )

    async def handle_selection_broadcast(
        self,
        session_id: UUID,
        user_id: UUID,
        username: str,
        request: SelectionUpdateRequest,
    ) -> None:
        """Broadcast selection update to other participants."""
        session = await self.repository.get_by_id(session_id)
        if not session:
            return

        participant = session.get_participant(user_id)
        if not participant:
            return

        message = create_message(
            MessageType.SELECTION_UPDATE,
            {
                "user_id": str(user_id),
                "username": username,
                "start_line": request.start_line,
                "start_column": request.start_column,
                "end_line": request.end_line,
                "end_column": request.end_column,
                "color": participant.color,
            },
        )
        await connection_manager.broadcast_to_session(
            session_id,
            message,
            exclude_user_id=user_id,
        )

    async def handle_chat_broadcast(
        self,
        session_id: UUID,
        user_id: UUID,
        username: str,
        message_text: str,
    ) -> None:
        """Broadcast chat message to all participants."""
        message = create_message(
            MessageType.CHAT_MESSAGE,
            {
                "user_id": str(user_id),
                "username": username,
                "message": message_text,
            },
        )
        await connection_manager.broadcast_to_session(session_id, message)
