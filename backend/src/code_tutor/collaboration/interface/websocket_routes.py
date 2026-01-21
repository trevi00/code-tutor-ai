"""WebSocket routes for real-time collaboration."""

import json
from uuid import UUID

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from code_tutor.collaboration.application.dto import (
    CodeChangeRequest,
    CursorUpdateRequest,
    SelectionUpdateRequest,
)
from code_tutor.collaboration.application.services import CollaborationService
from code_tutor.collaboration.infrastructure.connection_manager import (
    Connection,
    MessageType,
    connection_manager,
    create_error_message,
)
from code_tutor.collaboration.infrastructure.repository import (
    SQLAlchemyCollaborationRepository,
)
from code_tutor.identity.interface.dependencies import get_current_user_ws
from code_tutor.shared.infrastructure.database import get_async_session
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/collaboration", tags=["collaboration-ws"])


async def get_collaboration_service(
    db=Depends(get_async_session),
) -> CollaborationService:
    """Get collaboration service with repository."""
    repository = SQLAlchemyCollaborationRepository(db)
    return CollaborationService(repository)


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: UUID,
):
    """WebSocket endpoint for real-time collaboration."""
    connection: Connection | None = None

    try:
        # Authenticate user from token in query params
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=4001, reason="Missing authentication token")
            return

        # Get user from token
        user = await get_current_user_ws(token)
        if not user:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return

        # Get database session
        async for db in get_async_session():
            repository = SQLAlchemyCollaborationRepository(db)
            service = CollaborationService(repository)

            # Verify session exists and user can join
            session_detail = await service.join_session(
                session_id, user.id, user.username
            )
            if not session_detail:
                await websocket.close(code=4004, reason="Session not found")
                return

            # Accept connection and add to manager
            connection = await connection_manager.connect(
                websocket, session_id, user.id, user.username
            )

            # Send initial state and broadcast join
            await service.handle_websocket_join(connection)

            # Process messages
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    msg_type = message.get("type")
                    msg_data = message.get("data", {})

                    await handle_message(
                        service,
                        connection,
                        msg_type,
                        msg_data,
                    )
                except WebSocketDisconnect:
                    break
                except json.JSONDecodeError:
                    await connection.send_json(
                        create_error_message("Invalid JSON", "parse_error")
                    )
                except Exception as e:
                    logger.exception(
                        "websocket_message_error",
                        session_id=str(session_id),
                        user_id=str(user.id),
                        error=str(e),
                    )
                    await connection.send_json(
                        create_error_message(str(e), "message_error")
                    )

    except ValueError as e:
        logger.warning(
            "websocket_join_error",
            session_id=str(session_id),
            error=str(e),
        )
        if connection:
            await connection.send_json(create_error_message(str(e), "join_error"))
        await websocket.close(code=4003, reason=str(e))

    except Exception as e:
        logger.exception(
            "websocket_error",
            session_id=str(session_id),
            error=str(e),
        )

    finally:
        if connection:
            # Get fresh service for cleanup
            async for db in get_async_session():
                repository = SQLAlchemyCollaborationRepository(db)
                service = CollaborationService(repository)

                # Leave session and broadcast
                await service.leave_session(session_id, connection.user_id)
                await service.handle_websocket_leave(connection)
                await connection_manager.disconnect(connection)
                break


async def handle_message(
    service: CollaborationService,
    connection: Connection,
    msg_type: str,
    data: dict,
) -> None:
    """Handle incoming WebSocket message."""
    session_id = connection.session_id
    user_id = connection.user_id
    username = connection.username

    if msg_type == MessageType.CODE_CHANGE:
        request = CodeChangeRequest(
            operation_type=data.get("operation_type", "insert"),
            position=data.get("position", 0),
            content=data.get("content", ""),
            length=data.get("length", 0),
        )
        version = await service.apply_code_change(session_id, user_id, request)
        if version is not None:
            await service.handle_code_change_broadcast(
                session_id, user_id, username, request, version
            )

    elif msg_type == MessageType.CURSOR_MOVE:
        request = CursorUpdateRequest(
            line=data.get("line", 0),
            column=data.get("column", 0),
        )
        await service.update_cursor(session_id, user_id, request)
        await service.handle_cursor_broadcast(session_id, user_id, username, request)

    elif msg_type == MessageType.SELECTION_CHANGE:
        request = SelectionUpdateRequest(
            start_line=data.get("start_line", 0),
            start_column=data.get("start_column", 0),
            end_line=data.get("end_line", 0),
            end_column=data.get("end_column", 0),
        )
        await service.update_selection(session_id, user_id, request)
        await service.handle_selection_broadcast(session_id, user_id, username, request)

    elif msg_type == MessageType.CHAT:
        message_text = data.get("message", "")
        if message_text:
            await service.handle_chat_broadcast(
                session_id, user_id, username, message_text
            )

    else:
        await connection.send_json(
            create_error_message(f"Unknown message type: {msg_type}", "unknown_type")
        )
