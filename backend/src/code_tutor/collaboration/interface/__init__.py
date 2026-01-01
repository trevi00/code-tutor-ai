"""Collaboration interface module."""

from code_tutor.collaboration.interface.http_routes import (
    router as http_router,
)
from code_tutor.collaboration.interface.websocket_routes import (
    router as websocket_router,
)

__all__ = [
    "http_router",
    "websocket_router",
]
