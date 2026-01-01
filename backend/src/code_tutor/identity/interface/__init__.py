"""Identity interface layer"""

from code_tutor.identity.interface.dependencies import (
    get_current_active_user,
    get_current_user,
)
from code_tutor.identity.interface.routes import router as auth_router

__all__ = ["auth_router", "get_current_user", "get_current_active_user"]
