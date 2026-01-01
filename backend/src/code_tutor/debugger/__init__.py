"""Debugger module.

Provides step-by-step code debugging with:
- Line-by-line execution tracing
- Variable state inspection
- Call stack visualization
- Breakpoint support
"""

from .interface.routes import router

__all__ = ["router"]
