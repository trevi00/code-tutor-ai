"""Debugger domain value objects."""

from enum import Enum
from typing import Any


class StepType(str, Enum):
    """Types of execution steps."""

    CALL = "call"  # Function call
    LINE = "line"  # Line execution
    RETURN = "return"  # Function return
    EXCEPTION = "exception"  # Exception raised


class VariableType(str, Enum):
    """Variable type categories."""

    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    TUPLE = "tuple"
    SET = "set"
    NONE = "none"
    OBJECT = "object"
    FUNCTION = "function"
    CLASS = "class"


class DebugStatus(str, Enum):
    """Debug session status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


def get_variable_type(value: Any) -> VariableType:
    """Determine variable type from Python value."""
    if value is None:
        return VariableType.NONE
    elif isinstance(value, bool):
        return VariableType.BOOLEAN
    elif isinstance(value, int):
        return VariableType.INT
    elif isinstance(value, float):
        return VariableType.FLOAT
    elif isinstance(value, str):
        return VariableType.STRING
    elif isinstance(value, list):
        return VariableType.LIST
    elif isinstance(value, dict):
        return VariableType.DICT
    elif isinstance(value, tuple):
        return VariableType.TUPLE
    elif isinstance(value, set):
        return VariableType.SET
    elif callable(value):
        return VariableType.FUNCTION
    elif isinstance(value, type):
        return VariableType.CLASS
    else:
        return VariableType.OBJECT


def format_variable_value(value: Any, max_length: int = 100) -> str:
    """Format variable value for display."""
    try:
        if value is None:
            return "None"
        elif isinstance(value, str):
            if len(value) > max_length:
                return f'"{value[:max_length]}..."'
            return f'"{value}"'
        elif isinstance(value, (list, tuple, set)):
            if len(value) > 10:
                items = [repr(v) for v in list(value)[:10]]
                return f"[{', '.join(items)}, ... ({len(value)} items)]"
            return repr(value)
        elif isinstance(value, dict):
            if len(value) > 5:
                items = [f"{repr(k)}: {repr(v)}" for k, v in list(value.items())[:5]]
                return "{" + ", ".join(items) + f", ... ({len(value)} items)" + "}"
            return repr(value)
        else:
            result = repr(value)
            if len(result) > max_length:
                return result[:max_length] + "..."
            return result
    except Exception:
        return "<unrepresentable>"


# Maximum limits for safety
MAX_STEPS = 1000
MAX_EXECUTION_TIME = 10  # seconds
MAX_STRING_LENGTH = 1000
MAX_COLLECTION_SIZE = 100
