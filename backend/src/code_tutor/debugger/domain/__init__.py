"""Debugger domain layer."""

from .entities import (
    DebugResult,
    DebugSession,
    ExecutionStep,
    StackFrame,
    Variable,
)
from .value_objects import (
    MAX_COLLECTION_SIZE,
    MAX_EXECUTION_TIME,
    MAX_STEPS,
    MAX_STRING_LENGTH,
    DebugStatus,
    StepType,
    VariableType,
    format_variable_value,
    get_variable_type,
)

__all__ = [
    # Value Objects
    "StepType",
    "VariableType",
    "DebugStatus",
    "get_variable_type",
    "format_variable_value",
    "MAX_STEPS",
    "MAX_EXECUTION_TIME",
    "MAX_STRING_LENGTH",
    "MAX_COLLECTION_SIZE",
    # Entities
    "Variable",
    "StackFrame",
    "ExecutionStep",
    "DebugSession",
    "DebugResult",
]
