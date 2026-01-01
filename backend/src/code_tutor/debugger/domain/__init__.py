"""Debugger domain layer."""

from .value_objects import (
    StepType,
    VariableType,
    DebugStatus,
    get_variable_type,
    format_variable_value,
    MAX_STEPS,
    MAX_EXECUTION_TIME,
    MAX_STRING_LENGTH,
    MAX_COLLECTION_SIZE,
)
from .entities import (
    Variable,
    StackFrame,
    ExecutionStep,
    DebugSession,
    DebugResult,
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
