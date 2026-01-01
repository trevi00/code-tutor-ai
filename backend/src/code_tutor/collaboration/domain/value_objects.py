"""Collaboration domain value objects."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SessionStatus(str, Enum):
    """Collaboration session status."""

    WAITING = "waiting"  # Waiting for participants
    ACTIVE = "active"  # Session in progress
    CLOSED = "closed"  # Session ended


class OperationType(str, Enum):
    """Code change operation type."""

    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


@dataclass(frozen=True)
class CursorPosition:
    """Cursor position in editor."""

    line: int
    column: int

    def to_dict(self) -> dict[str, int]:
        return {"line": self.line, "column": self.column}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "CursorPosition":
        return cls(line=data.get("line", 1), column=data.get("column", 1))


@dataclass(frozen=True)
class SelectionRange:
    """Selection range in editor."""

    start_line: int
    start_column: int
    end_line: int
    end_column: int

    def to_dict(self) -> dict[str, int]:
        return {
            "start_line": self.start_line,
            "start_column": self.start_column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "SelectionRange":
        return cls(
            start_line=data.get("start_line", 1),
            start_column=data.get("start_column", 1),
            end_line=data.get("end_line", 1),
            end_column=data.get("end_column", 1),
        )


@dataclass(frozen=True)
class CodeOperation:
    """Code change operation for OT."""

    operation_type: OperationType
    position: int  # Character offset
    content: str  # Content to insert or delete
    length: int  # Length of operation (for delete/replace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.operation_type.value,
            "position": self.position,
            "content": self.content,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeOperation":
        return cls(
            operation_type=OperationType(data.get("type", "insert")),
            position=data.get("position", 0),
            content=data.get("content", ""),
            length=data.get("length", 0),
        )


# Participant colors for multi-cursor display
PARTICIPANT_COLORS = [
    "#FF6B6B",  # Red
    "#4ECDC4",  # Teal
    "#45B7D1",  # Blue
    "#96CEB4",  # Green
    "#FFEAA7",  # Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Gold
]


def get_participant_color(index: int) -> str:
    """Get color for participant by index."""
    return PARTICIPANT_COLORS[index % len(PARTICIPANT_COLORS)]
