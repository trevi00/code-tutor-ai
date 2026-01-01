"""AI Tutor domain value objects"""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

from code_tutor.shared.domain.base import ValueObject
from code_tutor.shared.exceptions import ValidationError


class MessageRole(str, Enum):
    """Message sender role"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationType(str, Enum):
    """Type of tutoring conversation"""

    GENERAL = "general"  # General Q&A
    PROBLEM_HELP = "problem_help"  # Help with specific problem
    CODE_REVIEW = "code_review"  # Code review request
    CONCEPT = "concept"  # Concept explanation


@dataclass(frozen=True)
class ConversationId(ValueObject):
    """Conversation ID value object"""

    value: UUID

    @classmethod
    def generate(cls) -> "ConversationId":
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> "ConversationId":
        try:
            return cls(value=UUID(value))
        except ValueError:
            raise ValidationError(f"Invalid conversation ID format: {value}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class MessageId(ValueObject):
    """Message ID value object"""

    value: UUID

    @classmethod
    def generate(cls) -> "MessageId":
        return cls(value=uuid4())

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class CodeContext(ValueObject):
    """Code context for tutoring"""

    code: str
    language: str = "python"
    problem_id: UUID | None = None
    submission_id: UUID | None = None
