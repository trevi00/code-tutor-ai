"""AI Tutor domain entities"""

from uuid import UUID

from code_tutor.shared.domain.base import AggregateRoot, Entity
from code_tutor.tutor.domain.value_objects import (
    CodeContext,
    ConversationType,
    MessageRole,
)


class Message(Entity):
    """Message entity within a conversation"""

    def __init__(
        self,
        id: UUID | None = None,
        conversation_id: UUID | None = None,
        role: MessageRole = MessageRole.USER,
        content: str = "",
        code_context: CodeContext | None = None,
        tokens_used: int = 0,
    ) -> None:
        super().__init__(id)
        self._conversation_id = conversation_id
        self._role = role
        self._content = content
        self._code_context = code_context
        self._tokens_used = tokens_used

    @property
    def conversation_id(self) -> UUID | None:
        return self._conversation_id

    @property
    def role(self) -> MessageRole:
        return self._role

    @property
    def content(self) -> str:
        return self._content

    @property
    def code_context(self) -> CodeContext | None:
        return self._code_context

    @property
    def tokens_used(self) -> int:
        return self._tokens_used


class Conversation(AggregateRoot):
    """Conversation aggregate root"""

    def __init__(
        self,
        id: UUID | None = None,
        user_id: UUID | None = None,
        problem_id: UUID | None = None,
        conversation_type: ConversationType = ConversationType.GENERAL,
        title: str = "",
        messages: list[Message] | None = None,
        total_tokens: int = 0,
        is_active: bool = True,
    ) -> None:
        super().__init__(id)
        self._user_id = user_id
        self._problem_id = problem_id
        self._conversation_type = conversation_type
        self._title = title
        self._messages = messages or []
        self._total_tokens = total_tokens
        self._is_active = is_active

    @classmethod
    def create(
        cls,
        user_id: UUID,
        conversation_type: ConversationType = ConversationType.GENERAL,
        problem_id: UUID | None = None,
        title: str = "",
    ) -> "Conversation":
        """Factory method to create a new conversation"""
        if not title:
            title = f"New {conversation_type.value.replace('_', ' ').title()}"

        return cls(
            user_id=user_id,
            problem_id=problem_id,
            conversation_type=conversation_type,
            title=title,
        )

    # Properties
    @property
    def user_id(self) -> UUID | None:
        return self._user_id

    @property
    def problem_id(self) -> UUID | None:
        return self._problem_id

    @property
    def conversation_type(self) -> ConversationType:
        return self._conversation_type

    @property
    def title(self) -> str:
        return self._title

    @property
    def messages(self) -> list[Message]:
        return self._messages.copy()

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def message_count(self) -> int:
        return len(self._messages)

    # Behavior methods
    def add_user_message(
        self,
        content: str,
        code_context: CodeContext | None = None,
    ) -> Message:
        """Add a user message to the conversation"""
        message = Message(
            conversation_id=self.id,
            role=MessageRole.USER,
            content=content,
            code_context=code_context,
        )
        self._messages.append(message)
        self._touch()
        return message

    def add_assistant_message(
        self,
        content: str,
        tokens_used: int = 0,
    ) -> Message:
        """Add an assistant response to the conversation"""
        message = Message(
            conversation_id=self.id,
            role=MessageRole.ASSISTANT,
            content=content,
            tokens_used=tokens_used,
        )
        self._messages.append(message)
        self._total_tokens += tokens_used
        self._touch()
        return message

    def update_title(self, title: str) -> None:
        """Update conversation title"""
        self._title = title
        self._touch()

    def close(self) -> None:
        """Close the conversation"""
        self._is_active = False
        self._touch()

    def get_context_messages(self, max_messages: int = 10) -> list[Message]:
        """Get recent messages for context"""
        return self._messages[-max_messages:]
