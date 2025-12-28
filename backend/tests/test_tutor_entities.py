"""Unit tests for Tutor Domain Entities"""

import pytest
from uuid import uuid4
from datetime import datetime, timezone

from code_tutor.tutor.domain.entities import Conversation, Message
from code_tutor.tutor.domain.value_objects import (
    ConversationType,
    MessageRole,
    ConversationId,
    MessageId,
    CodeContext,
)


class TestConversation:
    """Tests for Conversation entity"""

    def test_create_conversation(self):
        """Test creating a conversation"""
        user_id = uuid4()
        conversation = Conversation.create(
            user_id=user_id,
            conversation_type=ConversationType.GENERAL,
        )
        assert conversation.user_id == user_id
        assert conversation.conversation_type == ConversationType.GENERAL
        assert conversation.is_active is True

    def test_conversation_add_user_message(self):
        """Test adding user message to conversation"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.CONCEPT,
        )

        message = conversation.add_user_message(
            content="What is a binary search?",
        )

        assert len(conversation.messages) == 1
        assert message.content == "What is a binary search?"
        assert message.role == MessageRole.USER

    def test_conversation_add_assistant_message(self):
        """Test adding assistant message to conversation"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.CONCEPT,
        )

        message = conversation.add_assistant_message(
            content="Binary search is an efficient algorithm...",
            tokens_used=50,
        )

        assert len(conversation.messages) == 1
        assert message.content == "Binary search is an efficient algorithm..."
        assert message.role == MessageRole.ASSISTANT
        assert conversation.total_tokens == 50

    def test_conversation_close(self):
        """Test closing a conversation"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.GENERAL,
        )
        assert conversation.is_active is True

        conversation.close()
        assert conversation.is_active is False

    def test_conversation_types(self):
        """Test different conversation types"""
        for conv_type in ConversationType:
            conversation = Conversation.create(
                user_id=uuid4(),
                conversation_type=conv_type,
            )
            assert conversation.conversation_type == conv_type

    def test_conversation_update_title(self):
        """Test updating conversation title"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.GENERAL,
            title="Original Title",
        )
        assert conversation.title == "Original Title"

        conversation.update_title("Updated Title")
        assert conversation.title == "Updated Title"

    def test_conversation_message_count(self):
        """Test message count property"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.GENERAL,
        )
        assert conversation.message_count == 0

        conversation.add_user_message("Hello")
        conversation.add_assistant_message("Hi there!")
        assert conversation.message_count == 2

    def test_conversation_get_context_messages(self):
        """Test getting context messages"""
        conversation = Conversation.create(
            user_id=uuid4(),
            conversation_type=ConversationType.GENERAL,
        )

        for i in range(15):
            conversation.add_user_message(f"Message {i}")

        context = conversation.get_context_messages(max_messages=5)
        assert len(context) == 5


class TestMessage:
    """Tests for Message entity"""

    def test_create_user_message(self):
        """Test creating a user message"""
        conversation_id = uuid4()
        message = Message(
            conversation_id=conversation_id,
            role=MessageRole.USER,
            content="Hello, I need help",
        )
        assert message.role == MessageRole.USER
        assert message.content == "Hello, I need help"

    def test_create_assistant_message(self):
        """Test creating an assistant message"""
        conversation_id = uuid4()
        message = Message(
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content="Sure, I can help you!",
        )
        assert message.role == MessageRole.ASSISTANT
        assert message.content == "Sure, I can help you!"

    def test_message_with_code_context(self):
        """Test message with code context"""
        code_ctx = CodeContext(
            code="def foo(): pass",
            language="python",
        )
        message = Message(
            conversation_id=uuid4(),
            role=MessageRole.USER,
            content="Review this code",
            code_context=code_ctx,
        )
        assert message.code_context is not None
        assert message.code_context.code == "def foo(): pass"


class TestConversationType:
    """Tests for ConversationType enum"""

    def test_conversation_type_values(self):
        """Test conversation type values"""
        assert ConversationType.GENERAL.value == "general"
        assert ConversationType.CONCEPT.value == "concept"
        assert ConversationType.PROBLEM_HELP.value == "problem_help"
        assert ConversationType.CODE_REVIEW.value == "code_review"


class TestMessageRole:
    """Tests for MessageRole enum"""

    def test_message_role_values(self):
        """Test message role values"""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"


class TestConversationId:
    """Tests for ConversationId value object"""

    def test_generate_conversation_id(self):
        """Test generating conversation ID"""
        conv_id = ConversationId.generate()
        assert conv_id.value is not None

    def test_conversation_id_from_string(self):
        """Test creating conversation ID from string"""
        uuid_str = str(uuid4())
        conv_id = ConversationId.from_string(uuid_str)
        assert str(conv_id) == uuid_str


class TestCodeContext:
    """Tests for CodeContext value object"""

    def test_code_context_creation(self):
        """Test creating code context"""
        ctx = CodeContext(
            code="def hello(): return 'world'",
            language="python",
        )
        assert ctx.code == "def hello(): return 'world'"
        assert ctx.language == "python"

    def test_code_context_with_problem_id(self):
        """Test code context with problem ID"""
        problem_id = uuid4()
        ctx = CodeContext(
            code="x = 1",
            language="python",
            problem_id=problem_id,
        )
        assert ctx.problem_id == problem_id
