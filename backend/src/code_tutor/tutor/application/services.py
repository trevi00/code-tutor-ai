"""AI Tutor application services"""

from uuid import UUID

from code_tutor.shared.exceptions import NotFoundError
from code_tutor.shared.infrastructure.logging import get_logger
from code_tutor.tutor.application.dto import (
    ChatRequest,
    ChatResponse,
    CodeContextRequest,
    ConversationResponse,
    ConversationSummaryResponse,
    MessageResponse,
)
from code_tutor.tutor.domain.entities import Conversation, Message
from code_tutor.tutor.domain.repository import ConversationRepository
from code_tutor.tutor.domain.value_objects import CodeContext, ConversationType
from code_tutor.tutor.infrastructure.llm_service import LLMService, get_llm_service

logger = get_logger(__name__)


class TutorService:
    """AI Tutoring service"""

    def __init__(
        self,
        conversation_repository: ConversationRepository,
        llm_service: LLMService | None = None,
    ) -> None:
        self._conversation_repo = conversation_repository
        self._llm_service = llm_service or get_llm_service()

    async def chat(
        self,
        user_id: UUID,
        request: ChatRequest,
    ) -> ChatResponse:
        """Process a chat message and generate response"""
        is_new = False

        # Get or create conversation
        if request.conversation_id:
            conversation = await self._conversation_repo.get_by_id(request.conversation_id)
            if conversation is None:
                raise NotFoundError("Conversation", str(request.conversation_id))
            if conversation.user_id != user_id:
                raise NotFoundError("Conversation", str(request.conversation_id))
        else:
            # Create new conversation
            conversation = Conversation.create(
                user_id=user_id,
                conversation_type=request.conversation_type,
                problem_id=request.problem_id,
            )
            is_new = True

        # Add user message
        code_context = None
        if request.code_context:
            code_context = CodeContext(
                code=request.code_context.code,
                language=request.code_context.language,
                problem_id=request.code_context.problem_id,
                submission_id=request.code_context.submission_id,
            )

        conversation.add_user_message(
            content=request.message,
            code_context=code_context,
        )

        # Generate AI response
        # TODO: Integrate with actual LLM service (EEVE-Korean or other)
        ai_response = await self._generate_response(conversation)

        # Add assistant message
        assistant_message = conversation.add_assistant_message(
            content=ai_response,
            tokens_used=len(ai_response.split()) * 2,  # Rough estimate
        )

        # Save conversation
        if is_new:
            saved = await self._conversation_repo.add(conversation)
        else:
            saved = await self._conversation_repo.update(conversation)

        logger.info(
            "Chat processed",
            conversation_id=str(saved.id),
            user_id=str(user_id),
            is_new=is_new,
        )

        return ChatResponse(
            conversation_id=saved.id,
            message=self._message_to_response(assistant_message),
            is_new_conversation=is_new,
        )

    async def get_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ConversationResponse:
        """Get conversation by ID"""
        conversation = await self._conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise NotFoundError("Conversation", str(conversation_id))
        if conversation.user_id != user_id:
            raise NotFoundError("Conversation", str(conversation_id))
        return self._to_response(conversation)

    async def list_conversations(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[ConversationSummaryResponse]:
        """List user's conversations"""
        conversations = await self._conversation_repo.get_by_user(user_id, limit, offset)
        return [self._to_summary(c) for c in conversations]

    async def close_conversation(
        self,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ConversationResponse:
        """Close a conversation"""
        conversation = await self._conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise NotFoundError("Conversation", str(conversation_id))
        if conversation.user_id != user_id:
            raise NotFoundError("Conversation", str(conversation_id))

        conversation.close()
        updated = await self._conversation_repo.update(conversation)
        return self._to_response(updated)

    async def _generate_response(self, conversation: Conversation) -> str:
        """Generate AI response using LLM service"""
        # Get recent context
        recent_messages = conversation.get_context_messages(max_messages=5)

        # Build user message from last message
        last_message = recent_messages[-1] if recent_messages else None
        if not last_message:
            return "무엇을 도와드릴까요?"

        user_message = last_message.content

        # Build code context if available
        code_context = None
        if last_message.code_context:
            code_context = f"```{last_message.code_context.language}\n{last_message.code_context.code}\n```"

        # Build conversation history for context
        conversation_history = [
            {"role": msg.role.value, "content": msg.content}
            for msg in recent_messages[:-1]  # Exclude the last message
        ]

        # Generate response using LLM service
        try:
            response = await self._llm_service.generate_response(
                user_message=user_message,
                context=code_context,
                conversation_history=conversation_history,
            )
            return response
        except Exception as e:
            logger.error(f"LLM service error: {e}")
            return self._generate_fallback_response(conversation.conversation_type)

    def _generate_fallback_response(self, conversation_type: ConversationType) -> str:
        """Generate fallback response when LLM service fails"""
        if conversation_type == ConversationType.CODE_REVIEW:
            return "코드를 공유해 주시면 리뷰해 드리겠습니다."
        elif conversation_type == ConversationType.PROBLEM_HELP:
            return (
                "이 문제를 함께 풀어보겠습니다.\n\n"
                "어디서 막히셨는지 구체적으로 알려주시면 더 도움을 드릴 수 있습니다!"
            )
        elif conversation_type == ConversationType.CONCEPT:
            return "어떤 개념에 대해 알고 싶으신가요?"
        else:
            return (
                "안녕하세요! 알고리즘 학습을 도와드리는 AI 튜터입니다.\n\n"
                "무엇을 도와드릴까요?"
            )

    def _to_response(self, conversation: Conversation) -> ConversationResponse:
        """Convert Conversation entity to ConversationResponse"""
        return ConversationResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            problem_id=conversation.problem_id,
            conversation_type=conversation.conversation_type.value,
            title=conversation.title,
            messages=[self._message_to_response(m) for m in conversation.messages],
            total_tokens=conversation.total_tokens,
            is_active=conversation.is_active,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
        )

    def _to_summary(self, conversation: Conversation) -> ConversationSummaryResponse:
        """Convert Conversation entity to ConversationSummaryResponse"""
        return ConversationSummaryResponse(
            id=conversation.id,
            problem_id=conversation.problem_id,
            conversation_type=conversation.conversation_type.value,
            title=conversation.title,
            message_count=conversation.message_count,
            is_active=conversation.is_active,
            updated_at=conversation.updated_at,
        )

    def _message_to_response(self, message: Message) -> MessageResponse:
        """Convert Message entity to MessageResponse"""
        code_context = None
        if message.code_context:
            code_context = CodeContextRequest(
                code=message.code_context.code,
                language=message.code_context.language,
                problem_id=message.code_context.problem_id,
                submission_id=message.code_context.submission_id,
            )

        return MessageResponse(
            id=message.id,
            role=message.role.value,
            content=message.content,
            code_context=code_context,
            tokens_used=message.tokens_used,
            created_at=message.created_at,
        )
