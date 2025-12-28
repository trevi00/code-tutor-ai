"""AI Tutor repository implementations"""

from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from code_tutor.tutor.domain.entities import Conversation, Message
from code_tutor.tutor.domain.repository import ConversationRepository
from code_tutor.tutor.domain.value_objects import CodeContext, MessageRole
from code_tutor.tutor.infrastructure.models import ConversationModel, MessageModel


class SQLAlchemyConversationRepository(ConversationRepository):
    """SQLAlchemy implementation of ConversationRepository"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_entity(self, model: ConversationModel) -> Conversation:
        """Convert SQLAlchemy model to domain entity"""
        messages = [
            Message(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=msg.role,
                content=msg.content,
                code_context=CodeContext(**msg.code_context) if msg.code_context else None,
                tokens_used=msg.tokens_used,
            )
            for msg in model.messages
        ]

        # Restore message timestamps
        for msg, model_msg in zip(messages, model.messages):
            msg._created_at = model_msg.created_at

        conversation = Conversation(
            id=model.id,
            user_id=model.user_id,
            problem_id=model.problem_id,
            conversation_type=model.conversation_type,
            title=model.title,
            messages=messages,
            total_tokens=model.total_tokens,
            is_active=model.is_active,
        )
        conversation._created_at = model.created_at
        conversation._updated_at = model.updated_at
        return conversation

    def _to_model(self, entity: Conversation) -> ConversationModel:
        """Convert domain entity to SQLAlchemy model"""
        model = ConversationModel(
            id=entity.id,
            user_id=entity.user_id,
            problem_id=entity.problem_id,
            conversation_type=entity.conversation_type,
            title=entity.title,
            total_tokens=entity.total_tokens,
            is_active=entity.is_active,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

        model.messages = [
            MessageModel(
                id=msg.id,
                conversation_id=entity.id,
                role=msg.role,
                content=msg.content,
                code_context=self._code_context_to_dict(msg.code_context),
                tokens_used=msg.tokens_used,
                created_at=msg.created_at,
            )
            for msg in entity.messages
        ]

        return model

    def _code_context_to_dict(self, ctx: CodeContext | None) -> dict | None:
        """Convert CodeContext to dict for JSON storage"""
        if ctx is None:
            return None
        return {
            "code": ctx.code,
            "language": ctx.language,
            "problem_id": str(ctx.problem_id) if ctx.problem_id else None,
            "submission_id": str(ctx.submission_id) if ctx.submission_id else None,
        }

    async def get_by_id(self, id: UUID) -> Conversation | None:
        """Get conversation by ID with messages"""
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.id == id)
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def add(self, conversation: Conversation) -> Conversation:
        """Add a new conversation"""
        model = self._to_model(conversation)
        self._session.add(model)
        await self._session.flush()

        # Reload with messages
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.id == model.id)
        )
        result = await self._session.execute(stmt)
        saved_model = result.scalar_one()
        return self._to_entity(saved_model)

    async def update(self, conversation: Conversation) -> Conversation:
        """Update an existing conversation"""
        model = self._to_model(conversation)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def delete(self, id: UUID) -> bool:
        """Delete conversation by ID"""
        model = await self._session.get(ConversationModel, id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def get_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Conversation]:
        """Get conversations by user"""
        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def get_active_by_user(
        self,
        user_id: UUID,
        problem_id: UUID | None = None,
    ) -> Conversation | None:
        """Get active conversation for user"""
        conditions = [
            ConversationModel.user_id == user_id,
            ConversationModel.is_active == True,
        ]

        if problem_id:
            conditions.append(ConversationModel.problem_id == problem_id)

        stmt = (
            select(ConversationModel)
            .options(selectinload(ConversationModel.messages))
            .where(and_(*conditions))
            .order_by(ConversationModel.updated_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)
