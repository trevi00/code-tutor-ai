"""Playground repository implementations."""

import json
from uuid import UUID

from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.playground.domain.entities import (
    CodeTemplate,
    ExecutionHistory,
    Playground,
)
from code_tutor.playground.domain.repository import (
    ExecutionHistoryRepository,
    PlaygroundRepository,
    TemplateRepository,
)
from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)
from code_tutor.playground.infrastructure.models import (
    CodeTemplateModel,
    ExecutionHistoryModel,
    PlaygroundModel,
)


class SQLAlchemyPlaygroundRepository(PlaygroundRepository):
    """SQLAlchemy implementation of PlaygroundRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _model_to_entity(self, model: PlaygroundModel) -> Playground:
        """Convert SQLAlchemy model to domain entity."""
        return Playground(
            id=model.id,
            owner_id=model.owner_id,
            title=model.title,
            description=model.description,
            code=model.code,
            language=PlaygroundLanguage(model.language),
            visibility=PlaygroundVisibility(model.visibility),
            share_code=model.share_code,
            stdin=model.stdin,
            is_forked=model.is_forked,
            forked_from_id=model.forked_from_id,
            run_count=model.run_count,
            fork_count=model.fork_count,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _entity_to_model(
        self,
        entity: Playground,
        existing: PlaygroundModel | None = None,
    ) -> PlaygroundModel:
        """Convert domain entity to SQLAlchemy model."""
        if existing:
            existing.title = entity.title
            existing.description = entity.description
            existing.code = entity.code
            existing.language = entity.language.value
            existing.visibility = entity.visibility.value
            existing.stdin = entity.stdin
            existing.run_count = entity.run_count
            existing.fork_count = entity.fork_count
            existing.updated_at = entity.updated_at
            return existing

        return PlaygroundModel(
            id=entity.id,
            owner_id=entity.owner_id,
            title=entity.title,
            description=entity.description,
            code=entity.code,
            language=entity.language.value,
            visibility=entity.visibility.value,
            share_code=entity.share_code,
            stdin=entity.stdin,
            is_forked=entity.is_forked,
            forked_from_id=entity.forked_from_id,
            run_count=entity.run_count,
            fork_count=entity.fork_count,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    async def get_by_id(self, playground_id: UUID) -> Playground | None:
        """Get playground by ID."""
        query = select(PlaygroundModel).where(PlaygroundModel.id == playground_id)
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()
        return self._model_to_entity(model) if model else None

    async def get_by_share_code(self, share_code: str) -> Playground | None:
        """Get playground by share code."""
        query = select(PlaygroundModel).where(
            PlaygroundModel.share_code == share_code
        )
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()
        return self._model_to_entity(model) if model else None

    async def save(self, playground: Playground) -> Playground:
        """Save or update a playground."""
        query = select(PlaygroundModel).where(PlaygroundModel.id == playground.id)
        result = await self.session.execute(query)
        existing = result.scalar_one_or_none()

        model = self._entity_to_model(playground, existing)
        if not existing:
            self.session.add(model)

        await self.session.commit()
        await self.session.refresh(model)
        return self._model_to_entity(model)

    async def delete(self, playground_id: UUID) -> None:
        """Delete a playground."""
        query = select(PlaygroundModel).where(PlaygroundModel.id == playground_id)
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model:
            await self.session.delete(model)
            await self.session.commit()

    async def get_user_playgrounds(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Playground]:
        """Get playgrounds owned by a user."""
        query = (
            select(PlaygroundModel)
            .where(PlaygroundModel.owner_id == user_id)
            .order_by(desc(PlaygroundModel.updated_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def get_public_playgrounds(
        self,
        language: PlaygroundLanguage | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Playground]:
        """Get public playgrounds."""
        query = select(PlaygroundModel).where(
            PlaygroundModel.visibility == PlaygroundVisibility.PUBLIC.value
        )

        if language:
            query = query.where(PlaygroundModel.language == language.value)

        query = (
            query.order_by(desc(PlaygroundModel.created_at))
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def get_popular_playgrounds(self, limit: int = 10) -> list[Playground]:
        """Get most popular public playgrounds."""
        query = (
            select(PlaygroundModel)
            .where(PlaygroundModel.visibility == PlaygroundVisibility.PUBLIC.value)
            .order_by(desc(PlaygroundModel.run_count))
            .limit(limit)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def search_playgrounds(
        self,
        query_text: str,
        language: PlaygroundLanguage | None = None,
        limit: int = 20,
    ) -> list[Playground]:
        """Search public playgrounds."""
        search_pattern = f"%{query_text}%"
        query = select(PlaygroundModel).where(
            PlaygroundModel.visibility == PlaygroundVisibility.PUBLIC.value,
            or_(
                PlaygroundModel.title.ilike(search_pattern),
                PlaygroundModel.description.ilike(search_pattern),
            ),
        )

        if language:
            query = query.where(PlaygroundModel.language == language.value)

        query = query.order_by(desc(PlaygroundModel.run_count)).limit(limit)

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]


class SQLAlchemyTemplateRepository(TemplateRepository):
    """SQLAlchemy implementation of TemplateRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _model_to_entity(self, model: CodeTemplateModel) -> CodeTemplate:
        """Convert SQLAlchemy model to domain entity."""
        tags = json.loads(model.tags) if model.tags else []
        return CodeTemplate(
            id=model.id,
            title=model.title,
            description=model.description,
            code=model.code,
            language=PlaygroundLanguage(model.language),
            category=TemplateCategory(model.category),
            tags=tags,
            usage_count=model.usage_count,
            created_at=model.created_at,
        )

    def _entity_to_model(self, entity: CodeTemplate) -> CodeTemplateModel:
        """Convert domain entity to SQLAlchemy model."""
        return CodeTemplateModel(
            id=entity.id,
            title=entity.title,
            description=entity.description,
            code=entity.code,
            language=entity.language.value,
            category=entity.category.value,
            tags=json.dumps(entity.tags),
            usage_count=entity.usage_count,
            created_at=entity.created_at,
        )

    async def get_by_id(self, template_id: UUID) -> CodeTemplate | None:
        """Get template by ID."""
        query = select(CodeTemplateModel).where(CodeTemplateModel.id == template_id)
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()
        return self._model_to_entity(model) if model else None

    async def get_all(
        self,
        category: TemplateCategory | None = None,
        language: PlaygroundLanguage | None = None,
    ) -> list[CodeTemplate]:
        """Get all templates with optional filtering."""
        query = select(CodeTemplateModel)

        if category:
            query = query.where(CodeTemplateModel.category == category.value)
        if language:
            query = query.where(CodeTemplateModel.language == language.value)

        query = query.order_by(desc(CodeTemplateModel.usage_count))

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def get_popular(self, limit: int = 10) -> list[CodeTemplate]:
        """Get most popular templates."""
        query = (
            select(CodeTemplateModel)
            .order_by(desc(CodeTemplateModel.usage_count))
            .limit(limit)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def save(self, template: CodeTemplate) -> CodeTemplate:
        """Save a template."""
        model = self._entity_to_model(template)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return self._model_to_entity(model)


class SQLAlchemyExecutionHistoryRepository(ExecutionHistoryRepository):
    """SQLAlchemy implementation of ExecutionHistoryRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _model_to_entity(self, model: ExecutionHistoryModel) -> ExecutionHistory:
        """Convert SQLAlchemy model to domain entity."""
        return ExecutionHistory(
            id=model.id,
            playground_id=model.playground_id,
            user_id=model.user_id,
            code=model.code,
            stdin=model.stdin,
            stdout=model.stdout,
            stderr=model.stderr,
            exit_code=model.exit_code,
            execution_time_ms=model.execution_time_ms,
            is_success=model.is_success,
            executed_at=model.executed_at,
        )

    def _entity_to_model(self, entity: ExecutionHistory) -> ExecutionHistoryModel:
        """Convert domain entity to SQLAlchemy model."""
        return ExecutionHistoryModel(
            id=entity.id,
            playground_id=entity.playground_id,
            user_id=entity.user_id,
            code=entity.code,
            stdin=entity.stdin,
            stdout=entity.stdout,
            stderr=entity.stderr,
            exit_code=entity.exit_code,
            execution_time_ms=entity.execution_time_ms,
            is_success=entity.is_success,
            executed_at=entity.executed_at,
        )

    async def save(self, history: ExecutionHistory) -> ExecutionHistory:
        """Save execution history."""
        model = self._entity_to_model(history)
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return self._model_to_entity(model)

    async def get_playground_history(
        self,
        playground_id: UUID,
        limit: int = 10,
    ) -> list[ExecutionHistory]:
        """Get recent execution history for a playground."""
        query = (
            select(ExecutionHistoryModel)
            .where(ExecutionHistoryModel.playground_id == playground_id)
            .order_by(desc(ExecutionHistoryModel.executed_at))
            .limit(limit)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def get_user_history(
        self,
        user_id: UUID,
        limit: int = 20,
    ) -> list[ExecutionHistory]:
        """Get recent execution history for a user."""
        query = (
            select(ExecutionHistoryModel)
            .where(ExecutionHistoryModel.user_id == user_id)
            .order_by(desc(ExecutionHistoryModel.executed_at))
            .limit(limit)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]
