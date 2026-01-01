"""Collaboration repository implementation."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from code_tutor.collaboration.domain.entities import (
    CodeChange,
    CollaborationSession,
    Participant,
)
from code_tutor.collaboration.domain.repository import CollaborationRepository
from code_tutor.collaboration.domain.value_objects import (
    CodeOperation,
    CursorPosition,
    OperationType,
    SelectionRange,
    SessionStatus,
)
from code_tutor.collaboration.infrastructure.models import (
    CodeChangeModel,
    CollaborationSessionModel,
    SessionParticipantModel,
)


class SQLAlchemyCollaborationRepository(CollaborationRepository):
    """SQLAlchemy implementation of CollaborationRepository."""

    def __init__(self, session: AsyncSession):
        self.session = session

    def _model_to_entity(self, model: CollaborationSessionModel) -> CollaborationSession:
        """Convert SQLAlchemy model to domain entity."""
        participants = []
        for p in model.participants:
            cursor = None
            if p.cursor_position:
                cursor = CursorPosition.from_dict(p.cursor_position)

            selection = None
            if p.selection_range:
                selection = SelectionRange.from_dict(p.selection_range)

            participants.append(
                Participant(
                    id=p.id,
                    user_id=p.user_id,
                    session_id=p.session_id,
                    username=p.username,
                    cursor_position=cursor,
                    selection_range=selection,
                    is_active=p.is_active,
                    color=p.color,
                    joined_at=p.joined_at,
                )
            )

        return CollaborationSession(
            id=model.id,
            problem_id=model.problem_id,
            host_id=model.host_id,
            title=model.title,
            status=SessionStatus(model.status),
            code_content=model.code_content,
            language=model.language,
            version=model.version,
            participants=participants,
            max_participants=model.max_participants,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )

    def _entity_to_model(
        self, entity: CollaborationSession, existing_model: CollaborationSessionModel | None = None
    ) -> CollaborationSessionModel:
        """Convert domain entity to SQLAlchemy model."""
        if existing_model:
            model = existing_model
            model.title = entity.title
            model.status = entity.status.value
            model.code_content = entity.code_content
            model.language = entity.language
            model.version = entity.version
            model.updated_at = entity.updated_at
        else:
            model = CollaborationSessionModel(
                id=entity.id,
                problem_id=entity.problem_id,
                host_id=entity.host_id,
                title=entity.title,
                status=entity.status.value,
                code_content=entity.code_content,
                language=entity.language,
                version=entity.version,
                max_participants=entity.max_participants,
                created_at=entity.created_at,
                updated_at=entity.updated_at,
            )

        return model

    async def get_by_id(self, session_id: UUID) -> CollaborationSession | None:
        """Get session by ID."""
        query = (
            select(CollaborationSessionModel)
            .options(selectinload(CollaborationSessionModel.participants))
            .where(CollaborationSessionModel.id == session_id)
        )
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model:
            return self._model_to_entity(model)
        return None

    async def save(self, session: CollaborationSession) -> CollaborationSession:
        """Save or update a session."""
        # Check if exists
        query = select(CollaborationSessionModel).where(
            CollaborationSessionModel.id == session.id
        )
        result = await self.session.execute(query)
        existing = result.scalar_one_or_none()

        if existing:
            model = self._entity_to_model(session, existing)
            # Update participants
            await self._sync_participants(session, existing)
        else:
            model = self._entity_to_model(session)
            self.session.add(model)

        await self.session.commit()
        await self.session.refresh(model)

        # Reload with participants
        return await self.get_by_id(session.id)  # type: ignore

    async def _sync_participants(
        self, entity: CollaborationSession, model: CollaborationSessionModel
    ) -> None:
        """Sync participants between entity and model."""
        existing_ids = {p.id for p in model.participants}
        entity_ids = {p.id for p in entity.participants}

        # Add new participants
        for p in entity.participants:
            if p.id not in existing_ids:
                participant_model = SessionParticipantModel(
                    id=p.id,
                    session_id=entity.id,
                    user_id=p.user_id,
                    username=p.username,
                    cursor_position=p.cursor_position.to_dict() if p.cursor_position else None,
                    selection_range=p.selection_range.to_dict() if p.selection_range else None,
                    is_active=p.is_active,
                    color=p.color,
                    joined_at=p.joined_at,
                )
                self.session.add(participant_model)
            else:
                # Update existing
                for pm in model.participants:
                    if pm.id == p.id:
                        pm.cursor_position = p.cursor_position.to_dict() if p.cursor_position else None
                        pm.selection_range = p.selection_range.to_dict() if p.selection_range else None
                        pm.is_active = p.is_active
                        break

    async def delete(self, session_id: UUID) -> None:
        """Delete a session."""
        query = select(CollaborationSessionModel).where(
            CollaborationSessionModel.id == session_id
        )
        result = await self.session.execute(query)
        model = result.scalar_one_or_none()

        if model:
            await self.session.delete(model)
            await self.session.commit()

    async def get_user_sessions(
        self, user_id: UUID, active_only: bool = True
    ) -> list[CollaborationSession]:
        """Get sessions for a user (as host or participant)."""
        # Get sessions where user is host
        host_query = (
            select(CollaborationSessionModel)
            .options(selectinload(CollaborationSessionModel.participants))
            .where(CollaborationSessionModel.host_id == user_id)
        )
        if active_only:
            host_query = host_query.where(
                CollaborationSessionModel.status != SessionStatus.CLOSED.value
            )

        result = await self.session.execute(host_query)
        host_sessions = result.scalars().all()

        # Get sessions where user is participant
        participant_query = (
            select(CollaborationSessionModel)
            .options(selectinload(CollaborationSessionModel.participants))
            .join(SessionParticipantModel)
            .where(
                SessionParticipantModel.user_id == user_id,
                SessionParticipantModel.is_active == True,
            )
        )
        if active_only:
            participant_query = participant_query.where(
                CollaborationSessionModel.status != SessionStatus.CLOSED.value
            )

        result = await self.session.execute(participant_query)
        participant_sessions = result.scalars().all()

        # Combine and deduplicate
        all_models = {m.id: m for m in list(host_sessions) + list(participant_sessions)}
        return [self._model_to_entity(m) for m in all_models.values()]

    async def get_active_sessions(self, limit: int = 10) -> list[CollaborationSession]:
        """Get active sessions."""
        query = (
            select(CollaborationSessionModel)
            .options(selectinload(CollaborationSessionModel.participants))
            .where(CollaborationSessionModel.status == SessionStatus.ACTIVE.value)
            .order_by(CollaborationSessionModel.updated_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()
        return [self._model_to_entity(m) for m in models]

    async def save_code_change(self, change: CodeChange) -> None:
        """Save a code change to history."""
        model = CodeChangeModel(
            id=change.id,
            session_id=change.session_id,
            user_id=change.user_id,
            operation=change.operation.to_dict(),
            version=change.version,
            timestamp=change.timestamp,
        )
        self.session.add(model)
        await self.session.commit()

    async def get_session_changes(
        self, session_id: UUID, from_version: int = 0
    ) -> list[CodeChange]:
        """Get code changes for a session from a specific version."""
        query = (
            select(CodeChangeModel)
            .where(
                CodeChangeModel.session_id == session_id,
                CodeChangeModel.version > from_version,
            )
            .order_by(CodeChangeModel.version)
        )

        result = await self.session.execute(query)
        models = result.scalars().all()

        return [
            CodeChange(
                id=m.id,
                session_id=m.session_id,
                user_id=m.user_id,
                operation=CodeOperation.from_dict(m.operation),
                version=m.version,
                timestamp=m.timestamp,
            )
            for m in models
        ]
