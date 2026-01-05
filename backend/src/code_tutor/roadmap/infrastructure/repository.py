"""Repository implementations for Learning Roadmap."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from code_tutor.roadmap.domain.entities import (
    LearningPath,
    Module,
    Lesson,
    UserPathProgress,
    UserLessonProgress,
)
from code_tutor.roadmap.domain.value_objects import (
    PathLevel,
    LessonType,
    ProgressStatus,
)
from code_tutor.roadmap.domain.repository import (
    LearningPathRepository,
    ModuleRepository,
    LessonRepository,
    UserProgressRepository,
)
from code_tutor.roadmap.infrastructure.models import (
    LearningPathModel,
    ModuleModel,
    LessonModel,
    UserPathProgressModel,
    UserLessonProgressModel,
)


class SQLAlchemyLearningPathRepository(LearningPathRepository):
    """SQLAlchemy implementation of LearningPathRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_entity(self, model: LearningPathModel) -> LearningPath:
        """Convert model to entity."""
        modules = [
            Module(
                id=UUID(m.id),
                path_id=UUID(m.path_id),
                title=m.title,
                description=m.description,
                order=m.order,
                lessons=[
                    Lesson(
                        id=UUID(l.id),
                        module_id=UUID(l.module_id),
                        title=l.title,
                        description=l.description,
                        lesson_type=LessonType(l.lesson_type),
                        content=l.content,
                        content_id=UUID(l.content_id) if l.content_id else None,
                        order=l.order,
                        xp_reward=l.xp_reward,
                        estimated_minutes=l.estimated_minutes,
                    )
                    for l in m.lessons
                ],
            )
            for m in model.modules
        ]

        prerequisites = [UUID(p.prerequisite_id) for p in model.prerequisites]

        path = LearningPath(
            id=UUID(model.id),
            level=PathLevel(model.level),
            title=model.title,
            description=model.description,
            icon=model.icon,
            order=model.order,
            estimated_hours=model.estimated_hours,
            prerequisites=prerequisites,
            modules=modules,
            is_published=model.is_published,
        )
        path._created_at = model.created_at
        path._updated_at = model.updated_at
        return path

    async def get_by_id(self, path_id: UUID) -> Optional[LearningPath]:
        """Get a learning path by ID."""
        result = await self.session.execute(
            select(LearningPathModel)
            .options(
                selectinload(LearningPathModel.modules).selectinload(ModuleModel.lessons),
                selectinload(LearningPathModel.prerequisites),
            )
            .where(LearningPathModel.id == str(path_id))
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_level(self, level: PathLevel) -> Optional[LearningPath]:
        """Get a learning path by level."""
        result = await self.session.execute(
            select(LearningPathModel)
            .options(
                selectinload(LearningPathModel.modules).selectinload(ModuleModel.lessons),
                selectinload(LearningPathModel.prerequisites),
            )
            .where(LearningPathModel.level == level.value)
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def list_all(self, include_unpublished: bool = False) -> list[LearningPath]:
        """List all learning paths."""
        query = (
            select(LearningPathModel)
            .options(
                selectinload(LearningPathModel.modules).selectinload(ModuleModel.lessons),
                selectinload(LearningPathModel.prerequisites),
            )
            .order_by(LearningPathModel.order)
        )
        if not include_unpublished:
            query = query.where(LearningPathModel.is_published == True)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def save(self, path: LearningPath) -> LearningPath:
        """Save a learning path."""
        model = LearningPathModel(
            id=str(path.id),
            level=path.level.value,
            title=path.title,
            description=path.description,
            icon=path.icon,
            order=path.order,
            estimated_hours=path.estimated_hours,
            is_published=path.is_published,
        )
        self.session.add(model)
        await self.session.flush()
        return path

    async def delete(self, path_id: UUID) -> bool:
        """Delete a learning path."""
        result = await self.session.execute(
            select(LearningPathModel).where(LearningPathModel.id == str(path_id))
        )
        model = result.scalar_one_or_none()
        if model:
            await self.session.delete(model)
            return True
        return False


class SQLAlchemyModuleRepository(ModuleRepository):
    """SQLAlchemy implementation of ModuleRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_entity(self, model: ModuleModel) -> Module:
        """Convert model to entity."""
        return Module(
            id=UUID(model.id),
            path_id=UUID(model.path_id),
            title=model.title,
            description=model.description,
            order=model.order,
            lessons=[
                Lesson(
                    id=UUID(l.id),
                    module_id=UUID(l.module_id),
                    title=l.title,
                    description=l.description,
                    lesson_type=LessonType(l.lesson_type),
                    content=l.content,
                    content_id=UUID(l.content_id) if l.content_id else None,
                    order=l.order,
                    xp_reward=l.xp_reward,
                    estimated_minutes=l.estimated_minutes,
                )
                for l in model.lessons
            ],
        )

    async def get_by_id(self, module_id: UUID) -> Optional[Module]:
        """Get a module by ID."""
        result = await self.session.execute(
            select(ModuleModel)
            .options(selectinload(ModuleModel.lessons))
            .where(ModuleModel.id == str(module_id))
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_path_id(self, path_id: UUID) -> list[Module]:
        """Get all modules for a path."""
        result = await self.session.execute(
            select(ModuleModel)
            .options(selectinload(ModuleModel.lessons))
            .where(ModuleModel.path_id == str(path_id))
            .order_by(ModuleModel.order)
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def save(self, module: Module) -> Module:
        """Save a module."""
        model = ModuleModel(
            id=str(module.id),
            path_id=str(module.path_id) if module.path_id else None,
            title=module.title,
            description=module.description,
            order=module.order,
        )
        self.session.add(model)
        await self.session.flush()
        return module


class SQLAlchemyLessonRepository(LessonRepository):
    """SQLAlchemy implementation of LessonRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _to_entity(self, model: LessonModel) -> Lesson:
        """Convert model to entity."""
        return Lesson(
            id=UUID(model.id),
            module_id=UUID(model.module_id),
            title=model.title,
            description=model.description,
            lesson_type=LessonType(model.lesson_type),
            content=model.content,
            content_id=UUID(model.content_id) if model.content_id else None,
            order=model.order,
            xp_reward=model.xp_reward,
            estimated_minutes=model.estimated_minutes,
        )

    async def get_by_id(self, lesson_id: UUID) -> Optional[Lesson]:
        """Get a lesson by ID."""
        result = await self.session.execute(
            select(LessonModel).where(LessonModel.id == str(lesson_id))
        )
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def get_by_module_id(self, module_id: UUID) -> list[Lesson]:
        """Get all lessons for a module."""
        result = await self.session.execute(
            select(LessonModel)
            .where(LessonModel.module_id == str(module_id))
            .order_by(LessonModel.order)
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def get_by_path_id(self, path_id: UUID) -> list[Lesson]:
        """Get all lessons for a path."""
        result = await self.session.execute(
            select(LessonModel)
            .join(ModuleModel)
            .where(ModuleModel.path_id == str(path_id))
            .order_by(ModuleModel.order, LessonModel.order)
        )
        return [self._to_entity(m) for m in result.scalars().all()]

    async def save(self, lesson: Lesson) -> Lesson:
        """Save a lesson."""
        model = LessonModel(
            id=str(lesson.id),
            module_id=str(lesson.module_id) if lesson.module_id else None,
            title=lesson.title,
            description=lesson.description,
            lesson_type=lesson.lesson_type.value,
            content=lesson.content,
            content_id=str(lesson.content_id) if lesson.content_id else None,
            order=lesson.order,
            xp_reward=lesson.xp_reward,
            estimated_minutes=lesson.estimated_minutes,
        )
        self.session.add(model)
        await self.session.flush()
        return lesson


class SQLAlchemyUserProgressRepository(UserProgressRepository):
    """SQLAlchemy implementation of UserProgressRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _path_progress_to_entity(self, model: UserPathProgressModel) -> UserPathProgress:
        """Convert path progress model to entity."""
        progress = UserPathProgress(
            id=UUID(model.id),
            user_id=UUID(model.user_id),
            path_id=UUID(model.path_id),
            status=ProgressStatus(model.status),
            started_at=model.started_at,
            completed_at=model.completed_at,
            completed_lessons=model.completed_lessons,
            total_lessons=model.total_lessons,
        )
        progress._created_at = model.created_at
        progress._updated_at = model.updated_at
        return progress

    def _lesson_progress_to_entity(self, model: UserLessonProgressModel) -> UserLessonProgress:
        """Convert lesson progress model to entity."""
        progress = UserLessonProgress(
            id=UUID(model.id),
            user_id=UUID(model.user_id),
            lesson_id=UUID(model.lesson_id),
            status=ProgressStatus(model.status),
            started_at=model.started_at,
            completed_at=model.completed_at,
            score=model.score,
            attempts=model.attempts,
        )
        progress._created_at = model.created_at
        progress._updated_at = model.updated_at
        return progress

    async def get_path_progress(
        self, user_id: UUID, path_id: UUID
    ) -> Optional[UserPathProgress]:
        """Get user's progress on a path."""
        result = await self.session.execute(
            select(UserPathProgressModel).where(
                UserPathProgressModel.user_id == str(user_id),
                UserPathProgressModel.path_id == str(path_id),
            )
        )
        model = result.scalar_one_or_none()
        return self._path_progress_to_entity(model) if model else None

    async def get_all_path_progress(self, user_id: UUID) -> list[UserPathProgress]:
        """Get user's progress on all paths."""
        result = await self.session.execute(
            select(UserPathProgressModel).where(
                UserPathProgressModel.user_id == str(user_id)
            )
        )
        return [self._path_progress_to_entity(m) for m in result.scalars().all()]

    async def save_path_progress(
        self, progress: UserPathProgress
    ) -> UserPathProgress:
        """Save path progress."""
        # Check if exists
        result = await self.session.execute(
            select(UserPathProgressModel).where(
                UserPathProgressModel.user_id == str(progress.user_id),
                UserPathProgressModel.path_id == str(progress.path_id),
            )
        )
        model = result.scalar_one_or_none()

        if model:
            # Update existing
            model.status = progress.status.value
            model.started_at = progress.started_at
            model.completed_at = progress.completed_at
            model.completed_lessons = progress.completed_lessons
            model.total_lessons = progress.total_lessons
        else:
            # Create new
            model = UserPathProgressModel(
                id=str(progress.id),
                user_id=str(progress.user_id),
                path_id=str(progress.path_id),
                status=progress.status.value,
                started_at=progress.started_at,
                completed_at=progress.completed_at,
                completed_lessons=progress.completed_lessons,
                total_lessons=progress.total_lessons,
            )
            self.session.add(model)

        await self.session.flush()
        return progress

    async def get_lesson_progress(
        self, user_id: UUID, lesson_id: UUID
    ) -> Optional[UserLessonProgress]:
        """Get user's progress on a lesson."""
        result = await self.session.execute(
            select(UserLessonProgressModel).where(
                UserLessonProgressModel.user_id == str(user_id),
                UserLessonProgressModel.lesson_id == str(lesson_id),
            )
        )
        model = result.scalar_one_or_none()
        return self._lesson_progress_to_entity(model) if model else None

    async def get_module_lessons_progress(
        self, user_id: UUID, module_id: UUID
    ) -> list[UserLessonProgress]:
        """Get user's progress on all lessons in a module."""
        result = await self.session.execute(
            select(UserLessonProgressModel)
            .join(LessonModel)
            .where(
                UserLessonProgressModel.user_id == str(user_id),
                LessonModel.module_id == str(module_id),
            )
        )
        return [self._lesson_progress_to_entity(m) for m in result.scalars().all()]

    async def get_path_lessons_progress(
        self, user_id: UUID, path_id: UUID
    ) -> list[UserLessonProgress]:
        """Get user's progress on all lessons in a path."""
        result = await self.session.execute(
            select(UserLessonProgressModel)
            .join(LessonModel)
            .join(ModuleModel)
            .where(
                UserLessonProgressModel.user_id == str(user_id),
                ModuleModel.path_id == str(path_id),
            )
        )
        return [self._lesson_progress_to_entity(m) for m in result.scalars().all()]

    async def save_lesson_progress(
        self, progress: UserLessonProgress
    ) -> UserLessonProgress:
        """Save lesson progress."""
        # Check if exists
        result = await self.session.execute(
            select(UserLessonProgressModel).where(
                UserLessonProgressModel.user_id == str(progress.user_id),
                UserLessonProgressModel.lesson_id == str(progress.lesson_id),
            )
        )
        model = result.scalar_one_or_none()

        if model:
            # Update existing
            model.status = progress.status.value
            model.started_at = progress.started_at
            model.completed_at = progress.completed_at
            model.score = progress.score
            model.attempts = progress.attempts
        else:
            # Create new
            model = UserLessonProgressModel(
                id=str(progress.id),
                user_id=str(progress.user_id),
                lesson_id=str(progress.lesson_id),
                status=progress.status.value,
                started_at=progress.started_at,
                completed_at=progress.completed_at,
                score=progress.score,
                attempts=progress.attempts,
            )
            self.session.add(model)

        await self.session.flush()
        return progress

    async def get_completed_lesson_count(
        self, user_id: UUID, path_id: UUID
    ) -> int:
        """Get count of completed lessons in a path."""
        result = await self.session.execute(
            select(func.count(UserLessonProgressModel.id))
            .join(LessonModel)
            .join(ModuleModel)
            .where(
                UserLessonProgressModel.user_id == str(user_id),
                ModuleModel.path_id == str(path_id),
                UserLessonProgressModel.status == ProgressStatus.COMPLETED.value,
            )
        )
        return result.scalar() or 0

    async def get_next_lesson(
        self, user_id: UUID, path_id: UUID | None = None
    ) -> Optional[Lesson]:
        """Get the next incomplete lesson for user."""
        # Get completed lesson IDs
        completed_query = (
            select(UserLessonProgressModel.lesson_id)
            .where(
                UserLessonProgressModel.user_id == str(user_id),
                UserLessonProgressModel.status == ProgressStatus.COMPLETED.value,
            )
        )

        # Find next lesson not completed
        query = (
            select(LessonModel)
            .join(ModuleModel)
            .join(LearningPathModel)
            .where(
                LessonModel.id.notin_(completed_query),
                LearningPathModel.is_published == True,
            )
            .order_by(
                LearningPathModel.order,
                ModuleModel.order,
                LessonModel.order,
            )
        )

        if path_id:
            query = query.where(ModuleModel.path_id == str(path_id))

        result = await self.session.execute(query.limit(1))
        model = result.scalar_one_or_none()

        if model:
            return Lesson(
                id=UUID(model.id),
                module_id=UUID(model.module_id),
                title=model.title,
                description=model.description,
                lesson_type=LessonType(model.lesson_type),
                content=model.content,
                content_id=UUID(model.content_id) if model.content_id else None,
                order=model.order,
                xp_reward=model.xp_reward,
                estimated_minutes=model.estimated_minutes,
            )
        return None
