"""Application services for Learning Roadmap."""

from typing import Optional, Protocol
from uuid import UUID

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
from code_tutor.roadmap.application.dto import (
    LearningPathResponse,
    LearningPathListResponse,
    ModuleResponse,
    LessonResponse,
    UserProgressResponse,
    PathProgressResponse,
    LessonProgressResponse,
    CompleteLessonRequest,
)


class XPServiceProtocol(Protocol):
    """Protocol for XP service integration."""

    async def add_xp(
        self, user_id: UUID, action: str, custom_amount: Optional[int] = None
    ) -> None:
        """Add XP for an action."""
        ...

    async def set_path_level_completed(self, user_id: UUID, level: str) -> None:
        """Set a specific path level as completed for badge tracking."""
        ...


class RoadmapService:
    """Application service for Learning Roadmap."""

    def __init__(
        self,
        path_repo: LearningPathRepository,
        module_repo: ModuleRepository,
        lesson_repo: LessonRepository,
        progress_repo: UserProgressRepository,
        xp_service: Optional[XPServiceProtocol] = None,
    ) -> None:
        self.path_repo = path_repo
        self.module_repo = module_repo
        self.lesson_repo = lesson_repo
        self.progress_repo = progress_repo
        self.xp_service = xp_service

    # ============== Path Methods ==============

    async def list_paths(
        self, user_id: Optional[UUID] = None
    ) -> LearningPathListResponse:
        """List all learning paths with optional user progress."""
        paths = await self.path_repo.list_all()
        items = []

        for path in paths:
            response = await self._path_to_response(path, user_id)
            items.append(response)

        return LearningPathListResponse(items=items, total=len(items))

    async def get_path(
        self, path_id: UUID, user_id: Optional[UUID] = None
    ) -> Optional[LearningPathResponse]:
        """Get a learning path by ID with modules and lessons."""
        path = await self.path_repo.get_by_id(path_id)
        if not path:
            return None
        return await self._path_to_response(path, user_id, include_modules=True)

    async def get_path_by_level(
        self, level: PathLevel, user_id: Optional[UUID] = None
    ) -> Optional[LearningPathResponse]:
        """Get a learning path by level."""
        path = await self.path_repo.get_by_level(level)
        if not path:
            return None
        return await self._path_to_response(path, user_id, include_modules=True)

    # ============== Module Methods ==============

    async def get_module(
        self, module_id: UUID, user_id: Optional[UUID] = None
    ) -> Optional[ModuleResponse]:
        """Get a module by ID with lessons."""
        module = await self.module_repo.get_by_id(module_id)
        if not module:
            return None
        return await self._module_to_response(module, user_id, include_lessons=True)

    async def get_path_modules(
        self, path_id: UUID, user_id: Optional[UUID] = None
    ) -> list[ModuleResponse]:
        """Get all modules for a path."""
        modules = await self.module_repo.get_by_path_id(path_id)
        return [
            await self._module_to_response(m, user_id, include_lessons=True)
            for m in modules
        ]

    # ============== Lesson Methods ==============

    async def get_lesson(
        self, lesson_id: UUID, user_id: Optional[UUID] = None
    ) -> Optional[LessonResponse]:
        """Get a lesson by ID."""
        lesson = await self.lesson_repo.get_by_id(lesson_id)
        if not lesson:
            return None
        return await self._lesson_to_response(lesson, user_id)

    async def get_module_lessons(
        self, module_id: UUID, user_id: Optional[UUID] = None
    ) -> list[LessonResponse]:
        """Get all lessons for a module."""
        lessons = await self.lesson_repo.get_by_module_id(module_id)
        return [await self._lesson_to_response(l, user_id) for l in lessons]

    # ============== Progress Methods ==============

    async def get_user_progress(self, user_id: UUID) -> UserProgressResponse:
        """Get overall user progress across all paths."""
        paths = await self.path_repo.list_all()
        path_responses = []
        total_lessons = 0
        completed_lessons = 0
        completed_paths = 0
        in_progress_paths = 0
        current_path = None
        next_lesson = None

        for path in paths:
            response = await self._path_to_response(path, user_id)
            path_responses.append(response)
            total_lessons += path.lesson_count

            if response.status == ProgressStatus.COMPLETED:
                completed_paths += 1
                completed_lessons += path.lesson_count
            elif response.status == ProgressStatus.IN_PROGRESS:
                in_progress_paths += 1
                completed_lessons += response.completed_lessons
                if current_path is None:
                    current_path = response

        # Get next lesson
        next_lesson_entity = await self.progress_repo.get_next_lesson(user_id)
        if next_lesson_entity:
            next_lesson = await self._lesson_to_response(next_lesson_entity, user_id)

        # Calculate total XP earned (simplified - based on completed lessons)
        total_xp_earned = completed_lessons * 10  # Base XP per lesson

        return UserProgressResponse(
            total_paths=len(paths),
            completed_paths=completed_paths,
            in_progress_paths=in_progress_paths,
            total_lessons=total_lessons,
            completed_lessons=completed_lessons,
            total_xp_earned=total_xp_earned,
            current_path=current_path,
            next_lesson=next_lesson,
            paths=path_responses,
        )

    async def get_path_progress(
        self, user_id: UUID, path_id: UUID
    ) -> Optional[PathProgressResponse]:
        """Get user's progress on a specific path."""
        progress = await self.progress_repo.get_path_progress(user_id, path_id)
        if not progress:
            # Return default progress
            path = await self.path_repo.get_by_id(path_id)
            if not path:
                return None
            return PathProgressResponse(
                path_id=path_id,
                status=ProgressStatus.NOT_STARTED,
                completed_lessons=0,
                total_lessons=path.lesson_count,
                completion_rate=0.0,
            )

        return PathProgressResponse(
            path_id=path_id,
            status=progress.status,
            started_at=progress.started_at,
            completed_at=progress.completed_at,
            completed_lessons=progress.completed_lessons,
            total_lessons=progress.total_lessons,
            completion_rate=progress.completion_rate,
        )

    async def start_path(self, user_id: UUID, path_id: UUID) -> PathProgressResponse:
        """Start a learning path."""
        path = await self.path_repo.get_by_id(path_id)
        if not path:
            raise ValueError("Path not found")

        progress = await self.progress_repo.get_path_progress(user_id, path_id)
        if progress:
            # Already started
            return PathProgressResponse(
                path_id=path_id,
                status=progress.status,
                started_at=progress.started_at,
                completed_at=progress.completed_at,
                completed_lessons=progress.completed_lessons,
                total_lessons=progress.total_lessons,
                completion_rate=progress.completion_rate,
            )

        # Create new progress
        progress = UserPathProgress(
            user_id=user_id,
            path_id=path_id,
            total_lessons=path.lesson_count,
        )
        progress.start()
        await self.progress_repo.save_path_progress(progress)

        # Award XP for starting a path
        if self.xp_service:
            await self.xp_service.add_xp(user_id, "path_started")

        return PathProgressResponse(
            path_id=path_id,
            status=progress.status,
            started_at=progress.started_at,
            completed_at=progress.completed_at,
            completed_lessons=progress.completed_lessons,
            total_lessons=progress.total_lessons,
            completion_rate=progress.completion_rate,
        )

    async def complete_lesson(
        self,
        user_id: UUID,
        lesson_id: UUID,
        request: CompleteLessonRequest,
    ) -> LessonProgressResponse:
        """Complete a lesson."""
        lesson = await self.lesson_repo.get_by_id(lesson_id)
        if not lesson:
            raise ValueError("Lesson not found")

        # Get or create lesson progress
        progress = await self.progress_repo.get_lesson_progress(user_id, lesson_id)
        already_completed = progress and progress.status == ProgressStatus.COMPLETED

        if not progress:
            progress = UserLessonProgress(
                user_id=user_id,
                lesson_id=lesson_id,
            )
            progress.start()

        # Mark as completed
        progress.complete(score=request.score)
        await self.progress_repo.save_lesson_progress(progress)

        # Award XP for lesson completion (only if not already completed)
        if self.xp_service and not already_completed:
            # Award lesson XP (use lesson's xp_reward)
            await self.xp_service.add_xp(
                user_id, "lesson_completed", custom_amount=lesson.xp_reward
            )

        # Update path progress
        path_completed = False
        path = None
        module = await self.module_repo.get_by_id(lesson.module_id)
        if module:
            path_progress = await self.progress_repo.get_path_progress(
                user_id, module.path_id
            )
            if path_progress:
                old_status = path_progress.status
                completed_count = await self.progress_repo.get_completed_lesson_count(
                    user_id, module.path_id
                )
                path = await self.path_repo.get_by_id(module.path_id)
                if path:
                    path_progress.update_progress(completed_count, path.lesson_count)
                    await self.progress_repo.save_path_progress(path_progress)

                    # Check if path was just completed
                    if (
                        path_progress.status == ProgressStatus.COMPLETED
                        and old_status != ProgressStatus.COMPLETED
                    ):
                        path_completed = True

        # Award XP for path completion and set path level flag
        if self.xp_service and path_completed and path:
            await self.xp_service.add_xp(user_id, "path_completed")
            # Set the path level completion flag for badges
            await self.xp_service.set_path_level_completed(user_id, path.level.value)

        return LessonProgressResponse(
            lesson_id=lesson_id,
            status=progress.status,
            started_at=progress.started_at,
            completed_at=progress.completed_at,
            score=progress.score,
            attempts=progress.attempts,
        )

    async def get_next_lesson(
        self, user_id: UUID, path_id: Optional[UUID] = None
    ) -> Optional[LessonResponse]:
        """Get the next lesson for user to complete."""
        lesson = await self.progress_repo.get_next_lesson(user_id, path_id)
        if not lesson:
            return None
        return await self._lesson_to_response(lesson, user_id)

    # ============== Helper Methods ==============

    async def _path_to_response(
        self,
        path: LearningPath,
        user_id: Optional[UUID] = None,
        include_modules: bool = False,
    ) -> LearningPathResponse:
        """Convert path entity to response DTO."""
        modules = []
        if include_modules:
            for module in path.modules:
                modules.append(
                    await self._module_to_response(module, user_id, include_lessons=True)
                )

        # Get user progress
        status = ProgressStatus.NOT_STARTED
        completed_lessons = 0
        completion_rate = 0.0
        started_at = None
        completed_at = None

        if user_id:
            progress = await self.progress_repo.get_path_progress(user_id, path.id)
            if progress:
                status = progress.status
                completed_lessons = progress.completed_lessons
                completion_rate = progress.completion_rate
                started_at = progress.started_at
                completed_at = progress.completed_at

        return LearningPathResponse(
            id=path.id,
            level=path.level,
            level_display=path.level.display_name,
            title=path.title,
            description=path.description,
            icon=path.icon,
            order=path.order,
            estimated_hours=path.estimated_hours,
            module_count=path.module_count,
            lesson_count=path.lesson_count,
            total_xp=path.total_xp,
            prerequisites=path.prerequisites,
            modules=modules,
            status=status,
            completed_lessons=completed_lessons,
            completion_rate=completion_rate,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def _module_to_response(
        self,
        module: Module,
        user_id: Optional[UUID] = None,
        include_lessons: bool = False,
    ) -> ModuleResponse:
        """Convert module entity to response DTO."""
        lessons = []
        completed_count = 0

        if include_lessons:
            for lesson in module.lessons:
                lesson_response = await self._lesson_to_response(lesson, user_id)
                lessons.append(lesson_response)
                if lesson_response.status == ProgressStatus.COMPLETED:
                    completed_count += 1

        completion_rate = 0.0
        if module.lesson_count > 0:
            completion_rate = (completed_count / module.lesson_count) * 100

        return ModuleResponse(
            id=module.id,
            path_id=module.path_id,
            title=module.title,
            description=module.description,
            order=module.order,
            lesson_count=module.lesson_count,
            total_xp=module.total_xp,
            estimated_minutes=module.estimated_minutes,
            lessons=lessons,
            completed_lessons=completed_count,
            completion_rate=completion_rate,
        )

    async def _lesson_to_response(
        self, lesson: Lesson, user_id: Optional[UUID] = None
    ) -> LessonResponse:
        """Convert lesson entity to response DTO."""
        status = None
        completed_at = None
        score = None

        if user_id:
            progress = await self.progress_repo.get_lesson_progress(user_id, lesson.id)
            if progress:
                status = progress.status
                completed_at = progress.completed_at
                score = progress.score

        return LessonResponse(
            id=lesson.id,
            module_id=lesson.module_id,
            title=lesson.title,
            description=lesson.description,
            lesson_type=lesson.lesson_type,
            content=lesson.content,
            content_id=lesson.content_id,
            order=lesson.order,
            xp_reward=lesson.xp_reward,
            estimated_minutes=lesson.estimated_minutes,
            status=status,
            completed_at=completed_at,
            score=score,
        )
