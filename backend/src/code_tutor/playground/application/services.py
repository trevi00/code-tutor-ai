"""Playground application services."""

from uuid import UUID, uuid4

from code_tutor.execution.domain.value_objects import ExecutionRequest
from code_tutor.execution.infrastructure.sandbox import DockerSandbox, MockSandbox
from code_tutor.playground.application.dto import (
    CreatePlaygroundRequest,
    ExecutePlaygroundRequest,
    ExecutionResponse,
    ForkPlaygroundRequest,
    LanguageInfo,
    LanguagesResponse,
    PlaygroundDetailResponse,
    PlaygroundListResponse,
    PlaygroundResponse,
    TemplateListResponse,
    TemplateResponse,
    UpdatePlaygroundRequest,
)
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
    DEFAULT_CODE,
    LANGUAGE_CONFIG,
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)
from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import ForbiddenError, NotFoundError
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class PlaygroundService:
    """Service for managing playgrounds."""

    def __init__(
        self,
        playground_repo: PlaygroundRepository,
        history_repo: ExecutionHistoryRepository,
        use_docker: bool = True,
    ):
        self.playground_repo = playground_repo
        self.history_repo = history_repo
        self._settings = get_settings()
        self._sandbox = DockerSandbox() if use_docker else MockSandbox()

    def _to_response(self, playground: Playground) -> PlaygroundResponse:
        """Convert entity to response DTO."""
        return PlaygroundResponse(
            id=playground.id,
            owner_id=playground.owner_id,
            title=playground.title,
            description=playground.description,
            language=playground.language.value,
            visibility=playground.visibility.value,
            share_code=playground.share_code,
            is_forked=playground.is_forked,
            forked_from_id=playground.forked_from_id,
            run_count=playground.run_count,
            fork_count=playground.fork_count,
            created_at=playground.created_at,
            updated_at=playground.updated_at,
        )

    def _to_detail_response(self, playground: Playground) -> PlaygroundDetailResponse:
        """Convert entity to detailed response DTO."""
        return PlaygroundDetailResponse(
            id=playground.id,
            owner_id=playground.owner_id,
            title=playground.title,
            description=playground.description,
            code=playground.code,
            language=playground.language.value,
            visibility=playground.visibility.value,
            share_code=playground.share_code,
            stdin=playground.stdin,
            is_forked=playground.is_forked,
            forked_from_id=playground.forked_from_id,
            run_count=playground.run_count,
            fork_count=playground.fork_count,
            created_at=playground.created_at,
            updated_at=playground.updated_at,
        )

    async def create_playground(
        self,
        request: CreatePlaygroundRequest,
        user_id: UUID,
    ) -> PlaygroundDetailResponse:
        """Create a new playground."""
        language = PlaygroundLanguage(request.language)
        visibility = PlaygroundVisibility(request.visibility)

        # Use default code if not provided
        code = request.code or DEFAULT_CODE.get(language, "")

        playground = Playground.create(
            owner_id=user_id,
            title=request.title,
            code=code,
            language=language,
            description=request.description,
            visibility=visibility,
            stdin=request.stdin,
        )

        saved = await self.playground_repo.save(playground)

        logger.info(
            "playground_created",
            playground_id=str(saved.id),
            user_id=str(user_id),
            language=language.value,
        )

        return self._to_detail_response(saved)

    async def get_playground(
        self,
        playground_id: UUID,
        user_id: UUID | None = None,
    ) -> PlaygroundDetailResponse:
        """Get playground by ID."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_view(user_id):
            raise ForbiddenError("You don't have permission to view this playground")

        return self._to_detail_response(playground)

    async def get_playground_by_share_code(
        self,
        share_code: str,
    ) -> PlaygroundDetailResponse:
        """Get playground by share code."""
        playground = await self.playground_repo.get_by_share_code(share_code)
        if not playground:
            raise NotFoundError("Playground", share_code)

        return self._to_detail_response(playground)

    async def update_playground(
        self,
        playground_id: UUID,
        request: UpdatePlaygroundRequest,
        user_id: UUID,
    ) -> PlaygroundDetailResponse:
        """Update a playground."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_edit(user_id):
            raise ForbiddenError("You don't have permission to edit this playground")

        language = PlaygroundLanguage(request.language) if request.language else None
        visibility = PlaygroundVisibility(request.visibility) if request.visibility else None

        playground.update(
            title=request.title,
            description=request.description,
            code=request.code,
            language=language,
            visibility=visibility,
            stdin=request.stdin,
        )

        saved = await self.playground_repo.save(playground)
        return self._to_detail_response(saved)

    async def delete_playground(
        self,
        playground_id: UUID,
        user_id: UUID,
    ) -> None:
        """Delete a playground."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_edit(user_id):
            raise ForbiddenError("You don't have permission to delete this playground")

        await self.playground_repo.delete(playground_id)

        logger.info(
            "playground_deleted",
            playground_id=str(playground_id),
            user_id=str(user_id),
        )

    async def fork_playground(
        self,
        playground_id: UUID,
        user_id: UUID,
        request: ForkPlaygroundRequest | None = None,
    ) -> PlaygroundDetailResponse:
        """Fork a playground."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_view(user_id):
            raise ForbiddenError("You don't have permission to fork this playground")

        # Create fork
        forked = playground.fork(user_id)
        if request and request.title:
            forked.title = request.title

        # Save both (original with incremented fork_count, and new fork)
        await self.playground_repo.save(playground)
        saved = await self.playground_repo.save(forked)

        logger.info(
            "playground_forked",
            original_id=str(playground_id),
            fork_id=str(saved.id),
            user_id=str(user_id),
        )

        return self._to_detail_response(saved)

    async def execute_playground(
        self,
        playground_id: UUID,
        request: ExecutePlaygroundRequest,
        user_id: UUID | None = None,
    ) -> ExecutionResponse:
        """Execute playground code."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_view(user_id):
            raise ForbiddenError("You don't have permission to run this playground")

        # Use request code or playground code
        code = request.code or playground.code
        stdin = request.stdin or playground.stdin

        # Execute code
        exec_request = ExecutionRequest(
            code=code,
            language=playground.language.value,
            stdin=stdin,
            timeout_seconds=request.timeout_seconds,
            memory_limit_mb=self._settings.SANDBOX_MEMORY_LIMIT_MB,
            cpu_limit=self._settings.SANDBOX_CPU_LIMIT,
        )

        result = await self._sandbox.execute(exec_request)

        # Update run count
        playground.increment_run_count()
        await self.playground_repo.save(playground)

        # Save execution history
        history = ExecutionHistory(
            id=uuid4(),
            playground_id=playground_id,
            user_id=user_id,
            code=code,
            stdin=stdin,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            is_success=result.is_success,
        )
        await self.history_repo.save(history)

        logger.info(
            "playground_executed",
            playground_id=str(playground_id),
            user_id=str(user_id) if user_id else "anonymous",
            status=result.status.value,
        )

        return ExecutionResponse(
            execution_id=result.execution_id,
            status=result.status.value,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
            is_success=result.is_success,
        )

    async def get_user_playgrounds(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> PlaygroundListResponse:
        """Get user's playgrounds."""
        playgrounds = await self.playground_repo.get_user_playgrounds(
            user_id, limit, offset
        )
        return PlaygroundListResponse(
            playgrounds=[self._to_response(p) for p in playgrounds],
            total=len(playgrounds),
        )

    async def get_public_playgrounds(
        self,
        language: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> PlaygroundListResponse:
        """Get public playgrounds."""
        lang = PlaygroundLanguage(language) if language else None
        playgrounds = await self.playground_repo.get_public_playgrounds(
            lang, limit, offset
        )
        return PlaygroundListResponse(
            playgrounds=[self._to_response(p) for p in playgrounds],
            total=len(playgrounds),
        )

    async def get_popular_playgrounds(
        self,
        limit: int = 10,
    ) -> PlaygroundListResponse:
        """Get popular playgrounds."""
        playgrounds = await self.playground_repo.get_popular_playgrounds(limit)
        return PlaygroundListResponse(
            playgrounds=[self._to_response(p) for p in playgrounds],
            total=len(playgrounds),
        )

    async def search_playgrounds(
        self,
        query: str,
        language: str | None = None,
        limit: int = 20,
    ) -> PlaygroundListResponse:
        """Search public playgrounds."""
        lang = PlaygroundLanguage(language) if language else None
        playgrounds = await self.playground_repo.search_playgrounds(
            query, lang, limit
        )
        return PlaygroundListResponse(
            playgrounds=[self._to_response(p) for p in playgrounds],
            total=len(playgrounds),
        )

    async def regenerate_share_code(
        self,
        playground_id: UUID,
        user_id: UUID,
    ) -> str:
        """Regenerate share code for a playground."""
        playground = await self.playground_repo.get_by_id(playground_id)
        if not playground:
            raise NotFoundError("Playground", str(playground_id))

        if not playground.can_edit(user_id):
            raise ForbiddenError("You don't have permission to update this playground")

        new_code = playground.regenerate_share_code()
        await self.playground_repo.save(playground)
        return new_code

    def get_supported_languages(self) -> LanguagesResponse:
        """Get list of supported languages."""
        languages = [
            LanguageInfo(
                id=lang.value,
                display_name=config["display_name"],
                extension=config["extension"],
            )
            for lang, config in LANGUAGE_CONFIG.items()
        ]
        return LanguagesResponse(languages=languages)

    def get_default_code(self, language: str) -> str:
        """Get default code for a language."""
        lang = PlaygroundLanguage(language)
        return DEFAULT_CODE.get(lang, "")


class TemplateService:
    """Service for managing code templates."""

    def __init__(self, template_repo: TemplateRepository):
        self.template_repo = template_repo

    def _to_response(self, template: CodeTemplate) -> TemplateResponse:
        """Convert entity to response DTO."""
        return TemplateResponse(
            id=template.id,
            title=template.title,
            description=template.description,
            code=template.code,
            language=template.language.value,
            category=template.category.value,
            tags=template.tags,
            usage_count=template.usage_count,
        )

    async def get_templates(
        self,
        category: str | None = None,
        language: str | None = None,
    ) -> TemplateListResponse:
        """Get templates with optional filtering."""
        cat = TemplateCategory(category) if category else None
        lang = PlaygroundLanguage(language) if language else None

        templates = await self.template_repo.get_all(cat, lang)
        return TemplateListResponse(
            templates=[self._to_response(t) for t in templates],
            total=len(templates),
        )

    async def get_template(self, template_id: UUID) -> TemplateResponse:
        """Get a single template."""
        template = await self.template_repo.get_by_id(template_id)
        if not template:
            raise NotFoundError("Template", str(template_id))

        # Increment usage
        template.increment_usage()
        await self.template_repo.save(template)

        return self._to_response(template)

    async def get_popular_templates(self, limit: int = 10) -> TemplateListResponse:
        """Get popular templates."""
        templates = await self.template_repo.get_popular(limit)
        return TemplateListResponse(
            templates=[self._to_response(t) for t in templates],
            total=len(templates),
        )
