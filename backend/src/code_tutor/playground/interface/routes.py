"""Playground API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from code_tutor.identity.interface.dependencies import get_current_user
from code_tutor.identity.application.dto import UserResponse
from code_tutor.playground.application.dto import (
    CreatePlaygroundRequest,
    ExecutePlaygroundRequest,
    ExecutionResponse,
    ForkPlaygroundRequest,
    LanguagesResponse,
    PlaygroundDetailResponse,
    PlaygroundListResponse,
    TemplateListResponse,
    TemplateResponse,
    UpdatePlaygroundRequest,
)
from code_tutor.playground.application.services import (
    PlaygroundService,
    TemplateService,
)
from code_tutor.playground.infrastructure.repository import (
    SQLAlchemyExecutionHistoryRepository,
    SQLAlchemyPlaygroundRepository,
    SQLAlchemyTemplateRepository,
)
from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import ForbiddenError, NotFoundError
from code_tutor.shared.infrastructure.database import get_async_session

router = APIRouter(prefix="/playground", tags=["playground"])


async def get_playground_service(db=Depends(get_async_session)) -> PlaygroundService:
    """Get playground service with repositories."""
    playground_repo = SQLAlchemyPlaygroundRepository(db)
    history_repo = SQLAlchemyExecutionHistoryRepository(db)
    settings = get_settings()
    return PlaygroundService(
        playground_repo,
        history_repo,
        use_docker=settings.ENVIRONMENT != "test",
    )


async def get_template_service(db=Depends(get_async_session)) -> TemplateService:
    """Get template service with repository."""
    template_repo = SQLAlchemyTemplateRepository(db)
    return TemplateService(template_repo)


# === Playground CRUD ===


@router.post("", response_model=PlaygroundDetailResponse)
async def create_playground(
    request: CreatePlaygroundRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Create a new playground."""
    return await service.create_playground(request, current_user.id)


@router.get("/mine", response_model=PlaygroundListResponse)
async def list_my_playgrounds(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List user's playgrounds."""
    return await service.get_user_playgrounds(current_user.id, limit, offset)


@router.get("/public", response_model=PlaygroundListResponse)
async def list_public_playgrounds(
    language: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List public playgrounds."""
    return await service.get_public_playgrounds(language, limit, offset)


@router.get("/popular", response_model=PlaygroundListResponse)
async def list_popular_playgrounds(
    limit: int = Query(default=10, ge=1, le=50),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List popular playgrounds."""
    return await service.get_popular_playgrounds(limit)


@router.get("/search", response_model=PlaygroundListResponse)
async def search_playgrounds(
    q: str = Query(..., min_length=1, max_length=100),
    language: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """Search public playgrounds."""
    return await service.search_playgrounds(q, language, limit)


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages(
    service: PlaygroundService = Depends(get_playground_service),
) -> LanguagesResponse:
    """Get list of supported languages."""
    return service.get_supported_languages()


@router.get("/default-code")
async def get_default_code(
    language: str = Query(default="python"),
    service: PlaygroundService = Depends(get_playground_service),
) -> dict:
    """Get default code for a language."""
    code = service.get_default_code(language)
    return {"language": language, "code": code}


@router.get("/share/{share_code}", response_model=PlaygroundDetailResponse)
async def get_playground_by_share_code(
    share_code: str,
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Get playground by share code."""
    try:
        return await service.get_playground_by_share_code(share_code)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )


@router.get("/{playground_id}", response_model=PlaygroundDetailResponse)
async def get_playground(
    playground_id: UUID,
    current_user: UserResponse | None = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Get playground by ID."""
    try:
        user_id = current_user.id if current_user else None
        return await service.get_playground(playground_id, user_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this playground",
        )


@router.put("/{playground_id}", response_model=PlaygroundDetailResponse)
async def update_playground(
    playground_id: UUID,
    request: UpdatePlaygroundRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Update a playground."""
    try:
        return await service.update_playground(playground_id, request, current_user.id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to edit this playground",
        )


@router.delete("/{playground_id}")
async def delete_playground(
    playground_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> dict:
    """Delete a playground."""
    try:
        await service.delete_playground(playground_id, current_user.id)
        return {"status": "deleted"}
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this playground",
        )


# === Playground Actions ===


@router.post("/{playground_id}/execute", response_model=ExecutionResponse)
async def execute_playground(
    playground_id: UUID,
    request: ExecutePlaygroundRequest,
    current_user: UserResponse | None = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> ExecutionResponse:
    """Execute playground code."""
    try:
        user_id = current_user.id if current_user else None
        return await service.execute_playground(playground_id, request, user_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to run this playground",
        )


@router.post("/{playground_id}/fork", response_model=PlaygroundDetailResponse)
async def fork_playground(
    playground_id: UUID,
    request: ForkPlaygroundRequest | None = None,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Fork a playground."""
    try:
        return await service.fork_playground(playground_id, current_user.id, request)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to fork this playground",
        )


@router.post("/{playground_id}/regenerate-share-code")
async def regenerate_share_code(
    playground_id: UUID,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> dict:
    """Regenerate share code for a playground."""
    try:
        new_code = await service.regenerate_share_code(playground_id, current_user.id)
        return {"share_code": new_code}
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Playground not found",
        )
    except ForbiddenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update this playground",
        )


# === Templates ===


@router.get("/templates/list", response_model=TemplateListResponse)
async def list_templates(
    category: str | None = None,
    language: str | None = None,
    service: TemplateService = Depends(get_template_service),
) -> TemplateListResponse:
    """List code templates."""
    return await service.get_templates(category, language)


@router.get("/templates/popular", response_model=TemplateListResponse)
async def list_popular_templates(
    limit: int = Query(default=10, ge=1, le=50),
    service: TemplateService = Depends(get_template_service),
) -> TemplateListResponse:
    """List popular templates."""
    return await service.get_popular_templates(limit)


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(
    template_id: UUID,
    service: TemplateService = Depends(get_template_service),
) -> TemplateResponse:
    """Get a template by ID."""
    try:
        return await service.get_template(template_id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found",
        )
