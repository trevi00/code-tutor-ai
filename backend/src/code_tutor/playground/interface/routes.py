"""Playground API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status

from code_tutor.identity.application.dto import UserResponse
from code_tutor.identity.interface.dependencies import get_current_user
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

router = APIRouter(prefix="/playground", tags=["Playground"])


async def get_playground_service(db=Depends(get_async_session)) -> PlaygroundService:
    """Get playground service with repositories."""
    playground_repo = SQLAlchemyPlaygroundRepository(db)
    history_repo = SQLAlchemyExecutionHistoryRepository(db)
    settings = get_settings()
    return PlaygroundService(
        playground_repo,
        history_repo,
        use_docker=settings.ENVIRONMENT not in ("development", "test"),
    )


async def get_template_service(db=Depends(get_async_session)) -> TemplateService:
    """Get template service with repository."""
    template_repo = SQLAlchemyTemplateRepository(db)
    return TemplateService(template_repo)


# === Playground CRUD ===


@router.post(
    "",
    response_model=PlaygroundDetailResponse,
    summary="플레이그라운드 생성",
    description="새로운 코드 플레이그라운드를 생성합니다. 언어와 초기 코드를 지정할 수 있습니다.",
    responses={
        200: {"description": "플레이그라운드 생성 성공"},
        401: {"description": "인증 필요"},
    },
)
async def create_playground(
    request: CreatePlaygroundRequest,
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundDetailResponse:
    """Create a new playground."""
    return await service.create_playground(request, current_user.id)


@router.get(
    "/mine",
    response_model=PlaygroundListResponse,
    summary="내 플레이그라운드 목록",
    description="현재 사용자가 생성한 플레이그라운드 목록을 조회합니다.",
    responses={
        200: {"description": "플레이그라운드 목록 반환"},
        401: {"description": "인증 필요"},
    },
)
async def list_my_playgrounds(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: UserResponse = Depends(get_current_user),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List user's playgrounds."""
    return await service.get_user_playgrounds(current_user.id, limit, offset)


@router.get(
    "/public",
    response_model=PlaygroundListResponse,
    summary="공개 플레이그라운드 목록",
    description="공개된 플레이그라운드 목록을 조회합니다. 언어별 필터링이 가능합니다.",
    responses={
        200: {"description": "공개 플레이그라운드 목록 반환"},
    },
)
async def list_public_playgrounds(
    language: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List public playgrounds."""
    return await service.get_public_playgrounds(language, limit, offset)


@router.get(
    "/popular",
    response_model=PlaygroundListResponse,
    summary="인기 플레이그라운드 목록",
    description="포크 수가 많은 인기 플레이그라운드 목록을 조회합니다.",
    responses={
        200: {"description": "인기 플레이그라운드 목록 반환"},
    },
)
async def list_popular_playgrounds(
    limit: int = Query(default=10, ge=1, le=50),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """List popular playgrounds."""
    return await service.get_popular_playgrounds(limit)


@router.get(
    "/search",
    response_model=PlaygroundListResponse,
    summary="플레이그라운드 검색",
    description="제목이나 설명으로 공개 플레이그라운드를 검색합니다.",
    responses={
        200: {"description": "검색 결과 반환"},
    },
)
async def search_playgrounds(
    q: str = Query(..., min_length=1, max_length=100),
    language: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    service: PlaygroundService = Depends(get_playground_service),
) -> PlaygroundListResponse:
    """Search public playgrounds."""
    return await service.search_playgrounds(q, language, limit)


@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="지원 언어 목록",
    description="플레이그라운드에서 지원하는 프로그래밍 언어 목록을 조회합니다.",
    responses={
        200: {"description": "지원 언어 목록 반환"},
    },
)
async def list_languages(
    service: PlaygroundService = Depends(get_playground_service),
) -> LanguagesResponse:
    """Get list of supported languages."""
    return service.get_supported_languages()


@router.get(
    "/default-code",
    summary="기본 코드 조회",
    description="특정 언어의 기본 시작 코드를 조회합니다.",
    responses={
        200: {"description": "기본 코드 반환"},
    },
)
async def get_default_code(
    language: str = Query(default="python"),
    service: PlaygroundService = Depends(get_playground_service),
) -> dict:
    """Get default code for a language."""
    code = service.get_default_code(language)
    return {"language": language, "code": code}


@router.get(
    "/share/{share_code}",
    response_model=PlaygroundDetailResponse,
    summary="공유 코드로 조회",
    description="공유 코드를 통해 플레이그라운드를 조회합니다. 비공개 플레이그라운드도 공유 코드로 접근 가능합니다.",
    responses={
        200: {"description": "플레이그라운드 상세 정보"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.get(
    "/{playground_id}",
    response_model=PlaygroundDetailResponse,
    summary="플레이그라운드 상세 조회",
    description="플레이그라운드 ID로 상세 정보를 조회합니다. 비공개 플레이그라운드는 소유자만 조회 가능합니다.",
    responses={
        200: {"description": "플레이그라운드 상세 정보"},
        403: {"description": "접근 권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.put(
    "/{playground_id}",
    response_model=PlaygroundDetailResponse,
    summary="플레이그라운드 수정",
    description="플레이그라운드의 코드, 제목, 설명, 공개 여부 등을 수정합니다.",
    responses={
        200: {"description": "수정 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "수정 권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.delete(
    "/{playground_id}",
    summary="플레이그라운드 삭제",
    description="플레이그라운드를 삭제합니다. 소유자만 삭제할 수 있습니다.",
    responses={
        200: {"description": "삭제 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "삭제 권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.post(
    "/{playground_id}/execute",
    response_model=ExecutionResponse,
    summary="플레이그라운드 코드 실행",
    description="플레이그라운드의 코드를 샌드박스 환경에서 실행하고 결과를 반환합니다.",
    responses={
        200: {"description": "실행 결과 반환"},
        403: {"description": "실행 권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.post(
    "/{playground_id}/fork",
    response_model=PlaygroundDetailResponse,
    summary="플레이그라운드 포크",
    description="다른 사용자의 플레이그라운드를 복사하여 내 플레이그라운드로 생성합니다.",
    responses={
        200: {"description": "포크 성공"},
        401: {"description": "인증 필요"},
        403: {"description": "포크 권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.post(
    "/{playground_id}/regenerate-share-code",
    summary="공유 코드 재생성",
    description="플레이그라운드의 공유 코드를 새로 생성합니다. 기존 공유 링크는 무효화됩니다.",
    responses={
        200: {"description": "새 공유 코드 반환"},
        401: {"description": "인증 필요"},
        403: {"description": "권한 없음"},
        404: {"description": "플레이그라운드를 찾을 수 없음"},
    },
)
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


@router.get(
    "/templates/list",
    response_model=TemplateListResponse,
    summary="템플릿 목록 조회",
    description="사용 가능한 코드 템플릿 목록을 조회합니다. 카테고리와 언어로 필터링 가능합니다.",
    responses={
        200: {"description": "템플릿 목록 반환"},
    },
)
async def list_templates(
    category: str | None = None,
    language: str | None = None,
    service: TemplateService = Depends(get_template_service),
) -> TemplateListResponse:
    """List code templates."""
    return await service.get_templates(category, language)


@router.get(
    "/templates/popular",
    response_model=TemplateListResponse,
    summary="인기 템플릿 목록",
    description="사용 횟수가 많은 인기 코드 템플릿 목록을 조회합니다.",
    responses={
        200: {"description": "인기 템플릿 목록 반환"},
    },
)
async def list_popular_templates(
    limit: int = Query(default=10, ge=1, le=50),
    service: TemplateService = Depends(get_template_service),
) -> TemplateListResponse:
    """List popular templates."""
    return await service.get_popular_templates(limit)


@router.get(
    "/templates/{template_id}",
    response_model=TemplateResponse,
    summary="템플릿 상세 조회",
    description="특정 코드 템플릿의 상세 정보와 코드를 조회합니다.",
    responses={
        200: {"description": "템플릿 상세 정보"},
        404: {"description": "템플릿을 찾을 수 없음"},
    },
)
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
