"""FastAPI main application"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import ML pipeline models for database table creation
import code_tutor.ml.pipeline.models  # noqa: F401
# Import collaboration models for database table creation
import code_tutor.collaboration.infrastructure.models  # noqa: F401
# Import playground models for database table creation
import code_tutor.playground.infrastructure.models  # noqa: F401
# Import gamification models for database table creation
import code_tutor.gamification.infrastructure.models  # noqa: F401
# Import typing_practice models for database table creation
import code_tutor.typing_practice.infrastructure.models  # noqa: F401
# Import roadmap models for database table creation
import code_tutor.roadmap.infrastructure.models  # noqa: F401
from code_tutor.playground.infrastructure.template_seeder import seed_templates
from code_tutor.gamification.infrastructure.repository import SQLAlchemyBadgeRepository, SQLAlchemyUserBadgeRepository, SQLAlchemyUserStatsRepository
from code_tutor.gamification.application.services import BadgeService

from code_tutor.execution.interface.routes import router as execution_router

# Import routers
from code_tutor.identity.interface.routes import router as auth_router
from code_tutor.learning.interface.routes import router as learning_router
from code_tutor.collaboration.interface import http_router as collaboration_router
from code_tutor.collaboration.interface import websocket_router as collaboration_ws_router
from code_tutor.playground.interface import router as playground_router
from code_tutor.visualization.interface import router as visualization_router
from code_tutor.gamification.interface import router as gamification_router
from code_tutor.debugger.interface import router as debugger_router
from code_tutor.performance.interface import router as performance_router
from code_tutor.shared.api_response import success_response
from code_tutor.shared.config import get_settings
from code_tutor.shared.constants import RateLimiting
from code_tutor.shared.exception_handlers import register_exception_handlers
from code_tutor.shared.infrastructure.database import close_db, get_session_context, init_db
from code_tutor.shared.infrastructure.logging import configure_logging, get_logger
from code_tutor.shared.infrastructure.redis import close_redis, get_redis_client
from code_tutor.shared.middleware import RateLimitMiddleware
from code_tutor.shared.monitoring import setup_prometheus, get_health_checker
from code_tutor.tutor.interface.routes import router as tutor_router
from code_tutor.typing_practice.interface.routes import router as typing_practice_router
from code_tutor.roadmap.interface.routes import router as roadmap_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    # Startup
    configure_logging()
    logger.info("Starting Code Tutor AI backend...")

    settings = get_settings()
    logger.info(
        "Configuration loaded",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    # Initialize database (create tables if needed)
    # In production, use Alembic migrations instead
    if settings.ENVIRONMENT == "development":
        await init_db()
        logger.info("Database initialized")

        # Seed code templates and badges
        async with get_session_context() as session:
            count = await seed_templates(session)
            if count > 0:
                logger.info(f"Seeded {count} code templates")

            # Seed gamification badges
            badge_repo = SQLAlchemyBadgeRepository(session)
            user_badge_repo = SQLAlchemyUserBadgeRepository(session)
            user_stats_repo = SQLAlchemyUserStatsRepository(session)
            badge_service = BadgeService(badge_repo, user_badge_repo, user_stats_repo)
            badge_count = await badge_service.seed_badges()
            await session.commit()
            if badge_count > 0:
                logger.info(f"Seeded {badge_count} badges")

    logger.info("Application started successfully")

    yield

    # Shutdown
    logger.info("Shutting down...")
    await close_db()
    await close_redis()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()

    # OpenAPI tags metadata for API documentation
    openapi_tags = [
        {
            "name": "Health",
            "description": "서버 상태 확인 엔드포인트",
        },
        {
            "name": "Authentication",
            "description": "사용자 인증 및 계정 관리 (회원가입, 로그인, 토큰 갱신, 프로필)",
        },
        {
            "name": "Problems",
            "description": "알고리즘 문제 관리 및 제출 (문제 목록, 상세, 코드 제출, 채점)",
        },
        {
            "name": "AI Tutor",
            "description": "AI 기반 학습 도우미 (힌트, 코드 리뷰, 질문 답변)",
        },
        {
            "name": "Code Execution",
            "description": "코드 실행 및 테스트 (샌드박스 환경에서 안전한 코드 실행)",
        },
        {
            "name": "Typing Practice",
            "description": "타이핑 연습 (코드 타이핑 훈련, 속도/정확도 측정)",
        },
        {
            "name": "gamification",
            "description": "게이미피케이션 (XP, 레벨, 뱃지, 리더보드, 챌린지)",
        },
        {
            "name": "Roadmap",
            "description": "학습 로드맵 (체계적인 학습 경로, 진행 상황 추적)",
        },
        {
            "name": "Playground",
            "description": "코드 플레이그라운드 (자유로운 코드 작성 및 실행 환경)",
        },
        {
            "name": "Collaboration",
            "description": "실시간 협업 (페어 프로그래밍, 코드 공유)",
        },
        {
            "name": "Visualization",
            "description": "알고리즘 시각화 (정렬, 탐색 등 알고리즘 동작 시각화)",
        },
        {
            "name": "Debugger",
            "description": "코드 디버거 (단계별 실행, 변수 추적)",
        },
        {
            "name": "Performance",
            "description": "성능 분석 (시간/공간 복잡도 분석, 최적화 제안)",
        },
    ]

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
# Code Tutor AI API

AI 기반 Python 알고리즘 학습 플랫폼 API입니다.

## 주요 기능

- **🔐 인증**: JWT 기반 사용자 인증
- **📚 문제 풀이**: 난이도별 알고리즘 문제 제공 및 자동 채점
- **🤖 AI 튜터**: 힌트, 코드 리뷰, 질문 답변
- **⌨️ 타이핑 연습**: 코드 타이핑 속도/정확도 향상
- **🎮 게이미피케이션**: XP, 레벨, 뱃지, 리더보드
- **🗺️ 학습 로드맵**: 체계적인 학습 경로
- **🎨 시각화**: 알고리즘 동작 시각화
- **🐛 디버거**: 단계별 코드 실행

## 인증

대부분의 API는 JWT 토큰 인증이 필요합니다.
`Authorization: Bearer <access_token>` 헤더를 포함해주세요.

## Rate Limiting

API 요청은 분당 60회로 제한됩니다.
`X-RateLimit-Remaining` 헤더에서 남은 요청 수를 확인할 수 있습니다.
        """,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_tags=openapi_tags,
        lifespan=lifespan,
    )

    # Configure CORS with explicit allowed methods and headers
    # Note: Wildcard origins are blocked in production by config validation
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Origin",
            "X-Requested-With",
        ],
        expose_headers=[
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "Retry-After",
        ],
    )

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=RateLimiting.DEFAULT_REQUESTS_PER_MINUTE,
        burst_size=RateLimiting.DEFAULT_BURST_SIZE,
        exclude_paths=["/api/health", "/docs", "/redoc", "/openapi.json", "/"],
    )

    # Register exception handlers (PRD-compliant response format)
    register_exception_handlers(app)

    # Register routers
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(learning_router, prefix="/api/v1")
    app.include_router(tutor_router, prefix="/api/v1")
    app.include_router(execution_router, prefix="/api/v1")
    app.include_router(collaboration_router, prefix="/api/v1")
    app.include_router(collaboration_ws_router, prefix="/api/v1")
    app.include_router(playground_router, prefix="/api/v1")
    app.include_router(visualization_router, prefix="/api/v1")
    app.include_router(gamification_router, prefix="/api/v1")
    app.include_router(debugger_router, prefix="/api/v1")
    app.include_router(performance_router, prefix="/api/v1")
    app.include_router(typing_practice_router, prefix="/api/v1")
    app.include_router(roadmap_router, prefix="/api/v1")

    # Health check endpoint (enhanced with latency metrics)
    @app.get("/api/health", tags=["Health"])
    async def health_check() -> dict:
        """
        Health check endpoint with detailed service status.
        
        Returns latency metrics for each service and overall health status.
        """
        health_checker = get_health_checker()
        health_data = await health_checker.check_all()
        return success_response(health_data)

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """Root endpoint"""
        return success_response(
            {
                "name": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "docs": "/docs" if settings.DEBUG else "disabled",
            }
        )

    # Setup Prometheus metrics
    setup_prometheus(app)

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "code_tutor.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
