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
from code_tutor.shared.exception_handlers import register_exception_handlers
from code_tutor.shared.infrastructure.database import close_db, get_session_context, init_db
from code_tutor.shared.infrastructure.logging import configure_logging, get_logger
from code_tutor.shared.infrastructure.redis import close_redis
from code_tutor.shared.middleware import RateLimitMiddleware
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

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-based Python Algorithm Learning Platform API",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting middleware
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        burst_size=20,
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

    # Health check endpoint
    @app.get("/api/health", tags=["Health"])
    async def health_check() -> dict:
        """Health check endpoint"""
        return success_response(
            {
                "status": "healthy",
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
            }
        )

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
