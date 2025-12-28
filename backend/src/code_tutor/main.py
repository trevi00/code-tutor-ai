"""FastAPI main application"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from code_tutor.shared.config import get_settings
from code_tutor.shared.exception_handlers import register_exception_handlers
from code_tutor.shared.infrastructure.database import close_db, init_db
from code_tutor.shared.infrastructure.logging import configure_logging, get_logger
from code_tutor.shared.infrastructure.redis import close_redis
from code_tutor.shared.api_response import success_response

# Import routers
from code_tutor.identity.interface.routes import router as auth_router
from code_tutor.learning.interface.routes import router as learning_router
from code_tutor.tutor.interface.routes import router as tutor_router
from code_tutor.execution.interface.routes import router as execution_router

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

    # Register exception handlers (PRD-compliant response format)
    register_exception_handlers(app)

    # Register routers
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(learning_router, prefix="/api/v1")
    app.include_router(tutor_router, prefix="/api/v1")
    app.include_router(execution_router, prefix="/api/v1")

    # Health check endpoint
    @app.get("/api/health", tags=["Health"])
    async def health_check() -> dict:
        """Health check endpoint"""
        return success_response({
            "status": "healthy",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
        })

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """Root endpoint"""
        return success_response({
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs" if settings.DEBUG else "disabled",
        })

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
