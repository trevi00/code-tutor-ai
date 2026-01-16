"""Health check utilities for application monitoring"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import text

from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enum"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ServiceHealth:
    """Health status for a single service"""
    name: str
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2) if self.latency_ms else None,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
        }


class HealthChecker:
    """Comprehensive health checker for all services"""

    def __init__(self):
        self._checks: dict[str, ServiceHealth] = {}

    async def check_database(self) -> ServiceHealth:
        """Check database connectivity"""
        from code_tutor.shared.infrastructure.database import get_session_context

        start = time.perf_counter()
        try:
            async with get_session_context() as session:
                await session.execute(text("SELECT 1"))
            latency = (time.perf_counter() - start) * 1000

            health = ServiceHealth(
                name="database",
                status=HealthStatus.HEALTHY if latency < 100 else HealthStatus.DEGRADED,
                latency_ms=latency,
                message="Connected" if latency < 100 else "Slow response",
            )
        except Exception as e:
            health = ServiceHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e)[:100],
            )
            logger.warning(f"Database health check failed: {e}")

        self._checks["database"] = health
        return health

    async def check_redis(self) -> ServiceHealth:
        """Check Redis connectivity"""
        from code_tutor.shared.infrastructure.redis import get_redis_client

        start = time.perf_counter()
        try:
            redis = get_redis_client()
            if redis and redis._client:
                await redis._client.ping()
                latency = (time.perf_counter() - start) * 1000

                health = ServiceHealth(
                    name="redis",
                    status=HealthStatus.HEALTHY if latency < 50 else HealthStatus.DEGRADED,
                    latency_ms=latency,
                    message="Connected" if latency < 50 else "Slow response",
                )
            else:
                health = ServiceHealth(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Not configured (using in-memory fallback)",
                )
        except Exception as e:
            health = ServiceHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=str(e)[:100],
            )
            logger.warning(f"Redis health check failed: {e}")

        self._checks["redis"] = health
        return health

    async def check_all(self) -> dict[str, Any]:
        """Run all health checks concurrently"""
        from code_tutor.shared.config import get_settings

        settings = get_settings()
        start = time.perf_counter()

        # Run checks concurrently
        db_health, redis_health = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(db_health, Exception):
            db_health = ServiceHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(db_health)[:100],
            )
        if isinstance(redis_health, Exception):
            redis_health = ServiceHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=str(redis_health)[:100],
            )

        # Determine overall status
        statuses = [db_health.status, redis_health.status]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        total_latency = (time.perf_counter() - start) * 1000

        return {
            "status": overall_status.value,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_check_time_ms": round(total_latency, 2),
            "services": {
                "database": db_health.to_dict(),
                "redis": redis_health.to_dict(),
            },
        }

    def get_cached_status(self) -> dict[str, Any]:
        """Get cached health status without running checks"""
        return {
            name: health.to_dict()
            for name, health in self._checks.items()
        }


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get or create health checker singleton"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
