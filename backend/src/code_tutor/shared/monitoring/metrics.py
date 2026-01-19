"""Prometheus metrics for application monitoring"""

from collections.abc import Callable

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info

from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class MetricsManager:
    """Manages application metrics"""

    def __init__(self):
        self._instrumentator: Instrumentator | None = None
        self._custom_metrics: dict[str, Callable] = {}

    def setup(self, app: FastAPI) -> None:
        """Setup Prometheus metrics instrumentation"""
        self._instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=["/metrics", "/health", "/"],
            inprogress_name="http_requests_inprogress",
            inprogress_labels=True,
        )

        # Add default metrics
        self._instrumentator.add(
            metrics.default(
                metric_namespace="codetutor",
                metric_subsystem="api",
            )
        )

        # Add latency histogram
        self._instrumentator.add(
            metrics.latency(
                metric_namespace="codetutor",
                metric_subsystem="api",
                buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
        )

        # Add request size
        self._instrumentator.add(
            metrics.request_size(
                metric_namespace="codetutor",
                metric_subsystem="api",
            )
        )

        # Add response size
        self._instrumentator.add(
            metrics.response_size(
                metric_namespace="codetutor",
                metric_subsystem="api",
            )
        )

        # Add custom business metrics
        self._instrumentator.add(self._code_execution_metrics())
        self._instrumentator.add(self._ai_tutor_metrics())
        self._instrumentator.add(self._auth_metrics())

        # Instrument the app
        self._instrumentator.instrument(app)

        logger.info("Prometheus metrics instrumentation setup complete")

    def expose(self, app: FastAPI, endpoint: str = "/metrics") -> None:
        """Expose metrics endpoint"""
        if self._instrumentator:
            self._instrumentator.expose(
                app,
                endpoint=endpoint,
                include_in_schema=False,
            )
            logger.info(f"Metrics endpoint exposed at {endpoint}")

    def _code_execution_metrics(self) -> Callable[[Info], None]:
        """Custom metrics for code execution"""
        from prometheus_client import Counter, Histogram

        EXECUTION_TOTAL = Counter(
            "codetutor_code_executions_total",
            "Total code executions",
            ["language", "status"],
        )

        EXECUTION_DURATION = Histogram(
            "codetutor_code_execution_duration_seconds",
            "Code execution duration in seconds",
            ["language"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
        )

        def instrumentation(info: Info) -> None:
            if info.request.url.path == "/api/v1/execute/run":
                # Track execution count
                status = "success" if info.response.status_code == 200 else "error"
                EXECUTION_TOTAL.labels(language="python", status=status).inc()

        return instrumentation

    def _ai_tutor_metrics(self) -> Callable[[Info], None]:
        """Custom metrics for AI tutor"""
        from prometheus_client import Counter, Histogram

        CHAT_TOTAL = Counter(
            "codetutor_ai_chat_total",
            "Total AI chat messages",
            ["status"],
        )

        REVIEW_TOTAL = Counter(
            "codetutor_ai_review_total",
            "Total AI code reviews",
            ["status"],
        )

        AI_LATENCY = Histogram(
            "codetutor_ai_response_duration_seconds",
            "AI response duration in seconds",
            ["endpoint"],
            buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        def instrumentation(info: Info) -> None:
            path = info.request.url.path
            status = "success" if info.response.status_code == 200 else "error"

            if path == "/api/v1/tutor/chat":
                CHAT_TOTAL.labels(status=status).inc()
                if info.modified_duration:
                    AI_LATENCY.labels(endpoint="chat").observe(info.modified_duration)
            elif path == "/api/v1/tutor/review":
                REVIEW_TOTAL.labels(status=status).inc()
                if info.modified_duration:
                    AI_LATENCY.labels(endpoint="review").observe(info.modified_duration)

        return instrumentation

    def _auth_metrics(self) -> Callable[[Info], None]:
        """Custom metrics for authentication"""
        from prometheus_client import Counter

        AUTH_TOTAL = Counter(
            "codetutor_auth_total",
            "Total authentication attempts",
            ["action", "status"],
        )

        def instrumentation(info: Info) -> None:
            path = info.request.url.path
            status = "success" if info.response.status_code in (200, 201) else "error"

            if path == "/api/v1/auth/login":
                AUTH_TOTAL.labels(action="login", status=status).inc()
            elif path == "/api/v1/auth/register":
                AUTH_TOTAL.labels(action="register", status=status).inc()
            elif path == "/api/v1/auth/refresh":
                AUTH_TOTAL.labels(action="refresh", status=status).inc()

        return instrumentation


# Singleton instance
_metrics_manager: MetricsManager | None = None


def get_metrics_manager() -> MetricsManager:
    """Get or create metrics manager singleton"""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


def setup_prometheus(app: FastAPI) -> None:
    """Setup Prometheus metrics for FastAPI app"""
    manager = get_metrics_manager()
    manager.setup(app)
    manager.expose(app)
