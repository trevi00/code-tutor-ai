"""Tests for ML Pipeline Models and Cache."""

from datetime import date, datetime
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_tutor.ml.pipeline.models import (
    DailyStatsModel,
    UserInteractionModel,
    ModelTrainingLogModel,
    CodeQualityAnalysisModel,
    QualityTrendModel,
)


@pytest_asyncio.fixture
async def async_engine():
    """Create async engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.close()

    async with engine.begin() as conn:
        # Create required tables
        await conn.execute(
            text("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT,
                hashed_password TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT
            )
        """)
        )

        await conn.execute(
            text("""
            CREATE TABLE problems (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                difficulty TEXT,
                category TEXT,
                created_at TEXT
            )
        """)
        )

        await conn.execute(
            text("""
            CREATE TABLE submissions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                problem_id TEXT,
                code TEXT,
                status TEXT,
                created_at TEXT
            )
        """)
        )

        # Daily stats table
        await conn.execute(
            text("""
            CREATE TABLE daily_stats (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                stats_date TEXT NOT NULL,
                problems_attempted INTEGER DEFAULT 0,
                problems_solved INTEGER DEFAULT 0,
                total_submissions INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                avg_time_to_solve_ms REAL DEFAULT 0.0,
                avg_memory_usage_mb REAL DEFAULT 0.0,
                easy_solved INTEGER DEFAULT 0,
                medium_solved INTEGER DEFAULT 0,
                hard_solved INTEGER DEFAULT 0,
                categories_attempted INTEGER DEFAULT 0,
                category_breakdown TEXT DEFAULT '{}',
                streak_days INTEGER DEFAULT 0,
                study_minutes INTEGER DEFAULT 0,
                is_active_day INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(user_id, stats_date)
            )
        """)
        )

        # User interactions table
        await conn.execute(
            text("""
            CREATE TABLE user_interactions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                is_solved INTEGER DEFAULT 0,
                attempt_count INTEGER DEFAULT 0,
                best_execution_time_ms REAL,
                best_memory_usage_mb REAL,
                first_attempt_at TEXT,
                solved_at TEXT,
                time_to_solve_seconds INTEGER,
                interaction_score REAL DEFAULT 0.0,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(user_id, problem_id)
            )
        """)
        )

        # Model training logs table
        await conn.execute(
            text("""
            CREATE TABLE model_training_logs (
                id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_path TEXT,
                training_started_at TEXT NOT NULL,
                training_completed_at TEXT,
                training_samples INTEGER DEFAULT 0,
                epochs_completed INTEGER DEFAULT 0,
                metrics TEXT DEFAULT '{}',
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                is_active INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        )

        # Code quality analyses table
        await conn.execute(
            text("""
            CREATE TABLE code_quality_analyses (
                id TEXT PRIMARY KEY,
                submission_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                correctness_score INTEGER DEFAULT 0,
                efficiency_score INTEGER DEFAULT 0,
                readability_score INTEGER DEFAULT 0,
                best_practices_score INTEGER DEFAULT 0,
                overall_score INTEGER DEFAULT 0,
                overall_grade TEXT DEFAULT 'C',
                code_smells TEXT DEFAULT '[]',
                code_smells_count INTEGER DEFAULT 0,
                cyclomatic_complexity INTEGER DEFAULT 1,
                cognitive_complexity INTEGER DEFAULT 0,
                max_nesting_depth INTEGER DEFAULT 0,
                lines_of_code INTEGER DEFAULT 0,
                detected_patterns TEXT DEFAULT '[]',
                suggestions TEXT DEFAULT '[]',
                suggestions_count INTEGER DEFAULT 0,
                language TEXT DEFAULT 'python',
                analyzer_version TEXT DEFAULT '1.0.0',
                analyzed_at TEXT
            )
        """)
        )

        # Quality trends table
        await conn.execute(
            text("""
            CREATE TABLE quality_trends (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                trend_date TEXT NOT NULL,
                avg_overall_score REAL DEFAULT 0.0,
                avg_correctness REAL DEFAULT 0.0,
                avg_efficiency REAL DEFAULT 0.0,
                avg_readability REAL DEFAULT 0.0,
                avg_best_practices REAL DEFAULT 0.0,
                avg_cyclomatic REAL DEFAULT 0.0,
                avg_cognitive REAL DEFAULT 0.0,
                submissions_analyzed INTEGER DEFAULT 0,
                total_smells INTEGER DEFAULT 0,
                total_suggestions INTEGER DEFAULT 0,
                improved_count INTEGER DEFAULT 0,
                grade_distribution TEXT DEFAULT '{}',
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(user_id, trend_date)
            )
        """)
        )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine):
    """Create async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_maker() as session:
        yield session


class TestDailyStatsModel:
    """Tests for DailyStatsModel."""

    @pytest.mark.asyncio
    async def test_create_daily_stats(self, async_session):
        """Test creating daily stats record."""
        user_id = uuid4()
        stats = DailyStatsModel(
            id=uuid4(),
            user_id=user_id,
            stats_date=date.today(),
            problems_attempted=5,
            problems_solved=4,
            total_submissions=6,
            success_rate=80.0,
            avg_time_to_solve_ms=1500.0,
            easy_solved=2,
            medium_solved=2,
            hard_solved=0,
            categories_attempted=3,
            category_breakdown={"array": 2, "string": 2},
            streak_days=7,
            study_minutes=60,
            is_active_day=True,
        )

        async_session.add(stats)
        await async_session.commit()

        # Query back
        result = await async_session.get(DailyStatsModel, stats.id)

        assert result is not None
        assert result.user_id == user_id
        assert result.problems_attempted == 5
        assert result.problems_solved == 4
        assert result.success_rate == 80.0
        assert result.category_breakdown == {"array": 2, "string": 2}

    @pytest.mark.asyncio
    async def test_daily_stats_defaults(self, async_session):
        """Test daily stats default values."""
        stats = DailyStatsModel(
            id=uuid4(),
            user_id=uuid4(),
            stats_date=date.today(),
        )

        async_session.add(stats)
        await async_session.commit()

        result = await async_session.get(DailyStatsModel, stats.id)

        assert result.problems_attempted == 0
        assert result.problems_solved == 0
        assert result.success_rate == 0.0
        assert result.streak_days == 0
        assert result.is_active_day is False


class TestUserInteractionModel:
    """Tests for UserInteractionModel."""

    @pytest.mark.asyncio
    async def test_create_interaction(self, async_session):
        """Test creating user interaction record."""
        user_id = uuid4()
        problem_id = uuid4()
        interaction = UserInteractionModel(
            id=uuid4(),
            user_id=user_id,
            problem_id=problem_id,
            is_solved=True,
            attempt_count=3,
            best_execution_time_ms=150.0,
            best_memory_usage_mb=32.5,
            first_attempt_at=datetime.utcnow(),
            solved_at=datetime.utcnow(),
            time_to_solve_seconds=1800,
            interaction_score=0.85,
        )

        async_session.add(interaction)
        await async_session.commit()

        result = await async_session.get(UserInteractionModel, interaction.id)

        assert result is not None
        assert result.user_id == user_id
        assert result.problem_id == problem_id
        assert result.is_solved is True
        assert result.attempt_count == 3
        assert result.interaction_score == 0.85

    @pytest.mark.asyncio
    async def test_interaction_defaults(self, async_session):
        """Test interaction default values."""
        interaction = UserInteractionModel(
            id=uuid4(),
            user_id=uuid4(),
            problem_id=uuid4(),
        )

        async_session.add(interaction)
        await async_session.commit()

        result = await async_session.get(UserInteractionModel, interaction.id)

        assert result.is_solved is False
        assert result.attempt_count == 0
        assert result.interaction_score == 0.0
        assert result.best_execution_time_ms is None


class TestModelTrainingLogModel:
    """Tests for ModelTrainingLogModel."""

    @pytest.mark.asyncio
    async def test_create_training_log(self, async_session):
        """Test creating training log record."""
        log = ModelTrainingLogModel(
            id=uuid4(),
            model_type="ncf",
            model_version="1.0.0",
            model_path="/models/ncf_model.pt",
            training_started_at=datetime.utcnow(),
            training_completed_at=datetime.utcnow(),
            training_samples=10000,
            epochs_completed=20,
            metrics={"loss": 0.25, "auc": 0.85, "hit_rate@10": 0.72},
            status="completed",
            is_active=True,
        )

        async_session.add(log)
        await async_session.commit()

        result = await async_session.get(ModelTrainingLogModel, log.id)

        assert result is not None
        assert result.model_type == "ncf"
        assert result.model_version == "1.0.0"
        assert result.training_samples == 10000
        assert result.metrics["auc"] == 0.85
        assert result.is_active is True

    @pytest.mark.asyncio
    async def test_training_log_defaults(self, async_session):
        """Test training log default values."""
        log = ModelTrainingLogModel(
            id=uuid4(),
            model_type="lstm",
            model_version="1.0.0",
            training_started_at=datetime.utcnow(),
        )

        async_session.add(log)
        await async_session.commit()

        result = await async_session.get(ModelTrainingLogModel, log.id)

        assert result.status == "pending"
        assert result.training_samples == 0
        assert result.epochs_completed == 0
        assert result.is_active is False

    @pytest.mark.asyncio
    async def test_training_log_failed_status(self, async_session):
        """Test training log with failed status."""
        log = ModelTrainingLogModel(
            id=uuid4(),
            model_type="ncf",
            model_version="1.0.0",
            training_started_at=datetime.utcnow(),
            status="failed",
            error_message="Out of memory",
        )

        async_session.add(log)
        await async_session.commit()

        result = await async_session.get(ModelTrainingLogModel, log.id)

        assert result.status == "failed"
        assert result.error_message == "Out of memory"


class TestCodeQualityAnalysisModel:
    """Tests for CodeQualityAnalysisModel."""

    @pytest.mark.asyncio
    async def test_create_quality_analysis(self, async_session):
        """Test creating code quality analysis record."""
        analysis = CodeQualityAnalysisModel(
            id=uuid4(),
            submission_id=uuid4(),
            user_id=uuid4(),
            problem_id=uuid4(),
            correctness_score=90,
            efficiency_score=85,
            readability_score=80,
            best_practices_score=75,
            overall_score=82,
            overall_grade="B",
            code_smells=[
                {"type": "long_function", "severity": "warning", "line": 10}
            ],
            code_smells_count=1,
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            max_nesting_depth=3,
            lines_of_code=45,
            detected_patterns=["two-pointers", "binary-search"],
            suggestions=[
                {"type": "efficiency", "message": "Consider using hash map", "priority": "medium"}
            ],
            suggestions_count=1,
            language="python",
            analyzer_version="1.0.0",
        )

        async_session.add(analysis)
        await async_session.commit()

        result = await async_session.get(CodeQualityAnalysisModel, analysis.id)

        assert result is not None
        assert result.overall_score == 82
        assert result.overall_grade == "B"
        assert result.code_smells_count == 1
        assert "two-pointers" in result.detected_patterns
        assert result.cyclomatic_complexity == 5

    @pytest.mark.asyncio
    async def test_quality_analysis_defaults(self, async_session):
        """Test quality analysis default values."""
        analysis = CodeQualityAnalysisModel(
            id=uuid4(),
            submission_id=uuid4(),
            user_id=uuid4(),
            problem_id=uuid4(),
        )

        async_session.add(analysis)
        await async_session.commit()

        result = await async_session.get(CodeQualityAnalysisModel, analysis.id)

        assert result.overall_score == 0
        assert result.overall_grade == "C"
        assert result.code_smells == []
        assert result.cyclomatic_complexity == 1
        assert result.language == "python"


class TestQualityTrendModel:
    """Tests for QualityTrendModel."""

    @pytest.mark.asyncio
    async def test_create_quality_trend(self, async_session):
        """Test creating quality trend record."""
        trend = QualityTrendModel(
            id=uuid4(),
            user_id=uuid4(),
            trend_date=date.today(),
            avg_overall_score=78.5,
            avg_correctness=85.0,
            avg_efficiency=72.0,
            avg_readability=80.0,
            avg_best_practices=77.0,
            avg_cyclomatic=4.5,
            avg_cognitive=6.2,
            submissions_analyzed=10,
            total_smells=5,
            total_suggestions=12,
            improved_count=7,
            grade_distribution={"A": 2, "B": 5, "C": 3},
        )

        async_session.add(trend)
        await async_session.commit()

        result = await async_session.get(QualityTrendModel, trend.id)

        assert result is not None
        assert result.avg_overall_score == 78.5
        assert result.submissions_analyzed == 10
        assert result.improved_count == 7
        assert result.grade_distribution["B"] == 5

    @pytest.mark.asyncio
    async def test_quality_trend_defaults(self, async_session):
        """Test quality trend default values."""
        trend = QualityTrendModel(
            id=uuid4(),
            user_id=uuid4(),
            trend_date=date.today(),
        )

        async_session.add(trend)
        await async_session.commit()

        result = await async_session.get(QualityTrendModel, trend.id)

        assert result.avg_overall_score == 0.0
        assert result.submissions_analyzed == 0
        assert result.improved_count == 0


class TestMLConfigIntegration:
    """Integration tests for ML configuration."""

    def test_ml_config_paths(self):
        """Test ML config path settings."""
        from code_tutor.ml.config import MLConfig

        config = MLConfig()

        assert config.MODEL_CACHE_DIR is not None
        assert config.NCF_MODEL_PATH is not None
        assert config.LSTM_MODEL_PATH is not None

    def test_ml_config_model_settings(self):
        """Test ML config model settings."""
        from code_tutor.ml.config import MLConfig

        config = MLConfig()

        # NCF settings
        assert config.NCF_EMBEDDING_DIM > 0
        assert len(config.NCF_HIDDEN_LAYERS) > 0

        # LSTM settings
        assert config.LSTM_HIDDEN_SIZE > 0
        assert config.LSTM_NUM_LAYERS >= 1
        assert config.LSTM_SEQUENCE_LENGTH > 0

        # RAG settings
        assert config.RAG_TOP_K > 0
        assert 0 <= config.RAG_SIMILARITY_THRESHOLD <= 1

    def test_ml_config_embedding_settings(self):
        """Test ML config embedding settings."""
        from code_tutor.ml.config import MLConfig

        config = MLConfig()

        assert config.EMBEDDING_DIMENSION > 0
        assert config.CODE_EMBEDDING_DIMENSION > 0
        assert len(config.EMBEDDING_MODEL) > 0
        assert len(config.CODE_EMBEDDING_MODEL) > 0


class TestRecommendationCacheMocked:
    """Tests for RecommendationCache with mocking."""

    def test_cache_class_exists(self):
        """Test RecommendationCache class can be imported."""
        from code_tutor.ml.pipeline.cache import RecommendationCache

        # Just test that the class exists and can be imported
        assert RecommendationCache is not None

    @pytest.mark.asyncio
    async def test_cache_create_method(self):
        """Test RecommendationCache has create method."""
        from code_tutor.ml.pipeline.cache import RecommendationCache

        # Test that create is an async method
        assert hasattr(RecommendationCache, "create")
        assert callable(RecommendationCache.create)
