"""ML Pipeline Database Models

Models for storing aggregated learning statistics and user interactions.
"""

from datetime import date, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from code_tutor.shared.infrastructure.database import Base


class DailyStatsModel(Base):
    """Daily learning statistics for each user.

    Used for LSTM learning prediction model.
    Aggregated from submissions table daily.
    """

    __tablename__ = "daily_stats"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stats_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Problem attempt statistics
    problems_attempted: Mapped[int] = mapped_column(Integer, default=0)
    problems_solved: Mapped[int] = mapped_column(Integer, default=0)
    total_submissions: Mapped[int] = mapped_column(Integer, default=0)

    # Success metrics
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    avg_time_to_solve_ms: Mapped[float] = mapped_column(Float, default=0.0)
    avg_memory_usage_mb: Mapped[float] = mapped_column(Float, default=0.0)

    # Difficulty distribution
    easy_solved: Mapped[int] = mapped_column(Integer, default=0)
    medium_solved: Mapped[int] = mapped_column(Integer, default=0)
    hard_solved: Mapped[int] = mapped_column(Integer, default=0)

    # Category diversity
    categories_attempted: Mapped[int] = mapped_column(Integer, default=0)
    category_breakdown: Mapped[dict] = mapped_column(JSON, default={})

    # Streak and engagement
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    study_minutes: Mapped[int] = mapped_column(Integer, default=0)
    is_active_day: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("user_id", "stats_date", name="uq_daily_stats_user_date"),
        Index("ix_daily_stats_user_date", "user_id", "stats_date"),
    )


class UserInteractionModel(Base):
    """User-Problem interaction records for NCF model.

    Stores aggregated interaction data between users and problems.
    Used for collaborative filtering recommendations.
    """

    __tablename__ = "user_interactions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Interaction metrics
    is_solved: Mapped[bool] = mapped_column(Boolean, default=False)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    best_execution_time_ms: Mapped[float] = mapped_column(Float, nullable=True)
    best_memory_usage_mb: Mapped[float] = mapped_column(Float, nullable=True)

    # Engagement signals
    first_attempt_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    solved_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    time_to_solve_seconds: Mapped[int] = mapped_column(Integer, nullable=True)

    # Implicit feedback score (computed)
    interaction_score: Mapped[float] = mapped_column(Float, default=0.0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "problem_id", name="uq_user_interactions_user_problem"
        ),
        Index("ix_user_interactions_user_problem", "user_id", "problem_id"),
        Index("ix_user_interactions_solved", "is_solved"),
    )


class ModelTrainingLogModel(Base):
    """Log of ML model training runs.

    Tracks model versions, performance metrics, and training history.
    """

    __tablename__ = "model_training_logs"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    model_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )  # 'ncf', 'lstm'
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_path: Mapped[str] = mapped_column(String(500), nullable=True)

    # Training details
    training_started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    training_completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)

    # Performance metrics
    metrics: Mapped[dict] = mapped_column(JSON, default={})
    # For NCF: {'loss': 0.5, 'auc': 0.85, 'hit_rate@10': 0.7}
    # For LSTM: {'loss': 0.3, 'mae': 0.1, 'mse': 0.02}

    # Status
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, training, completed, failed
    error_message: Mapped[str] = mapped_column(String(1000), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_model_training_type_active", "model_type", "is_active"),
    )


class CodeQualityAnalysisModel(Base):
    """Code quality analysis results for submissions.

    Stores multi-dimensional quality scores, code smells,
    complexity metrics, and improvement suggestions.
    Used for tracking code quality improvements over time.
    """

    __tablename__ = "code_quality_analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    submission_id: Mapped[UUID] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Multi-dimensional quality scores (0-100)
    correctness_score: Mapped[int] = mapped_column(Integer, default=0)
    efficiency_score: Mapped[int] = mapped_column(Integer, default=0)
    readability_score: Mapped[int] = mapped_column(Integer, default=0)
    best_practices_score: Mapped[int] = mapped_column(Integer, default=0)
    overall_score: Mapped[int] = mapped_column(Integer, default=0)
    overall_grade: Mapped[str] = mapped_column(String(2), default="C")  # A, B, C, D, F

    # Code smells detected
    # [{"type": "long_function", "severity": "warning", "line": 10, "message": "..."}]
    code_smells: Mapped[list] = mapped_column(JSON, default=[])
    code_smells_count: Mapped[int] = mapped_column(Integer, default=0)

    # Complexity metrics
    cyclomatic_complexity: Mapped[int] = mapped_column(Integer, default=1)
    cognitive_complexity: Mapped[int] = mapped_column(Integer, default=0)
    max_nesting_depth: Mapped[int] = mapped_column(Integer, default=0)
    lines_of_code: Mapped[int] = mapped_column(Integer, default=0)

    # Detected algorithm patterns
    # ["two-pointers", "sliding-window"]
    detected_patterns: Mapped[list] = mapped_column(JSON, default=[])

    # Improvement suggestions
    # [{"type": "efficiency", "message": "...", "priority": "high"}]
    suggestions: Mapped[list] = mapped_column(JSON, default=[])
    suggestions_count: Mapped[int] = mapped_column(Integer, default=0)

    # Analysis metadata
    language: Mapped[str] = mapped_column(String(20), default="python")
    analyzer_version: Mapped[str] = mapped_column(String(20), default="1.0.0")

    # Timestamps
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_quality_user_analyzed", "user_id", "analyzed_at"),
        Index("ix_quality_problem", "problem_id"),
        Index("ix_quality_overall_score", "overall_score"),
    )


class QualityTrendModel(Base):
    """Daily aggregated quality metrics for users.

    Tracks quality score trends over time for analytics.
    Aggregated from CodeQualityAnalysisModel daily.
    """

    __tablename__ = "quality_trends"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    trend_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)

    # Average scores for the day
    avg_overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    avg_correctness: Mapped[float] = mapped_column(Float, default=0.0)
    avg_efficiency: Mapped[float] = mapped_column(Float, default=0.0)
    avg_readability: Mapped[float] = mapped_column(Float, default=0.0)
    avg_best_practices: Mapped[float] = mapped_column(Float, default=0.0)

    # Complexity averages
    avg_cyclomatic: Mapped[float] = mapped_column(Float, default=0.0)
    avg_cognitive: Mapped[float] = mapped_column(Float, default=0.0)

    # Counts
    submissions_analyzed: Mapped[int] = mapped_column(Integer, default=0)
    total_smells: Mapped[int] = mapped_column(Integer, default=0)
    total_suggestions: Mapped[int] = mapped_column(Integer, default=0)

    # Improvement tracking
    improved_count: Mapped[int] = mapped_column(
        Integer, default=0
    )  # Submissions with higher score than previous
    grade_distribution: Mapped[dict] = mapped_column(
        JSON, default={}
    )  # {"A": 2, "B": 3, "C": 1}

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("user_id", "trend_date", name="uq_quality_trends_user_date"),
        Index("ix_quality_trends_user_date", "user_id", "trend_date"),
    )
