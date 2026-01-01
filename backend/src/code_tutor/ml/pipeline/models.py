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
