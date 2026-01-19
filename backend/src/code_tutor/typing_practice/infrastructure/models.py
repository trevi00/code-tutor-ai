"""SQLAlchemy models for typing practice."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from code_tutor.shared.infrastructure.database import Base
from code_tutor.typing_practice.domain.value_objects import (
    AttemptStatus,
    Difficulty,
    ExerciseCategory,
)


def generate_uuid():
    """Generate UUID string."""
    return str(uuid4())


class TypingExerciseModel(Base):
    """SQLAlchemy model for typing exercises."""

    __tablename__ = "typing_exercises"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    source_code = Column(Text, nullable=False)
    language = Column(String(50), default="python")
    category = Column(String(50), default=ExerciseCategory.TEMPLATE.value)
    difficulty = Column(String(20), default=Difficulty.EASY.value)
    description = Column(Text, default="")
    required_completions = Column(Integer, default=5)
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    attempts = relationship("TypingAttemptModel", back_populates="exercise")


class TypingAttemptModel(Base):
    """SQLAlchemy model for typing attempts."""

    __tablename__ = "typing_attempts"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    exercise_id = Column(String(36), ForeignKey("typing_exercises.id"), nullable=False)
    attempt_number = Column(Integer, nullable=False)
    user_code = Column(Text, default="")
    accuracy = Column(Float, default=0.0)
    wpm = Column(Float, default=0.0)
    time_seconds = Column(Float, default=0.0)
    status = Column(String(20), default=AttemptStatus.IN_PROGRESS.value)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    exercise = relationship("TypingExerciseModel", back_populates="attempts")
