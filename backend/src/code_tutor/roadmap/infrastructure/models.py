"""SQLAlchemy models for Learning Roadmap."""

from datetime import datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from code_tutor.roadmap.domain.value_objects import (
    LessonType,
    PathLevel,
    ProgressStatus,
)
from code_tutor.shared.infrastructure.database import Base


def generate_uuid() -> str:
    """Generate UUID string."""
    return str(uuid4())


class LearningPathModel(Base):
    """SQLAlchemy model for learning paths."""

    __tablename__ = "learning_paths"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    level = Column(String(20), default=PathLevel.BEGINNER.value, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="")
    icon = Column(String(50), default="")
    order = Column(Integer, default=0)
    estimated_hours = Column(Integer, default=0)
    is_published = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    modules = relationship(
        "ModuleModel",
        back_populates="path",
        order_by="ModuleModel.order",
        cascade="all, delete-orphan",
    )
    prerequisites = relationship(
        "PathPrerequisiteModel",
        back_populates="path",
        foreign_keys="PathPrerequisiteModel.path_id",
        cascade="all, delete-orphan",
    )
    user_progress = relationship(
        "UserPathProgressModel",
        back_populates="path",
        cascade="all, delete-orphan",
    )


class PathPrerequisiteModel(Base):
    """SQLAlchemy model for path prerequisites (many-to-many)."""

    __tablename__ = "path_prerequisites"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    path_id = Column(String(36), ForeignKey("learning_paths.id"), nullable=False)
    prerequisite_id = Column(String(36), ForeignKey("learning_paths.id"), nullable=False)

    # Relationships
    path = relationship(
        "LearningPathModel",
        back_populates="prerequisites",
        foreign_keys=[path_id],
    )


class ModuleModel(Base):
    """SQLAlchemy model for modules."""

    __tablename__ = "roadmap_modules"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    path_id = Column(String(36), ForeignKey("learning_paths.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="")
    order = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    path = relationship("LearningPathModel", back_populates="modules")
    lessons = relationship(
        "LessonModel",
        back_populates="module",
        order_by="LessonModel.order",
        cascade="all, delete-orphan",
    )


class LessonModel(Base):
    """SQLAlchemy model for lessons."""

    __tablename__ = "roadmap_lessons"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    module_id = Column(String(36), ForeignKey("roadmap_modules.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="")
    lesson_type = Column(String(20), default=LessonType.CONCEPT.value, nullable=False)
    content = Column(Text, default="")
    content_id = Column(String(36), nullable=True)
    order = Column(Integer, default=0)
    xp_reward = Column(Integer, default=10)
    estimated_minutes = Column(Integer, default=10)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    module = relationship("ModuleModel", back_populates="lessons")
    user_progress = relationship(
        "UserLessonProgressModel",
        back_populates="lesson",
        cascade="all, delete-orphan",
    )


class UserPathProgressModel(Base):
    """SQLAlchemy model for user path progress."""

    __tablename__ = "user_path_progress"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    path_id = Column(String(36), ForeignKey("learning_paths.id"), nullable=False)
    status = Column(String(20), default=ProgressStatus.NOT_STARTED.value)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    completed_lessons = Column(Integer, default=0)
    total_lessons = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    path = relationship("LearningPathModel", back_populates="user_progress")


class UserLessonProgressModel(Base):
    """SQLAlchemy model for user lesson progress."""

    __tablename__ = "user_lesson_progress"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    lesson_id = Column(String(36), ForeignKey("roadmap_lessons.id"), nullable=False)
    status = Column(String(20), default=ProgressStatus.NOT_STARTED.value)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    score = Column(Integer, nullable=True)
    attempts = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    lesson = relationship("LessonModel", back_populates="user_progress")
