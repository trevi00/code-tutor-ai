"""Gamification SQLAlchemy models."""

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
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from code_tutor.gamification.domain.value_objects import (
    BadgeCategory,
    BadgeRarity,
    ChallengeStatus,
    ChallengeType,
)
from code_tutor.shared.infrastructure.database import Base


class BadgeModel(Base):
    """Badge database model."""

    __tablename__ = "badges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    icon = Column(String(50), nullable=False)
    rarity = Column(SQLEnum(BadgeRarity), nullable=False)
    category = Column(SQLEnum(BadgeCategory), nullable=False)
    requirement = Column(String(100), nullable=False)
    requirement_value = Column(Integer, nullable=False)
    xp_reward = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user_badges = relationship("UserBadgeModel", back_populates="badge")


class UserBadgeModel(Base):
    """User badge database model."""

    __tablename__ = "user_badges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    badge_id = Column(UUID(as_uuid=True), ForeignKey("badges.id"), nullable=False)
    earned_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    badge = relationship("BadgeModel", back_populates="user_badges")


class UserStatsModel(Base):
    """User stats database model."""

    __tablename__ = "user_stats"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), unique=True, nullable=False
    )
    total_xp = Column(Integer, default=0)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    problems_solved = Column(Integer, default=0)
    problems_solved_first_try = Column(Integer, default=0)
    patterns_mastered = Column(Integer, default=0)
    collaborations_count = Column(Integer, default=0)
    playgrounds_created = Column(Integer, default=0)
    playgrounds_shared = Column(Integer, default=0)
    # Roadmap progress
    lessons_completed = Column(Integer, default=0)
    paths_completed = Column(Integer, default=0)
    # Path level completion flags
    beginner_path_completed = Column(Boolean, default=False)
    elementary_path_completed = Column(Boolean, default=False)
    intermediate_path_completed = Column(Boolean, default=False)
    advanced_path_completed = Column(Boolean, default=False)
    last_activity_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChallengeModel(Base):
    """Challenge database model."""

    __tablename__ = "challenges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    challenge_type = Column(SQLEnum(ChallengeType), nullable=False)
    target_action = Column(String(50), nullable=False)
    target_value = Column(Integer, nullable=False)
    xp_reward = Column(Integer, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user_challenges = relationship("UserChallengeModel", back_populates="challenge")


class UserChallengeModel(Base):
    """User challenge progress database model."""

    __tablename__ = "user_challenges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    challenge_id = Column(
        UUID(as_uuid=True), ForeignKey("challenges.id"), nullable=False
    )
    current_progress = Column(Integer, default=0)
    status = Column(SQLEnum(ChallengeStatus), default=ChallengeStatus.ACTIVE)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    challenge = relationship("ChallengeModel", back_populates="user_challenges")
