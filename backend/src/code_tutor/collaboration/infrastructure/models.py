"""Collaboration SQLAlchemy models."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from code_tutor.collaboration.domain.value_objects import SessionStatus
from code_tutor.shared.infrastructure.database import Base


class CollaborationSessionModel(Base):
    """SQLAlchemy model for collaboration sessions."""

    __tablename__ = "collaboration_sessions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    problem_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("problems.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    host_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default=SessionStatus.WAITING.value, index=True
    )
    code_content: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(20), default="python")
    version: Mapped[int] = mapped_column(Integer, default=0)
    max_participants: Mapped[int] = mapped_column(Integer, default=5)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    participants: Mapped[list["SessionParticipantModel"]] = relationship(
        "SessionParticipantModel",
        back_populates="session",
        cascade="all, delete-orphan",
    )
    code_changes: Mapped[list["CodeChangeModel"]] = relationship(
        "CodeChangeModel",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_collab_sessions_status_created", "status", "created_at"),
    )


class SessionParticipantModel(Base):
    """SQLAlchemy model for session participants."""

    __tablename__ = "session_participants"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("collaboration_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    cursor_position: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    selection_range: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    color: Mapped[str] = mapped_column(String(20), default="#4ECDC4")

    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    session: Mapped["CollaborationSessionModel"] = relationship(
        "CollaborationSessionModel", back_populates="participants"
    )

    __table_args__ = (
        Index("ix_participants_session_active", "session_id", "is_active"),
    )


class CodeChangeModel(Base):
    """SQLAlchemy model for code change history."""

    __tablename__ = "code_changes"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("collaboration_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    operation: Mapped[dict] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)

    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    session: Mapped["CollaborationSessionModel"] = relationship(
        "CollaborationSessionModel", back_populates="code_changes"
    )

    __table_args__ = (
        Index("ix_code_changes_session_version", "session_id", "version"),
    )
