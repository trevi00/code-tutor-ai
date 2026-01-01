"""Playground SQLAlchemy models."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)
from code_tutor.shared.infrastructure.database import Base


class PlaygroundModel(Base):
    """SQLAlchemy model for playgrounds."""

    __tablename__ = "playgrounds"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    owner_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(
        String(20),
        default=PlaygroundLanguage.PYTHON.value,
    )
    visibility: Mapped[str] = mapped_column(
        String(20),
        default=PlaygroundVisibility.PRIVATE.value,
        index=True,
    )
    share_code: Mapped[str] = mapped_column(
        String(32),
        unique=True,
        nullable=False,
        index=True,
    )
    stdin: Mapped[str] = mapped_column(Text, default="")
    is_forked: Mapped[bool] = mapped_column(Boolean, default=False)
    forked_from_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("playgrounds.id", ondelete="SET NULL"),
        nullable=True,
    )
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    fork_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    __table_args__ = (
        Index("ix_playgrounds_visibility_created", "visibility", "created_at"),
        Index("ix_playgrounds_owner_updated", "owner_id", "updated_at"),
    )


class CodeTemplateModel(Base):
    """SQLAlchemy model for code templates."""

    __tablename__ = "code_templates"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(
        String(20),
        default=PlaygroundLanguage.PYTHON.value,
        index=True,
    )
    category: Mapped[str] = mapped_column(
        String(30),
        default=TemplateCategory.SNIPPET.value,
        index=True,
    )
    tags: Mapped[str] = mapped_column(Text, default="")  # JSON array as string
    usage_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_templates_category_usage", "category", "usage_count"),
    )


class ExecutionHistoryModel(Base):
    """SQLAlchemy model for execution history."""

    __tablename__ = "playground_executions"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    playground_id: Mapped[UUID] = mapped_column(
        ForeignKey("playgrounds.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    code: Mapped[str] = mapped_column(Text, nullable=False)
    stdin: Mapped[str] = mapped_column(Text, default="")
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    exit_code: Mapped[int] = mapped_column(Integer, default=0)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    is_success: Mapped[bool] = mapped_column(Boolean, default=False)

    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_executions_playground_time", "playground_id", "executed_at"),
    )
