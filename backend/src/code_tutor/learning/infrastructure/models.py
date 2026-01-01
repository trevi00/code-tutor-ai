"""Learning SQLAlchemy models"""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from code_tutor.learning.domain.value_objects import (
    Category,
    Difficulty,
    SubmissionStatus,
)
from code_tutor.shared.infrastructure.database import Base


class ProblemModel(Base):
    """SQLAlchemy model for Problem entity"""

    __tablename__ = "problems"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    difficulty: Mapped[Difficulty] = mapped_column(
        Enum(Difficulty),
        nullable=False,
    )
    category: Mapped[Category] = mapped_column(
        Enum(Category),
        nullable=False,
        index=True,
    )
    constraints: Mapped[str] = mapped_column(Text, default="")
    hints: Mapped[list[str]] = mapped_column(JSON, default=[])
    solution_template: Mapped[str] = mapped_column(Text, default="")
    reference_solution: Mapped[str] = mapped_column(Text, default="")
    time_limit_ms: Mapped[int] = mapped_column(Integer, default=1000)
    memory_limit_mb: Mapped[int] = mapped_column(Integer, default=256)
    is_published: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    # Pattern-related fields
    pattern_ids: Mapped[list[str]] = mapped_column(JSON, default=[])
    pattern_explanation: Mapped[str] = mapped_column(Text, default="")
    approach_hint: Mapped[str] = mapped_column(Text, default="")
    time_complexity_hint: Mapped[str] = mapped_column(String(50), default="")
    space_complexity_hint: Mapped[str] = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    test_cases: Mapped[list["TestCaseModel"]] = relationship(
        "TestCaseModel",
        back_populates="problem",
        cascade="all, delete-orphan",
        order_by="TestCaseModel.order",
    )
    submissions: Mapped[list["SubmissionModel"]] = relationship(
        "SubmissionModel",
        back_populates="problem",
    )


class TestCaseModel(Base):
    """SQLAlchemy model for TestCase entity"""

    __tablename__ = "test_cases"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    problem_id: Mapped[UUID] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_data: Mapped[str] = mapped_column(Text, nullable=False)
    expected_output: Mapped[str] = mapped_column(Text, nullable=False)
    is_sample: Mapped[bool] = mapped_column(Boolean, default=False)
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )

    # Relationships
    problem: Mapped["ProblemModel"] = relationship(
        "ProblemModel",
        back_populates="test_cases",
    )


class SubmissionModel(Base):
    """SQLAlchemy model for Submission entity"""

    __tablename__ = "submissions"

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
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(20), default="python")
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus),
        default=SubmissionStatus.PENDING,
        index=True,
    )
    test_results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    total_tests: Mapped[int] = mapped_column(Integer, default=0)
    passed_tests: Mapped[int] = mapped_column(Integer, default=0)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    memory_usage_mb: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        index=True,
    )
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    problem: Mapped["ProblemModel"] = relationship(
        "ProblemModel",
        back_populates="submissions",
    )
