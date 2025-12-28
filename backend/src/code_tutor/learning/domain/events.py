"""Learning domain events"""

from dataclasses import dataclass, field
from uuid import UUID, uuid4

from code_tutor.shared.domain.events import DomainEvent


@dataclass(frozen=True)
class ProblemCreated(DomainEvent):
    """Event raised when a new problem is created"""

    problem_id: UUID = field(default_factory=uuid4)
    title: str = ""
    category: str = ""


@dataclass(frozen=True)
class ProblemPublished(DomainEvent):
    """Event raised when a problem is published"""

    problem_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class SubmissionCreated(DomainEvent):
    """Event raised when a new submission is created"""

    submission_id: UUID = field(default_factory=uuid4)
    user_id: UUID = field(default_factory=uuid4)
    problem_id: UUID = field(default_factory=uuid4)


@dataclass(frozen=True)
class SubmissionEvaluated(DomainEvent):
    """Event raised when a submission is evaluated"""

    submission_id: UUID = field(default_factory=uuid4)
    status: str = ""
    passed_tests: int = 0
    total_tests: int = 0
