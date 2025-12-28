"""Learning domain repository interfaces"""

from abc import ABC, abstractmethod
from uuid import UUID

from code_tutor.learning.domain.entities import Problem, Submission
from code_tutor.learning.domain.value_objects import Category, Difficulty


class ProblemRepository(ABC):
    """Abstract repository interface for Problem aggregate"""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Problem | None:
        """Get problem by ID"""
        ...

    @abstractmethod
    async def add(self, problem: Problem) -> Problem:
        """Add a new problem"""
        ...

    @abstractmethod
    async def update(self, problem: Problem) -> Problem:
        """Update an existing problem"""
        ...

    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """Delete problem by ID"""
        ...

    @abstractmethod
    async def get_published(
        self,
        category: Category | None = None,
        difficulty: Difficulty | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Problem]:
        """Get published problems with optional filters"""
        ...

    @abstractmethod
    async def count_published(
        self,
        category: Category | None = None,
        difficulty: Difficulty | None = None,
    ) -> int:
        """Count published problems"""
        ...


class SubmissionRepository(ABC):
    """Abstract repository interface for Submission aggregate"""

    @abstractmethod
    async def get_by_id(self, id: UUID) -> Submission | None:
        """Get submission by ID"""
        ...

    @abstractmethod
    async def add(self, submission: Submission) -> Submission:
        """Add a new submission"""
        ...

    @abstractmethod
    async def update(self, submission: Submission) -> Submission:
        """Update an existing submission"""
        ...

    @abstractmethod
    async def get_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Submission]:
        """Get submissions by user"""
        ...

    @abstractmethod
    async def get_by_problem(
        self,
        problem_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Submission]:
        """Get submissions for a problem"""
        ...

    @abstractmethod
    async def get_user_problem_submissions(
        self,
        user_id: UUID,
        problem_id: UUID,
        limit: int = 10,
    ) -> list[Submission]:
        """Get user's submissions for a specific problem"""
        ...

    @abstractmethod
    async def has_user_solved(self, user_id: UUID, problem_id: UUID) -> bool:
        """Check if user has solved a problem"""
        ...
