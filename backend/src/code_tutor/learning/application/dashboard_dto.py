"""Dashboard DTOs"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class CategoryProgress(BaseModel):
    """Progress for a specific category"""

    category: str
    total_problems: int
    solved_problems: int
    success_rate: float = Field(ge=0, le=100)


class RecentSubmission(BaseModel):
    """Recent submission summary"""

    id: UUID
    problem_id: UUID
    problem_title: str
    status: str
    submitted_at: datetime


class StreakInfo(BaseModel):
    """Streak information"""

    current_streak: int = 0
    longest_streak: int = 0
    last_activity_date: datetime | None = None


class UserStats(BaseModel):
    """User statistics"""

    total_problems_attempted: int
    total_problems_solved: int
    total_submissions: int
    overall_success_rate: float = Field(ge=0, le=100)
    easy_solved: int
    medium_solved: int
    hard_solved: int
    streak: StreakInfo


class DashboardResponse(BaseModel):
    """Dashboard response"""

    stats: UserStats
    category_progress: list[CategoryProgress]
    recent_submissions: list[RecentSubmission]
