"""Learning DTOs (Data Transfer Objects)"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from code_tutor.learning.domain.value_objects import Category, Difficulty, SubmissionStatus


# Request DTOs
class CreateTestCaseRequest(BaseModel):
    """Create test case request"""
    input_data: str
    expected_output: str
    is_sample: bool = False


class CreateProblemRequest(BaseModel):
    """Create problem request"""
    title: str = Field(..., min_length=1, max_length=255)
    description: str
    difficulty: Difficulty
    category: Category
    constraints: str = ""
    hints: list[str] = []
    solution_template: str = ""
    reference_solution: str = ""
    time_limit_ms: int = Field(default=1000, ge=100, le=10000)
    memory_limit_mb: int = Field(default=256, ge=32, le=512)
    test_cases: list[CreateTestCaseRequest] = []
    # Pattern-related fields
    pattern_ids: list[str] = []
    pattern_explanation: str = ""
    approach_hint: str = ""
    time_complexity_hint: str = ""
    space_complexity_hint: str = ""


class UpdateProblemRequest(BaseModel):
    """Update problem request"""
    title: str | None = None
    description: str | None = None
    constraints: str | None = None
    hints: list[str] | None = None


class CreateSubmissionRequest(BaseModel):
    """Create submission request"""
    problem_id: UUID
    code: str = Field(..., min_length=1)
    language: str = Field(default="python", pattern="^(python)$")


class ProblemFilterParams(BaseModel):
    """Problem filter parameters"""
    category: Category | None = None
    difficulty: Difficulty | None = None
    pattern_id: str | None = None
    page: int = Field(default=1, ge=1)
    size: int = Field(default=20, ge=1, le=100)


# Response DTOs
class TestCaseResponse(BaseModel):
    """Test case response"""
    id: UUID
    input_data: str
    expected_output: str
    is_sample: bool
    order: int

    class Config:
        from_attributes = True


class ProblemResponse(BaseModel):
    """Full problem response"""
    id: UUID
    title: str
    description: str
    difficulty: str
    category: str
    constraints: str
    hints: list[str]
    solution_template: str
    time_limit_ms: int
    memory_limit_mb: int
    is_published: bool
    test_cases: list[TestCaseResponse]
    # Pattern-related fields
    pattern_ids: list[str] = []
    pattern_explanation: str = ""
    approach_hint: str = ""
    time_complexity_hint: str = ""
    space_complexity_hint: str = ""
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ProblemSummaryResponse(BaseModel):
    """Problem summary for list views"""
    id: UUID
    title: str
    difficulty: str
    category: str
    is_published: bool
    pattern_ids: list[str] = []
    created_at: datetime


class ProblemListResponse(BaseModel):
    """Paginated problem list response"""
    items: list[ProblemSummaryResponse]
    total: int
    page: int
    size: int
    pages: int


class TestResultResponse(BaseModel):
    """Test result response"""
    test_case_id: UUID
    input_data: str
    expected_output: str
    actual_output: str
    is_passed: bool
    execution_time_ms: float
    error_message: str | None = None


class SubmissionResponse(BaseModel):
    """Submission response"""
    id: UUID
    user_id: UUID
    problem_id: UUID
    code: str
    language: str
    status: str
    test_results: list[TestResultResponse]
    total_tests: int
    passed_tests: int
    execution_time_ms: float
    memory_usage_mb: float
    error_message: str | None
    submitted_at: datetime
    evaluated_at: datetime | None

    class Config:
        from_attributes = True


class SubmissionSummaryResponse(BaseModel):
    """Submission summary for list views"""
    id: UUID
    problem_id: UUID
    status: str
    passed_tests: int
    total_tests: int
    submitted_at: datetime


class HintsResponse(BaseModel):
    """Problem hints response"""
    problem_id: UUID
    hints: list[str]
    total_hints: int


class RecommendedProblemResponse(BaseModel):
    """Recommended problem with reason"""
    id: UUID
    title: str
    difficulty: str
    category: str
    reason: str
