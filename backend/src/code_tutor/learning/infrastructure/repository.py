"""Learning repository implementations"""

from uuid import UUID

from sqlalchemy import and_, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from code_tutor.learning.domain.entities import Problem, Submission, TestCase
from code_tutor.learning.domain.repository import ProblemRepository, SubmissionRepository
from code_tutor.learning.domain.value_objects import Category, Difficulty, SubmissionStatus, TestResult
from code_tutor.learning.infrastructure.models import (
    ProblemModel,
    SubmissionModel,
    TestCaseModel,
)


class SQLAlchemyProblemRepository(ProblemRepository):
    """SQLAlchemy implementation of ProblemRepository"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_entity(self, model: ProblemModel) -> Problem:
        """Convert SQLAlchemy model to domain entity"""
        test_cases = [
            TestCase(
                id=tc.id,
                problem_id=tc.problem_id,
                input_data=tc.input_data,
                expected_output=tc.expected_output,
                is_sample=tc.is_sample,
                order=tc.order,
            )
            for tc in model.test_cases
        ]

        problem = Problem(
            id=model.id,
            title=model.title,
            description=model.description,
            difficulty=model.difficulty,
            category=model.category,
            constraints=model.constraints,
            hints=model.hints or [],
            solution_template=model.solution_template,
            reference_solution=model.reference_solution,
            time_limit_ms=model.time_limit_ms,
            memory_limit_mb=model.memory_limit_mb,
            is_published=model.is_published,
            test_cases=test_cases,
        )
        problem._created_at = model.created_at
        problem._updated_at = model.updated_at
        return problem

    def _to_model(self, entity: Problem) -> ProblemModel:
        """Convert domain entity to SQLAlchemy model"""
        model = ProblemModel(
            id=entity.id,
            title=entity.title,
            description=entity.description,
            difficulty=entity.difficulty,
            category=entity.category,
            constraints=entity.constraints,
            hints=entity.hints,
            solution_template=entity.solution_template,
            reference_solution=entity.reference_solution,
            time_limit_ms=entity.time_limit_ms,
            memory_limit_mb=entity.memory_limit_mb,
            is_published=entity.is_published,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

        model.test_cases = [
            TestCaseModel(
                id=tc.id,
                problem_id=entity.id,
                input_data=tc.input_data,
                expected_output=tc.expected_output,
                is_sample=tc.is_sample,
                order=tc.order,
            )
            for tc in entity.test_cases
        ]

        return model

    async def get_by_id(self, id: UUID) -> Problem | None:
        """Get problem by ID with test cases"""
        stmt = (
            select(ProblemModel)
            .options(selectinload(ProblemModel.test_cases))
            .where(ProblemModel.id == id)
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def add(self, problem: Problem) -> Problem:
        """Add a new problem"""
        model = self._to_model(problem)
        self._session.add(model)
        await self._session.flush()

        # Reload with test cases
        stmt = (
            select(ProblemModel)
            .options(selectinload(ProblemModel.test_cases))
            .where(ProblemModel.id == model.id)
        )
        result = await self._session.execute(stmt)
        saved_model = result.scalar_one()
        return self._to_entity(saved_model)

    async def update(self, problem: Problem) -> Problem:
        """Update an existing problem"""
        model = self._to_model(problem)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def delete(self, id: UUID) -> bool:
        """Delete problem by ID"""
        model = await self._session.get(ProblemModel, id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def get_published(
        self,
        category: Category | None = None,
        difficulty: Difficulty | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Problem]:
        """Get published problems with optional filters"""
        conditions = [ProblemModel.is_published == True]

        if category:
            conditions.append(ProblemModel.category == category)
        if difficulty:
            conditions.append(ProblemModel.difficulty == difficulty)

        stmt = (
            select(ProblemModel)
            .options(selectinload(ProblemModel.test_cases))
            .where(and_(*conditions))
            .order_by(ProblemModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def count_published(
        self,
        category: Category | None = None,
        difficulty: Difficulty | None = None,
    ) -> int:
        """Count published problems"""
        conditions = [ProblemModel.is_published == True]

        if category:
            conditions.append(ProblemModel.category == category)
        if difficulty:
            conditions.append(ProblemModel.difficulty == difficulty)

        stmt = select(func.count()).select_from(ProblemModel).where(and_(*conditions))
        result = await self._session.execute(stmt)
        return result.scalar() or 0


class SQLAlchemySubmissionRepository(SubmissionRepository):
    """SQLAlchemy implementation of SubmissionRepository"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_entity(self, model: SubmissionModel) -> Submission:
        """Convert SQLAlchemy model to domain entity"""
        test_results = []
        if model.test_results:
            for tr in model.test_results:
                test_results.append(
                    TestResult(
                        test_case_id=UUID(tr["test_case_id"]),
                        input_data=tr["input_data"],
                        expected_output=tr["expected_output"],
                        actual_output=tr["actual_output"],
                        is_passed=tr["is_passed"],
                        execution_time_ms=tr["execution_time_ms"],
                        error_message=tr.get("error_message"),
                    )
                )

        submission = Submission(
            id=model.id,
            user_id=model.user_id,
            problem_id=model.problem_id,
            code=model.code,
            language=model.language,
            status=model.status,
            test_results=test_results,
            total_tests=model.total_tests,
            passed_tests=model.passed_tests,
            execution_time_ms=model.execution_time_ms,
            memory_usage_mb=model.memory_usage_mb,
            error_message=model.error_message,
            submitted_at=model.submitted_at,
            evaluated_at=model.evaluated_at,
        )
        submission._created_at = model.created_at
        submission._updated_at = model.updated_at
        return submission

    def _to_model(self, entity: Submission) -> SubmissionModel:
        """Convert domain entity to SQLAlchemy model"""
        test_results_json = None
        if entity.test_results:
            test_results_json = [
                {
                    "test_case_id": str(tr.test_case_id),
                    "input_data": tr.input_data,
                    "expected_output": tr.expected_output,
                    "actual_output": tr.actual_output,
                    "is_passed": tr.is_passed,
                    "execution_time_ms": tr.execution_time_ms,
                    "error_message": tr.error_message,
                }
                for tr in entity.test_results
            ]

        return SubmissionModel(
            id=entity.id,
            user_id=entity.user_id,
            problem_id=entity.problem_id,
            code=entity.code,
            language=entity.language,
            status=entity.status,
            test_results=test_results_json,
            total_tests=entity.total_tests,
            passed_tests=entity.passed_tests,
            execution_time_ms=entity.execution_time_ms,
            memory_usage_mb=entity.memory_usage_mb,
            error_message=entity.error_message,
            submitted_at=entity.submitted_at,
            evaluated_at=entity.evaluated_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    async def get_by_id(self, id: UUID) -> Submission | None:
        """Get submission by ID"""
        model = await self._session.get(SubmissionModel, id)
        if model is None:
            return None
        return self._to_entity(model)

    async def add(self, submission: Submission) -> Submission:
        """Add a new submission"""
        model = self._to_model(submission)
        self._session.add(model)
        await self._session.flush()
        return self._to_entity(model)

    async def update(self, submission: Submission) -> Submission:
        """Update an existing submission"""
        model = self._to_model(submission)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def get_by_user(
        self,
        user_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Submission]:
        """Get submissions by user"""
        stmt = (
            select(SubmissionModel)
            .where(SubmissionModel.user_id == user_id)
            .order_by(SubmissionModel.submitted_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def get_by_problem(
        self,
        problem_id: UUID,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Submission]:
        """Get submissions for a problem"""
        stmt = (
            select(SubmissionModel)
            .where(SubmissionModel.problem_id == problem_id)
            .order_by(SubmissionModel.submitted_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def get_user_problem_submissions(
        self,
        user_id: UUID,
        problem_id: UUID,
        limit: int = 10,
    ) -> list[Submission]:
        """Get user's submissions for a specific problem"""
        stmt = (
            select(SubmissionModel)
            .where(
                and_(
                    SubmissionModel.user_id == user_id,
                    SubmissionModel.problem_id == problem_id,
                )
            )
            .order_by(SubmissionModel.submitted_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        models = result.scalars().all()
        return [self._to_entity(m) for m in models]

    async def has_user_solved(self, user_id: UUID, problem_id: UUID) -> bool:
        """Check if user has solved a problem"""
        stmt = (
            select(SubmissionModel.id)
            .where(
                and_(
                    SubmissionModel.user_id == user_id,
                    SubmissionModel.problem_id == problem_id,
                    SubmissionModel.status == SubmissionStatus.ACCEPTED,
                )
            )
            .limit(1)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None
