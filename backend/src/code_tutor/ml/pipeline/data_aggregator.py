"""Data Aggregator for ML Models

Aggregates user-problem interactions for NCF collaborative filtering.
"""

from datetime import UTC, datetime
from uuid import UUID


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(UTC)

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.learning.domain.value_objects import SubmissionStatus
from code_tutor.learning.infrastructure.models import ProblemModel, SubmissionModel
from code_tutor.ml.pipeline.models import UserInteractionModel
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DataAggregator:
    """Aggregates data for ML model training and inference."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def aggregate_user_interactions(
        self,
        user_id: UUID | None = None,
    ) -> int:
        """Aggregate user-problem interactions from submissions.

        Creates/updates UserInteractionModel records for NCF model.

        Args:
            user_id: Optional specific user to aggregate

        Returns:
            Number of interactions updated
        """
        logger.info(
            "aggregating_user_interactions",
            user_id=str(user_id) if user_id else "all",
        )

        # Build query for grouped submission stats
        query = select(
            SubmissionModel.user_id,
            SubmissionModel.problem_id,
            func.count(SubmissionModel.id).label("attempt_count"),
            func.max(SubmissionModel.status == SubmissionStatus.ACCEPTED).label(
                "is_solved"
            ),
            func.min(SubmissionModel.submitted_at).label("first_attempt"),
            func.min(SubmissionModel.execution_time_ms)
            .filter(SubmissionModel.status == SubmissionStatus.ACCEPTED)
            .label("best_time"),
            func.min(SubmissionModel.memory_usage_mb)
            .filter(SubmissionModel.status == SubmissionStatus.ACCEPTED)
            .label("best_memory"),
        ).group_by(SubmissionModel.user_id, SubmissionModel.problem_id)

        if user_id:
            query = query.where(SubmissionModel.user_id == user_id)

        result = await self.session.execute(query)
        rows = result.all()

        if not rows:
            logger.info("no_interactions_found")
            return 0

        # Get first accepted submission times for each user-problem pair
        solved_times = await self._get_solved_times(user_id)

        # Upsert interactions
        interactions_count = 0
        for row in rows:
            uid, pid, attempts, is_solved, first_attempt, best_time, best_memory = row

            # Calculate interaction score (implicit feedback)
            # Score based on: solved status, attempt efficiency, time
            score = self._calculate_interaction_score(
                is_solved=bool(is_solved),
                attempt_count=attempts,
                best_time_ms=best_time,
            )

            # Get solved timestamp
            solved_at = solved_times.get((uid, pid))
            time_to_solve = None
            if solved_at and first_attempt:
                time_to_solve = int((solved_at - first_attempt).total_seconds())

            interaction = {
                "user_id": uid,
                "problem_id": pid,
                "is_solved": bool(is_solved),
                "attempt_count": attempts,
                "best_execution_time_ms": best_time,
                "best_memory_usage_mb": best_memory,
                "first_attempt_at": first_attempt,
                "solved_at": solved_at,
                "time_to_solve_seconds": time_to_solve,
                "interaction_score": score,
            }

            await self._upsert_interaction(interaction)
            interactions_count += 1

        await self.session.flush()
        logger.info("interactions_aggregated", count=interactions_count)

        return interactions_count

    async def _get_solved_times(
        self,
        user_id: UUID | None = None,
    ) -> dict[tuple[UUID, UUID], datetime]:
        """Get first solved timestamp for each user-problem pair."""
        query = (
            select(
                SubmissionModel.user_id,
                SubmissionModel.problem_id,
                func.min(SubmissionModel.submitted_at).label("solved_at"),
            )
            .where(SubmissionModel.status == SubmissionStatus.ACCEPTED)
            .group_by(SubmissionModel.user_id, SubmissionModel.problem_id)
        )

        if user_id:
            query = query.where(SubmissionModel.user_id == user_id)

        result = await self.session.execute(query)
        return {(row[0], row[1]): row[2] for row in result.all()}

    def _calculate_interaction_score(
        self,
        is_solved: bool,
        attempt_count: int,
        best_time_ms: float | None,
    ) -> float:
        """Calculate implicit feedback score for NCF.

        Higher score = stronger positive signal.

        Score components:
        - Solved: +1.0
        - Efficiency bonus: +0.5 if solved in <= 3 attempts
        - Speed bonus: +0.3 if solved quickly (< 100ms)
        - Attempt penalty: -0.1 per attempt beyond 5
        """
        score = 0.0

        if is_solved:
            score += 1.0

            # Efficiency bonus
            if attempt_count <= 3:
                score += 0.5
            elif attempt_count > 5:
                score -= min(0.3, (attempt_count - 5) * 0.1)

            # Speed bonus
            if best_time_ms is not None and best_time_ms < 100:
                score += 0.3

        else:
            # Attempted but not solved - still some engagement signal
            score += 0.2 if attempt_count >= 3 else 0.1

        return max(0.0, min(2.0, score))  # Clamp to [0, 2]

    async def _upsert_interaction(self, interaction: dict) -> None:
        """Insert or update interaction record."""
        try:
            stmt = pg_insert(UserInteractionModel).values(**interaction)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_user_interactions_user_problem",
                set_={
                    "is_solved": stmt.excluded.is_solved,
                    "attempt_count": stmt.excluded.attempt_count,
                    "best_execution_time_ms": stmt.excluded.best_execution_time_ms,
                    "best_memory_usage_mb": stmt.excluded.best_memory_usage_mb,
                    "solved_at": stmt.excluded.solved_at,
                    "time_to_solve_seconds": stmt.excluded.time_to_solve_seconds,
                    "interaction_score": stmt.excluded.interaction_score,
                    "updated_at": utc_now(),
                },
            )
            await self.session.execute(stmt)
        except Exception:
            # Fallback for SQLite
            stmt = sqlite_insert(UserInteractionModel).values(**interaction)
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "problem_id"],
                set_={
                    "is_solved": stmt.excluded.is_solved,
                    "attempt_count": stmt.excluded.attempt_count,
                    "best_execution_time_ms": stmt.excluded.best_execution_time_ms,
                    "best_memory_usage_mb": stmt.excluded.best_memory_usage_mb,
                    "solved_at": stmt.excluded.solved_at,
                    "time_to_solve_seconds": stmt.excluded.time_to_solve_seconds,
                    "interaction_score": stmt.excluded.interaction_score,
                    "updated_at": utc_now(),
                },
            )
            await self.session.execute(stmt)

    async def get_ncf_training_data(self) -> tuple[list[dict], list[dict], list[tuple]]:
        """Get data formatted for NCF model training.

        Returns:
            Tuple of (problems, user_histories, interactions)
        """
        # Get all problems
        problems_query = select(ProblemModel).where(ProblemModel.is_published == True)
        problems_result = await self.session.execute(problems_query)
        problems = [
            {
                "id": str(p.id),
                "title": p.title,
                "difficulty": p.difficulty.value,
                "category": p.category.value,
                "pattern_ids": p.pattern_ids or [],
            }
            for p in problems_result.scalars().all()
        ]

        # Get user interactions
        interactions_query = select(UserInteractionModel)
        interactions_result = await self.session.execute(interactions_query)
        interactions_list = interactions_result.scalars().all()

        # Group by user for histories
        user_solved: dict[str, list[str]] = {}
        interactions = []

        for inter in interactions_list:
            uid = str(inter.user_id)
            pid = str(inter.problem_id)

            if uid not in user_solved:
                user_solved[uid] = []

            if inter.is_solved:
                user_solved[uid].append(pid)

            # NCF format: (user_id, problem_id, is_solved)
            interactions.append((uid, pid, inter.is_solved))

        user_histories = [
            {"user_id": uid, "solved_problems": pids}
            for uid, pids in user_solved.items()
        ]

        logger.info(
            "ncf_training_data_prepared",
            problems=len(problems),
            users=len(user_histories),
            interactions=len(interactions),
        )

        return problems, user_histories, interactions

    async def get_interaction_matrix(self) -> dict:
        """Get sparse interaction matrix for NCF.

        Returns:
            Dict with user_ids, problem_ids, and interaction data
        """
        query = select(UserInteractionModel)
        result = await self.session.execute(query)
        interactions = result.scalars().all()

        # Build ID mappings
        user_ids = list({str(i.user_id) for i in interactions})
        problem_ids = list({str(i.problem_id) for i in interactions})

        user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
        problem_to_idx = {pid: idx for idx, pid in enumerate(problem_ids)}

        # Build sparse data
        rows = []  # user indices
        cols = []  # problem indices
        values = []  # interaction scores

        for inter in interactions:
            rows.append(user_to_idx[str(inter.user_id)])
            cols.append(problem_to_idx[str(inter.problem_id)])
            values.append(inter.interaction_score)

        return {
            "user_ids": user_ids,
            "problem_ids": problem_ids,
            "user_to_idx": user_to_idx,
            "problem_to_idx": problem_to_idx,
            "rows": rows,
            "cols": cols,
            "values": values,
            "shape": (len(user_ids), len(problem_ids)),
        }
