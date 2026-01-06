"""Daily Stats Service

Aggregates user learning statistics from submissions for ML models.
"""

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from uuid import UUID


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.learning.domain.value_objects import Difficulty, SubmissionStatus
from code_tutor.learning.infrastructure.models import ProblemModel, SubmissionModel
from code_tutor.ml.pipeline.models import DailyStatsModel
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DailyStatsService:
    """Service for aggregating daily learning statistics."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def aggregate_daily_stats(
        self,
        target_date: date | None = None,
        user_id: UUID | None = None,
    ) -> int:
        """Aggregate daily statistics from submissions.

        Args:
            target_date: Date to aggregate (defaults to yesterday)
            user_id: Optional specific user to aggregate

        Returns:
            Number of records created/updated
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        logger.info(
            "aggregating_daily_stats",
            target_date=str(target_date),
            user_id=str(user_id) if user_id else None,
        )

        # Get date range
        start_dt = datetime.combine(target_date, datetime.min.time())
        end_dt = datetime.combine(target_date + timedelta(days=1), datetime.min.time())

        # Build query for submissions on target date
        query = (
            select(SubmissionModel)
            .where(SubmissionModel.submitted_at >= start_dt)
            .where(SubmissionModel.submitted_at < end_dt)
        )

        if user_id:
            query = query.where(SubmissionModel.user_id == user_id)

        result = await self.session.execute(query)
        submissions = result.scalars().all()

        if not submissions:
            logger.info("no_submissions_found", target_date=str(target_date))
            return 0

        # Group submissions by user
        user_submissions: dict[UUID, list] = defaultdict(list)
        for sub in submissions:
            user_submissions[sub.user_id].append(sub)

        # Get problem details for difficulty info
        problem_ids = {sub.problem_id for sub in submissions}
        problems_query = select(ProblemModel).where(ProblemModel.id.in_(problem_ids))
        problems_result = await self.session.execute(problems_query)
        problems = {p.id: p for p in problems_result.scalars().all()}

        # Aggregate stats for each user
        records_processed = 0
        for uid, user_subs in user_submissions.items():
            stats = self._calculate_user_stats(uid, user_subs, problems, target_date)
            await self._upsert_daily_stats(stats)
            records_processed += 1

        await self.session.flush()
        logger.info(
            "daily_stats_aggregated",
            target_date=str(target_date),
            records=records_processed,
        )

        return records_processed

    def _calculate_user_stats(
        self,
        user_id: UUID,
        submissions: list[SubmissionModel],
        problems: dict[UUID, ProblemModel],
        stats_date: date,
    ) -> dict:
        """Calculate statistics for a single user's daily submissions."""
        # Basic counts
        total_submissions = len(submissions)
        problems_attempted = len({s.problem_id for s in submissions})

        # Solved problems (accepted submissions)
        solved_problem_ids = {
            s.problem_id for s in submissions if s.status == SubmissionStatus.ACCEPTED
        }
        problems_solved = len(solved_problem_ids)

        # Success rate
        success_rate = (
            problems_solved / problems_attempted if problems_attempted > 0 else 0.0
        )

        # Execution time and memory (for accepted submissions)
        accepted_subs = [
            s for s in submissions if s.status == SubmissionStatus.ACCEPTED
        ]
        avg_time = (
            sum(s.execution_time_ms for s in accepted_subs) / len(accepted_subs)
            if accepted_subs
            else 0.0
        )
        avg_memory = (
            sum(s.memory_usage_mb for s in accepted_subs) / len(accepted_subs)
            if accepted_subs
            else 0.0
        )

        # Difficulty breakdown
        easy_solved = medium_solved = hard_solved = 0
        for pid in solved_problem_ids:
            if pid in problems:
                diff = problems[pid].difficulty
                if diff == Difficulty.EASY:
                    easy_solved += 1
                elif diff == Difficulty.MEDIUM:
                    medium_solved += 1
                elif diff == Difficulty.HARD:
                    hard_solved += 1

        # Category breakdown
        category_counts: dict[str, int] = defaultdict(int)
        for pid in solved_problem_ids:
            if pid in problems:
                cat = problems[pid].category.value
                category_counts[cat] += 1

        categories_attempted = len(
            {
                problems[s.problem_id].category
                for s in submissions
                if s.problem_id in problems
            }
        )

        # Estimate study time (rough estimate based on submissions)
        # Assume average 10 minutes per submission
        study_minutes = total_submissions * 10

        return {
            "user_id": user_id,
            "stats_date": stats_date,
            "problems_attempted": problems_attempted,
            "problems_solved": problems_solved,
            "total_submissions": total_submissions,
            "success_rate": success_rate,
            "avg_time_to_solve_ms": avg_time,
            "avg_memory_usage_mb": avg_memory,
            "easy_solved": easy_solved,
            "medium_solved": medium_solved,
            "hard_solved": hard_solved,
            "categories_attempted": categories_attempted,
            "category_breakdown": dict(category_counts),
            "study_minutes": study_minutes,
            "is_active_day": total_submissions > 0,
        }

    async def _upsert_daily_stats(self, stats: dict) -> None:
        """Insert or update daily stats record."""
        # Try PostgreSQL-style upsert first, fall back to SQLite
        try:
            stmt = pg_insert(DailyStatsModel).values(**stats)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_daily_stats_user_date",
                set_={
                    "problems_attempted": stmt.excluded.problems_attempted,
                    "problems_solved": stmt.excluded.problems_solved,
                    "total_submissions": stmt.excluded.total_submissions,
                    "success_rate": stmt.excluded.success_rate,
                    "avg_time_to_solve_ms": stmt.excluded.avg_time_to_solve_ms,
                    "avg_memory_usage_mb": stmt.excluded.avg_memory_usage_mb,
                    "easy_solved": stmt.excluded.easy_solved,
                    "medium_solved": stmt.excluded.medium_solved,
                    "hard_solved": stmt.excluded.hard_solved,
                    "categories_attempted": stmt.excluded.categories_attempted,
                    "category_breakdown": stmt.excluded.category_breakdown,
                    "study_minutes": stmt.excluded.study_minutes,
                    "is_active_day": stmt.excluded.is_active_day,
                    "updated_at": utc_now(),
                },
            )
            await self.session.execute(stmt)
        except Exception:
            # Fallback for SQLite
            stmt = sqlite_insert(DailyStatsModel).values(**stats)
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "stats_date"],
                set_={
                    "problems_attempted": stmt.excluded.problems_attempted,
                    "problems_solved": stmt.excluded.problems_solved,
                    "total_submissions": stmt.excluded.total_submissions,
                    "success_rate": stmt.excluded.success_rate,
                    "avg_time_to_solve_ms": stmt.excluded.avg_time_to_solve_ms,
                    "avg_memory_usage_mb": stmt.excluded.avg_memory_usage_mb,
                    "easy_solved": stmt.excluded.easy_solved,
                    "medium_solved": stmt.excluded.medium_solved,
                    "hard_solved": stmt.excluded.hard_solved,
                    "categories_attempted": stmt.excluded.categories_attempted,
                    "category_breakdown": stmt.excluded.category_breakdown,
                    "study_minutes": stmt.excluded.study_minutes,
                    "is_active_day": stmt.excluded.is_active_day,
                    "updated_at": utc_now(),
                },
            )
            await self.session.execute(stmt)

    async def update_streak(self, user_id: UUID) -> int:
        """Calculate and update streak for a user.

        Returns the current streak count.
        """
        today = date.today()

        # Get recent daily stats ordered by date desc
        query = (
            select(DailyStatsModel)
            .where(DailyStatsModel.user_id == user_id)
            .where(DailyStatsModel.is_active_day == True)
            .order_by(DailyStatsModel.stats_date.desc())
            .limit(365)
        )

        result = await self.session.execute(query)
        active_days = [r.stats_date for r in result.scalars().all()]

        if not active_days:
            return 0

        # Calculate streak
        streak = 0
        check_date = today

        # Check if today is active, otherwise check yesterday
        if active_days[0] != today:
            check_date = today - timedelta(days=1)
            if active_days[0] != check_date:
                return 0  # Streak broken

        for active_date in active_days:
            if active_date == check_date:
                streak += 1
                check_date -= timedelta(days=1)
            elif active_date < check_date:
                break  # Gap found, streak ends

        # Update streak in today's stats
        update_query = (
            select(DailyStatsModel)
            .where(DailyStatsModel.user_id == user_id)
            .where(DailyStatsModel.stats_date == today)
        )
        result = await self.session.execute(update_query)
        today_stats = result.scalar_one_or_none()

        if today_stats:
            today_stats.streak_days = streak
            await self.session.flush()

        return streak

    async def get_user_stats_sequence(
        self,
        user_id: UUID,
        days: int = 30,
    ) -> list[dict]:
        """Get sequence of daily stats for LSTM model.

        Returns list of feature dictionaries for each day.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        query = (
            select(DailyStatsModel)
            .where(DailyStatsModel.user_id == user_id)
            .where(DailyStatsModel.stats_date >= start_date)
            .where(DailyStatsModel.stats_date <= end_date)
            .order_by(DailyStatsModel.stats_date.asc())
        )

        result = await self.session.execute(query)
        stats_records = result.scalars().all()

        # Create date-indexed map
        stats_map = {s.stats_date: s for s in stats_records}

        # Generate sequence with zero-filled gaps
        sequence = []
        current_date = start_date

        while current_date <= end_date:
            if current_date in stats_map:
                s = stats_map[current_date]
                sequence.append(
                    {
                        "problems_attempted": s.problems_attempted,
                        "problems_solved": s.problems_solved,
                        "success_rate": s.success_rate,
                        "avg_time_to_solve": s.avg_time_to_solve_ms,
                        "difficulty_easy": s.easy_solved,
                        "difficulty_medium": s.medium_solved,
                        "difficulty_hard": s.hard_solved,
                        "categories_attempted": s.categories_attempted,
                        "streak_days": s.streak_days,
                        "total_study_minutes": s.study_minutes,
                    }
                )
            else:
                # Zero-fill for inactive days
                sequence.append(
                    {
                        "problems_attempted": 0,
                        "problems_solved": 0,
                        "success_rate": 0.0,
                        "avg_time_to_solve": 0.0,
                        "difficulty_easy": 0,
                        "difficulty_medium": 0,
                        "difficulty_hard": 0,
                        "categories_attempted": 0,
                        "streak_days": 0,
                        "total_study_minutes": 0,
                    }
                )
            current_date += timedelta(days=1)

        return sequence

    async def get_all_user_ids_with_activity(self) -> list[UUID]:
        """Get all user IDs that have at least one submission."""
        query = select(SubmissionModel.user_id).distinct()
        result = await self.session.execute(query)
        return [row[0] for row in result.all()]

    async def backfill_stats(
        self,
        days_back: int = 30,
        user_id: UUID | None = None,
    ) -> int:
        """Backfill daily stats for the specified number of days.

        Args:
            days_back: Number of days to backfill
            user_id: Optional specific user to backfill

        Returns:
            Total number of records created/updated
        """
        total_records = 0
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        logger.info(
            "starting_backfill",
            start_date=str(start_date),
            end_date=str(end_date),
            user_id=str(user_id) if user_id else "all",
        )

        current_date = start_date
        while current_date < end_date:
            records = await self.aggregate_daily_stats(current_date, user_id)
            total_records += records
            current_date += timedelta(days=1)

        logger.info("backfill_completed", total_records=total_records)
        return total_records
