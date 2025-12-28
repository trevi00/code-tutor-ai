"""Dashboard application service"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import and_, case, distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.learning.application.dashboard_dto import (
    CategoryProgress,
    DashboardResponse,
    HeatmapData,
    RecentSubmission,
    SkillPrediction,
    StreakInfo,
    UserStats,
)
from code_tutor.learning.domain.value_objects import Category, Difficulty, SubmissionStatus
from code_tutor.learning.infrastructure.models import ProblemModel, SubmissionModel
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class DashboardService:
    """Service for user dashboard data"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_dashboard(self, user_id: UUID) -> DashboardResponse:
        """Get dashboard data for a user"""
        stats = await self._get_user_stats(user_id)
        category_progress = await self._get_category_progress(user_id)
        recent_submissions = await self._get_recent_submissions(user_id)
        heatmap = await self._get_heatmap_data(user_id)
        skill_predictions = self._generate_skill_predictions(category_progress)

        return DashboardResponse(
            stats=stats,
            category_progress=category_progress,
            recent_submissions=recent_submissions,
            heatmap=heatmap,
            skill_predictions=skill_predictions,
        )

    async def _get_user_stats(self, user_id: UUID) -> UserStats:
        """Get user statistics"""
        # Get submission counts
        submission_query = select(
            func.count(SubmissionModel.id).label("total_submissions"),
            func.count(
                case(
                    (SubmissionModel.status == SubmissionStatus.ACCEPTED, 1),
                    else_=None,
                )
            ).label("accepted_submissions"),
        ).where(SubmissionModel.user_id == user_id)

        result = await self._session.execute(submission_query)
        row = result.first()
        total_submissions = row.total_submissions if row else 0
        accepted_submissions = row.accepted_submissions if row else 0

        # Get unique problems attempted and solved
        problems_query = select(
            func.count(distinct(SubmissionModel.problem_id)).label("attempted"),
            func.count(
                distinct(
                    case(
                        (SubmissionModel.status == SubmissionStatus.ACCEPTED, SubmissionModel.problem_id),
                        else_=None,
                    )
                )
            ).label("solved"),
        ).where(SubmissionModel.user_id == user_id)

        result = await self._session.execute(problems_query)
        row = result.first()
        problems_attempted = row.attempted if row else 0
        problems_solved = row.solved if row else 0

        # Get solved by difficulty
        difficulty_query = (
            select(
                ProblemModel.difficulty,
                func.count(distinct(SubmissionModel.problem_id)).label("count"),
            )
            .join(ProblemModel, SubmissionModel.problem_id == ProblemModel.id)
            .where(
                and_(
                    SubmissionModel.user_id == user_id,
                    SubmissionModel.status == SubmissionStatus.ACCEPTED,
                )
            )
            .group_by(ProblemModel.difficulty)
        )

        result = await self._session.execute(difficulty_query)
        difficulty_counts = {row[0]: row[1] for row in result.all()}

        # Get streak info
        streak = await self._get_streak_info(user_id)

        success_rate = (
            (accepted_submissions / total_submissions * 100) if total_submissions > 0 else 0
        )

        return UserStats(
            total_problems_attempted=problems_attempted,
            total_problems_solved=problems_solved,
            total_submissions=total_submissions,
            overall_success_rate=round(success_rate, 1),
            easy_solved=difficulty_counts.get(Difficulty.EASY, 0),
            medium_solved=difficulty_counts.get(Difficulty.MEDIUM, 0),
            hard_solved=difficulty_counts.get(Difficulty.HARD, 0),
            streak=streak,
        )

    async def _get_category_progress(self, user_id: UUID) -> list[CategoryProgress]:
        """Get progress by category"""
        # Get total problems per category
        total_query = (
            select(
                ProblemModel.category,
                func.count(ProblemModel.id).label("total"),
            )
            .where(ProblemModel.is_published == True)
            .group_by(ProblemModel.category)
        )

        result = await self._session.execute(total_query)
        total_by_category = {row[0]: row[1] for row in result.all()}

        # Get solved problems per category
        solved_query = (
            select(
                ProblemModel.category,
                func.count(distinct(SubmissionModel.problem_id)).label("solved"),
            )
            .join(ProblemModel, SubmissionModel.problem_id == ProblemModel.id)
            .where(
                and_(
                    SubmissionModel.user_id == user_id,
                    SubmissionModel.status == SubmissionStatus.ACCEPTED,
                )
            )
            .group_by(ProblemModel.category)
        )

        result = await self._session.execute(solved_query)
        solved_by_category = {row[0]: row[1] for row in result.all()}

        progress_list = []
        for category in Category:
            total = total_by_category.get(category, 0)
            solved = solved_by_category.get(category, 0)
            success_rate = (solved / total * 100) if total > 0 else 0

            if total > 0:  # Only include categories with problems
                progress_list.append(
                    CategoryProgress(
                        category=category.value,
                        total_problems=total,
                        solved_problems=solved,
                        success_rate=round(success_rate, 1),
                    )
                )

        # Sort by solved problems (descending)
        progress_list.sort(key=lambda x: x.solved_problems, reverse=True)
        return progress_list

    async def _get_recent_submissions(
        self, user_id: UUID, limit: int = 10
    ) -> list[RecentSubmission]:
        """Get recent submissions"""
        query = (
            select(
                SubmissionModel.id,
                SubmissionModel.problem_id,
                ProblemModel.title,
                SubmissionModel.status,
                SubmissionModel.submitted_at,
            )
            .join(ProblemModel, SubmissionModel.problem_id == ProblemModel.id)
            .where(SubmissionModel.user_id == user_id)
            .order_by(SubmissionModel.submitted_at.desc())
            .limit(limit)
        )

        result = await self._session.execute(query)
        rows = result.all()

        return [
            RecentSubmission(
                id=row[0],
                problem_id=row[1],
                problem_title=row[2],
                status=row[3].value if hasattr(row[3], "value") else str(row[3]),
                submitted_at=row[4],
            )
            for row in rows
        ]

    async def _get_streak_info(self, user_id: UUID) -> StreakInfo:
        """Calculate user's streak information"""
        # Get unique submission dates
        query = (
            select(func.date(SubmissionModel.submitted_at).label("date"))
            .where(SubmissionModel.user_id == user_id)
            .distinct()
            .order_by(func.date(SubmissionModel.submitted_at).desc())
        )

        result = await self._session.execute(query)
        dates = [row[0] for row in result.all()]

        if not dates:
            return StreakInfo()

        today = datetime.now(timezone.utc).date()
        last_activity = dates[0]

        # Calculate current streak
        current_streak = 0
        check_date = today

        for date in dates:
            if date == check_date or date == check_date - timedelta(days=1):
                current_streak += 1
                check_date = date - timedelta(days=1)
            else:
                break

        # Calculate longest streak
        longest_streak = 1
        current = 1

        for i in range(1, len(dates)):
            if dates[i - 1] - dates[i] == timedelta(days=1):
                current += 1
                longest_streak = max(longest_streak, current)
            else:
                current = 1

        return StreakInfo(
            current_streak=current_streak,
            longest_streak=max(longest_streak, current_streak),
            last_activity_date=datetime.combine(
                last_activity, datetime.min.time(), tzinfo=timezone.utc
            ),
        )

    async def _get_heatmap_data(self, user_id: UUID, days: int = 365) -> list[HeatmapData]:
        """Get activity heatmap data for the past year"""
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)

        # Get submission counts per day
        query = (
            select(
                func.date(SubmissionModel.submitted_at).label("date"),
                func.count(SubmissionModel.id).label("count"),
            )
            .where(
                and_(
                    SubmissionModel.user_id == user_id,
                    func.date(SubmissionModel.submitted_at) >= start_date,
                )
            )
            .group_by(func.date(SubmissionModel.submitted_at))
            .order_by(func.date(SubmissionModel.submitted_at))
        )

        result = await self._session.execute(query)
        submission_counts = {row[0]: row[1] for row in result.all()}

        # Generate heatmap data for all days
        heatmap = []
        current_date = start_date

        while current_date <= end_date:
            count = submission_counts.get(current_date, 0)
            level = self._calculate_activity_level(count)
            heatmap.append(
                HeatmapData(
                    date=current_date.isoformat(),
                    count=count,
                    level=level,
                )
            )
            current_date += timedelta(days=1)

        return heatmap

    def _calculate_activity_level(self, count: int) -> int:
        """Calculate activity level (0-4) based on submission count"""
        if count == 0:
            return 0
        elif count <= 2:
            return 1
        elif count <= 5:
            return 2
        elif count <= 10:
            return 3
        else:
            return 4

    def _generate_skill_predictions(
        self, category_progress: list[CategoryProgress]
    ) -> list[SkillPrediction]:
        """Generate skill predictions based on category progress"""
        predictions = []

        for progress in category_progress:
            if progress.total_problems == 0:
                continue

            # Calculate current skill level based on success rate and completion
            completion_rate = progress.solved_problems / progress.total_problems
            current_level = (progress.success_rate * 0.6 + completion_rate * 100 * 0.4)

            # Predict future level (simple linear extrapolation)
            # In production, this could use ML models
            growth_factor = 1.1 if current_level < 50 else 1.05
            predicted_level = min(100, current_level * growth_factor)

            # Calculate confidence based on number of solved problems
            confidence = min(1.0, progress.solved_problems / 10)

            # Recommend focus if completion rate is low
            recommended_focus = completion_rate < 0.3 and progress.total_problems >= 5

            predictions.append(
                SkillPrediction(
                    category=progress.category,
                    current_level=round(current_level, 1),
                    predicted_level=round(predicted_level, 1),
                    confidence=round(confidence, 2),
                    recommended_focus=recommended_focus,
                )
            )

        # Sort by recommended focus first, then by current level (ascending)
        predictions.sort(key=lambda x: (-x.recommended_focus, x.current_level))

        return predictions
