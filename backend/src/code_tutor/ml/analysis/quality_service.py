"""Code Quality Analysis Service

Integrates code analysis, classification, and persistence.
Provides comprehensive code quality tracking and trend analysis.
"""

from datetime import date, datetime, timedelta, timezone
from uuid import UUID


def utc_now() -> datetime:
    """Get current UTC time (timezone-aware)"""
    return datetime.now(timezone.utc)

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.ml.pipeline.models import CodeQualityAnalysisModel, QualityTrendModel

logger = structlog.get_logger()

# Version for tracking analyzer changes
ANALYZER_VERSION = "1.0.0"


class CodeQualityService:
    """Service for code quality analysis with persistence."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._analyzer = None
        self._classifier = None

    def _get_analyzer(self):
        """Lazy load code analyzer."""
        if self._analyzer is None:
            from code_tutor.ml.analysis.code_analyzer import CodeAnalyzer

            self._analyzer = CodeAnalyzer()
        return self._analyzer

    def _get_classifier(self):
        """Lazy load code classifier."""
        if self._classifier is None:
            from code_tutor.ml.analysis.code_classifier import CodeQualityClassifier

            self._classifier = CodeQualityClassifier()
        return self._classifier

    async def analyze_submission(
        self,
        submission_id: UUID,
        user_id: UUID,
        problem_id: UUID,
        code: str,
        language: str = "python",
    ) -> CodeQualityAnalysisModel:
        """Analyze code quality for a submission and store results.

        Args:
            submission_id: Submission UUID
            user_id: User UUID
            problem_id: Problem UUID
            code: Source code to analyze
            language: Programming language

        Returns:
            CodeQualityAnalysisModel with analysis results
        """
        # Check if already analyzed
        existing = await self.get_submission_quality(submission_id)
        if existing:
            logger.debug(
                "submission_already_analyzed", submission_id=str(submission_id)
            )
            return existing

        try:
            # Run analyzer for code smells, complexity, patterns
            analyzer = self._get_analyzer()
            analysis = analyzer.analyze(code, language)

            # Run classifier for multi-dimensional quality scores
            classifier = self._get_classifier()
            classification = classifier.classify(code, language)
            suggestions = classifier.get_improvement_suggestions(classification)

            # Extract dimensions
            dims = classification.get("dimensions", {})

            # Create analysis record
            quality_analysis = CodeQualityAnalysisModel(
                submission_id=submission_id,
                user_id=user_id,
                problem_id=problem_id,
                # Quality scores
                correctness_score=int(dims.get("correctness", {}).get("score", 50)),
                efficiency_score=int(dims.get("efficiency", {}).get("score", 50)),
                readability_score=int(dims.get("readability", {}).get("score", 50)),
                best_practices_score=int(
                    dims.get("best_practices", {}).get("score", 50)
                ),
                overall_score=int(classification.get("overall_score", 50)),
                overall_grade=classification.get("overall_grade", "C"),
                # Code smells
                code_smells=analysis.get("code_smells", []),
                code_smells_count=len(analysis.get("code_smells", [])),
                # Complexity
                cyclomatic_complexity=analysis.get("complexity", {}).get(
                    "cyclomatic", 1
                ),
                cognitive_complexity=analysis.get("complexity", {}).get("cognitive", 0),
                max_nesting_depth=analysis.get("complexity", {}).get(
                    "nesting_depth", 0
                ),
                lines_of_code=len(code.strip().split("\n")),
                # Patterns
                detected_patterns=[
                    p["pattern_id"] for p in analysis.get("patterns", [])
                ],
                # Suggestions
                suggestions=suggestions,
                suggestions_count=len(suggestions),
                # Metadata
                language=language,
                analyzer_version=ANALYZER_VERSION,
                analyzed_at=utc_now(),
            )

            self.session.add(quality_analysis)
            await self.session.commit()
            await self.session.refresh(quality_analysis)

            logger.info(
                "submission_quality_analyzed",
                submission_id=str(submission_id),
                overall_score=quality_analysis.overall_score,
                grade=quality_analysis.overall_grade,
            )

            return quality_analysis

        except Exception as e:
            logger.error(
                "quality_analysis_failed",
                submission_id=str(submission_id),
                error=str(e),
            )
            # Create minimal analysis on failure
            quality_analysis = CodeQualityAnalysisModel(
                submission_id=submission_id,
                user_id=user_id,
                problem_id=problem_id,
                overall_score=50,
                overall_grade="C",
                language=language,
                analyzer_version=ANALYZER_VERSION,
            )
            self.session.add(quality_analysis)
            await self.session.commit()
            await self.session.refresh(quality_analysis)
            return quality_analysis

    async def get_submission_quality(
        self, submission_id: UUID
    ) -> CodeQualityAnalysisModel | None:
        """Get quality analysis for a submission."""
        query = select(CodeQualityAnalysisModel).where(
            CodeQualityAnalysisModel.submission_id == submission_id
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def get_user_quality_stats(self, user_id: UUID) -> dict:
        """Get aggregated quality statistics for a user.

        Returns:
            Dict with average scores, grade distribution, etc.
        """
        query = select(
            func.count(CodeQualityAnalysisModel.id).label("total_analyses"),
            func.avg(CodeQualityAnalysisModel.overall_score).label("avg_overall"),
            func.avg(CodeQualityAnalysisModel.correctness_score).label(
                "avg_correctness"
            ),
            func.avg(CodeQualityAnalysisModel.efficiency_score).label("avg_efficiency"),
            func.avg(CodeQualityAnalysisModel.readability_score).label(
                "avg_readability"
            ),
            func.avg(CodeQualityAnalysisModel.best_practices_score).label(
                "avg_best_practices"
            ),
            func.avg(CodeQualityAnalysisModel.cyclomatic_complexity).label(
                "avg_cyclomatic"
            ),
            func.sum(CodeQualityAnalysisModel.code_smells_count).label("total_smells"),
        ).where(CodeQualityAnalysisModel.user_id == user_id)

        result = await self.session.execute(query)
        row = result.one()

        if not row.total_analyses:
            return {
                "total_analyses": 0,
                "avg_overall": 0,
                "avg_correctness": 0,
                "avg_efficiency": 0,
                "avg_readability": 0,
                "avg_best_practices": 0,
                "avg_cyclomatic": 0,
                "total_smells": 0,
                "grade_distribution": {},
            }

        # Get grade distribution
        grade_query = (
            select(
                CodeQualityAnalysisModel.overall_grade,
                func.count(CodeQualityAnalysisModel.id).label("count"),
            )
            .where(CodeQualityAnalysisModel.user_id == user_id)
            .group_by(CodeQualityAnalysisModel.overall_grade)
        )
        grade_result = await self.session.execute(grade_query)
        grade_dist = {row.overall_grade: row.count for row in grade_result}

        return {
            "total_analyses": row.total_analyses,
            "avg_overall": round(float(row.avg_overall or 0), 1),
            "avg_correctness": round(float(row.avg_correctness or 0), 1),
            "avg_efficiency": round(float(row.avg_efficiency or 0), 1),
            "avg_readability": round(float(row.avg_readability or 0), 1),
            "avg_best_practices": round(float(row.avg_best_practices or 0), 1),
            "avg_cyclomatic": round(float(row.avg_cyclomatic or 0), 1),
            "total_smells": int(row.total_smells or 0),
            "grade_distribution": grade_dist,
        }

    async def get_quality_trends(self, user_id: UUID, days: int = 30) -> list[dict]:
        """Get quality score trends for a user.

        Args:
            user_id: User UUID
            days: Number of days to look back

        Returns:
            List of daily quality metrics
        """
        start_date = date.today() - timedelta(days=days)

        query = (
            select(QualityTrendModel)
            .where(
                QualityTrendModel.user_id == user_id,
                QualityTrendModel.trend_date >= start_date,
            )
            .order_by(QualityTrendModel.trend_date)
        )

        result = await self.session.execute(query)
        trends = result.scalars().all()

        return [
            {
                "date": str(t.trend_date),
                "avg_overall": t.avg_overall_score,
                "avg_correctness": t.avg_correctness,
                "avg_efficiency": t.avg_efficiency,
                "avg_readability": t.avg_readability,
                "avg_best_practices": t.avg_best_practices,
                "submissions_analyzed": t.submissions_analyzed,
                "improved_count": t.improved_count,
            }
            for t in trends
        ]

    async def aggregate_daily_trends(
        self, target_date: date | None = None, user_id: UUID | None = None
    ) -> int:
        """Aggregate quality analysis into daily trends.

        Args:
            target_date: Date to aggregate (default: yesterday)
            user_id: Specific user to aggregate (default: all users)

        Returns:
            Number of trend records created/updated
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)

        # Get users with analyses on target date
        user_filter = []
        if user_id:
            user_filter.append(CodeQualityAnalysisModel.user_id == user_id)

        query = (
            select(
                CodeQualityAnalysisModel.user_id,
                func.count(CodeQualityAnalysisModel.id).label("count"),
                func.avg(CodeQualityAnalysisModel.overall_score).label("avg_overall"),
                func.avg(CodeQualityAnalysisModel.correctness_score).label(
                    "avg_correctness"
                ),
                func.avg(CodeQualityAnalysisModel.efficiency_score).label(
                    "avg_efficiency"
                ),
                func.avg(CodeQualityAnalysisModel.readability_score).label(
                    "avg_readability"
                ),
                func.avg(CodeQualityAnalysisModel.best_practices_score).label(
                    "avg_best_practices"
                ),
                func.avg(CodeQualityAnalysisModel.cyclomatic_complexity).label(
                    "avg_cyclomatic"
                ),
                func.avg(CodeQualityAnalysisModel.cognitive_complexity).label(
                    "avg_cognitive"
                ),
                func.sum(CodeQualityAnalysisModel.code_smells_count).label(
                    "total_smells"
                ),
                func.sum(CodeQualityAnalysisModel.suggestions_count).label(
                    "total_suggestions"
                ),
            )
            .where(
                func.date(CodeQualityAnalysisModel.analyzed_at) == target_date,
                *user_filter,
            )
            .group_by(CodeQualityAnalysisModel.user_id)
        )

        result = await self.session.execute(query)
        rows = result.all()

        count = 0
        for row in rows:
            # Check if trend exists
            existing_query = select(QualityTrendModel).where(
                QualityTrendModel.user_id == row.user_id,
                QualityTrendModel.trend_date == target_date,
            )
            existing_result = await self.session.execute(existing_query)
            existing = existing_result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.avg_overall_score = float(row.avg_overall or 0)
                existing.avg_correctness = float(row.avg_correctness or 0)
                existing.avg_efficiency = float(row.avg_efficiency or 0)
                existing.avg_readability = float(row.avg_readability or 0)
                existing.avg_best_practices = float(row.avg_best_practices or 0)
                existing.avg_cyclomatic = float(row.avg_cyclomatic or 0)
                existing.avg_cognitive = float(row.avg_cognitive or 0)
                existing.submissions_analyzed = row.count
                existing.total_smells = int(row.total_smells or 0)
                existing.total_suggestions = int(row.total_suggestions or 0)
            else:
                # Create new
                trend = QualityTrendModel(
                    user_id=row.user_id,
                    trend_date=target_date,
                    avg_overall_score=float(row.avg_overall or 0),
                    avg_correctness=float(row.avg_correctness or 0),
                    avg_efficiency=float(row.avg_efficiency or 0),
                    avg_readability=float(row.avg_readability or 0),
                    avg_best_practices=float(row.avg_best_practices or 0),
                    avg_cyclomatic=float(row.avg_cyclomatic or 0),
                    avg_cognitive=float(row.avg_cognitive or 0),
                    submissions_analyzed=row.count,
                    total_smells=int(row.total_smells or 0),
                    total_suggestions=int(row.total_suggestions or 0),
                )
                self.session.add(trend)

            count += 1

        await self.session.commit()
        logger.info(
            "quality_trends_aggregated",
            date=str(target_date),
            records=count,
        )

        return count

    async def get_recent_analyses(
        self, user_id: UUID, limit: int = 10
    ) -> list[CodeQualityAnalysisModel]:
        """Get recent quality analyses for a user."""
        query = (
            select(CodeQualityAnalysisModel)
            .where(CodeQualityAnalysisModel.user_id == user_id)
            .order_by(CodeQualityAnalysisModel.analyzed_at.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_problem_quality_stats(self, problem_id: UUID) -> dict:
        """Get quality statistics for a problem."""
        query = select(
            func.count(CodeQualityAnalysisModel.id).label("total"),
            func.avg(CodeQualityAnalysisModel.overall_score).label("avg_score"),
            func.avg(CodeQualityAnalysisModel.cyclomatic_complexity).label(
                "avg_complexity"
            ),
        ).where(CodeQualityAnalysisModel.problem_id == problem_id)

        result = await self.session.execute(query)
        row = result.one()

        return {
            "total_submissions": row.total or 0,
            "avg_quality_score": round(float(row.avg_score or 0), 1),
            "avg_complexity": round(float(row.avg_complexity or 0), 1),
        }


async def get_quality_service(session: AsyncSession) -> CodeQualityService:
    """Factory function to get quality service."""
    return CodeQualityService(session)
