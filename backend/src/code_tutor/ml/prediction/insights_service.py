"""Learning Insights Service

Integrates LSTM predictions with dashboard insights.
"""

from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.ml.pipeline.cache import RecommendationCache
from code_tutor.ml.pipeline.daily_stats_service import DailyStatsService
from code_tutor.ml.prediction.learning_predictor import LearningPredictor
from code_tutor.ml.recommendation import RecommenderService

logger = structlog.get_logger()

# Global predictor instance (singleton)
_predictor: LearningPredictor | None = None


class InsightsService:
    """Service for generating ML-based learning insights."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._stats_service = DailyStatsService(session)
        self._cache: RecommendationCache | None = None

    async def _get_cache(self) -> RecommendationCache:
        """Get or create cache instance."""
        if self._cache is None:
            self._cache = await RecommendationCache.create()
        return self._cache

    def _get_predictor(self) -> LearningPredictor:
        """Get or create predictor instance."""
        global _predictor
        if _predictor is None:
            _predictor = LearningPredictor()
            _predictor.initialize()
        return _predictor

    async def get_full_insights(
        self,
        user_id: UUID,
        use_cache: bool = True,
    ) -> dict:
        """Get comprehensive learning insights for a user.

        Args:
            user_id: User UUID
            use_cache: Whether to use cache

        Returns:
            Complete insights dictionary
        """
        cache = await self._get_cache()

        # Check cache first
        if use_cache and cache.is_available:
            cached = await cache.get_predictions(user_id, "insights")
            if cached:
                logger.debug("insights_cache_hit", user_id=str(user_id))
                return cached

        # Get user's daily stats sequence
        user_history = await self._stats_service.get_user_stats_sequence(
            user_id, days=30
        )

        # Get predictor
        predictor = self._get_predictor()

        # Generate insights
        velocity = predictor.analyze_learning_velocity(user_history)
        prediction = predictor.predict_success_rate(user_history)
        schedule = predictor.recommend_study_schedule(user_history)
        insights = predictor.get_insights(user_history)

        # Get skill gaps from recommender
        skill_gaps = await self._get_skill_gaps(user_id)

        result = {
            "velocity": velocity,
            "prediction": prediction,
            "schedule": schedule,
            "skill_gaps": skill_gaps,
            "insights": insights,
            "study_recommendations": schedule.get("recommendations", []),
        }

        # Cache results
        if use_cache and cache.is_available:
            await cache.set_predictions(user_id, result, "insights")

        return result

    async def get_velocity_analysis(self, user_id: UUID) -> dict:
        """Get learning velocity analysis.

        Args:
            user_id: User UUID

        Returns:
            Velocity analysis
        """
        user_history = await self._stats_service.get_user_stats_sequence(
            user_id, days=30
        )
        predictor = self._get_predictor()
        return predictor.analyze_learning_velocity(user_history)

    async def get_success_prediction(
        self,
        user_id: UUID,
        days_ahead: int = 7,
    ) -> dict:
        """Get success rate prediction.

        Args:
            user_id: User UUID
            days_ahead: Days to predict

        Returns:
            Prediction results
        """
        cache = await self._get_cache()

        # Check cache
        if cache.is_available:
            cached = await cache.get_predictions(user_id, "success_rate")
            if cached:
                return cached

        user_history = await self._stats_service.get_user_stats_sequence(
            user_id, days=30
        )
        predictor = self._get_predictor()
        result = predictor.predict_success_rate(user_history, days_ahead)

        # Cache
        if cache.is_available:
            await cache.set_predictions(user_id, result, "success_rate")

        return result

    async def get_study_recommendations(
        self,
        user_id: UUID,
        target_success_rate: float = 80,
    ) -> dict:
        """Get study schedule recommendations.

        Args:
            user_id: User UUID
            target_success_rate: Target success rate

        Returns:
            Study recommendations
        """
        user_history = await self._stats_service.get_user_stats_sequence(
            user_id, days=30
        )
        predictor = self._get_predictor()
        return predictor.recommend_study_schedule(user_history, target_success_rate)

    async def _get_skill_gaps(self, user_id: UUID) -> list[dict]:
        """Get skill gaps from recommender.

        Args:
            user_id: User UUID

        Returns:
            List of skill gaps
        """
        try:
            recommender = RecommenderService(self.session)
            return await recommender.get_skill_gaps(user_id)
        except Exception as e:
            logger.warning("failed_to_get_skill_gaps", error=str(e))
            return []

    async def invalidate_cache(self, user_id: UUID) -> None:
        """Invalidate insights cache for user."""
        cache = await self._get_cache()
        if cache.is_available:
            await cache.invalidate_predictions(user_id)


async def get_insights_service(session: AsyncSession) -> InsightsService:
    """Factory function to get insights service."""
    return InsightsService(session)
