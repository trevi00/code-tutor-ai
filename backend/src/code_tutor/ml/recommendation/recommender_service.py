"""Recommender Service

High-level service for problem recommendations with auto-initialization,
caching, and integration with the data pipeline.
"""

import asyncio
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.learning.domain.value_objects import SubmissionStatus
from code_tutor.learning.infrastructure.models import ProblemModel, SubmissionModel
from code_tutor.ml.pipeline.cache import RecommendationCache
from code_tutor.ml.pipeline.data_aggregator import DataAggregator
from code_tutor.ml.recommendation.recommender import ProblemRecommender

logger = structlog.get_logger()

# Global recommender instance (singleton)
_recommender: ProblemRecommender | None = None
_recommender_lock = asyncio.Lock()
_is_initializing = False


class RecommenderService:
    """Service for problem recommendations with auto-initialization."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self._cache: RecommendationCache | None = None

    async def _get_cache(self) -> RecommendationCache:
        """Get or create cache instance."""
        if self._cache is None:
            self._cache = await RecommendationCache.create()
        return self._cache

    async def _ensure_initialized(self) -> ProblemRecommender:
        """Ensure recommender is initialized."""
        global _recommender, _is_initializing

        if _recommender is not None and _recommender._is_initialized:
            return _recommender

        async with _recommender_lock:
            # Double-check after acquiring lock
            if _recommender is not None and _recommender._is_initialized:
                return _recommender

            if _is_initializing:
                logger.warning("recommender_already_initializing")
                # Wait for initialization to complete
                while _is_initializing:
                    await asyncio.sleep(0.1)
                if _recommender is not None:
                    return _recommender

            _is_initializing = True
            try:
                _recommender = await self._initialize_recommender()
            finally:
                _is_initializing = False

        return _recommender

    async def _initialize_recommender(self) -> ProblemRecommender:
        """Initialize recommender with data from database."""
        logger.info("initializing_recommender")

        # Get all published problems
        problems_query = select(ProblemModel).where(ProblemModel.is_published == True)
        problems_result = await self.session.execute(problems_query)
        problem_models = problems_result.scalars().all()

        problems = [
            {
                "id": p.id,
                "title": p.title,
                "difficulty": p.difficulty.value,
                "category": p.category.value,
                "patterns": p.pattern_ids or [],
                "acceptance_rate": 0.5,  # Will be calculated from submissions
            }
            for p in problem_models
        ]

        # Get all submissions for interactions
        submissions_query = select(SubmissionModel)
        submissions_result = await self.session.execute(submissions_query)
        submission_models = submissions_result.scalars().all()

        # Calculate acceptance rates
        problem_stats: dict[UUID, dict] = {}
        for sub in submission_models:
            if sub.problem_id not in problem_stats:
                problem_stats[sub.problem_id] = {"total": 0, "accepted": 0}
            problem_stats[sub.problem_id]["total"] += 1
            if sub.status == SubmissionStatus.ACCEPTED:
                problem_stats[sub.problem_id]["accepted"] += 1

        for p in problems:
            if p["id"] in problem_stats:
                stats = problem_stats[p["id"]]
                p["acceptance_rate"] = (
                    stats["accepted"] / stats["total"] if stats["total"] > 0 else 0.5
                )

        # Build interactions list
        interactions = [
            {
                "user_id": sub.user_id,
                "problem_id": sub.problem_id,
                "is_solved": sub.status == SubmissionStatus.ACCEPTED,
            }
            for sub in submission_models
        ]

        logger.info(
            "recommender_data_loaded",
            problems=len(problems),
            interactions=len(interactions),
        )

        # Initialize recommender
        recommender = ProblemRecommender()
        recommender.initialize(
            problems=problems,
            interactions=interactions,
            force_retrain=False,
        )

        logger.info("recommender_initialized")
        return recommender

    async def get_recommendations(
        self,
        user_id: UUID,
        limit: int = 10,
        strategy: str = "hybrid",
        difficulty_filter: str | None = None,
        category_filter: str | None = None,
        use_cache: bool = True,
    ) -> list[dict]:
        """Get personalized problem recommendations.

        Args:
            user_id: User UUID
            limit: Number of recommendations
            strategy: "hybrid", "collaborative", or "content"
            difficulty_filter: Filter by difficulty
            category_filter: Filter by category
            use_cache: Whether to use cache

        Returns:
            List of recommended problems
        """
        cache = await self._get_cache()

        # Check cache first
        if use_cache and cache.is_available:
            cached = await cache.get_recommendations(user_id, strategy, limit)
            if cached:
                logger.debug("recommendations_cache_hit", user_id=str(user_id))
                return cached

        # Get recommendations
        recommender = await self._ensure_initialized()
        recommendations = recommender.recommend(
            user_id=user_id,
            top_k=limit,
            strategy=strategy,
            difficulty_filter=difficulty_filter,
            category_filter=category_filter,
        )

        # Enrich with problem details
        enriched = await self._enrich_recommendations(recommendations)

        # Cache results
        if use_cache and cache.is_available:
            await cache.set_recommendations(user_id, enriched, strategy, limit)

        return enriched

    async def _enrich_recommendations(self, recommendations: list[dict]) -> list[dict]:
        """Enrich recommendations with full problem details."""
        if not recommendations:
            return []

        problem_ids = [rec["problem_id"] for rec in recommendations]
        problems_query = select(ProblemModel).where(ProblemModel.id.in_(problem_ids))
        result = await self.session.execute(problems_query)
        problems = {p.id: p for p in result.scalars().all()}

        enriched = []
        for rec in recommendations:
            pid = rec["problem_id"]
            if pid in problems:
                p = problems[pid]
                enriched.append(
                    {
                        "id": str(pid),
                        "title": p.title,
                        "difficulty": p.difficulty.value,
                        "category": p.category.value,
                        "score": rec.get("score", 0.5),
                        "reason": rec.get("reason", "recommended"),
                        "pattern_ids": p.pattern_ids or [],
                    }
                )

        return enriched

    async def get_skill_gaps(self, user_id: UUID) -> list[dict]:
        """Get skill gaps for a user."""
        recommender = await self._ensure_initialized()
        return recommender.get_skill_gaps(user_id)

    async def get_next_challenge(self, user_id: UUID) -> dict | None:
        """Get next challenge problem for a user."""
        recommender = await self._ensure_initialized()
        challenge = recommender.get_next_challenge(user_id)

        if challenge:
            # Enrich with full details
            enriched = await self._enrich_recommendations([challenge])
            return enriched[0] if enriched else None

        return None

    async def invalidate_user_cache(self, user_id: UUID) -> None:
        """Invalidate recommendations cache for a user.

        Should be called when user solves a problem.
        """
        cache = await self._get_cache()
        if cache.is_available:
            await cache.invalidate_user_cache(user_id)
            logger.debug("user_cache_invalidated", user_id=str(user_id))

    async def update_user_interaction(
        self,
        user_id: UUID,
        problem_id: UUID,
        is_solved: bool,
    ) -> None:
        """Update user interaction after submission.

        Updates the recommender's user history and invalidates cache.
        """
        global _recommender

        if _recommender is not None and _recommender._is_initialized:
            # Update user history in memory
            if user_id not in _recommender._user_history:
                _recommender._user_history[user_id] = set()

            if is_solved:
                _recommender._user_history[user_id].add(problem_id)

        # Also update in database via aggregator
        aggregator = DataAggregator(self.session)
        await aggregator.aggregate_user_interactions(user_id)

        # Invalidate cache
        await self.invalidate_user_cache(user_id)

        logger.debug(
            "user_interaction_updated",
            user_id=str(user_id),
            problem_id=str(problem_id),
            is_solved=is_solved,
        )


async def get_recommender_service(session: AsyncSession) -> RecommenderService:
    """Factory function to get recommender service."""
    return RecommenderService(session)


def reset_recommender() -> None:
    """Reset the global recommender instance.

    Useful for testing or when model needs to be retrained.
    """
    global _recommender
    _recommender = None
    logger.info("recommender_reset")
