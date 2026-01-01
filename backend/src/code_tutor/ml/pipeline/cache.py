"""ML Recommendation Cache

Redis-based caching for ML model predictions and recommendations.
"""

from datetime import datetime
from uuid import UUID

import structlog

from code_tutor.shared.infrastructure.redis import RedisClient

logger = structlog.get_logger()


class RecommendationCache:
    """Cache layer for ML recommendations and predictions."""

    # Cache key prefixes
    PREFIX_RECOMMENDATIONS = "ml:recommendations"
    PREFIX_PREDICTIONS = "ml:predictions"
    PREFIX_USER_STATS = "ml:user_stats"
    PREFIX_INTERACTION_MATRIX = "ml:interaction_matrix"

    # Default TTLs (in seconds)
    TTL_RECOMMENDATIONS = 3600  # 1 hour
    TTL_PREDICTIONS = 21600  # 6 hours
    TTL_USER_STATS = 1800  # 30 minutes
    TTL_INTERACTION_MATRIX = 86400  # 24 hours

    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client

    @classmethod
    async def create(cls) -> "RecommendationCache":
        """Factory method to create RecommendationCache."""
        redis_client = await RedisClient.create()
        return cls(redis_client)

    @property
    def is_available(self) -> bool:
        """Check if cache is available."""
        return self.redis.is_available

    # ============== Recommendations Cache ==============

    async def get_recommendations(
        self,
        user_id: UUID,
        strategy: str = "hybrid",
        limit: int = 10,
    ) -> list[dict] | None:
        """Get cached recommendations for a user.

        Args:
            user_id: User UUID
            strategy: Recommendation strategy (hybrid, collaborative, content)
            limit: Number of recommendations

        Returns:
            List of recommended problems or None if not cached
        """
        key = f"{self.PREFIX_RECOMMENDATIONS}:{user_id}:{strategy}:{limit}"
        cached = await self.redis.get_json(key)

        if cached:
            logger.debug("cache_hit", key=key)
            return cached.get("recommendations")

        logger.debug("cache_miss", key=key)
        return None

    async def set_recommendations(
        self,
        user_id: UUID,
        recommendations: list[dict],
        strategy: str = "hybrid",
        limit: int = 10,
        ttl: int | None = None,
    ) -> bool:
        """Cache recommendations for a user.

        Args:
            user_id: User UUID
            recommendations: List of recommended problems
            strategy: Recommendation strategy
            limit: Number of recommendations
            ttl: Cache TTL in seconds (default: 1 hour)

        Returns:
            True if cached successfully
        """
        key = f"{self.PREFIX_RECOMMENDATIONS}:{user_id}:{strategy}:{limit}"
        data = {
            "recommendations": recommendations,
            "cached_at": datetime.utcnow().isoformat(),
            "strategy": strategy,
        }

        ttl = ttl or self.TTL_RECOMMENDATIONS
        success = await self.redis.set_json(key, data, expire_seconds=ttl)

        if success:
            logger.debug("cache_set", key=key, ttl=ttl)

        return success

    async def invalidate_recommendations(self, user_id: UUID) -> int:
        """Invalidate all recommendations for a user.

        Called when user solves a problem or makes a submission.
        """
        # Invalidate all strategies
        deleted = 0
        for strategy in ["hybrid", "collaborative", "content"]:
            for limit in [5, 10, 20]:
                key = f"{self.PREFIX_RECOMMENDATIONS}:{user_id}:{strategy}:{limit}"
                deleted += await self.redis.delete(key)

        if deleted > 0:
            logger.debug("cache_invalidated", user_id=str(user_id), count=deleted)

        return deleted

    # ============== Predictions Cache ==============

    async def get_predictions(
        self,
        user_id: UUID,
        prediction_type: str = "success_rate",
    ) -> dict | None:
        """Get cached predictions for a user.

        Args:
            user_id: User UUID
            prediction_type: Type of prediction (success_rate, velocity, etc.)

        Returns:
            Prediction data or None if not cached
        """
        key = f"{self.PREFIX_PREDICTIONS}:{user_id}:{prediction_type}"
        cached = await self.redis.get_json(key)

        if cached:
            logger.debug("prediction_cache_hit", key=key)
            return cached

        return None

    async def set_predictions(
        self,
        user_id: UUID,
        predictions: dict,
        prediction_type: str = "success_rate",
        ttl: int | None = None,
    ) -> bool:
        """Cache predictions for a user.

        Args:
            user_id: User UUID
            predictions: Prediction data
            prediction_type: Type of prediction
            ttl: Cache TTL in seconds (default: 6 hours)

        Returns:
            True if cached successfully
        """
        key = f"{self.PREFIX_PREDICTIONS}:{user_id}:{prediction_type}"
        predictions["cached_at"] = datetime.utcnow().isoformat()

        ttl = ttl or self.TTL_PREDICTIONS
        return await self.redis.set_json(key, predictions, expire_seconds=ttl)

    async def invalidate_predictions(self, user_id: UUID) -> int:
        """Invalidate all predictions for a user."""
        deleted = 0
        for pred_type in ["success_rate", "velocity", "insights"]:
            key = f"{self.PREFIX_PREDICTIONS}:{user_id}:{pred_type}"
            deleted += await self.redis.delete(key)

        return deleted

    # ============== User Stats Cache ==============

    async def get_user_stats_sequence(
        self,
        user_id: UUID,
        days: int = 30,
    ) -> list[dict] | None:
        """Get cached user stats sequence for LSTM.

        Args:
            user_id: User UUID
            days: Number of days in sequence

        Returns:
            Stats sequence or None if not cached
        """
        key = f"{self.PREFIX_USER_STATS}:{user_id}:{days}"
        cached = await self.redis.get_json(key)

        if cached:
            return cached.get("sequence")

        return None

    async def set_user_stats_sequence(
        self,
        user_id: UUID,
        sequence: list[dict],
        days: int = 30,
        ttl: int | None = None,
    ) -> bool:
        """Cache user stats sequence.

        Args:
            user_id: User UUID
            sequence: Stats sequence
            days: Number of days
            ttl: Cache TTL in seconds (default: 30 minutes)

        Returns:
            True if cached successfully
        """
        key = f"{self.PREFIX_USER_STATS}:{user_id}:{days}"
        data = {
            "sequence": sequence,
            "cached_at": datetime.utcnow().isoformat(),
        }

        ttl = ttl or self.TTL_USER_STATS
        return await self.redis.set_json(key, data, expire_seconds=ttl)

    # ============== Interaction Matrix Cache ==============

    async def get_interaction_matrix(self) -> dict | None:
        """Get cached interaction matrix for NCF.

        Returns:
            Interaction matrix data or None if not cached
        """
        key = self.PREFIX_INTERACTION_MATRIX
        return await self.redis.get_json(key)

    async def set_interaction_matrix(
        self,
        matrix_data: dict,
        ttl: int | None = None,
    ) -> bool:
        """Cache interaction matrix.

        Args:
            matrix_data: Interaction matrix with user/problem mappings
            ttl: Cache TTL in seconds (default: 24 hours)

        Returns:
            True if cached successfully
        """
        key = self.PREFIX_INTERACTION_MATRIX
        matrix_data["cached_at"] = datetime.utcnow().isoformat()

        ttl = ttl or self.TTL_INTERACTION_MATRIX
        return await self.redis.set_json(key, matrix_data, expire_seconds=ttl)

    async def invalidate_interaction_matrix(self) -> int:
        """Invalidate interaction matrix cache."""
        return await self.redis.delete(self.PREFIX_INTERACTION_MATRIX)

    # ============== Bulk Operations ==============

    async def invalidate_user_cache(self, user_id: UUID) -> int:
        """Invalidate all cache entries for a user.

        Called when user makes a submission that affects ML predictions.
        """
        total_deleted = 0
        total_deleted += await self.invalidate_recommendations(user_id)
        total_deleted += await self.invalidate_predictions(user_id)

        logger.info(
            "user_cache_invalidated",
            user_id=str(user_id),
            deleted=total_deleted,
        )

        return total_deleted

    async def warm_cache_for_user(
        self,
        user_id: UUID,
        recommendations: list[dict] | None = None,
        predictions: dict | None = None,
        stats_sequence: list[dict] | None = None,
    ) -> dict:
        """Pre-populate cache for a user.

        Args:
            user_id: User UUID
            recommendations: Optional recommendations to cache
            predictions: Optional predictions to cache
            stats_sequence: Optional stats sequence to cache

        Returns:
            Summary of what was cached
        """
        cached = {}

        if recommendations:
            await self.set_recommendations(user_id, recommendations)
            cached["recommendations"] = True

        if predictions:
            await self.set_predictions(user_id, predictions)
            cached["predictions"] = True

        if stats_sequence:
            await self.set_user_stats_sequence(user_id, stats_sequence)
            cached["stats_sequence"] = True

        return cached
