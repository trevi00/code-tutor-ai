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


class WeaknessAnalyzer:
    """
    사용자 약점 분석 및 맞춤형 문제 추천.

    제출 기록을 분석하여 약점 패턴을 식별하고,
    약점 강화를 위한 문제를 추천합니다.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_weakness_patterns(self, user_id: UUID) -> list[dict]:
        """
        사용자의 약점 패턴을 분석합니다.

        Args:
            user_id: 사용자 ID

        Returns:
            약점 패턴 리스트 (낮은 성공률 순)
        """
        # 사용자 제출 기록 조회
        submissions_query = select(SubmissionModel).where(
            SubmissionModel.user_id == user_id
        )
        result = await self.session.execute(submissions_query)
        submissions = result.scalars().all()

        if not submissions:
            return []

        # 문제 ID 수집
        problem_ids = list(set(sub.problem_id for sub in submissions))

        # 문제 정보 조회
        problems_query = select(ProblemModel).where(ProblemModel.id.in_(problem_ids))
        problems_result = await self.session.execute(problems_query)
        problems = {p.id: p for p in problems_result.scalars().all()}

        # 카테고리 및 패턴별 통계 계산
        category_stats: dict[str, dict] = {}
        pattern_stats: dict[str, dict] = {}

        for sub in submissions:
            if sub.problem_id not in problems:
                continue

            problem = problems[sub.problem_id]
            category = problem.category.value

            # 카테고리 통계
            if category not in category_stats:
                category_stats[category] = {"attempts": 0, "solved": 0}
            category_stats[category]["attempts"] += 1
            if sub.status == SubmissionStatus.ACCEPTED:
                category_stats[category]["solved"] += 1

            # 패턴 통계
            for pattern in problem.pattern_ids or []:
                if pattern not in pattern_stats:
                    pattern_stats[pattern] = {"attempts": 0, "solved": 0, "problems": set()}
                pattern_stats[pattern]["attempts"] += 1
                pattern_stats[pattern]["problems"].add(sub.problem_id)
                if sub.status == SubmissionStatus.ACCEPTED:
                    pattern_stats[pattern]["solved"] += 1

        # 약점 패턴 식별 (성공률 < 50%)
        weaknesses = []

        for category, stats in category_stats.items():
            if stats["attempts"] >= 2:  # 최소 2번 시도
                success_rate = stats["solved"] / stats["attempts"]
                if success_rate < 0.5:
                    weaknesses.append({
                        "type": "category",
                        "name": category,
                        "success_rate": success_rate,
                        "attempts": stats["attempts"],
                        "solved": stats["solved"],
                        "severity": "high" if success_rate < 0.3 else "medium",
                        "recommendation": f"{category} 카테고리의 기초 문제부터 다시 연습해보세요.",
                    })

        for pattern, stats in pattern_stats.items():
            if stats["attempts"] >= 2:
                success_rate = stats["solved"] / stats["attempts"]
                if success_rate < 0.5:
                    weaknesses.append({
                        "type": "pattern",
                        "name": pattern,
                        "success_rate": success_rate,
                        "attempts": stats["attempts"],
                        "solved": stats["solved"],
                        "problem_count": len(stats["problems"]),
                        "severity": "high" if success_rate < 0.3 else "medium",
                        "recommendation": f"{pattern} 패턴을 집중적으로 연습해보세요.",
                    })

        # 성공률 오름차순 정렬 (가장 약한 것 먼저)
        weaknesses.sort(key=lambda x: x["success_rate"])

        return weaknesses

    async def get_targeted_recommendations(
        self,
        user_id: UUID,
        limit: int = 5,
        focus_weakness: str | None = None,
    ) -> list[dict]:
        """
        약점 기반 맞춤형 문제 추천.

        Args:
            user_id: 사용자 ID
            limit: 추천 문제 수
            focus_weakness: 특정 약점에 집중 (카테고리 또는 패턴 이름)

        Returns:
            추천 문제 리스트
        """
        weaknesses = await self.get_weakness_patterns(user_id)

        if not weaknesses:
            # 약점이 없으면 일반 추천
            return []

        # 가장 약한 패턴/카테고리 선택
        if focus_weakness:
            target = next(
                (w for w in weaknesses if w["name"] == focus_weakness),
                weaknesses[0]
            )
        else:
            target = weaknesses[0]

        # 해당 카테고리/패턴의 문제 조회
        if target["type"] == "category":
            from code_tutor.learning.domain.value_objects import ProblemCategory
            try:
                category_enum = ProblemCategory(target["name"])
                problems_query = select(ProblemModel).where(
                    ProblemModel.category == category_enum,
                    ProblemModel.is_published == True,
                )
            except ValueError:
                problems_query = select(ProblemModel).where(
                    ProblemModel.is_published == True
                )
        else:
            # 패턴 기반 조회
            problems_query = select(ProblemModel).where(
                ProblemModel.is_published == True,
            )

        result = await self.session.execute(problems_query)
        problems = result.scalars().all()

        # 사용자가 이미 풀었던 문제 조회
        solved_query = select(SubmissionModel.problem_id).where(
            SubmissionModel.user_id == user_id,
            SubmissionModel.status == SubmissionStatus.ACCEPTED,
        ).distinct()
        solved_result = await self.session.execute(solved_query)
        solved_ids = set(r[0] for r in solved_result.all())

        # 미풀이 문제 필터링 및 난이도순 정렬
        recommendations = []
        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}

        for p in problems:
            if p.id in solved_ids:
                continue

            # 패턴 매칭 확인
            if target["type"] == "pattern":
                if target["name"] not in (p.pattern_ids or []):
                    continue

            recommendations.append({
                "id": str(p.id),
                "title": p.title,
                "difficulty": p.difficulty.value,
                "category": p.category.value,
                "pattern_ids": p.pattern_ids or [],
                "reason": f"약점 보강: {target['name']} ({target['type']})",
                "weakness_info": {
                    "success_rate": target["success_rate"],
                    "attempts": target["attempts"],
                },
            })

        # 난이도순 정렬 (쉬운 것부터)
        recommendations.sort(
            key=lambda x: difficulty_order.get(x["difficulty"], 1)
        )

        return recommendations[:limit]

    async def get_learning_path(self, user_id: UUID) -> dict:
        """
        사용자 맞춤형 학습 경로 제안.

        Args:
            user_id: 사용자 ID

        Returns:
            학습 경로 정보
        """
        weaknesses = await self.get_weakness_patterns(user_id)

        # 제출 통계 조회
        submissions_query = select(SubmissionModel).where(
            SubmissionModel.user_id == user_id
        )
        result = await self.session.execute(submissions_query)
        submissions = result.scalars().all()

        total_submissions = len(submissions)
        solved_count = sum(
            1 for s in submissions if s.status == SubmissionStatus.ACCEPTED
        )

        # 학습 단계 결정
        if total_submissions < 10:
            stage = "beginner"
            stage_description = "기초 단계"
            next_milestone = "10문제 풀이"
        elif solved_count < 20:
            stage = "intermediate"
            stage_description = "중급 단계"
            next_milestone = "20문제 해결"
        elif solved_count < 50:
            stage = "advanced"
            stage_description = "고급 단계"
            next_milestone = "50문제 해결"
        else:
            stage = "expert"
            stage_description = "전문가 단계"
            next_milestone = "대회 문제 도전"

        # 추천 학습 순서
        if weaknesses:
            priority_focus = [w["name"] for w in weaknesses[:3]]
        else:
            priority_focus = ["array", "string", "dynamic_programming"]

        return {
            "current_stage": stage,
            "stage_description": stage_description,
            "total_submissions": total_submissions,
            "problems_solved": solved_count,
            "success_rate": solved_count / total_submissions if total_submissions > 0 else 0,
            "next_milestone": next_milestone,
            "weaknesses": weaknesses[:5],
            "priority_focus": priority_focus,
            "recommended_next_steps": [
                f"약점 보강: {weaknesses[0]['name']}" if weaknesses else "기초 문제 연습",
                "일일 1문제 풀이 도전",
                "풀이 시간 단축 연습",
            ],
        }


async def get_weakness_analyzer(session: AsyncSession) -> WeaknessAnalyzer:
    """Factory function to get weakness analyzer."""
    return WeaknessAnalyzer(session)
