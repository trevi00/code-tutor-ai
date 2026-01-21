"""Quality-Based Recommender Service

Provides problem recommendations based on code quality analysis.
Identifies weak quality dimensions and suggests problems to improve them.
"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.learning.infrastructure.models import ProblemModel
from code_tutor.ml.pipeline.models import CodeQualityAnalysisModel
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)

# Quality dimension thresholds
WEAK_DIMENSION_THRESHOLD = 60  # Below this is considered weak
STRONG_DIMENSION_THRESHOLD = 80  # Above this is considered strong

# Problem patterns that help with specific quality dimensions
DIMENSION_PATTERNS: dict[str, list[str]] = {
    "correctness": [
        "edge-cases",
        "boundary-conditions",
        "input-validation",
        "test-driven",
    ],
    "efficiency": [
        "two-pointers",
        "sliding-window",
        "binary-search",
        "dynamic-programming",
        "memoization",
        "greedy",
    ],
    "readability": [
        "clean-code",
        "modular-design",
        "helper-functions",
        "descriptive-naming",
    ],
    "best_practices": [
        "error-handling",
        "defensive-programming",
        "code-organization",
        "dry-principle",
    ],
}

# Category to quality dimension mapping
CATEGORY_DIMENSION_BOOST: dict[str, dict[str, float]] = {
    "array": {"efficiency": 0.3, "correctness": 0.2},
    "string": {"readability": 0.2, "efficiency": 0.2},
    "hash_table": {"efficiency": 0.4},
    "linked_list": {"correctness": 0.3, "best_practices": 0.2},
    "stack": {"best_practices": 0.2, "readability": 0.2},
    "queue": {"best_practices": 0.2, "readability": 0.2},
    "tree": {"correctness": 0.3, "efficiency": 0.2},
    "graph": {"efficiency": 0.3, "correctness": 0.3},
    "dp": {"efficiency": 0.5, "correctness": 0.2},
    "greedy": {"efficiency": 0.3, "best_practices": 0.2},
    "binary_search": {"efficiency": 0.4, "correctness": 0.3},
    "sorting": {"efficiency": 0.3, "readability": 0.2},
    "math": {"correctness": 0.4, "readability": 0.2},
    "bit_manipulation": {"efficiency": 0.3, "best_practices": 0.2},
    "recursion": {"readability": 0.3, "correctness": 0.3},
}

# Difficulty to complexity expectation
DIFFICULTY_COMPLEXITY: dict[str, dict[str, int]] = {
    "easy": {"max_cyclomatic": 5, "max_cognitive": 8},
    "medium": {"max_cyclomatic": 10, "max_cognitive": 15},
    "hard": {"max_cyclomatic": 15, "max_cognitive": 25},
}


class QualityRecommender:
    """Recommender based on code quality analysis."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_quality_profile(self, user_id: UUID) -> dict:
        """Get user's quality profile based on their submissions.

        Returns:
            Dict with dimension averages, weak areas, and improvement trends
        """
        # Get recent quality analyses (last 20)
        query = (
            select(CodeQualityAnalysisModel)
            .where(CodeQualityAnalysisModel.user_id == user_id)
            .order_by(CodeQualityAnalysisModel.analyzed_at.desc())
            .limit(20)
        )
        result = await self.session.execute(query)
        analyses = list(result.scalars().all())

        if not analyses:
            return {
                "has_data": False,
                "dimensions": {
                    "correctness": 50,
                    "efficiency": 50,
                    "readability": 50,
                    "best_practices": 50,
                },
                "weak_areas": [],
                "strong_areas": [],
                "common_smells": [],
                "avg_complexity": 1,
                "improvement_trend": "new_user",
            }

        # Calculate dimension averages
        dimensions = {
            "correctness": sum(a.correctness_score for a in analyses) / len(analyses),
            "efficiency": sum(a.efficiency_score for a in analyses) / len(analyses),
            "readability": sum(a.readability_score for a in analyses) / len(analyses),
            "best_practices": sum(a.best_practices_score for a in analyses)
            / len(analyses),
        }

        # Identify weak and strong areas
        weak_areas = [
            dim for dim, score in dimensions.items() if score < WEAK_DIMENSION_THRESHOLD
        ]
        strong_areas = [
            dim
            for dim, score in dimensions.items()
            if score >= STRONG_DIMENSION_THRESHOLD
        ]

        # Aggregate common code smells
        smell_counts: dict[str, int] = {}
        for analysis in analyses:
            for smell in analysis.code_smells or []:
                smell_type = smell.get("type", "unknown")
                smell_counts[smell_type] = smell_counts.get(smell_type, 0) + 1

        common_smells = sorted(smell_counts.items(), key=lambda x: -x[1])[:5]

        # Calculate average complexity
        avg_complexity = sum(a.cyclomatic_complexity for a in analyses) / len(analyses)

        # Determine improvement trend (compare first half vs second half)
        if len(analyses) >= 4:
            first_half = analyses[len(analyses) // 2 :]
            second_half = analyses[: len(analyses) // 2]
            first_avg = sum(a.overall_score for a in first_half) / len(first_half)
            second_avg = sum(a.overall_score for a in second_half) / len(second_half)
            if second_avg > first_avg + 5:
                trend = "improving"
            elif second_avg < first_avg - 5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "has_data": True,
            "dimensions": dimensions,
            "weak_areas": weak_areas,
            "strong_areas": strong_areas,
            "common_smells": common_smells,
            "avg_complexity": avg_complexity,
            "improvement_trend": trend,
            "total_analyses": len(analyses),
        }

    async def get_quality_recommendations(
        self,
        user_id: UUID,
        limit: int = 5,
    ) -> list[dict]:
        """Get problem recommendations based on quality profile.

        Recommends problems that will help improve weak quality dimensions.

        Args:
            user_id: User UUID
            limit: Max number of recommendations

        Returns:
            List of recommended problems with quality focus reasons
        """
        profile = await self.get_quality_profile(user_id)

        if not profile["has_data"]:
            # Return beginner-friendly problems
            return await self._get_beginner_recommendations(limit)

        weak_areas = profile["weak_areas"]
        dimensions = profile["dimensions"]

        # Get all published problems
        problems_query = select(ProblemModel).where(ProblemModel.is_published == True)
        result = await self.session.execute(problems_query)
        problems = list(result.scalars().all())

        # Get user's solved problems
        solved_query = (
            select(CodeQualityAnalysisModel.problem_id)
            .where(CodeQualityAnalysisModel.user_id == user_id)
            .distinct()
        )
        solved_result = await self.session.execute(solved_query)
        solved_ids = {row[0] for row in solved_result}

        # Score problems based on quality improvement potential
        scored_problems = []
        for problem in problems:
            if problem.id in solved_ids:
                continue  # Skip already attempted

            score, reasons = self._calculate_quality_score(
                problem, weak_areas, dimensions, profile["avg_complexity"]
            )

            if score > 0:
                scored_problems.append(
                    {
                        "problem_id": problem.id,
                        "title": problem.title,
                        "difficulty": problem.difficulty.value,
                        "category": problem.category.value,
                        "score": score,
                        "quality_focus": reasons,
                        "pattern_ids": problem.pattern_ids or [],
                    }
                )

        # Sort by score and return top recommendations
        scored_problems.sort(key=lambda x: -x["score"])

        recommendations = []
        for p in scored_problems[:limit]:
            recommendations.append(
                {
                    "id": str(p["problem_id"]),
                    "title": p["title"],
                    "difficulty": p["difficulty"],
                    "category": p["category"],
                    "score": round(p["score"], 2),
                    "reason": f"품질 향상: {', '.join(p['quality_focus'][:2])}",
                    "quality_focus": p["quality_focus"],
                    "pattern_ids": p["pattern_ids"],
                }
            )

        return recommendations

    def _calculate_quality_score(
        self,
        problem: ProblemModel,
        weak_areas: list[str],
        dimensions: dict[str, float],
        avg_complexity: float,
    ) -> tuple[float, list[str]]:
        """Calculate quality improvement score for a problem.

        Returns:
            Tuple of (score, list of focus reasons)
        """
        score = 0.0
        reasons = []
        category = problem.category.value
        patterns = problem.pattern_ids or []

        # Score based on category's dimension boost
        if category in CATEGORY_DIMENSION_BOOST:
            boosts = CATEGORY_DIMENSION_BOOST[category]
            for dim, boost in boosts.items():
                if dim in weak_areas:
                    score += boost * 1.5  # Extra boost for weak areas
                    reasons.append(self._get_dimension_label(dim))
                else:
                    score += boost * 0.5

        # Score based on patterns that help with dimensions
        for dim, helpful_patterns in DIMENSION_PATTERNS.items():
            overlap = set(patterns) & set(helpful_patterns)
            if overlap:
                if dim in weak_areas:
                    score += len(overlap) * 0.3
                    if self._get_dimension_label(dim) not in reasons:
                        reasons.append(self._get_dimension_label(dim))
                else:
                    score += len(overlap) * 0.1

        # Adjust score based on difficulty progression
        difficulty = problem.difficulty.value
        complexity_limits = DIFFICULTY_COMPLEXITY.get(difficulty, {})
        max_cyclomatic = complexity_limits.get("max_cyclomatic", 10)

        # Recommend problems that challenge but don't overwhelm
        if avg_complexity < max_cyclomatic * 0.7:
            score += 0.2  # Good complexity progression
        elif avg_complexity > max_cyclomatic:
            score -= 0.3  # Too complex for current level

        # Boost scores for weakest dimension
        if weak_areas:
            weakest = min(weak_areas, key=lambda d: dimensions.get(d, 50))
            if any(weakest in CATEGORY_DIMENSION_BOOST.get(category, {}) for _ in [1]):
                score += 0.3
                if self._get_dimension_label(weakest) not in reasons:
                    reasons.insert(0, self._get_dimension_label(weakest))

        return score, reasons

    def _get_dimension_label(self, dimension: str) -> str:
        """Get Korean label for dimension."""
        labels = {
            "correctness": "정확성 향상",
            "efficiency": "효율성 향상",
            "readability": "가독성 향상",
            "best_practices": "모범사례 학습",
        }
        return labels.get(dimension, dimension)

    async def _get_beginner_recommendations(self, limit: int) -> list[dict]:
        """Get recommendations for new users."""
        # Get easy problems with good patterns
        query = (
            select(ProblemModel)
            .where(
                ProblemModel.is_published == True,
                ProblemModel.difficulty == "easy",
            )
            .limit(limit)
        )
        result = await self.session.execute(query)
        problems = result.scalars().all()

        return [
            {
                "id": str(p.id),
                "title": p.title,
                "difficulty": p.difficulty.value,
                "category": p.category.value,
                "score": 0.7,
                "reason": "초보자 추천 문제",
                "quality_focus": ["기초 다지기"],
                "pattern_ids": p.pattern_ids or [],
            }
            for p in problems
        ]

    async def get_improvement_suggestions(self, user_id: UUID) -> list[dict]:
        """Get personalized improvement suggestions based on quality analysis.

        Returns:
            List of actionable suggestions
        """
        profile = await self.get_quality_profile(user_id)

        if not profile["has_data"]:
            return [
                {
                    "type": "start",
                    "priority": 1,
                    "message": "코드를 제출하면 품질 분석을 받을 수 있어요!",
                    "action": "문제 풀기 시작",
                }
            ]

        suggestions = []
        dimensions = profile["dimensions"]
        weak_areas = profile["weak_areas"]
        common_smells = profile["common_smells"]

        # Suggestions for weak dimensions
        dimension_suggestions = {
            "correctness": {
                "message": "정확성 점수가 낮습니다. 엣지 케이스와 경계 조건을 더 신경 써보세요.",
                "tips": [
                    "입력이 빈 경우를 확인하세요",
                    "최소/최대 값을 테스트하세요",
                    "null/undefined 처리를 확인하세요",
                ],
            },
            "efficiency": {
                "message": "효율성 점수가 낮습니다. 시간/공간 복잡도를 최적화해보세요.",
                "tips": [
                    "불필요한 중복 계산을 제거하세요",
                    "적절한 자료구조를 선택하세요",
                    "반복문 중첩을 줄여보세요",
                ],
            },
            "readability": {
                "message": "가독성 점수가 낮습니다. 코드를 더 명확하게 작성해보세요.",
                "tips": [
                    "변수명을 의미있게 지으세요",
                    "함수를 작게 나누세요",
                    "주석으로 의도를 설명하세요",
                ],
            },
            "best_practices": {
                "message": "모범사례 점수가 낮습니다. 코딩 컨벤션을 따라보세요.",
                "tips": [
                    "매직 넘버 대신 상수를 사용하세요",
                    "에러 처리를 추가하세요",
                    "코드 중복을 제거하세요",
                ],
            },
        }

        for dim in weak_areas:
            if dim in dimension_suggestions:
                sug = dimension_suggestions[dim]
                suggestions.append(
                    {
                        "type": "dimension",
                        "dimension": dim,
                        "priority": 1,
                        "score": dimensions[dim],
                        "message": sug["message"],
                        "tips": sug["tips"],
                    }
                )

        # Suggestions for common code smells
        smell_suggestions = {
            "long_function": "함수가 너무 깁니다. 작은 함수로 분리해보세요.",
            "deep_nesting": "중첩이 깊습니다. Early return을 사용해보세요.",
            "magic_number": "매직 넘버가 있습니다. 상수로 정의하세요.",
            "long_line": "라인이 너무 깁니다. 적절히 줄바꿈하세요.",
            "multiple_return": "return이 많습니다. 구조를 단순화하세요.",
            "unused_variable": "사용하지 않는 변수가 있습니다. 제거하세요.",
            "complex_condition": "조건이 복잡합니다. 변수로 추출하세요.",
        }

        for smell_type, count in common_smells[:3]:
            if smell_type in smell_suggestions:
                suggestions.append(
                    {
                        "type": "smell",
                        "smell_type": smell_type,
                        "priority": 2,
                        "count": count,
                        "message": smell_suggestions[smell_type],
                    }
                )

        # Complexity suggestion
        if profile["avg_complexity"] > 10:
            suggestions.append(
                {
                    "type": "complexity",
                    "priority": 2,
                    "value": profile["avg_complexity"],
                    "message": f"평균 복잡도가 {profile['avg_complexity']:.1f}로 높습니다. 함수를 분리하고 조건을 단순화해보세요.",
                }
            )

        # Trend-based suggestion
        if profile["improvement_trend"] == "improving":
            suggestions.append(
                {
                    "type": "encouragement",
                    "priority": 3,
                    "message": "코드 품질이 향상되고 있어요! 계속 노력하세요! 🎉",
                }
            )
        elif profile["improvement_trend"] == "declining":
            suggestions.append(
                {
                    "type": "warning",
                    "priority": 1,
                    "message": "최근 코드 품질이 하락하고 있어요. 기본기를 다시 점검해보세요.",
                }
            )

        # Sort by priority
        suggestions.sort(key=lambda x: x["priority"])

        return suggestions


async def get_quality_recommender(session: AsyncSession) -> QualityRecommender:
    """Factory function to get quality recommender."""
    return QualityRecommender(session)
