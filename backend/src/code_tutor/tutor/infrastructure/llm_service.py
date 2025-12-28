"""LLM Service for AI Tutor responses"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class LLMService(ABC):
    """Abstract base class for LLM services"""

    @abstractmethod
    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a response to the user's message"""
        pass

    @abstractmethod
    async def analyze_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Analyze code and provide feedback"""
        pass


class PatternBasedLLMService(LLMService):
    """
    Pattern-based LLM service using pre-defined algorithm patterns.
    This can be upgraded to use actual LLM models later.
    """

    def __init__(self, patterns_dir: Path | None = None) -> None:
        self._patterns_dir = patterns_dir or Path("docs/patterns")
        self._patterns: dict[str, str] = {}
        self._load_patterns()

    def _load_patterns(self) -> None:
        """Load algorithm patterns from markdown files"""
        if not self._patterns_dir.exists():
            logger.warning(f"Patterns directory not found: {self._patterns_dir}")
            return

        for pattern_file in self._patterns_dir.glob("*.md"):
            try:
                content = pattern_file.read_text(encoding="utf-8")
                pattern_name = pattern_file.stem
                self._patterns[pattern_name] = content
                logger.debug(f"Loaded pattern: {pattern_name}")
            except Exception as e:
                logger.error(f"Failed to load pattern {pattern_file}: {e}")

        logger.info(f"Loaded {len(self._patterns)} algorithm patterns")

    def _find_relevant_patterns(self, query: str) -> list[tuple[str, str]]:
        """Find patterns relevant to the query"""
        query_lower = query.lower()
        relevant = []

        # Keyword mapping to patterns
        keyword_patterns = {
            "two pointer": "01_two_pointers",
            "투 포인터": "01_two_pointers",
            "sliding window": "02_sliding_window",
            "슬라이딩 윈도우": "02_sliding_window",
            "bfs": "07_bfs",
            "너비 우선": "07_bfs",
            "dfs": "08_dfs",
            "깊이 우선": "08_dfs",
            "binary search": "09_binary_search",
            "이진 탐색": "09_binary_search",
            "dp": "12_dp_01_knapsack",
            "dynamic programming": "12_dp_01_knapsack",
            "동적 프로그래밍": "12_dp_01_knapsack",
            "배낭": "12_dp_01_knapsack",
            "knapsack": "12_dp_01_knapsack",
            "backtracking": "14_backtracking",
            "백트래킹": "14_backtracking",
            "greedy": "15_greedy",
            "그리디": "15_greedy",
            "탐욕": "15_greedy",
            "union find": "16_union_find",
            "유니온 파인드": "16_union_find",
            "dijkstra": "17_shortest_path",
            "다익스트라": "17_shortest_path",
            "최단 경로": "17_shortest_path",
            "trie": "18_trie",
            "트라이": "18_trie",
        }

        for keyword, pattern_name in keyword_patterns.items():
            if keyword in query_lower and pattern_name in self._patterns:
                relevant.append((pattern_name, self._patterns[pattern_name]))

        return relevant[:3]  # Return top 3 matches

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a response based on algorithm patterns"""
        # Find relevant patterns
        relevant_patterns = self._find_relevant_patterns(user_message)

        if relevant_patterns:
            # Extract key information from the first matching pattern
            pattern_name, pattern_content = relevant_patterns[0]
            return self._format_pattern_response(pattern_name, pattern_content, user_message)

        # Default response when no pattern is found
        return self._generate_default_response(user_message, context)

    def _format_pattern_response(
        self,
        pattern_name: str,
        pattern_content: str,
        user_message: str,
    ) -> str:
        """Format a response using pattern content"""
        # Extract title from pattern
        lines = pattern_content.split("\n")
        title = lines[0].replace("#", "").strip() if lines else pattern_name

        # Extract key sections
        sections = {
            "개요": "",
            "언제 사용": "",
            "템플릿": "",
            "시간 복잡도": "",
        }

        current_section = ""
        for line in lines:
            if line.startswith("##"):
                section_name = line.replace("#", "").strip()
                for key in sections:
                    if key in section_name:
                        current_section = key
                        break
            elif current_section and line.strip():
                sections[current_section] += line + "\n"

        response = f"""## {title}

{sections.get('개요', '').strip()[:500]}

### 언제 사용하나요?
{sections.get('언제 사용', '').strip()[:300] or '이 패턴은 특정 조건에서 효율적인 솔루션을 제공합니다.'}

더 자세한 설명이 필요하시면 말씀해주세요! 템플릿 코드나 예제 문제도 보여드릴 수 있습니다."""

        return response

    def _generate_default_response(
        self,
        user_message: str,
        context: str | None,
    ) -> str:
        """Generate a default response when no pattern matches"""
        lower_msg = user_message.lower()

        if any(word in lower_msg for word in ["안녕", "hello", "hi"]):
            return """안녕하세요! 알고리즘 학습을 도와드리는 AI 튜터입니다.

다음과 같은 도움을 드릴 수 있습니다:
- **알고리즘 패턴 설명**: Two Pointers, Sliding Window, DP 등 25개 핵심 패턴
- **코드 리뷰**: 복잡도 분석, 개선 제안
- **문제 풀이 힌트**: 어떤 접근법을 사용해야 하는지

무엇을 도와드릴까요?"""

        if any(word in lower_msg for word in ["시간 복잡도", "big o", "복잡도"]):
            return """## 시간 복잡도 분석

알고리즘의 효율성을 측정하는 방법입니다.

**일반적인 시간 복잡도:**
- O(1): 상수 시간 - 배열 인덱싱
- O(log n): 로그 시간 - 이진 탐색
- O(n): 선형 시간 - 배열 순회
- O(n log n): 로그 선형 - 병합 정렬, 퀵 정렬
- O(n²): 제곱 시간 - 이중 루프, 버블 정렬
- O(2^n): 지수 시간 - 부분집합 생성

특정 알고리즘의 복잡도가 궁금하시면 말씀해주세요!"""

        if context and "code" in lower_msg:
            return f"""코드를 분석해보겠습니다.

{context[:200] if context else ''}

**리뷰 포인트:**
1. 변수명이 명확한지 확인하세요
2. 엣지 케이스를 처리하고 있는지 확인하세요
3. 시간 복잡도를 개선할 수 있는지 고려해보세요

구체적인 질문이 있으시면 알려주세요!"""

        return """좋은 질문입니다!

알고리즘 문제를 풀 때는 다음 단계를 추천드립니다:

1. **문제 이해**: 입력과 출력을 명확히 파악
2. **예제 분석**: 손으로 풀어보며 패턴 발견
3. **접근법 선택**: 적합한 알고리즘 패턴 선택
4. **구현**: 코드 작성
5. **검증**: 엣지 케이스 테스트

어떤 알고리즘 패턴에 대해 알고 싶으신가요?
(예: Two Pointers, DP, BFS/DFS, Binary Search 등)"""

    async def analyze_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Analyze code and provide feedback"""
        analysis = {
            "suggestions": [],
            "complexity": {"time": "O(?)", "space": "O(?)"},
            "issues": [],
            "score": 70,
        }

        lines = code.split("\n")
        num_lines = len(lines)

        # Basic analysis
        if num_lines > 50:
            analysis["suggestions"].append("함수를 더 작은 단위로 분리하는 것을 고려해보세요.")

        # Check for common patterns
        if "for" in code and "for" in code[code.index("for") + 3:]:
            analysis["complexity"]["time"] = "O(n²)"
            analysis["suggestions"].append("이중 루프를 발견했습니다. 더 효율적인 방법이 있는지 확인해보세요.")

        if "while True" in code:
            analysis["issues"].append("무한 루프 가능성이 있습니다. 종료 조건을 확인하세요.")

        if "pass" in code:
            analysis["issues"].append("구현되지 않은 부분(pass)이 있습니다.")

        # Score adjustment
        analysis["score"] -= len(analysis["issues"]) * 10
        analysis["score"] = max(0, min(100, analysis["score"]))

        return analysis


def get_llm_service() -> LLMService:
    """Factory function to get the appropriate LLM service"""
    settings = get_settings()

    # For now, always use PatternBasedLLMService
    # In the future, we can add conditions for real LLM models
    patterns_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "docs" / "patterns"
    return PatternBasedLLMService(patterns_dir)
