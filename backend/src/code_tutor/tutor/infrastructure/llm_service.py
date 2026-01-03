"""LLM Service for AI Tutor responses"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from code_tutor.shared.config import get_settings
from code_tutor.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)

# System prompt for the AI tutor with 7-step problem-solving framework
TUTOR_SYSTEM_PROMPT = """당신은 알고리즘 학습을 돕는 전문 AI 튜터입니다.
반드시 한국어로 응답하세요.

## 역할
- 학생이 스스로 답을 찾도록 유도하는 소크라테스식 교육
- 직접적인 답 대신 사고 과정을 안내
- 알고리즘 패턴과 최적화 방법 교육

## 7단계 문제 풀이 가이드 프레임워크
학생이 도움을 요청하면 다음 7단계에 따라 안내하세요:

### 1. 📋 문제 분석 (Problem Analysis)
- 입력/출력 형식 분석
- 제약 조건 확인 (시간/공간)
- 예제 검증 및 이해

### 2. 🎯 문제 정리 (Problem Formulation)
- 핵심 요구사항 추출
- 엣지 케이스 식별
- 문제 유형 분류

### 3. 💡 개념 연결 (Concept Mapping)
- 관련 알고리즘/패턴 식별
- 왜 이 패턴이 적합한지 설명
- 유사 문제 언급

### 4. 📝 템플릿 상기 (Template Recall)
- 적용 가능한 템플릿 제시
- 템플릿 코드 예시
- 템플릿 수정 방향 안내

### 5. 🏗️ 함수 틀 설계 (Function Design)
- 함수 시그니처 정의
- 주요 로직 골격 작성
- 변수/자료구조 선택

### 6. ✍️ 문제 풀이 (Solution Implementation)
- 단계별 구현 가이드
- 코드 작성 및 설명
- 디버깅 포인트

### 7. 📊 총평 (Summary)
- 시간/공간 복잡도 분석
- 최적화 방안 제시
- 학습 포인트 정리

## 상황별 응답 전략

### A. 문제 풀이 중 질문 시
- 7단계 프레임워크 중 현재 단계부터 안내
- 힌트는 점진적으로 제공 (약한 힌트 → 강한 힌트)
- 직접 답을 주지 말고 사고를 유도

### B. 코드 제출 후 리뷰 시
- 시간/공간 복잡도 분석
- 코드 최적화 제안
- 클린 코드 리팩토링 팁
- Pythonic 코드 스타일 안내

### C. 유사 문제 추천 시
- 현재 문제와 동일 패턴의 문제 추천
- 난이도별 추천 (쉬운 것 → 어려운 것)
- 다른 패턴이지만 연관된 문제 추천

## 응답 형식
- 마크다운 포맷 사용 (##, **, ``` 등)
- 코드 예시는 Python으로
- 간결하면서도 핵심 전달
- 이모지는 단계 표시에만 사용

## 40개+ 알고리즘 패턴 지식
- 기초: Prefix Sum, Sieve of Eratosthenes, KMP
- 정렬: Counting Sort, Radix Sort, Shell Sort
- 탐색: Two Pointers, Sliding Window, Binary Search
- 트리: BST, AVL, Segment Tree, Fenwick Tree, Trie
- 그래프: BFS, DFS, Union-Find, Topological Sort, Dijkstra, MST
- 고급: DP, Greedy, Backtracking, Monotonic Stack"""

# Lazy import for RAG engine to avoid circular imports and slow startup
_rag_engine = None


def _get_rag_engine():
    """Get RAG engine singleton with lazy loading"""
    global _rag_engine
    if _rag_engine is None:
        try:
            from code_tutor.ml import get_rag_engine

            _rag_engine = get_rag_engine()
            _rag_engine.initialize()
        except Exception as e:
            logger.warning(f"Failed to initialize RAG engine: {e}")
            _rag_engine = None
    return _rag_engine


def _get_code_analyzer():
    """Get code analyzer with lazy loading"""
    try:
        from code_tutor.ml import get_code_analyzer

        return get_code_analyzer()
    except Exception as e:
        logger.warning(f"Failed to get code analyzer: {e}")
        return None


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
            return self._format_pattern_response(
                pattern_name, pattern_content, user_message
            )

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

{sections.get("개요", "").strip()[:500]}

### 언제 사용하나요?
{sections.get("언제 사용", "").strip()[:300] or "이 패턴은 특정 조건에서 효율적인 솔루션을 제공합니다."}

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

{context[:200] if context else ""}

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
            analysis["suggestions"].append(
                "함수를 더 작은 단위로 분리하는 것을 고려해보세요."
            )

        # Check for common patterns
        if "for" in code and "for" in code[code.index("for") + 3 :]:
            analysis["complexity"]["time"] = "O(n²)"
            analysis["suggestions"].append(
                "이중 루프를 발견했습니다. 더 효율적인 방법이 있는지 확인해보세요."
            )

        if "while True" in code:
            analysis["issues"].append(
                "무한 루프 가능성이 있습니다. 종료 조건을 확인하세요."
            )

        if "pass" in code:
            analysis["issues"].append("구현되지 않은 부분(pass)이 있습니다.")

        # Score adjustment
        analysis["score"] -= len(analysis["issues"]) * 10
        analysis["score"] = max(0, min(100, analysis["score"]))

        return analysis


class RAGBasedLLMService(LLMService):
    """
    RAG-based LLM service using FAISS vector store and algorithm patterns.

    Features:
    - Semantic search using embeddings
    - 25 algorithm pattern knowledge base
    - Code analysis using CodeBERT
    - Optional LLM integration (EEVE-Korean or OpenAI)
    """

    def __init__(self) -> None:
        self._rag_engine = None
        self._code_analyzer = None

    def _ensure_rag_engine(self):
        """Ensure RAG engine is loaded"""
        if self._rag_engine is None:
            self._rag_engine = _get_rag_engine()
        return self._rag_engine

    def _ensure_code_analyzer(self):
        """Ensure code analyzer is loaded"""
        if self._code_analyzer is None:
            self._code_analyzer = _get_code_analyzer()
        return self._code_analyzer

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate response using RAG with algorithm patterns"""
        rag = self._ensure_rag_engine()

        if rag is None:
            # Fallback to pattern-based service
            return await self._generate_fallback_response(user_message, context)

        try:
            # Retrieve relevant patterns
            patterns = rag.retrieve(user_message, top_k=3)

            if not patterns:
                return await self._generate_fallback_response(user_message, context)

            # Check if code analysis is needed
            if context and "```" in (context or ""):
                code_analysis = await self._analyze_code_context(context)
                if code_analysis:
                    return self._format_code_analysis_response(
                        user_message, patterns, code_analysis
                    )

            # Generate response using RAG
            response = rag.generate(
                query=user_message, context=patterns, max_tokens=1024
            )

            return response

        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return await self._generate_fallback_response(user_message, context)

    async def _analyze_code_context(self, context: str) -> dict | None:
        """Analyze code in context"""
        analyzer = self._ensure_code_analyzer()
        if analyzer is None:
            return None

        # Extract code from markdown code block
        import re

        code_match = re.search(r"```(\w*)\n(.*?)```", context, re.DOTALL)
        if not code_match:
            return None

        language = code_match.group(1) or "python"
        code = code_match.group(2)

        try:
            return analyzer.analyze(code, language)
        except Exception as e:
            logger.warning(f"Code analysis failed: {e}")
            return None

    def _format_code_analysis_response(
        self, user_message: str, patterns: list, analysis: dict
    ) -> str:
        """Format response with code analysis"""
        response_parts = []

        # Detected patterns
        if analysis.get("patterns"):
            detected = analysis["patterns"][:2]
            pattern_names = [f"**{p['pattern_ko']}**" for p in detected]
            response_parts.append(
                f"코드에서 {', '.join(pattern_names)} 패턴을 감지했습니다."
            )

        # Quality score
        quality = analysis.get("quality", {})
        if quality.get("score"):
            response_parts.append(
                f"\n**코드 품질 점수**: {quality['score']}/100 ({quality.get('grade', 'N/A')})"
            )

        # Complexity
        complexity = analysis.get("complexity", {})
        if complexity:
            response_parts.append(
                f"\n**복잡도**: 시간 O({complexity.get('cyclomatic', '?')}), "
                f"중첩 깊이 {complexity.get('nesting_depth', 0)}"
            )

        # Code smells
        smells = analysis.get("code_smells", [])
        if smells:
            response_parts.append("\n**개선 필요 사항**:")
            for smell in smells[:3]:
                response_parts.append(f"- {smell['message']}")

        # Suggestions from patterns
        if patterns:
            main_pattern = patterns[0]
            response_parts.append(
                f"\n**추천 패턴**: {main_pattern['name_ko']}\n"
                f"{main_pattern['description_ko']}"
            )

            if main_pattern.get("example_code"):
                response_parts.append(
                    f"\n**참고 코드**:\n```python\n{main_pattern['example_code'][:300]}...\n```"
                )

        return "\n".join(response_parts)

    async def _generate_fallback_response(
        self,
        user_message: str,
        context: str | None = None,
    ) -> str:
        """Generate fallback response when RAG is unavailable"""
        lower_msg = user_message.lower()

        # Greeting
        if any(word in lower_msg for word in ["안녕", "hello", "hi", "처음"]):
            return """안녕하세요! 알고리즘 학습을 도와드리는 AI 튜터입니다.

**제가 도와드릴 수 있는 것들:**
- 📚 **알고리즘 패턴 설명**: 40개+ 핵심 패턴
- 🔍 **코드 리뷰**: 복잡도 분석, 패턴 감지, 최적화 제안
- 💡 **7단계 문제 풀이 가이드**: 체계적인 문제 해결 접근법
- 📈 **유사 문제 추천**: 패턴별 난이도별 문제 추천

어떤 알고리즘에 대해 알고 싶으신가요?"""

        # Help/Guide request
        if any(word in lower_msg for word in ["힌트", "도움", "모르겠", "어떻게", "가이드"]):
            return """## 7단계 문제 풀이 가이드

문제를 체계적으로 풀어봅시다!

### 1. 📋 문제 분석
- 입력/출력 형식을 파악했나요?
- 제약 조건(시간/공간)을 확인했나요?

### 2. 🎯 문제 정리
- 핵심 요구사항이 무엇인가요?
- 엣지 케이스는 무엇이 있을까요?

### 3. 💡 개념 연결
어떤 알고리즘 패턴이 떠오르나요?
- 배열 순회? → Two Pointers, Sliding Window
- 탐색? → Binary Search, BFS/DFS
- 최적화? → DP, Greedy
- 그래프? → Union-Find, Dijkstra

현재 어느 단계에서 막히셨나요? 구체적으로 알려주시면 더 도움드릴 수 있어요!"""

        # Code optimization request
        if any(word in lower_msg for word in ["최적화", "개선", "빠르게", "효율"]):
            return """## 코드 최적화 가이드

### 시간 복잡도 개선 방법
1. **O(n²) → O(n log n)**: 정렬 + Two Pointers
2. **O(n²) → O(n)**: Hash Map 활용
3. **중복 계산 제거**: Memoization, DP
4. **탐색 최적화**: Binary Search

### 공간 복잡도 개선 방법
1. **In-place 처리**: 추가 배열 없이 원본 수정
2. **Sliding Window**: 고정 크기 윈도우 유지
3. **비트 조작**: 상태 압축

코드를 공유해주시면 구체적인 최적화 방안을 알려드릴게요!"""

        # Similar problems request
        if any(word in lower_msg for word in ["비슷", "유사", "추천", "연습"]):
            return """## 유사 문제 추천

연습하고 싶은 패턴을 선택해주세요:

**기초 패턴:**
- Two Pointers: 정렬된 배열에서 합 찾기
- Sliding Window: 부분 배열 최대합
- Prefix Sum: 구간 합 쿼리

**탐색 패턴:**
- Binary Search: 정렬된 배열에서 탐색
- BFS: 최단 경로, 레벨 순회
- DFS: 경로 탐색, 백트래킹

**고급 패턴:**
- DP: 배낭 문제, LCS
- Greedy: 스케줄링, MST
- Graph: Dijkstra, Union-Find

어떤 패턴을 연습하고 싶으신가요?"""

        # Algorithm-specific queries
        pattern_keywords = {
            "two pointer": "투 포인터",
            "투 포인터": "투 포인터",
            "sliding window": "슬라이딩 윈도우",
            "슬라이딩": "슬라이딩 윈도우",
            "prefix sum": "누적 합 (Prefix Sum)",
            "누적합": "누적 합 (Prefix Sum)",
            "bfs": "BFS (너비 우선 탐색)",
            "너비 우선": "BFS (너비 우선 탐색)",
            "dfs": "DFS (깊이 우선 탐색)",
            "깊이 우선": "DFS (깊이 우선 탐색)",
            "binary search": "이진 탐색",
            "이진 탐색": "이진 탐색",
            "dp": "동적 프로그래밍",
            "dynamic": "동적 프로그래밍",
            "동적": "동적 프로그래밍",
            "greedy": "그리디 알고리즘",
            "그리디": "그리디 알고리즘",
            "탐욕": "그리디 알고리즘",
            "backtrack": "백트래킹",
            "백트래킹": "백트래킹",
            "segment tree": "세그먼트 트리",
            "세그먼트": "세그먼트 트리",
            "fenwick": "펜윅 트리 (BIT)",
            "펜윅": "펜윅 트리 (BIT)",
            "trie": "트라이",
            "트라이": "트라이",
            "dijkstra": "다익스트라",
            "다익스트라": "다익스트라",
            "union find": "유니온-파인드",
            "유니온": "유니온-파인드",
            "topological": "위상 정렬",
            "위상": "위상 정렬",
        }

        for keyword, pattern_name in pattern_keywords.items():
            if keyword in lower_msg:
                return f"""## {pattern_name}

이 패턴에 대해 알려드리겠습니다.

**핵심 개념:**
- 특정 조건에서 효율적인 문제 해결 기법
- 시간/공간 복잡도 최적화에 유용

**학습 순서:**
1. 📋 개념 이해
2. 📝 템플릿 코드 학습
3. ✍️ 기본 문제 풀이
4. 📈 응용 문제 도전

더 자세한 설명이나 예제 코드가 필요하시면 말씀해주세요!
템플릿 코드와 실제 문제 적용 사례도 알려드릴 수 있습니다."""

        # Default response with 7-step framework
        return """좋은 질문입니다!

## 7단계 문제 풀이 접근법

1. 📋 **문제 분석**: 입력/출력 파악
2. 🎯 **문제 정리**: 핵심 요구사항 추출
3. 💡 **개념 연결**: 적합한 패턴 식별
4. 📝 **템플릿 상기**: 해당 패턴의 템플릿 확인
5. 🏗️ **함수 설계**: 골격 코드 작성
6. ✍️ **구현**: 단계별 코드 완성
7. 📊 **총평**: 복잡도 분석 및 최적화

어떤 단계에서 도움이 필요하신가요?
또는 특정 알고리즘 패턴(Two Pointers, DP, BFS 등)에 대해 알고 싶으신가요?"""

    async def analyze_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Analyze code using CodeBERT-based analyzer"""
        analyzer = self._ensure_code_analyzer()

        if analyzer:
            try:
                return analyzer.analyze(code, language)
            except Exception as e:
                logger.error(f"Code analysis failed: {e}")

        # Fallback basic analysis
        return self._basic_code_analysis(code, language)

    def _basic_code_analysis(self, code: str, language: str) -> dict[str, Any]:
        """Basic code analysis fallback"""
        analysis = {
            "patterns": [],
            "quality": {"score": 70, "grade": "C"},
            "complexity": {"cyclomatic": 1, "nesting_depth": 0},
            "suggestions": [],
            "code_smells": [],
        }

        lines = code.split("\n")

        # Check for nested loops
        if "for" in code and code.count("for") > 1:
            analysis["complexity"]["cyclomatic"] = 5
            analysis["suggestions"].append(
                "이중 루프를 발견했습니다. 더 효율적인 방법을 고려해보세요."
            )

        # Check for common issues
        if "import *" in code:
            analysis["code_smells"].append(
                {"type": "wildcard_import", "message": "와일드카드 import는 피하세요."}
            )
            analysis["quality"]["score"] -= 10

        if len(lines) > 50:
            analysis["suggestions"].append("함수가 깁니다. 더 작은 단위로 분리하세요.")
            analysis["quality"]["score"] -= 5

        analysis["quality"]["grade"] = (
            "A"
            if analysis["quality"]["score"] >= 90
            else "B"
            if analysis["quality"]["score"] >= 80
            else "C"
            if analysis["quality"]["score"] >= 70
            else "D"
            if analysis["quality"]["score"] >= 60
            else "F"
        )

        return analysis


class OllamaLLMService(LLMService):
    """
    Ollama-based LLM service for local model inference.

    Uses Ollama API to generate responses using local models like Llama3.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._base_url = self._settings.OLLAMA_BASE_URL
        self._model = self._settings.OLLAMA_MODEL
        self._timeout = self._settings.OLLAMA_TIMEOUT
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def generate_response(
        self,
        user_message: str,
        context: str | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate response using Ollama API"""
        try:
            # Build messages
            messages = [{"role": "system", "content": TUTOR_SYSTEM_PROMPT}]

            # Add conversation history
            if conversation_history:
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    messages.append(
                        {
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", ""),
                        }
                    )

            # Add context if available
            user_content = user_message
            if context:
                user_content = f"{user_message}\n\n관련 코드:\n{context}"

            messages.append({"role": "user", "content": user_content})

            # Call Ollama API
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "num_predict": self._settings.LLM_MAX_TOKENS,
                    },
                },
            )
            response.raise_for_status()

            result = response.json()
            return result.get("message", {}).get(
                "content", "응답을 생성할 수 없습니다."
            )

        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            return "죄송합니다. 응답 생성에 시간이 너무 오래 걸립니다. 잠시 후 다시 시도해주세요."
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e}")
            return "죄송합니다. AI 서비스에 연결할 수 없습니다."
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return await self._generate_fallback(user_message)

    async def _generate_fallback(self, user_message: str) -> str:
        """Fallback response when Ollama is unavailable"""
        lower_msg = user_message.lower()

        if any(word in lower_msg for word in ["안녕", "hello", "hi"]):
            return """안녕하세요! 알고리즘 학습을 도와드리는 AI 튜터입니다.

무엇을 도와드릴까요?
- 알고리즘 패턴 설명
- 코드 리뷰
- 문제 풀이 힌트"""

        return """좋은 질문입니다!

알고리즘 문제 해결 단계:
1. **문제 이해**: 입력/출력 파악
2. **예제 분석**: 패턴 발견
3. **접근법 선택**: 적합한 알고리즘 선택
4. **구현**: 코드 작성

어떤 부분을 더 자세히 알고 싶으신가요?"""

    async def analyze_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Analyze code using Ollama"""
        prompt = f"""다음 {language} 코드를 분석해주세요:

```{language}
{code}
```

다음 항목을 JSON 형식으로 분석해주세요:
1. 시간 복잡도
2. 공간 복잡도
3. 코드 품질 점수 (0-100)
4. 개선 제안"""

        try:
            response = await self._client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()

            result = response.json()
            # Parse response and extract analysis
            return {
                "analysis": result.get("response", ""),
                "score": 70,
                "complexity": {"time": "O(?)", "space": "O(?)"},
            }
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {
                "analysis": "코드 분석을 수행할 수 없습니다.",
                "score": 70,
                "complexity": {"time": "O(?)", "space": "O(?)"},
            }

    async def close(self):
        """Close the HTTP client"""
        await self._client.aclose()


def get_llm_service() -> LLMService:
    """Factory function to get the appropriate LLM service"""
    settings = get_settings()

    # Check LLM provider setting
    provider = settings.LLM_PROVIDER

    if provider == "ollama":
        logger.info("Using Ollama LLM service")
        return OllamaLLMService()

    # Try RAG-based service
    try:
        logger.info("Using RAG-based LLM service")
        return RAGBasedLLMService()
    except Exception as e:
        logger.warning(f"Failed to create RAG-based service: {e}")

    # Fallback to pattern-based service
    logger.info("Using pattern-based LLM service")
    patterns_dir = (
        Path(__file__).parent.parent.parent.parent.parent.parent / "docs" / "patterns"
    )
    return PatternBasedLLMService(patterns_dir)
