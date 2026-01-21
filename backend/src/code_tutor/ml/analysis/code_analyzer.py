"""Code Analyzer using CodeBERT for semantic code understanding"""

import ast
import logging
import re
from typing import Any

from code_tutor.ml.config import get_ml_config
from code_tutor.ml.embeddings import CodeEmbedder
from code_tutor.ml.rag import PatternKnowledgeBase

logger = logging.getLogger(__name__)


class CodeAnalyzer:
    """
    Comprehensive code analyzer using CodeBERT and pattern matching.

    Features:
    - Pattern detection using code embeddings
    - Code quality scoring
    - Complexity analysis
    - Bug pattern detection
    - Improvement suggestions
    """

    # Common code smells and anti-patterns
    CODE_SMELLS = {
        "long_function": {
            "pattern": r"def\s+\w+\s*\([^)]*\):\s*\n((?:(?!def\s)[\s\S])*)",
            "threshold": 50,  # lines
            "message": "함수가 너무 깁니다. 작은 함수로 분리하는 것을 고려하세요.",
            "severity": "warning",
        },
        "deep_nesting": {
            "pattern": r"^(\s+)",
            "threshold": 4,  # levels
            "message": "중첩 레벨이 너무 깊습니다. 조기 반환(early return)을 사용하세요.",
            "severity": "warning",
        },
        "magic_numbers": {
            "pattern": r"(?<!['\"])\b(?!0\b|1\b|-1\b)[0-9]+(?:\.[0-9]+)?\b(?!['\"])",
            "message": "매직 넘버를 상수로 정의하세요.",
            "severity": "info",
        },
        "empty_except": {
            "pattern": r"except\s*(?:\w+\s*)?:\s*(?:pass|\.\.\.)",
            "message": "빈 except 블록은 오류를 숨깁니다. 적절한 에러 처리를 추가하세요.",
            "severity": "error",
        },
        "print_debugging": {
            "pattern": r"\bprint\s*\([^)]*\)",
            "message": "디버깅용 print 문이 남아있습니다.",
            "severity": "info",
        },
    }

    # Common algorithm patterns for detection
    ALGORITHM_SIGNATURES = {
        "two_pointers": [r"left.*right", r"i.*j.*while", r"\[left\].*\[right\]"],
        "sliding_window": [
            r"window",
            r"sum\s*\(\s*\w+\s*\[\s*\w+\s*:\s*\w+\s*\]",
            r"start.*end.*while",
        ],
        "binary_search": [r"left.*right.*mid", r"lo.*hi.*mid", r"//\s*2", r"bisect"],
        "dfs": [r"def\s+dfs\s*\(", r"stack\s*=\s*\[", r"visited.*add", r"recursive"],
        "bfs": [
            r"from\s+collections\s+import\s+deque",
            r"queue\s*=.*deque",
            r"queue\.popleft\s*\(\)",
            r"level.*order",
        ],
        "dp": [
            r"dp\s*=\s*\[",
            r"memo",
            r"@cache",
            r"@lru_cache",
            r"dp\s*\[\s*i\s*\]\s*\[\s*j\s*\]",
        ],
        "greedy": [r"sort\s*\(", r"max\s*\(|min\s*\(", r"for.*in.*sorted"],
        "recursion": [r"return\s+\w+\s*\(", r"def\s+(\w+).*\1\s*\("],
    }

    def __init__(self, config=None):
        self.config = config or get_ml_config()

        self._code_embedder: CodeEmbedder | None = None
        self._pattern_kb: PatternKnowledgeBase | None = None

    @property
    def code_embedder(self) -> CodeEmbedder:
        """Get code embedder, creating if needed"""
        if self._code_embedder is None:
            self._code_embedder = CodeEmbedder(
                model_name=self.config.CODE_EMBEDDING_MODEL,
                cache_dir=str(self.config.MODEL_CACHE_DIR),
            )
        return self._code_embedder

    @property
    def pattern_kb(self) -> PatternKnowledgeBase:
        """Get pattern knowledge base"""
        if self._pattern_kb is None:
            self._pattern_kb = PatternKnowledgeBase()
        return self._pattern_kb

    def analyze(
        self,
        code: str,
        language: str = "python",
        include_patterns: bool = True,
        include_quality: bool = True,
        include_complexity: bool = True,
        include_suggestions: bool = True,
    ) -> dict[str, Any]:
        """
        Perform comprehensive code analysis.

        Args:
            code: Source code to analyze
            language: Programming language
            include_patterns: Include pattern detection
            include_quality: Include quality scoring
            include_complexity: Include complexity analysis
            include_suggestions: Include improvement suggestions

        Returns:
            Analysis results dict
        """
        result = {
            "language": language,
            "lines_of_code": self._count_lines(code),
            "patterns": [],
            "quality": {},
            "complexity": {},
            "suggestions": [],
            "code_smells": [],
        }

        # Pattern detection
        if include_patterns:
            result["patterns"] = self._detect_patterns(code, language)

        # Quality analysis
        if include_quality:
            result["quality"] = self._analyze_quality(code, language)
            result["code_smells"] = result["quality"].get("smells", [])

        # Complexity analysis
        if include_complexity:
            result["complexity"] = self._analyze_complexity(code, language)

        # Generate suggestions
        if include_suggestions:
            result["suggestions"] = self._generate_suggestions(result)

        return result

    def _count_lines(self, code: str) -> dict[str, int]:
        """Count lines of code"""
        lines = code.split("\n")
        total = len(lines)
        blank = sum(1 for line in lines if not line.strip())
        comment = sum(1 for line in lines if line.strip().startswith("#"))
        code_lines = total - blank - comment

        return {"total": total, "code": code_lines, "blank": blank, "comment": comment}

    def _detect_patterns(self, code: str, language: str) -> list[dict]:
        """Detect algorithm patterns in code"""
        detected = []

        # Rule-based pattern detection
        for pattern_name, signatures in self.ALGORITHM_SIGNATURES.items():
            matches = 0
            for sig in signatures:
                if re.search(sig, code, re.IGNORECASE):
                    matches += 1

            if matches >= 2:  # At least 2 signature matches
                confidence = min(0.9, 0.5 + matches * 0.15)
                pattern_info = self.pattern_kb.get_pattern(pattern_name)

                detected.append(
                    {
                        "pattern": pattern_name,
                        "pattern_ko": pattern_info["name_ko"]
                        if pattern_info
                        else pattern_name,
                        "confidence": confidence,
                        "detection_method": "rule_based",
                        "description": pattern_info["description_ko"]
                        if pattern_info
                        else "",
                    }
                )

        # ML-based pattern detection using CodeBERT
        try:
            ml_patterns = self._detect_patterns_ml(code, language)
            for mp in ml_patterns:
                # Check if already detected by rules
                existing = next(
                    (p for p in detected if p["pattern"] == mp["pattern"]), None
                )
                if existing:
                    # Boost confidence if ML agrees
                    existing["confidence"] = min(0.95, existing["confidence"] + 0.1)
                    existing["detection_method"] = "hybrid"
                else:
                    detected.append(mp)
        except Exception as e:
            logger.warning(f"ML pattern detection failed: {e}")

        # Sort by confidence
        detected.sort(key=lambda x: x["confidence"], reverse=True)

        return detected[:5]  # Return top 5

    def _detect_patterns_ml(self, code: str, language: str) -> list[dict]:
        """Detect patterns using ML embeddings"""
        # Build pattern embeddings if needed
        if self._pattern_kb._embeddings is None:
            self._pattern_kb.build_embeddings(code_embedder=self.code_embedder)

        # Find similar patterns
        similar = self._pattern_kb.find_similar_by_code(
            code=code, language=language, top_k=3, threshold=0.6
        )

        return [
            {
                "pattern": p["id"],
                "pattern_ko": p["name_ko"],
                "confidence": p["similarity"],
                "detection_method": "ml_embedding",
                "description": p["description_ko"],
            }
            for p in similar
        ]

    def _analyze_quality(self, code: str, language: str) -> dict:
        """Analyze code quality"""
        smells = []
        score = 100  # Start with perfect score

        # Check for code smells
        for smell_name, smell_info in self.CODE_SMELLS.items():
            matches = re.findall(smell_info["pattern"], code, re.MULTILINE)

            if smell_name == "long_function":
                for match in matches:
                    lines = match.count("\n") + 1
                    if lines > smell_info["threshold"]:
                        smells.append(
                            {
                                "type": smell_name,
                                "message": smell_info["message"],
                                "severity": smell_info["severity"],
                                "details": f"함수 길이: {lines}줄",
                            }
                        )
                        score -= 10

            elif smell_name == "deep_nesting":
                max_indent = 0
                for line in code.split("\n"):
                    if line.strip():
                        indent = len(line) - len(line.lstrip())
                        spaces = indent // 4  # Assuming 4 spaces per level
                        max_indent = max(max_indent, spaces)

                if max_indent > smell_info["threshold"]:
                    smells.append(
                        {
                            "type": smell_name,
                            "message": smell_info["message"],
                            "severity": smell_info["severity"],
                            "details": f"최대 중첩 레벨: {max_indent}",
                        }
                    )
                    score -= 10

            elif matches:
                smells.append(
                    {
                        "type": smell_name,
                        "message": smell_info["message"],
                        "severity": smell_info["severity"],
                        "occurrences": len(matches),
                    }
                )
                score -= 5 * len(matches)

        # Python-specific checks
        if language == "python":
            py_issues = self._check_python_quality(code)
            smells.extend(py_issues)
            score -= len(py_issues) * 5

        return {
            "score": max(0, score),
            "grade": self._score_to_grade(score),
            "smells": smells,
        }

    def _check_python_quality(self, code: str) -> list[dict]:
        """Python-specific quality checks"""
        issues = []

        try:
            tree = ast.parse(code)

            # Check for mutable default arguments
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            issues.append(
                                {
                                    "type": "mutable_default",
                                    "message": f"함수 '{node.name}'에 변경 가능한 기본 인자가 있습니다.",
                                    "severity": "warning",
                                }
                            )

            # Check for unused variables (simple check)
            assigned = set()
            used = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        assigned.add(node.id)
                    elif isinstance(node.ctx, ast.Load):
                        used.add(node.id)

            unused = assigned - used - {"_", "__"}
            if unused:
                issues.append(
                    {
                        "type": "unused_variables",
                        "message": f"사용되지 않는 변수: {', '.join(list(unused)[:3])}",
                        "severity": "info",
                    }
                )

        except SyntaxError as e:
            issues.append(
                {
                    "type": "syntax_error",
                    "message": f"구문 오류: {str(e)}",
                    "severity": "error",
                }
            )

        return issues

    def _analyze_complexity(self, code: str, language: str) -> dict:
        """Analyze code complexity"""
        complexity = {
            "cyclomatic": 1,
            "cognitive": 0,
            "nesting_depth": 0,
            "function_count": 0,
            "class_count": 0,
        }

        if language == "python":
            try:
                tree = ast.parse(code)

                # Count functions and classes
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity["function_count"] += 1
                    elif isinstance(node, ast.ClassDef):
                        complexity["class_count"] += 1

                # Calculate cyclomatic complexity
                complexity["cyclomatic"] = self._calculate_cyclomatic(tree)

                # Calculate cognitive complexity
                complexity["cognitive"] = self._calculate_cognitive(tree)

            except SyntaxError:
                pass

        # Calculate nesting depth
        complexity["nesting_depth"] = self._calculate_nesting_depth(code)

        # Complexity rating
        if complexity["cyclomatic"] <= 5:
            complexity["rating"] = "low"
        elif complexity["cyclomatic"] <= 10:
            complexity["rating"] = "medium"
        else:
            complexity["rating"] = "high"

        return complexity

    def _calculate_cyclomatic(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        decision_nodes = (
            ast.If,
            ast.While,
            ast.For,
            ast.ExceptHandler,
            ast.With,
            ast.Assert,
            ast.comprehension,
        )

        for node in ast.walk(tree):
            if isinstance(node, decision_nodes):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def _calculate_cognitive(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate cognitive complexity"""
        complexity = 0

        nesting_nodes = (ast.If, ast.While, ast.For, ast.With, ast.Try)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, nesting_nodes):
                complexity += 1 + depth  # Nesting penalty
                complexity += self._calculate_cognitive(node, depth + 1)
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            else:
                complexity += self._calculate_cognitive(node, depth)

        return complexity

    def _calculate_nesting_depth(self, code: str) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        current_depth = 0

        for line in code.split("\n"):
            stripped = line.lstrip()
            if not stripped:
                continue

            indent = len(line) - len(stripped)
            current_depth = indent // 4

            max_depth = max(max_depth, current_depth)

        return max_depth

    def _score_to_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_suggestions(self, analysis: dict) -> list[dict]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []

        # Pattern-based suggestions
        patterns = analysis.get("patterns", [])
        if patterns:
            main_pattern = patterns[0]
            if main_pattern["confidence"] < 0.7:
                suggestions.append(
                    {
                        "type": "pattern_clarity",
                        "priority": "medium",
                        "message": "패턴이 명확하지 않습니다. 알고리즘 패턴을 더 명시적으로 구현하세요.",
                        "pattern": main_pattern["pattern"],
                    }
                )

        # Quality-based suggestions
        quality = analysis.get("quality", {})
        if quality.get("score", 100) < 70:
            suggestions.append(
                {
                    "type": "code_quality",
                    "priority": "high",
                    "message": "코드 품질 점수가 낮습니다. 코드 스멜을 수정하세요.",
                }
            )

        # Complexity-based suggestions
        complexity = analysis.get("complexity", {})
        if complexity.get("cyclomatic", 0) > 10:
            suggestions.append(
                {
                    "type": "complexity",
                    "priority": "high",
                    "message": "순환 복잡도가 높습니다. 함수를 더 작은 단위로 분리하세요.",
                }
            )

        if complexity.get("nesting_depth", 0) > 4:
            suggestions.append(
                {
                    "type": "nesting",
                    "priority": "medium",
                    "message": "중첩 깊이가 깊습니다. 조기 반환 패턴을 사용하세요.",
                }
            )

        # Code smell suggestions
        for smell in analysis.get("code_smells", []):
            if smell["severity"] in ["error", "warning"]:
                suggestions.append(
                    {
                        "type": f"fix_{smell['type']}",
                        "priority": "high"
                        if smell["severity"] == "error"
                        else "medium",
                        "message": smell["message"],
                    }
                )

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return suggestions[:5]  # Return top 5 suggestions

    def compare_codes(
        self, code1: str, code2: str, language: str = "python"
    ) -> dict[str, Any]:
        """
        Compare two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            language: Programming language

        Returns:
            Comparison results
        """
        # Get embeddings
        emb1 = self.code_embedder.embed(code1, language)
        emb2 = self.code_embedder.embed(code2, language)

        # Calculate similarity
        import numpy as np

        similarity = float(np.dot(emb1, emb2))

        # Analyze both
        analysis1 = self.analyze(code1, language)
        analysis2 = self.analyze(code2, language)

        return {
            "similarity": similarity,
            "similarity_interpretation": self._interpret_similarity(similarity),
            "code1_analysis": analysis1,
            "code2_analysis": analysis2,
            "comparison": {
                "lines_difference": (
                    analysis1["lines_of_code"]["code"]
                    - analysis2["lines_of_code"]["code"]
                ),
                "complexity_difference": (
                    analysis1["complexity"].get("cyclomatic", 0)
                    - analysis2["complexity"].get("cyclomatic", 0)
                ),
                "quality_difference": (
                    analysis1["quality"].get("score", 0)
                    - analysis2["quality"].get("score", 0)
                ),
                "same_patterns": [
                    p["pattern"]
                    for p in analysis1["patterns"]
                    if any(
                        p2["pattern"] == p["pattern"] for p2 in analysis2["patterns"]
                    )
                ],
            },
        }

    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret similarity score"""
        if similarity >= 0.95:
            return "거의 동일한 코드입니다."
        elif similarity >= 0.8:
            return "매우 유사한 코드입니다."
        elif similarity >= 0.6:
            return "유사한 접근 방식을 사용합니다."
        elif similarity >= 0.4:
            return "일부 유사점이 있습니다."
        else:
            return "다른 접근 방식입니다."

    def get_pattern_suggestion(
        self, code: str, problem_description: str, language: str = "python"
    ) -> dict[str, Any]:
        """
        Suggest optimal pattern for given problem.

        Args:
            code: Current code attempt
            problem_description: Problem description
            language: Programming language

        Returns:
            Pattern suggestion with explanation
        """
        # Detect current patterns in code
        current_patterns = self._detect_patterns(code, language)

        # Find patterns matching problem description
        suggested_patterns = self.pattern_kb.find_similar_by_text(
            problem_description, top_k=3
        )

        # Compare current vs suggested
        current_names = {p["pattern"] for p in current_patterns}
        suggestions = []

        for sp in suggested_patterns:
            if sp["id"] not in current_names:
                suggestions.append(
                    {
                        "pattern": sp["id"],
                        "pattern_ko": sp["name_ko"],
                        "relevance": sp["similarity"],
                        "description": sp["description_ko"],
                        "example_code": sp["example_code"],
                        "reason": f"이 문제는 {sp['name_ko']} 패턴이 효과적일 수 있습니다.",
                    }
                )

        return {
            "current_patterns": current_patterns,
            "suggested_patterns": suggestions,
            "recommendation": suggestions[0] if suggestions else None,
        }

    # ============== Enhanced Analysis Methods ==============

    # Time/Space complexity patterns
    COMPLEXITY_PATTERNS = {
        "O(1)": {
            "indicators": [r"return\s+\w+\s*$", r"^\s*\w+\s*=\s*\w+\[\d+\]"],
            "description": "상수 시간 - 인덱스 접근, 직접 계산",
        },
        "O(log n)": {
            "indicators": [r"//\s*2", r"bisect", r"left.*right.*mid", r">>.*1"],
            "description": "로그 시간 - 이분 탐색, 분할",
        },
        "O(n)": {
            "indicators": [r"for\s+\w+\s+in\s+", r"while.*<.*len"],
            "description": "선형 시간 - 단일 순회",
        },
        "O(n log n)": {
            "indicators": [r"\.sort\(\)", r"sorted\(", r"heapq"],
            "description": "로그 선형 시간 - 정렬, 힙",
        },
        "O(n²)": {
            "indicators": [r"for.*:\s*\n\s*for", r"for.*for.*in"],
            "description": "제곱 시간 - 이중 루프",
        },
        "O(n³)": {
            "indicators": [r"for.*:\s*\n\s*for.*:\s*\n\s*for"],
            "description": "세제곱 시간 - 삼중 루프",
        },
        "O(2^n)": {
            "indicators": [
                r"def\s+\w+.*\n.*return.*\w+\(.*-\s*1\).*\+.*\w+\(.*-\s*1\)"
            ],
            "description": "지수 시간 - 재귀적 분기",
        },
    }

    SPACE_PATTERNS = {
        "O(1)": {
            "indicators": [r"^\s*\w+\s*=\s*0", r"^\s*\w+\s*=\s*None"],
            "description": "상수 공간 - 변수만 사용",
        },
        "O(n)": {
            "indicators": [r"\[\s*\]\s*$", r"set\(\)", r"dict\(\)", r"\{\s*\}"],
            "description": "선형 공간 - 리스트, 집합, 딕셔너리",
        },
        "O(n²)": {
            "indicators": [r"\[\s*\[", r"for.*\[\].*for"],
            "description": "제곱 공간 - 2D 배열",
        },
    }

    # Anti-patterns specific to algorithmic problems
    ALGORITHM_ANTIPATTERNS = {
        "unnecessary_sort": {
            "pattern": r"\.sort\(\).*for.*for",
            "message": "정렬 후 이중 루프는 비효율적입니다. 정렬만으로 해결할 수 있는지 확인하세요.",
            "suggestion": "정렬 후 투 포인터나 이분 탐색 사용을 고려하세요.",
            "severity": "warning",
        },
        "repeated_search": {
            "pattern": r"for.*:\s*\n\s*.*in\s+\w+",
            "message": "루프 내 반복 검색은 O(n²)입니다.",
            "suggestion": "해시맵(dict, set)을 사용하여 O(1) 검색으로 개선하세요.",
            "severity": "warning",
        },
        "string_concatenation_loop": {
            "pattern": r"for.*:\s*\n\s*.*\+\=\s*['\"]",
            "message": "루프 내 문자열 연결은 O(n²)입니다.",
            "suggestion": "리스트에 추가 후 ''.join()을 사용하세요.",
            "severity": "warning",
        },
        "list_as_queue": {
            "pattern": r"\.pop\s*\(\s*0\s*\)",
            "message": "리스트의 pop(0)은 O(n)입니다.",
            "suggestion": "collections.deque를 사용하여 O(1) popleft()를 사용하세요.",
            "severity": "error",
        },
        "redundant_copy": {
            "pattern": r"\w+\[:\]",
            "message": "불필요한 리스트 복사가 있을 수 있습니다.",
            "suggestion": "복사가 필요한지 확인하세요. 인플레이스 처리가 가능한지 검토하세요.",
            "severity": "info",
        },
        "global_state": {
            "pattern": r"global\s+\w+",
            "message": "전역 변수 사용은 버그 유발 가능성이 높습니다.",
            "suggestion": "함수 매개변수와 반환값을 사용하세요.",
            "severity": "warning",
        },
        "recursive_without_memo": {
            "pattern": r"def\s+(\w+).*:\s*\n(?:(?!@cache|@lru_cache|memo)[\s\S])*return.*\1\s*\(",
            "message": "메모이제이션 없는 재귀는 중복 계산이 발생합니다.",
            "suggestion": "@lru_cache 또는 딕셔너리 메모이제이션을 추가하세요.",
            "severity": "warning",
        },
        "n_squared_distinct": {
            "pattern": r"list\s*\(\s*set\s*\(",
            "message": "set을 list로 변환하는 것이 필요한지 확인하세요.",
            "suggestion": "set으로 유지하거나, 처음부터 리스트 중복 제거 로직을 개선하세요.",
            "severity": "info",
        },
    }

    def estimate_complexity(
        self, code: str, language: str = "python"
    ) -> dict[str, Any]:
        """
        코드의 시간/공간 복잡도를 추정합니다.

        Args:
            code: 분석할 코드
            language: 프로그래밍 언어

        Returns:
            복잡도 추정 결과
        """
        result = {
            "time_complexity": {
                "estimate": "O(?)",
                "confidence": 0.0,
                "explanation": "",
                "factors": [],
            },
            "space_complexity": {
                "estimate": "O(?)",
                "confidence": 0.0,
                "explanation": "",
                "factors": [],
            },
        }

        # Time complexity estimation
        time_scores = {}
        for complexity, info in self.COMPLEXITY_PATTERNS.items():
            score = 0
            for indicator in info["indicators"]:
                if re.search(indicator, code, re.MULTILINE):
                    score += 1
            if score > 0:
                time_scores[complexity] = score

        if time_scores:
            # 가장 높은 복잡도 선택 (보수적 추정)
            complexity_order = [
                "O(2^n)",
                "O(n³)",
                "O(n²)",
                "O(n log n)",
                "O(n)",
                "O(log n)",
                "O(1)",
            ]
            for comp in complexity_order:
                if comp in time_scores:
                    result["time_complexity"]["estimate"] = comp
                    result["time_complexity"]["confidence"] = min(
                        0.9, time_scores[comp] * 0.3
                    )
                    result["time_complexity"]["explanation"] = self.COMPLEXITY_PATTERNS[
                        comp
                    ]["description"]
                    break

        # 추가 분석: 루프 중첩 깊이
        loop_depth = self._count_nested_loops(code)
        if loop_depth >= 3:
            result["time_complexity"]["estimate"] = "O(n³) 이상"
            result["time_complexity"]["factors"].append(f"루프 중첩 깊이: {loop_depth}")
        elif loop_depth == 2:
            if result["time_complexity"]["estimate"] not in [
                "O(n²)",
                "O(n³)",
                "O(2^n)",
            ]:
                result["time_complexity"]["estimate"] = "O(n²)"
            result["time_complexity"]["factors"].append("이중 루프 감지")

        # Space complexity estimation
        space_scores = {}
        for complexity, info in self.SPACE_PATTERNS.items():
            score = 0
            for indicator in info["indicators"]:
                if re.search(indicator, code, re.MULTILINE):
                    score += 1
            if score > 0:
                space_scores[complexity] = score

        if space_scores:
            space_order = ["O(n²)", "O(n)", "O(1)"]
            for comp in space_order:
                if comp in space_scores:
                    result["space_complexity"]["estimate"] = comp
                    result["space_complexity"]["confidence"] = min(
                        0.9, space_scores[comp] * 0.3
                    )
                    result["space_complexity"]["explanation"] = self.SPACE_PATTERNS[
                        comp
                    ]["description"]
                    break

        # 재귀 함수 검사 (스택 공간)
        if re.search(r"def\s+(\w+).*:\s*\n.*return.*\1\s*\(", code, re.DOTALL):
            result["space_complexity"]["factors"].append(
                "재귀 함수 - 호출 스택 O(n) 또는 O(log n)"
            )

        return result

    def _count_nested_loops(self, code: str) -> int:
        """중첩 루프 깊이 계산"""
        max_depth = 0
        current_depth = 0

        for line in code.split("\n"):
            stripped = line.strip()
            if re.match(r"(for|while)\s+", stripped):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif stripped.startswith("return") or stripped.startswith("break"):
                current_depth = max(0, current_depth - 1)

        return max_depth

    def suggest_optimizations(self, code: str, language: str = "python") -> list[dict]:
        """
        코드 최적화 제안을 생성합니다.

        Args:
            code: 분석할 코드
            language: 프로그래밍 언어

        Returns:
            최적화 제안 리스트
        """
        suggestions = []

        # 감지된 패턴 기반 최적화 (TODO: use detected_patterns for optimization)
        self._detect_patterns(code, language)
        complexity = self.estimate_complexity(code, language)

        # 시간 복잡도 기반 제안
        time_est = complexity["time_complexity"]["estimate"]
        if "n²" in time_est:
            suggestions.append(
                {
                    "type": "time_optimization",
                    "priority": "high",
                    "current": time_est,
                    "message": "O(n²) 복잡도가 감지되었습니다.",
                    "suggestions": [
                        "해시맵을 사용하여 검색을 O(1)로 개선",
                        "정렬 + 투 포인터로 O(n log n) 가능 여부 확인",
                        "이분 탐색 적용 가능 여부 확인",
                    ],
                }
            )
        elif "n³" in time_est:
            suggestions.append(
                {
                    "type": "time_optimization",
                    "priority": "critical",
                    "current": time_est,
                    "message": "O(n³) 복잡도가 감지되었습니다. 심각한 성능 문제가 예상됩니다.",
                    "suggestions": [
                        "알고리즘 접근법 자체를 재검토하세요",
                        "동적 프로그래밍 적용 가능 여부 확인",
                        "삼중 루프 중 하나를 해시맵으로 대체",
                    ],
                }
            )

        # 패턴별 최적화 제안
        pattern_optimizations = {
            "dfs": {
                "check": r"def\s+dfs.*:.*\n(?:(?!visited)[\s\S]){0,200}dfs\s*\(",
                "message": "DFS에서 방문 체크가 누락된 것 같습니다.",
                "suggestion": "visited 집합을 사용하여 중복 방문을 방지하세요.",
            },
            "bfs": {
                "check": r"queue.*append.*\n(?:(?!visited)[\s\S]){0,100}queue",
                "message": "BFS에서 방문 체크가 누락된 것 같습니다.",
                "suggestion": "노드 추가 시 visited 체크를 수행하세요.",
            },
            "dp": {
                "check": r"dp\s*=\s*\[\s*\[.*\]\s*\*\s*\w+",
                "message": "2D DP 배열 초기화 방식이 잘못될 수 있습니다.",
                "suggestion": "[[0]*m for _ in range(n)] 형태로 초기화하세요.",
            },
        }

        for pattern_name, opt_info in pattern_optimizations.items():
            if re.search(opt_info["check"], code, re.MULTILINE | re.IGNORECASE):
                suggestions.append(
                    {
                        "type": "pattern_specific",
                        "pattern": pattern_name,
                        "priority": "medium",
                        "message": opt_info["message"],
                        "suggestion": opt_info["suggestion"],
                    }
                )

        # 일반적인 최적화 제안
        if re.search(r"for.*range\s*\(\s*len\s*\(", code):
            suggestions.append(
                {
                    "type": "pythonic",
                    "priority": "low",
                    "message": "range(len(x)) 대신 enumerate()를 사용하세요.",
                    "suggestion": "for i, item in enumerate(items): 형태가 더 Pythonic합니다.",
                }
            )

        if re.search(r"\+\s*=\s*\[", code):
            suggestions.append(
                {
                    "type": "performance",
                    "priority": "low",
                    "message": "리스트 += 보다 extend()가 더 효율적입니다.",
                    "suggestion": "list1 += list2 대신 list1.extend(list2)를 사용하세요.",
                }
            )

        return sorted(
            suggestions,
            key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                x["priority"], 4
            ),
        )

    def detect_antipatterns(self, code: str, language: str = "python") -> list[dict]:
        """
        알고리즘 안티패턴을 탐지합니다.

        Args:
            code: 분석할 코드
            language: 프로그래밍 언어

        Returns:
            탐지된 안티패턴 리스트
        """
        detected = []

        for name, info in self.ALGORITHM_ANTIPATTERNS.items():
            if re.search(info["pattern"], code, re.MULTILINE | re.DOTALL):
                detected.append(
                    {
                        "antipattern": name,
                        "message": info["message"],
                        "suggestion": info["suggestion"],
                        "severity": info["severity"],
                    }
                )

        # 정렬 후 반환
        severity_order = {"error": 0, "warning": 1, "info": 2}
        detected.sort(key=lambda x: severity_order.get(x["severity"], 3))

        return detected

    def full_analysis(
        self, code: str, language: str = "python", problem_info: dict | None = None
    ) -> dict[str, Any]:
        """
        코드의 전체 분석을 수행합니다 (기존 analyze + 새로운 기능).

        Args:
            code: 분석할 코드
            language: 프로그래밍 언어
            problem_info: 문제 정보 (옵션)

        Returns:
            종합 분석 결과
        """
        # 기본 분석
        base_analysis = self.analyze(code, language)

        # 복잡도 추정
        complexity_estimate = self.estimate_complexity(code, language)

        # 최적화 제안
        optimizations = self.suggest_optimizations(code, language)

        # 안티패턴 탐지
        antipatterns = self.detect_antipatterns(code, language)

        # 결과 통합
        result = {
            **base_analysis,
            "complexity_estimate": complexity_estimate,
            "optimizations": optimizations,
            "antipatterns": antipatterns,
        }

        # 종합 점수 계산
        base_score = base_analysis.get("quality", {}).get("score", 70)

        # 안티패턴 감점
        for ap in antipatterns:
            if ap["severity"] == "error":
                base_score -= 15
            elif ap["severity"] == "warning":
                base_score -= 10
            else:
                base_score -= 5

        # 복잡도 기반 감점
        time_comp = complexity_estimate["time_complexity"]["estimate"]
        if "n³" in time_comp or "2^n" in time_comp:
            base_score -= 20
        elif "n²" in time_comp:
            base_score -= 10

        result["overall_score"] = max(0, min(100, base_score))
        result["overall_grade"] = self._score_to_grade(result["overall_score"])

        # 요약 생성
        result["summary"] = self._generate_summary(result)

        return result

    def _generate_summary(self, analysis: dict) -> str:
        """분석 결과 요약 생성"""
        parts = []

        # 복잡도
        time_comp = analysis.get("complexity_estimate", {}).get("time_complexity", {})
        space_comp = analysis.get("complexity_estimate", {}).get("space_complexity", {})
        parts.append(
            f"**복잡도**: 시간 {time_comp.get('estimate', 'O(?)')}, 공간 {space_comp.get('estimate', 'O(?)')}"
        )

        # 점수
        score = analysis.get("overall_score", 0)
        grade = analysis.get("overall_grade", "?")
        parts.append(f"**품질 점수**: {score}/100 ({grade}등급)")

        # 패턴
        patterns = analysis.get("patterns", [])
        if patterns:
            pattern_names = [
                p.get("pattern_ko", p.get("pattern", "")) for p in patterns[:2]
            ]
            parts.append(f"**감지된 패턴**: {', '.join(pattern_names)}")

        # 주요 문제
        antipatterns = analysis.get("antipatterns", [])
        critical_issues = [
            ap for ap in antipatterns if ap["severity"] in ["error", "warning"]
        ]
        if critical_issues:
            parts.append(f"**주요 문제**: {len(critical_issues)}개 발견")

        # 최적화
        optimizations = analysis.get("optimizations", [])
        high_priority = [
            o for o in optimizations if o.get("priority") in ["critical", "high"]
        ]
        if high_priority:
            parts.append(f"**최적화 필요**: {len(high_priority)}개 항목")

        return "\n".join(parts)


# ============== Debugging Assistant ==============


class DebuggingAssistant:
    """
    실시간 디버깅 도움 시스템.

    오류 패턴 분석 및 해결 가이드 제공.
    """

    # 런타임 에러 패턴
    ERROR_PATTERNS = {
        "IndexError": {
            "description": "배열 범위 초과 오류",
            "common_causes": [
                "인덱스가 배열 길이 이상",
                "음수 인덱스 오류",
                "빈 배열에 접근",
                "off-by-one 오류 (경계값 실수)",
            ],
            "debugging_steps": [
                "1. 배열 길이와 접근하는 인덱스를 print로 확인하세요",
                "2. 루프의 range 범위가 올바른지 확인하세요 (len-1 vs len)",
                "3. 빈 배열이 입력될 수 있는지 확인하세요",
                "4. 배열 접근 전 bounds check를 추가하세요",
            ],
            "code_fix_hints": [
                "if i < len(arr): arr[i]  # bounds check",
                "for i in range(len(arr)):  # 0 to len-1",
                "if arr:  # 빈 배열 체크",
            ],
        },
        "KeyError": {
            "description": "딕셔너리 키 없음 오류",
            "common_causes": [
                "존재하지 않는 키 접근",
                "키 타입 불일치 (문자열 vs 정수)",
                "대소문자 차이",
            ],
            "debugging_steps": [
                "1. 딕셔너리의 키 목록을 print로 확인하세요",
                "2. 접근하려는 키의 타입과 값을 확인하세요",
                "3. get() 메서드로 기본값을 설정하세요",
            ],
            "code_fix_hints": [
                "value = d.get(key, default_value)",
                "if key in d: value = d[key]",
                "from collections import defaultdict",
            ],
        },
        "RecursionError": {
            "description": "재귀 깊이 초과 오류",
            "common_causes": [
                "종료 조건 누락 또는 잘못됨",
                "종료 조건에 도달하지 못함",
                "무한 재귀",
                "입력 크기가 너무 큼 (반복문으로 변환 필요)",
            ],
            "debugging_steps": [
                "1. 재귀 함수의 종료 조건을 확인하세요",
                "2. 재귀 호출 시 인자가 종료 조건에 수렴하는지 확인하세요",
                "3. 작은 입력으로 재귀 호출 흐름을 추적하세요",
                "4. sys.setrecursionlimit() 증가 또는 반복문으로 변환 고려",
            ],
            "code_fix_hints": [
                "if base_case: return result  # 종료 조건",
                "sys.setrecursionlimit(10**6)  # 깊이 제한 증가",
                "# 꼬리 재귀 → 반복문 변환 고려",
            ],
        },
        "TimeoutError": {
            "description": "시간 초과 오류",
            "common_causes": [
                "알고리즘 복잡도가 너무 높음",
                "불필요한 중복 계산",
                "비효율적인 자료구조 사용",
                "무한 루프",
            ],
            "debugging_steps": [
                "1. 알고리즘 시간 복잡도를 분석하세요",
                "2. 루프 내 불필요한 연산이 있는지 확인하세요",
                "3. 메모이제이션을 적용할 수 있는지 확인하세요",
                "4. 더 효율적인 자료구조(해시맵, 힙 등)를 고려하세요",
            ],
            "code_fix_hints": [
                "@lru_cache(maxsize=None)  # 메모이제이션",
                "# O(n²) → O(n log n) 또는 O(n) 개선",
                "# list.pop(0) → deque.popleft()",
            ],
        },
        "MemoryError": {
            "description": "메모리 초과 오류",
            "common_causes": [
                "너무 큰 자료구조 생성",
                "불필요한 데이터 저장",
                "재귀 스택 오버플로우",
            ],
            "debugging_steps": [
                "1. 생성하는 자료구조 크기를 확인하세요",
                "2. 제너레이터를 사용하여 메모리 사용을 줄이세요",
                "3. 필요한 데이터만 저장하세요",
            ],
            "code_fix_hints": [
                "# 리스트 대신 제너레이터 사용",
                "for x in range(n):  # range는 제너레이터",
                "# DP 공간 최적화: 2D → 1D",
            ],
        },
        "ValueError": {
            "description": "잘못된 값 오류",
            "common_causes": [
                "형변환 실패 (문자열 → 숫자)",
                "잘못된 함수 인자",
                "빈 시퀀스에서 max/min 호출",
            ],
            "debugging_steps": [
                "1. 입력 데이터 형식을 확인하세요",
                "2. 형변환 전 데이터를 검증하세요",
                "3. 빈 시퀀스 체크를 추가하세요",
            ],
            "code_fix_hints": [
                "if s.isdigit(): num = int(s)",
                "if arr: result = max(arr)",
                "try: num = int(s) except ValueError: ...",
            ],
        },
        "TypeError": {
            "description": "타입 오류",
            "common_causes": [
                "연산 불가능한 타입 간 연산",
                "None 값에 연산 수행",
                "함수 인자 타입 불일치",
            ],
            "debugging_steps": [
                "1. 변수의 타입을 type()으로 확인하세요",
                "2. None이 반환될 수 있는 경우를 확인하세요",
                "3. 함수의 예상 인자 타입을 확인하세요",
            ],
            "code_fix_hints": [
                "if value is not None: ...",
                "str(num) + ' ' + other_str  # 타입 변환",
                "print(type(variable))  # 타입 확인",
            ],
        },
        "WrongAnswer": {
            "description": "오답 (로직 오류)",
            "common_causes": [
                "엣지 케이스 미처리",
                "off-by-one 오류",
                "정수 오버플로우",
                "부동소수점 정밀도 문제",
                "잘못된 알고리즘 선택",
            ],
            "debugging_steps": [
                "1. 엣지 케이스를 테스트하세요 (빈 입력, 최대값, 최소값)",
                "2. 경계값 조건을 확인하세요 (< vs <=)",
                "3. 중간 결과를 print로 확인하세요",
                "4. 손으로 작은 예제를 따라가 보세요",
            ],
            "code_fix_hints": [
                "# 엣지 케이스: 빈 배열, 단일 원소, 동일 원소",
                "# < vs <= 확인",
                "# 부동소수점: abs(a - b) < 1e-9",
            ],
        },
        "RuntimeError": {
            "description": "런타임 오류",
            "common_causes": [
                "일반적인 실행 중 오류",
                "스택 오버플로우",
                "잘못된 상태",
            ],
            "debugging_steps": [
                "1. 전체 에러 메시지를 확인하세요",
                "2. 에러 발생 지점의 변수 값을 확인하세요",
                "3. 특정 케이스에서만 발생하는지 확인하세요",
            ],
            "code_fix_hints": [
                "try: ... except Exception as e: print(e)",
                "import traceback; traceback.print_exc()",
            ],
        },
    }

    # 일반적인 알고리즘 실수 패턴
    COMMON_MISTAKES = {
        "off_by_one": {
            "pattern": r"range\s*\(\s*\d+\s*,\s*\w+\s*\)|range\s*\(\s*len\s*\(",
            "description": "경계값 오류 (off-by-one)",
            "check_points": [
                "range(n)은 0부터 n-1까지",
                "range(1, n+1)은 1부터 n까지",
                "arr[len(arr)]는 오류 (마지막 원소는 arr[len(arr)-1])",
            ],
        },
        "integer_division": {
            "pattern": r"/(?!/)|\bdiv\b",
            "description": "정수 나눗셈 오류",
            "check_points": [
                "Python 3에서 /는 float, //가 정수 나눗셈",
                "음수 나눗셈 주의: -7 // 2 = -4 (내림)",
            ],
        },
        "mutable_default": {
            "pattern": r"def\s+\w+\s*\([^)]*=\s*\[\s*\]|def\s+\w+\s*\([^)]*=\s*\{\s*\}",
            "description": "변경 가능한 기본 인자",
            "check_points": [
                "리스트나 딕셔너리를 기본 인자로 사용하면 공유됨",
                "def f(arr=None): arr = arr or [] 패턴 사용",
            ],
        },
        "shallow_copy": {
            "pattern": r"\w+\s*=\s*\w+(?:\s*$|\s*#)",
            "description": "얕은 복사로 인한 참조 공유",
            "check_points": [
                "리스트 복사: new_list = old_list[:] 또는 copy.deepcopy()",
                "2D 배열: [[0]*m for _ in range(n)], [row[:] for row in matrix]",
            ],
        },
        "float_comparison": {
            "pattern": r"==\s*\d+\.\d+|!=\s*\d+\.\d+",
            "description": "부동소수점 비교 오류",
            "check_points": [
                "== 대신 abs(a - b) < 1e-9 사용",
                "정수로 변환 가능하면 정수 사용",
            ],
        },
        "string_immutable": {
            "pattern": r"\w+\[\d+\]\s*=\s*['\"]",
            "description": "문자열 불변성 오류",
            "check_points": [
                "문자열은 수정 불가: s[0] = 'a' 불가능",
                "리스트로 변환 후 수정: list(s), ''.join()",
            ],
        },
        "global_variable": {
            "pattern": r"global\s+\w+",
            "description": "전역 변수 의존",
            "check_points": [
                "전역 변수는 디버깅을 어렵게 함",
                "함수 매개변수와 반환값 사용 권장",
                "클래스나 클로저로 상태 관리",
            ],
        },
    }

    # 테스트 케이스 타입별 디버깅 힌트
    TEST_CASE_HINTS = {
        "empty_input": {
            "description": "빈 입력",
            "hints": [
                "빈 배열/문자열 체크: if not arr: return ...",
                "len(arr) == 0 조건 처리",
            ],
        },
        "single_element": {
            "description": "단일 원소",
            "hints": [
                "원소가 하나일 때 특수 처리 필요한지 확인",
                "투 포인터에서 left == right 케이스",
            ],
        },
        "all_same": {
            "description": "모든 원소 동일",
            "hints": [
                "중복 원소 처리 로직 확인",
                "정렬 후 연속된 동일 원소 처리",
            ],
        },
        "sorted_input": {
            "description": "정렬된 입력",
            "hints": [
                "이미 정렬된 입력에서 알고리즘 동작 확인",
                "역순 정렬된 입력도 테스트",
            ],
        },
        "negative_numbers": {
            "description": "음수 포함",
            "hints": [
                "음수 인덱스 접근 주의",
                "음수 나눗셈/나머지 연산 주의",
                "합이 0이 되는 케이스",
            ],
        },
        "large_numbers": {
            "description": "큰 수",
            "hints": [
                "정수 오버플로우 (Python은 자동 처리)",
                "시간 초과 가능성",
                "부동소수점 정밀도",
            ],
        },
        "boundary_values": {
            "description": "경계값",
            "hints": [
                "최대/최소 제한값 테스트",
                "0, -1, n, n-1 등 경계값 확인",
            ],
        },
    }

    def __init__(self, code_analyzer: CodeAnalyzer | None = None):
        """
        DebuggingAssistant 초기화

        Args:
            code_analyzer: 코드 분석기 (옵션)
        """
        self.code_analyzer = code_analyzer or CodeAnalyzer()

    def analyze_error(
        self,
        code: str,
        error_type: str,
        error_message: str = "",
        test_case: dict | None = None,
        language: str = "python",
    ) -> dict:
        """
        에러를 분석하고 디버깅 가이드를 제공합니다.

        Args:
            code: 오류 발생 코드
            error_type: 에러 타입 (예: "IndexError", "WrongAnswer")
            error_message: 상세 에러 메시지
            test_case: 실패한 테스트 케이스 (input, expected_output)
            language: 프로그래밍 언어

        Returns:
            디버깅 가이드 딕셔너리
        """
        result = {
            "error_type": error_type,
            "error_message": error_message,
            "diagnosis": {},
            "debugging_guide": {},
            "code_specific_issues": [],
            "test_case_analysis": {},
            "suggested_fixes": [],
        }

        # 1. 에러 타입 기반 진단
        if error_type in self.ERROR_PATTERNS:
            pattern_info = self.ERROR_PATTERNS[error_type]
            result["diagnosis"] = {
                "description": pattern_info["description"],
                "common_causes": pattern_info["common_causes"],
            }
            result["debugging_guide"] = {
                "steps": pattern_info["debugging_steps"],
                "code_hints": pattern_info["code_fix_hints"],
            }

        # 2. 코드 내 문제점 분석
        code_issues = self._analyze_code_issues(code, error_type, language)
        result["code_specific_issues"] = code_issues

        # 3. 테스트 케이스 분석
        if test_case:
            test_analysis = self._analyze_test_case(test_case, error_type)
            result["test_case_analysis"] = test_analysis

        # 4. 수정 제안 생성
        suggested_fixes = self._generate_fix_suggestions(
            code, error_type, code_issues, test_case, language
        )
        result["suggested_fixes"] = suggested_fixes

        # 5. 요약 생성
        result["summary"] = self._generate_debug_summary(result)

        return result

    def _analyze_code_issues(
        self, code: str, error_type: str, language: str
    ) -> list[dict]:
        """코드 내 잠재적 문제점 분석"""
        issues = []

        # 일반적인 실수 패턴 검사
        for mistake_name, mistake_info in self.COMMON_MISTAKES.items():
            if re.search(mistake_info["pattern"], code, re.MULTILINE):
                issues.append(
                    {
                        "type": mistake_name,
                        "description": mistake_info["description"],
                        "check_points": mistake_info["check_points"],
                        "relevance": self._calculate_relevance(
                            mistake_name, error_type
                        ),
                    }
                )

        # 에러 타입별 특정 검사
        if error_type == "IndexError":
            # 배열 접근 패턴 분석
            array_accesses = re.findall(r"\w+\s*\[\s*([^]]+)\s*\]", code)
            for access in array_accesses:
                if re.search(r"\+\s*1|len\s*\(", access):
                    issues.append(
                        {
                            "type": "suspicious_index",
                            "description": f"의심스러운 인덱스 접근: [{access}]",
                            "check_points": [
                                "인덱스가 배열 범위 내인지 확인",
                                "+1 이 필요한지 재검토",
                            ],
                            "relevance": "high",
                        }
                    )

        elif error_type == "RecursionError":
            # 재귀 함수 분석
            recursive_funcs = re.findall(r"def\s+(\w+)\s*\([^)]*\):", code)
            for func in recursive_funcs:
                if re.search(rf"{func}\s*\(", code):
                    # 종료 조건 확인
                    if not re.search(r"if\s+.*:\s*\n\s*return", code):
                        issues.append(
                            {
                                "type": "missing_base_case",
                                "description": f"함수 '{func}'에 명확한 종료 조건이 없을 수 있습니다",
                                "check_points": [
                                    "재귀의 종료 조건이 있는지 확인",
                                    "종료 조건에 도달할 수 있는지 확인",
                                ],
                                "relevance": "high",
                            }
                        )

        elif error_type == "TimeoutError":
            # 복잡도 분석
            analysis = self.code_analyzer.estimate_complexity(code, language)
            time_comp = analysis["time_complexity"]["estimate"]

            if "n²" in time_comp or "n³" in time_comp:
                issues.append(
                    {
                        "type": "high_complexity",
                        "description": f"높은 시간 복잡도: {time_comp}",
                        "check_points": [
                            "더 효율적인 알고리즘 사용 가능 여부",
                            "불필요한 중복 계산 제거",
                            "적절한 자료구조 사용",
                        ],
                        "relevance": "high",
                    }
                )

        # 관련도 순 정렬
        relevance_order = {"high": 0, "medium": 1, "low": 2}
        issues.sort(key=lambda x: relevance_order.get(x.get("relevance", "low"), 3))

        return issues

    def _analyze_test_case(self, test_case: dict, error_type: str) -> dict:
        """테스트 케이스 분석"""
        analysis = {
            "input_characteristics": [],
            "potential_edge_cases": [],
            "debugging_suggestions": [],
        }

        test_input = test_case.get("input", "")
        expected = test_case.get("expected_output", "")
        actual = test_case.get("actual_output", "")

        # 입력 특성 분석
        input_str = str(test_input)

        # 빈 입력 체크
        if not test_input or input_str in ["[]", "{}", '""', "''"]:
            analysis["input_characteristics"].append("empty_input")
            analysis["potential_edge_cases"].extend(
                self.TEST_CASE_HINTS["empty_input"]["hints"]
            )

        # 숫자 분석
        numbers = re.findall(r"-?\d+", input_str)
        if numbers:
            nums = [int(n) for n in numbers]

            # 단일 원소
            if len(nums) == 1:
                analysis["input_characteristics"].append("single_element")
                analysis["potential_edge_cases"].extend(
                    self.TEST_CASE_HINTS["single_element"]["hints"]
                )

            # 모든 원소 동일
            if len(set(nums)) == 1 and len(nums) > 1:
                analysis["input_characteristics"].append("all_same")
                analysis["potential_edge_cases"].extend(
                    self.TEST_CASE_HINTS["all_same"]["hints"]
                )

            # 음수 포함
            if any(n < 0 for n in nums):
                analysis["input_characteristics"].append("negative_numbers")
                analysis["potential_edge_cases"].extend(
                    self.TEST_CASE_HINTS["negative_numbers"]["hints"]
                )

            # 큰 수
            if any(abs(n) > 10**6 for n in nums):
                analysis["input_characteristics"].append("large_numbers")
                analysis["potential_edge_cases"].extend(
                    self.TEST_CASE_HINTS["large_numbers"]["hints"]
                )

            # 정렬 여부
            if nums == sorted(nums) or nums == sorted(nums, reverse=True):
                analysis["input_characteristics"].append("sorted_input")
                analysis["potential_edge_cases"].extend(
                    self.TEST_CASE_HINTS["sorted_input"]["hints"]
                )

        # 오답인 경우 출력 비교
        if error_type == "WrongAnswer" and expected and actual:
            analysis["debugging_suggestions"].append(f"예상 출력: {expected}")
            analysis["debugging_suggestions"].append(f"실제 출력: {actual}")

            # 차이점 분석
            if str(expected) != str(actual):
                try:
                    exp_nums = [int(n) for n in re.findall(r"-?\d+", str(expected))]
                    act_nums = [int(n) for n in re.findall(r"-?\d+", str(actual))]

                    if exp_nums and act_nums:
                        if len(exp_nums) != len(act_nums):
                            analysis["debugging_suggestions"].append(
                                f"출력 길이 차이: 예상 {len(exp_nums)}개, 실제 {len(act_nums)}개"
                            )
                        else:
                            diffs = [
                                (i, e, a)
                                for i, (e, a) in enumerate(zip(exp_nums, act_nums))
                                if e != a
                            ]
                            if diffs:
                                first_diff = diffs[0]
                                analysis["debugging_suggestions"].append(
                                    f"첫 번째 차이점: 인덱스 {first_diff[0]}에서 "
                                    f"예상 {first_diff[1]}, 실제 {first_diff[2]}"
                                )
                except (ValueError, TypeError):
                    pass

        return analysis

    def _calculate_relevance(self, mistake_type: str, error_type: str) -> str:
        """실수 타입과 에러 타입의 관련도 계산"""
        relevance_map = {
            ("off_by_one", "IndexError"): "high",
            ("off_by_one", "WrongAnswer"): "high",
            ("integer_division", "WrongAnswer"): "high",
            ("mutable_default", "WrongAnswer"): "medium",
            ("shallow_copy", "WrongAnswer"): "high",
            ("float_comparison", "WrongAnswer"): "high",
            ("string_immutable", "TypeError"): "high",
            ("global_variable", "WrongAnswer"): "medium",
        }
        return relevance_map.get((mistake_type, error_type), "low")

    def _generate_fix_suggestions(
        self,
        code: str,
        error_type: str,
        code_issues: list,
        test_case: dict | None,
        language: str,
    ) -> list[dict]:
        """수정 제안 생성"""
        suggestions = []

        # 에러 타입별 기본 제안
        if error_type in self.ERROR_PATTERNS:
            for hint in self.ERROR_PATTERNS[error_type]["code_fix_hints"]:
                suggestions.append(
                    {
                        "type": "code_hint",
                        "priority": "medium",
                        "description": hint,
                    }
                )

        # 코드 이슈 기반 제안
        for issue in code_issues:
            if issue.get("relevance") == "high":
                for check in issue.get("check_points", []):
                    suggestions.append(
                        {
                            "type": "check_point",
                            "priority": "high",
                            "description": check,
                            "related_issue": issue["type"],
                        }
                    )

        # 중복 제거 및 우선순위 정렬
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = s["description"]
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_suggestions.sort(
            key=lambda x: priority_order.get(x.get("priority", "low"), 3)
        )

        return unique_suggestions[:10]  # 상위 10개

    def _generate_debug_summary(self, analysis: dict) -> str:
        """디버깅 분석 요약 생성"""
        parts = []

        error_type = analysis.get("error_type", "Unknown")
        diagnosis = analysis.get("diagnosis", {})

        # 에러 요약
        parts.append(f"**에러**: {error_type}")
        if diagnosis.get("description"):
            parts.append(f"**원인**: {diagnosis['description']}")

        # 주요 이슈
        code_issues = analysis.get("code_specific_issues", [])
        high_issues = [i for i in code_issues if i.get("relevance") == "high"]
        if high_issues:
            parts.append(f"**주요 문제점**: {len(high_issues)}개 발견")
            for issue in high_issues[:2]:
                parts.append(f"  - {issue['description']}")

        # 테스트 케이스 특성
        test_analysis = analysis.get("test_case_analysis", {})
        characteristics = test_analysis.get("input_characteristics", [])
        if characteristics:
            parts.append(f"**입력 특성**: {', '.join(characteristics)}")

        # 수정 제안
        fixes = analysis.get("suggested_fixes", [])
        if fixes:
            parts.append(f"**권장 조치**: {fixes[0]['description']}")

        return "\n".join(parts)

    def get_edge_case_suggestions(
        self, problem_type: str, constraints: dict | None = None
    ) -> list[dict]:
        """
        문제 타입에 맞는 엣지 케이스 제안

        Args:
            problem_type: 문제 타입 (예: "array", "string", "tree", "graph")
            constraints: 제약 조건 (예: {"n": 10**5, "values": [-10**9, 10**9]})

        Returns:
            테스트해야 할 엣지 케이스 리스트
        """
        edge_cases = []

        # 공통 엣지 케이스
        common_cases = [
            {"case": "빈 입력", "input": "[]", "description": "빈 배열/문자열"},
            {"case": "단일 원소", "input": "[1]", "description": "원소가 하나인 경우"},
            {
                "case": "두 원소",
                "input": "[1, 2]",
                "description": "최소 비교 가능 크기",
            },
        ]
        edge_cases.extend(common_cases)

        # 문제 타입별 엣지 케이스
        type_specific = {
            "array": [
                {"case": "모든 원소 동일", "input": "[5, 5, 5, 5]"},
                {"case": "오름차순 정렬", "input": "[1, 2, 3, 4, 5]"},
                {"case": "내림차순 정렬", "input": "[5, 4, 3, 2, 1]"},
                {"case": "음수 포함", "input": "[-3, -1, 0, 1, 3]"},
            ],
            "string": [
                {"case": "빈 문자열", "input": '""'},
                {"case": "한 글자", "input": '"a"'},
                {"case": "모든 글자 동일", "input": '"aaaa"'},
                {"case": "팰린드롬", "input": '"abba"'},
            ],
            "tree": [
                {"case": "루트만", "input": "root만 있는 트리"},
                {"case": "편향 트리", "input": "일자형 트리 (linked list)"},
                {"case": "완전 이진 트리", "input": "균형 잡힌 트리"},
            ],
            "graph": [
                {"case": "노드 1개", "input": "단일 노드"},
                {"case": "연결 없음", "input": "간선이 없는 그래프"},
                {"case": "완전 그래프", "input": "모든 노드가 연결"},
                {"case": "사이클", "input": "사이클이 있는 경우"},
            ],
        }

        if problem_type in type_specific:
            edge_cases.extend(type_specific[problem_type])

        # 제약 조건 기반 엣지 케이스
        if constraints:
            n_max = constraints.get("n")
            if n_max:
                edge_cases.append(
                    {
                        "case": "최대 크기",
                        "input": f"n = {n_max}",
                        "description": "시간/공간 제한 테스트",
                    }
                )

            value_range = constraints.get("values")
            if value_range:
                edge_cases.append(
                    {
                        "case": "최소값",
                        "input": f"모든 값이 {value_range[0]}",
                    }
                )
                edge_cases.append(
                    {
                        "case": "최대값",
                        "input": f"모든 값이 {value_range[1]}",
                    }
                )

        return edge_cases


# Factory function
def get_debugging_assistant(
    code_analyzer: CodeAnalyzer | None = None,
) -> DebuggingAssistant:
    """DebuggingAssistant 인스턴스 생성 헬퍼"""
    return DebuggingAssistant(code_analyzer)
