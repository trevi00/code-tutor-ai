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
