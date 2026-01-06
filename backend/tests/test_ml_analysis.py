"""Tests for ML Analysis Module (code_analyzer, code_classifier, quality_service)."""

import ast
from datetime import date, datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_tutor.ml.analysis.code_analyzer import (
    CodeAnalyzer,
    DebuggingAssistant,
    get_debugging_assistant,
)
from code_tutor.ml.analysis.code_classifier import CodeQualityClassifier


# ============== CodeAnalyzer Tests ==============


class TestCodeAnalyzerInit:
    """Tests for CodeAnalyzer initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        analyzer = CodeAnalyzer()
        assert analyzer.config is not None
        assert analyzer._code_embedder is None
        assert analyzer._pattern_kb is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        mock_config = MagicMock()
        analyzer = CodeAnalyzer(config=mock_config)
        assert analyzer.config == mock_config

    def test_code_smells_defined(self):
        """Test that code smells patterns are defined."""
        assert "long_function" in CodeAnalyzer.CODE_SMELLS
        assert "deep_nesting" in CodeAnalyzer.CODE_SMELLS
        assert "magic_numbers" in CodeAnalyzer.CODE_SMELLS
        assert "empty_except" in CodeAnalyzer.CODE_SMELLS
        assert "print_debugging" in CodeAnalyzer.CODE_SMELLS

    def test_algorithm_signatures_defined(self):
        """Test that algorithm signatures are defined."""
        assert "two_pointers" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "sliding_window" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "binary_search" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "dfs" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "bfs" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "dp" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "greedy" in CodeAnalyzer.ALGORITHM_SIGNATURES
        assert "recursion" in CodeAnalyzer.ALGORITHM_SIGNATURES


class TestCodeAnalyzerCountLines:
    """Tests for line counting functionality."""

    def test_count_lines_simple(self):
        """Test counting lines in simple code."""
        analyzer = CodeAnalyzer()
        code = "x = 1\ny = 2\nz = 3"
        result = analyzer._count_lines(code)

        assert result["total"] == 3
        assert result["code"] == 3
        assert result["blank"] == 0
        assert result["comment"] == 0

    def test_count_lines_with_blanks(self):
        """Test counting lines with blank lines."""
        analyzer = CodeAnalyzer()
        code = "x = 1\n\ny = 2\n\nz = 3"
        result = analyzer._count_lines(code)

        assert result["total"] == 5
        assert result["blank"] == 2

    def test_count_lines_with_comments(self):
        """Test counting lines with comments."""
        analyzer = CodeAnalyzer()
        code = "# Comment\nx = 1\n# Another comment\ny = 2"
        result = analyzer._count_lines(code)

        assert result["total"] == 4
        assert result["comment"] == 2
        assert result["code"] == 2

    def test_count_lines_empty_code(self):
        """Test counting lines in empty code."""
        analyzer = CodeAnalyzer()
        code = ""
        result = analyzer._count_lines(code)

        assert result["total"] == 1  # Empty string split creates one element


class TestCodeAnalyzerPatternDetection:
    """Tests for algorithm pattern detection."""

    def test_detect_two_pointers_pattern(self):
        """Test detecting two pointers pattern."""
        analyzer = CodeAnalyzer()
        code = """
def two_sum(arr, target):
    left = 0
    right = len(arr) - 1
    while left < right:
        if arr[left] + arr[right] == target:
            return [left, right]
        elif arr[left] + arr[right] < target:
            left += 1
        else:
            right -= 1
"""
        patterns = analyzer._detect_patterns(code, "python")
        pattern_names = [p["pattern"] for p in patterns]
        assert "two_pointers" in pattern_names

    def test_detect_binary_search_pattern(self):
        """Test detecting binary search pattern."""
        analyzer = CodeAnalyzer()
        # Code with clear binary search signatures
        code = """
import bisect
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return bisect.bisect_left(arr, target)
"""
        patterns = analyzer._detect_patterns(code, "python")
        pattern_names = [p["pattern"] for p in patterns]
        assert "binary_search" in pattern_names

    def test_detect_bfs_pattern(self):
        """Test detecting BFS pattern."""
        analyzer = CodeAnalyzer()
        code = """
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append(neighbor)
"""
        patterns = analyzer._detect_patterns(code, "python")
        pattern_names = [p["pattern"] for p in patterns]
        assert "bfs" in pattern_names

    def test_detect_dp_pattern(self):
        """Test detecting DP pattern."""
        analyzer = CodeAnalyzer()
        code = """
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

dp = [0] * 100
memo = {}
"""
        patterns = analyzer._detect_patterns(code, "python")
        pattern_names = [p["pattern"] for p in patterns]
        assert "dp" in pattern_names

    def test_no_pattern_detected(self):
        """Test when no clear pattern is detected."""
        analyzer = CodeAnalyzer()
        code = """
x = 1
y = 2
print(x + y)
"""
        patterns = analyzer._detect_patterns(code, "python")
        assert len(patterns) == 0 or all(p["confidence"] < 0.5 for p in patterns)


class TestCodeAnalyzerQualityAnalysis:
    """Tests for code quality analysis."""

    def test_analyze_quality_perfect_code(self):
        """Test quality analysis on clean code."""
        analyzer = CodeAnalyzer()
        code = """
def add(a, b):
    return a + b
"""
        result = analyzer._analyze_quality(code, "python")

        assert "score" in result
        assert "grade" in result
        assert "smells" in result
        assert result["score"] >= 70  # Clean code should score well

    def test_analyze_quality_with_code_smells(self):
        """Test quality analysis with code smells."""
        analyzer = CodeAnalyzer()
        code = """
def bad_function():
    try:
        x = 100
        y = 200
        print("debugging")
    except:
        pass
"""
        result = analyzer._analyze_quality(code, "python")

        assert len(result["smells"]) > 0
        smell_types = [s["type"] for s in result["smells"]]
        assert "empty_except" in smell_types or "print_debugging" in smell_types

    def test_analyze_quality_deep_nesting(self):
        """Test detection of deep nesting."""
        analyzer = CodeAnalyzer()
        code = """
def deeply_nested():
    if True:
        if True:
            if True:
                if True:
                    if True:
                        x = 1
"""
        result = analyzer._analyze_quality(code, "python")

        smell_types = [s["type"] for s in result["smells"]]
        assert "deep_nesting" in smell_types

    def test_score_to_grade_a(self):
        """Test score to grade conversion for A."""
        analyzer = CodeAnalyzer()
        assert analyzer._score_to_grade(95) == "A"
        assert analyzer._score_to_grade(90) == "A"

    def test_score_to_grade_b(self):
        """Test score to grade conversion for B."""
        analyzer = CodeAnalyzer()
        assert analyzer._score_to_grade(85) == "B"
        assert analyzer._score_to_grade(80) == "B"

    def test_score_to_grade_c(self):
        """Test score to grade conversion for C."""
        analyzer = CodeAnalyzer()
        assert analyzer._score_to_grade(75) == "C"
        assert analyzer._score_to_grade(70) == "C"

    def test_score_to_grade_d(self):
        """Test score to grade conversion for D."""
        analyzer = CodeAnalyzer()
        assert analyzer._score_to_grade(65) == "D"
        assert analyzer._score_to_grade(60) == "D"

    def test_score_to_grade_f(self):
        """Test score to grade conversion for F."""
        analyzer = CodeAnalyzer()
        assert analyzer._score_to_grade(55) == "F"
        assert analyzer._score_to_grade(0) == "F"


class TestCodeAnalyzerComplexityAnalysis:
    """Tests for complexity analysis."""

    def test_analyze_complexity_simple(self):
        """Test complexity analysis on simple code."""
        analyzer = CodeAnalyzer()
        code = """
def simple():
    return 1
"""
        result = analyzer._analyze_complexity(code, "python")

        assert result["cyclomatic"] >= 1
        assert result["rating"] == "low"

    def test_analyze_complexity_with_conditions(self):
        """Test complexity analysis with conditions."""
        analyzer = CodeAnalyzer()
        code = """
def complex_func(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return 1
            else:
                return 2
        else:
            return 3
    else:
        for i in range(10):
            while i > 0:
                i -= 1
        return 4
"""
        result = analyzer._analyze_complexity(code, "python")

        assert result["cyclomatic"] > 5
        assert result["function_count"] == 1

    def test_calculate_cyclomatic_complexity(self):
        """Test cyclomatic complexity calculation."""
        analyzer = CodeAnalyzer()
        code = """
def func():
    if a:
        pass
    elif b:
        pass
    for i in range(10):
        pass
    while True:
        break
"""
        tree = ast.parse(code)
        complexity = analyzer._calculate_cyclomatic(tree)

        assert complexity >= 4  # 1 base + if + for + while

    def test_calculate_nesting_depth(self):
        """Test nesting depth calculation."""
        analyzer = CodeAnalyzer()
        code = """
if True:
    if True:
        if True:
            x = 1
"""
        depth = analyzer._calculate_nesting_depth(code)

        assert depth >= 3

    def test_complexity_rating_medium(self):
        """Test medium complexity rating."""
        analyzer = CodeAnalyzer()
        code = """
def func(a, b, c, d, e, f, g):
    if a: pass
    if b: pass
    if c: pass
    if d: pass
    if e: pass
    if f: pass
    for i in range(10):
        pass
"""
        result = analyzer._analyze_complexity(code, "python")

        assert result["rating"] in ["medium", "high"]


class TestCodeAnalyzerPythonQuality:
    """Tests for Python-specific quality checks."""

    def test_check_mutable_default(self):
        """Test detection of mutable default arguments."""
        analyzer = CodeAnalyzer()
        code = """
def bad_func(items=[]):
    items.append(1)
    return items
"""
        issues = analyzer._check_python_quality(code)

        issue_types = [i["type"] for i in issues]
        assert "mutable_default" in issue_types

    def test_check_unused_variables(self):
        """Test detection of unused variables."""
        analyzer = CodeAnalyzer()
        code = """
def func():
    unused_var = 1
    used_var = 2
    return used_var
"""
        issues = analyzer._check_python_quality(code)

        issue_types = [i["type"] for i in issues]
        assert "unused_variables" in issue_types

    def test_check_syntax_error(self):
        """Test handling of syntax errors."""
        analyzer = CodeAnalyzer()
        code = """
def func(
    return 1
"""
        issues = analyzer._check_python_quality(code)

        issue_types = [i["type"] for i in issues]
        assert "syntax_error" in issue_types


class TestCodeAnalyzerFullAnalysis:
    """Tests for full code analysis."""

    def test_analyze_full(self):
        """Test full analysis method."""
        analyzer = CodeAnalyzer()
        code = """
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
        result = analyzer.analyze(code, "python")

        assert "language" in result
        assert "lines_of_code" in result
        assert "patterns" in result
        assert "quality" in result
        assert "complexity" in result
        assert "suggestions" in result
        assert "code_smells" in result

    def test_analyze_with_options(self):
        """Test analysis with selective options."""
        analyzer = CodeAnalyzer()
        code = "x = 1"

        result = analyzer.analyze(
            code,
            language="python",
            include_patterns=False,
            include_quality=False,
            include_complexity=False,
            include_suggestions=False,
        )

        assert result["patterns"] == []
        assert result["quality"] == {}
        assert result["complexity"] == {}
        assert result["suggestions"] == []


class TestCodeAnalyzerSuggestions:
    """Tests for suggestion generation."""

    def test_generate_suggestions_low_quality(self):
        """Test suggestions for low quality code."""
        analyzer = CodeAnalyzer()
        analysis = {
            "patterns": [],
            "quality": {"score": 50},
            "complexity": {"cyclomatic": 5, "nesting_depth": 2},
            "code_smells": [],
        }

        suggestions = analyzer._generate_suggestions(analysis)

        assert len(suggestions) > 0
        assert any(s["type"] == "code_quality" for s in suggestions)

    def test_generate_suggestions_high_complexity(self):
        """Test suggestions for high complexity code."""
        analyzer = CodeAnalyzer()
        analysis = {
            "patterns": [],
            "quality": {"score": 80},
            "complexity": {"cyclomatic": 15, "nesting_depth": 6},
            "code_smells": [],
        }

        suggestions = analyzer._generate_suggestions(analysis)

        high_priority = [s for s in suggestions if s["priority"] == "high"]
        assert len(high_priority) > 0

    def test_generate_suggestions_sorted_by_priority(self):
        """Test that suggestions are sorted by priority."""
        analyzer = CodeAnalyzer()
        analysis = {
            "patterns": [{"pattern": "test", "confidence": 0.5}],
            "quality": {"score": 60},
            "complexity": {"cyclomatic": 12, "nesting_depth": 5},
            "code_smells": [{"type": "test", "severity": "error", "message": "test"}],
        }

        suggestions = analyzer._generate_suggestions(analysis)

        if len(suggestions) > 1:
            priorities = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(suggestions) - 1):
                p1 = priorities.get(suggestions[i]["priority"], 3)
                p2 = priorities.get(suggestions[i + 1]["priority"], 3)
                assert p1 <= p2


class TestCodeAnalyzerComplexityEstimation:
    """Tests for time/space complexity estimation."""

    def test_estimate_complexity_linear(self):
        """Test linear complexity estimation."""
        analyzer = CodeAnalyzer()
        code = """
def linear(arr):
    total = 0
    for num in arr:
        total += num
    return total
"""
        result = analyzer.estimate_complexity(code)

        assert "time_complexity" in result
        assert "space_complexity" in result
        assert "O(n)" in result["time_complexity"]["estimate"]

    def test_estimate_complexity_quadratic(self):
        """Test quadratic complexity estimation."""
        analyzer = CodeAnalyzer()
        code = """
def quadratic(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            print(arr[i], arr[j])
"""
        result = analyzer.estimate_complexity(code)

        assert "n²" in result["time_complexity"]["estimate"]

    def test_estimate_complexity_logarithmic(self):
        """Test logarithmic complexity estimation."""
        analyzer = CodeAnalyzer()
        code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
"""
        result = analyzer.estimate_complexity(code)

        time_est = result["time_complexity"]["estimate"]
        assert "log" in time_est.lower() or "n" in time_est

    def test_count_nested_loops(self):
        """Test nested loop counting."""
        analyzer = CodeAnalyzer()
        code = """
for i in range(n):
    for j in range(n):
        for k in range(n):
            pass
"""
        depth = analyzer._count_nested_loops(code)

        assert depth == 3


class TestCodeAnalyzerOptimizations:
    """Tests for optimization suggestions."""

    def test_suggest_optimizations_quadratic(self):
        """Test optimization suggestions for quadratic code."""
        analyzer = CodeAnalyzer()
        code = """
def find_pair(arr, target):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if arr[i] + arr[j] == target:
                return [i, j]
"""
        suggestions = analyzer.suggest_optimizations(code)

        assert len(suggestions) > 0
        assert any("time_optimization" in s.get("type", "") for s in suggestions)

    def test_suggest_optimizations_range_len(self):
        """Test suggestion for range(len()) pattern."""
        analyzer = CodeAnalyzer()
        code = """
for i in range(len(items)):
    print(items[i])
"""
        suggestions = analyzer.suggest_optimizations(code)

        pythonic = [s for s in suggestions if s.get("type") == "pythonic"]
        assert len(pythonic) > 0


class TestCodeAnalyzerAntipatterns:
    """Tests for anti-pattern detection."""

    def test_detect_list_as_queue(self):
        """Test detection of list used as queue."""
        analyzer = CodeAnalyzer()
        code = """
queue = []
queue.append(1)
item = queue.pop(0)  # O(n) operation
"""
        antipatterns = analyzer.detect_antipatterns(code)

        ap_names = [ap["antipattern"] for ap in antipatterns]
        assert "list_as_queue" in ap_names

    def test_detect_string_concatenation_loop(self):
        """Test detection of string concatenation in loop."""
        analyzer = CodeAnalyzer()
        code = """
result = ""
for char in chars:
    result += "x"
"""
        antipatterns = analyzer.detect_antipatterns(code)

        ap_names = [ap["antipattern"] for ap in antipatterns]
        assert "string_concatenation_loop" in ap_names

    def test_detect_global_state(self):
        """Test detection of global variable usage."""
        analyzer = CodeAnalyzer()
        code = """
count = 0

def increment():
    global count
    count += 1
"""
        antipatterns = analyzer.detect_antipatterns(code)

        ap_names = [ap["antipattern"] for ap in antipatterns]
        assert "global_state" in ap_names


class TestCodeAnalyzerFullAnalysisEnhanced:
    """Tests for enhanced full analysis."""

    def test_full_analysis(self):
        """Test full_analysis method."""
        analyzer = CodeAnalyzer()
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        result = analyzer.full_analysis(code)

        assert "complexity_estimate" in result
        assert "optimizations" in result
        assert "antipatterns" in result
        assert "overall_score" in result
        assert "overall_grade" in result
        assert "summary" in result

    def test_generate_summary(self):
        """Test summary generation."""
        analyzer = CodeAnalyzer()
        analysis = {
            "complexity_estimate": {
                "time_complexity": {"estimate": "O(n)"},
                "space_complexity": {"estimate": "O(1)"},
            },
            "overall_score": 85,
            "overall_grade": "B",
            "patterns": [{"pattern": "two_pointers", "pattern_ko": "투 포인터"}],
            "antipatterns": [{"severity": "warning"}],
            "optimizations": [{"priority": "high"}],
        }

        summary = analyzer._generate_summary(analysis)

        assert "복잡도" in summary
        assert "품질 점수" in summary


class TestCodeAnalyzerSimilarity:
    """Tests for code similarity interpretation."""

    def test_interpret_similarity_identical(self):
        """Test similarity interpretation for identical code."""
        analyzer = CodeAnalyzer()
        assert "동일" in analyzer._interpret_similarity(0.96)

    def test_interpret_similarity_very_similar(self):
        """Test similarity interpretation for very similar code."""
        analyzer = CodeAnalyzer()
        assert "유사" in analyzer._interpret_similarity(0.85)

    def test_interpret_similarity_similar(self):
        """Test similarity interpretation for similar code."""
        analyzer = CodeAnalyzer()
        assert "유사" in analyzer._interpret_similarity(0.65)

    def test_interpret_similarity_different(self):
        """Test similarity interpretation for different code."""
        analyzer = CodeAnalyzer()
        assert "다른" in analyzer._interpret_similarity(0.3)


# ============== DebuggingAssistant Tests ==============


class TestDebuggingAssistantInit:
    """Tests for DebuggingAssistant initialization."""

    def test_init_default(self):
        """Test default initialization."""
        assistant = DebuggingAssistant()
        assert assistant.code_analyzer is not None

    def test_init_with_analyzer(self):
        """Test initialization with custom analyzer."""
        analyzer = CodeAnalyzer()
        assistant = DebuggingAssistant(code_analyzer=analyzer)
        assert assistant.code_analyzer == analyzer

    def test_error_patterns_defined(self):
        """Test that error patterns are defined."""
        assert "IndexError" in DebuggingAssistant.ERROR_PATTERNS
        assert "KeyError" in DebuggingAssistant.ERROR_PATTERNS
        assert "RecursionError" in DebuggingAssistant.ERROR_PATTERNS
        assert "TimeoutError" in DebuggingAssistant.ERROR_PATTERNS
        assert "WrongAnswer" in DebuggingAssistant.ERROR_PATTERNS

    def test_common_mistakes_defined(self):
        """Test that common mistakes are defined."""
        assert "off_by_one" in DebuggingAssistant.COMMON_MISTAKES
        assert "integer_division" in DebuggingAssistant.COMMON_MISTAKES
        assert "mutable_default" in DebuggingAssistant.COMMON_MISTAKES
        assert "shallow_copy" in DebuggingAssistant.COMMON_MISTAKES


class TestDebuggingAssistantAnalyzeError:
    """Tests for error analysis."""

    def test_analyze_index_error(self):
        """Test analysis of IndexError."""
        assistant = DebuggingAssistant()
        code = """
arr = [1, 2, 3]
print(arr[3])
"""
        result = assistant.analyze_error(code, "IndexError", "list index out of range")

        assert result["error_type"] == "IndexError"
        assert "diagnosis" in result
        assert "debugging_guide" in result
        assert "배열 범위" in result["diagnosis"]["description"]

    def test_analyze_recursion_error(self):
        """Test analysis of RecursionError."""
        assistant = DebuggingAssistant()
        code = """
def infinite():
    return infinite()
"""
        result = assistant.analyze_error(code, "RecursionError", "maximum recursion depth exceeded")

        assert "재귀" in result["diagnosis"]["description"]

    def test_analyze_wrong_answer(self):
        """Test analysis of WrongAnswer."""
        assistant = DebuggingAssistant()
        code = """
def add(a, b):
    return a - b  # Bug: should be a + b
"""
        test_case = {
            "input": "1 2",
            "expected_output": "3",
            "actual_output": "-1",
        }
        result = assistant.analyze_error(code, "WrongAnswer", "", test_case)

        assert "오답" in result["diagnosis"]["description"]
        assert "test_case_analysis" in result

    def test_analyze_timeout_error(self):
        """Test analysis of TimeoutError."""
        assistant = DebuggingAssistant()
        code = """
def slow(n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                pass
"""
        result = assistant.analyze_error(code, "TimeoutError", "Time limit exceeded")

        assert "시간 초과" in result["diagnosis"]["description"]


class TestDebuggingAssistantAnalyzeCodeIssues:
    """Tests for code issue analysis."""

    def test_analyze_code_issues_index_error(self):
        """Test code issue analysis for IndexError."""
        assistant = DebuggingAssistant()
        code = """
arr = [1, 2, 3]
print(arr[len(arr)])
"""
        issues = assistant._analyze_code_issues(code, "IndexError", "python")

        assert len(issues) > 0

    def test_analyze_code_issues_off_by_one(self):
        """Test detection of off-by-one pattern."""
        assistant = DebuggingAssistant()
        code = """
for i in range(1, n):
    print(i)
"""
        issues = assistant._analyze_code_issues(code, "WrongAnswer", "python")

        issue_types = [i["type"] for i in issues]
        assert "off_by_one" in issue_types


class TestDebuggingAssistantTestCaseAnalysis:
    """Tests for test case analysis."""

    def test_analyze_empty_input(self):
        """Test analysis of empty input."""
        assistant = DebuggingAssistant()
        test_case = {"input": "[]", "expected_output": "0"}

        result = assistant._analyze_test_case(test_case, "WrongAnswer")

        assert "empty_input" in result["input_characteristics"]

    def test_analyze_single_element(self):
        """Test analysis of single element input."""
        assistant = DebuggingAssistant()
        test_case = {"input": "[5]", "expected_output": "5"}

        result = assistant._analyze_test_case(test_case, "WrongAnswer")

        assert "single_element" in result["input_characteristics"]

    def test_analyze_negative_numbers(self):
        """Test analysis of negative number input."""
        assistant = DebuggingAssistant()
        test_case = {"input": "[-1, -2, 3]", "expected_output": "0"}

        result = assistant._analyze_test_case(test_case, "WrongAnswer")

        assert "negative_numbers" in result["input_characteristics"]

    def test_analyze_sorted_input(self):
        """Test analysis of sorted input."""
        assistant = DebuggingAssistant()
        test_case = {"input": "[1, 2, 3, 4, 5]", "expected_output": "15"}

        result = assistant._analyze_test_case(test_case, "WrongAnswer")

        assert "sorted_input" in result["input_characteristics"]


class TestDebuggingAssistantEdgeCases:
    """Tests for edge case suggestions."""

    def test_edge_cases_array(self):
        """Test edge case suggestions for array problems."""
        assistant = DebuggingAssistant()

        edge_cases = assistant.get_edge_case_suggestions("array")

        assert len(edge_cases) > 0
        case_names = [c["case"] for c in edge_cases]
        assert "빈 입력" in case_names
        assert "단일 원소" in case_names

    def test_edge_cases_string(self):
        """Test edge case suggestions for string problems."""
        assistant = DebuggingAssistant()

        edge_cases = assistant.get_edge_case_suggestions("string")

        assert len(edge_cases) > 0
        case_names = [c["case"] for c in edge_cases]
        assert "빈 문자열" in case_names

    def test_edge_cases_graph(self):
        """Test edge case suggestions for graph problems."""
        assistant = DebuggingAssistant()

        edge_cases = assistant.get_edge_case_suggestions("graph")

        assert len(edge_cases) > 0
        case_names = [c["case"] for c in edge_cases]
        assert "사이클" in case_names

    def test_edge_cases_with_constraints(self):
        """Test edge case suggestions with constraints."""
        assistant = DebuggingAssistant()
        constraints = {"n": 10**5, "values": [-10**9, 10**9]}

        edge_cases = assistant.get_edge_case_suggestions("array", constraints)

        case_names = [c["case"] for c in edge_cases]
        assert "최대 크기" in case_names
        assert "최소값" in case_names
        assert "최대값" in case_names


class TestDebuggingAssistantRelevance:
    """Tests for relevance calculation."""

    def test_calculate_relevance_high(self):
        """Test high relevance calculation."""
        assistant = DebuggingAssistant()

        relevance = assistant._calculate_relevance("off_by_one", "IndexError")
        assert relevance == "high"

    def test_calculate_relevance_medium(self):
        """Test medium relevance calculation."""
        assistant = DebuggingAssistant()

        relevance = assistant._calculate_relevance("mutable_default", "WrongAnswer")
        assert relevance == "medium"

    def test_calculate_relevance_low(self):
        """Test low relevance calculation."""
        assistant = DebuggingAssistant()

        relevance = assistant._calculate_relevance("unknown", "UnknownError")
        assert relevance == "low"


class TestDebuggingAssistantFactory:
    """Tests for debugging assistant factory."""

    def test_get_debugging_assistant_default(self):
        """Test factory function with default analyzer."""
        assistant = get_debugging_assistant()

        assert isinstance(assistant, DebuggingAssistant)
        assert assistant.code_analyzer is not None

    def test_get_debugging_assistant_with_analyzer(self):
        """Test factory function with custom analyzer."""
        analyzer = CodeAnalyzer()
        assistant = get_debugging_assistant(analyzer)

        assert assistant.code_analyzer == analyzer


# ============== CodeQualityClassifier Tests ==============


class TestCodeQualityClassifierInit:
    """Tests for CodeQualityClassifier initialization."""

    def test_init_default(self):
        """Test default initialization."""
        classifier = CodeQualityClassifier()

        assert classifier.model_name == "microsoft/codebert-base"
        assert classifier.num_labels == 4
        assert classifier._model is None
        assert classifier._tokenizer is None

    def test_init_custom(self):
        """Test custom initialization."""
        classifier = CodeQualityClassifier(
            model_name="custom/model",
            num_labels=5,
            device="cpu",
            cache_dir="/tmp/cache",
        )

        assert classifier.model_name == "custom/model"
        assert classifier.num_labels == 5
        assert classifier._device == "cpu"
        assert classifier.cache_dir == "/tmp/cache"

    def test_quality_labels(self):
        """Test quality labels are defined."""
        assert CodeQualityClassifier.QUALITY_LABELS == ["poor", "fair", "good", "excellent"]

    def test_quality_dimensions(self):
        """Test quality dimensions are defined."""
        expected = ["correctness", "efficiency", "readability", "best_practices"]
        assert CodeQualityClassifier.QUALITY_DIMENSIONS == expected


class TestCodeQualityClassifierHelpers:
    """Tests for classifier helper methods."""

    def test_score_to_grade_a(self):
        """Test score to grade A conversion."""
        classifier = CodeQualityClassifier()
        assert classifier._score_to_grade(90) == "A"
        assert classifier._score_to_grade(95) == "A"

    def test_score_to_grade_b(self):
        """Test score to grade B conversion."""
        classifier = CodeQualityClassifier()
        assert classifier._score_to_grade(80) == "B"
        assert classifier._score_to_grade(89) == "B"

    def test_score_to_grade_c(self):
        """Test score to grade C conversion."""
        classifier = CodeQualityClassifier()
        assert classifier._score_to_grade(70) == "C"
        assert classifier._score_to_grade(79) == "C"

    def test_score_to_grade_d(self):
        """Test score to grade D conversion."""
        classifier = CodeQualityClassifier()
        assert classifier._score_to_grade(60) == "D"
        assert classifier._score_to_grade(69) == "D"

    def test_score_to_grade_f(self):
        """Test score to grade F conversion."""
        classifier = CodeQualityClassifier()
        assert classifier._score_to_grade(59) == "F"
        assert classifier._score_to_grade(0) == "F"

    def test_dimension_to_korean(self):
        """Test dimension translation to Korean."""
        classifier = CodeQualityClassifier()

        assert classifier._dimension_to_korean("correctness") == "정확성"
        assert classifier._dimension_to_korean("efficiency") == "효율성"
        assert classifier._dimension_to_korean("readability") == "가독성"
        assert classifier._dimension_to_korean("best_practices") == "베스트 프랙티스"
        assert classifier._dimension_to_korean("unknown") == "unknown"


class TestCodeQualityClassifierSuggestions:
    """Tests for improvement suggestions."""

    def test_get_suggestion_poor_correctness(self):
        """Test suggestion for poor correctness."""
        classifier = CodeQualityClassifier()

        suggestion = classifier._get_suggestion_for_dimension("correctness", "poor")

        assert suggestion is not None
        assert suggestion["dimension"] == "correctness"
        assert suggestion["priority"] == 1
        assert len(suggestion["tips"]) > 0

    def test_get_suggestion_fair_efficiency(self):
        """Test suggestion for fair efficiency."""
        classifier = CodeQualityClassifier()

        suggestion = classifier._get_suggestion_for_dimension("efficiency", "fair")

        assert suggestion is not None
        assert suggestion["priority"] == 3

    def test_get_suggestion_good_returns_none(self):
        """Test that good/excellent labels return no suggestion."""
        classifier = CodeQualityClassifier()

        suggestion = classifier._get_suggestion_for_dimension("correctness", "good")

        assert suggestion is None

    def test_get_improvement_suggestions(self):
        """Test getting improvement suggestions from classification."""
        classifier = CodeQualityClassifier()

        classification_result = {
            "dimensions": {
                "correctness": {"score": 40, "label": "poor"},
                "efficiency": {"score": 45, "label": "poor"},
                "readability": {"score": 70, "label": "good"},
                "best_practices": {"score": 80, "label": "excellent"},
            }
        }

        suggestions = classifier.get_improvement_suggestions(classification_result)

        assert len(suggestions) >= 2  # At least correctness and efficiency
        # Should be sorted by priority
        for i in range(len(suggestions) - 1):
            assert suggestions[i]["priority"] <= suggestions[i + 1]["priority"]


class TestCodeQualityClassifierUnload:
    """Tests for model unloading."""

    def test_unload_when_not_loaded(self):
        """Test unload when model is not loaded."""
        classifier = CodeQualityClassifier()

        # Should not raise error
        classifier.unload()

        assert classifier._model is None
        assert classifier._tokenizer is None


# ============== Integration Tests ==============


class TestMLAnalysisIntegration:
    """Integration tests for ML analysis components."""

    def test_analyzer_full_workflow(self):
        """Test complete analyzer workflow."""
        analyzer = CodeAnalyzer()

        code = """
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
    return max_val
"""

        # Full analysis
        result = analyzer.full_analysis(code)

        assert result["overall_score"] > 0
        assert result["overall_grade"] in ["A", "B", "C", "D", "F"]
        assert "summary" in result

        # Complexity estimation
        complexity = analyzer.estimate_complexity(code)
        assert complexity["time_complexity"]["estimate"] != "O(?)"

        # Antipattern detection
        antipatterns = analyzer.detect_antipatterns(code)
        assert isinstance(antipatterns, list)

        # Optimization suggestions
        suggestions = analyzer.suggest_optimizations(code)
        assert isinstance(suggestions, list)

    def test_debugging_assistant_full_workflow(self):
        """Test complete debugging assistant workflow."""
        assistant = DebuggingAssistant()

        code = """
def binary_search(arr, target):
    left, right = 0, len(arr)
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""

        test_case = {
            "input": "[1, 2, 3, 4, 5], 3",
            "expected_output": "2",
            "actual_output": "IndexError",
        }

        result = assistant.analyze_error(code, "IndexError", "list index out of range", test_case)

        assert "diagnosis" in result
        assert "debugging_guide" in result
        assert "code_specific_issues" in result
        assert "test_case_analysis" in result
        assert "suggested_fixes" in result
        assert "summary" in result


# ============== Run Tests ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
