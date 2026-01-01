"""Static complexity analyzer using AST."""

import ast
from typing import Optional

from ..domain import (
    ComplexityAnalysis,
    ComplexityClass,
    FunctionInfo,
    IssueSeverity,
    LoopInfo,
    PerformanceIssue,
    PerformanceIssueType,
)


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analyzes code complexity using AST."""

    def __init__(self, code: str):
        self.code = code
        self.lines = code.split("\n")
        self.loops: list[LoopInfo] = []
        self.functions: list[FunctionInfo] = []
        self.issues: list[PerformanceIssue] = []

        # Tracking state
        self._current_nesting = 0
        self._max_nesting = 0
        self._current_function: Optional[str] = None
        self._function_calls: dict[str, set[str]] = {}  # caller -> callees
        self._defined_functions: set[str] = set()

    def analyze(self) -> ComplexityAnalysis:
        """Perform complexity analysis."""
        try:
            tree = ast.parse(self.code)
            self.visit(tree)
        except SyntaxError:
            return ComplexityAnalysis(
                time_complexity=ComplexityClass.UNKNOWN,
                space_complexity=ComplexityClass.UNKNOWN,
                time_explanation="구문 오류로 분석 불가",
                space_explanation="구문 오류로 분석 불가",
            )

        # Calculate complexity
        time_complexity = self._calculate_time_complexity()
        space_complexity = self._calculate_space_complexity()

        # Find recursive functions
        recursive_functions = self._find_recursive_functions()

        # Update function info with complexity and recursion
        for func in self.functions:
            if func.name in recursive_functions:
                func.is_recursive = True

        return ComplexityAnalysis(
            time_complexity=time_complexity,
            space_complexity=space_complexity,
            time_explanation=self._explain_time_complexity(time_complexity),
            space_explanation=self._explain_space_complexity(space_complexity),
            loops=self.loops,
            functions=self.functions,
            max_nesting_depth=self._max_nesting,
            recursive_functions=recursive_functions,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        self._defined_functions.add(node.name)
        prev_function = self._current_function
        self._current_function = node.name

        params = [arg.arg for arg in node.args.args]
        func_info = FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            parameters=params,
        )
        self.functions.append(func_info)

        if node.name not in self._function_calls:
            self._function_calls[node.name] = set()

        self.generic_visit(node)
        self._current_function = prev_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)  # type: ignore

    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call."""
        if self._current_function:
            if isinstance(node.func, ast.Name):
                callee = node.func.id
                self._function_calls[self._current_function].add(callee)

                # Count calls for each function
                for func in self.functions:
                    if func.name == callee:
                        func.calls_count += 1
                        break

        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Visit for loop."""
        self._current_nesting += 1
        self._max_nesting = max(self._max_nesting, self._current_nesting)

        # Extract iteration info
        iter_var = None
        if isinstance(node.target, ast.Name):
            iter_var = node.target.id

        iterable = None
        estimated = None
        if isinstance(node.iter, ast.Call):
            if isinstance(node.iter.func, ast.Name):
                if node.iter.func.id == "range":
                    iterable = "range"
                    if node.iter.args:
                        estimated = ast.unparse(node.iter.args[-1])

        loop_info = LoopInfo(
            line_number=node.lineno,
            loop_type="for",
            nesting_level=self._current_nesting,
            iteration_variable=iter_var,
            iterable=iterable,
            estimated_iterations=estimated,
        )
        self.loops.append(loop_info)

        # Check for nested loop issues
        if self._current_nesting >= 2:
            self._add_nested_loop_issue(node.lineno, self._current_nesting)

        self.generic_visit(node)
        self._current_nesting -= 1

    def visit_While(self, node: ast.While) -> None:
        """Visit while loop."""
        self._current_nesting += 1
        self._max_nesting = max(self._max_nesting, self._current_nesting)

        loop_info = LoopInfo(
            line_number=node.lineno,
            loop_type="while",
            nesting_level=self._current_nesting,
        )
        self.loops.append(loop_info)

        if self._current_nesting >= 2:
            self._add_nested_loop_issue(node.lineno, self._current_nesting)

        self.generic_visit(node)
        self._current_nesting -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """Visit list comprehension."""
        # Count generators as loops
        for gen in node.generators:
            self._current_nesting += 1
            self._max_nesting = max(self._max_nesting, self._current_nesting)

        if len(node.generators) >= 2:
            self._add_nested_loop_issue(node.lineno, len(node.generators))

        self.generic_visit(node)
        self._current_nesting -= len(node.generators)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Visit binary operation."""
        # Check for string concatenation in loop
        if isinstance(node.op, ast.Add):
            if isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str):
                if self._current_nesting > 0:
                    self.issues.append(
                        PerformanceIssue(
                            issue_type=PerformanceIssueType.STRING_CONCATENATION,
                            severity=IssueSeverity.WARNING,
                            line_number=node.lineno,
                            message="루프 내 문자열 연결은 비효율적입니다",
                            suggestion="리스트에 추가 후 ''.join()을 사용하세요",
                            code_snippet=self._get_line(node.lineno),
                        )
                    )

        self.generic_visit(node)

    def _add_nested_loop_issue(self, line_number: int, nesting_level: int) -> None:
        """Add a nested loop performance issue."""
        severity = IssueSeverity.WARNING if nesting_level == 2 else IssueSeverity.ERROR

        complexity_str = "O(n²)" if nesting_level == 2 else f"O(n^{nesting_level})"

        self.issues.append(
            PerformanceIssue(
                issue_type=PerformanceIssueType.NESTED_LOOP,
                severity=severity,
                line_number=line_number,
                message=f"{nesting_level}중 중첩 루프 감지 - 시간 복잡도 {complexity_str}",
                suggestion="알고리즘 최적화 또는 데이터 구조 변경을 고려하세요",
                code_snippet=self._get_line(line_number),
            )
        )

    def _find_recursive_functions(self) -> list[str]:
        """Find functions that call themselves (directly or indirectly)."""
        recursive = []

        for func_name in self._defined_functions:
            if self._is_recursive(func_name, set()):
                recursive.append(func_name)

        return recursive

    def _is_recursive(self, func_name: str, visited: set[str]) -> bool:
        """Check if a function is recursive."""
        if func_name in visited:
            return True

        visited.add(func_name)
        callees = self._function_calls.get(func_name, set())

        for callee in callees:
            if callee == func_name:
                return True
            if callee in self._defined_functions:
                if self._is_recursive(callee, visited.copy()):
                    return True

        return False

    def _calculate_time_complexity(self) -> ComplexityClass:
        """Calculate overall time complexity."""
        # Check for recursion first
        recursive_funcs = self._find_recursive_functions()
        if recursive_funcs:
            # Simple heuristic: recursive with no loop = O(2^n) or O(n)
            if self._max_nesting == 0:
                return ComplexityClass.EXPONENTIAL

        # Based on max nesting depth
        if self._max_nesting == 0:
            return ComplexityClass.LINEAR if self.loops else ComplexityClass.CONSTANT
        elif self._max_nesting == 1:
            return ComplexityClass.LINEAR
        elif self._max_nesting == 2:
            return ComplexityClass.QUADRATIC
        elif self._max_nesting == 3:
            return ComplexityClass.CUBIC
        else:
            return ComplexityClass.EXPONENTIAL

    def _calculate_space_complexity(self) -> ComplexityClass:
        """Calculate space complexity (simplified)."""
        # Check for recursion
        recursive_funcs = self._find_recursive_functions()
        if recursive_funcs:
            return ComplexityClass.LINEAR  # Call stack grows

        # Check for data structure growth in loops
        # This is a simplified heuristic
        if self._max_nesting >= 2:
            return ComplexityClass.QUADRATIC
        elif self._max_nesting == 1:
            return ComplexityClass.LINEAR

        return ComplexityClass.CONSTANT

    def _explain_time_complexity(self, complexity: ComplexityClass) -> str:
        """Explain the time complexity result."""
        explanations = {
            ComplexityClass.CONSTANT: "상수 시간 - 입력 크기에 관계없이 일정한 시간",
            ComplexityClass.LINEAR: "선형 시간 - 단일 루프 또는 순회",
            ComplexityClass.QUADRATIC: f"이차 시간 - {self._max_nesting}중 중첩 루프",
            ComplexityClass.CUBIC: f"삼차 시간 - {self._max_nesting}중 중첩 루프",
            ComplexityClass.EXPONENTIAL: "지수 시간 - 재귀 호출 또는 깊은 중첩",
        }

        base = explanations.get(complexity, "복잡도 분석 결과")

        if self.loops:
            base += f" (루프 {len(self.loops)}개 감지)"

        return base

    def _explain_space_complexity(self, complexity: ComplexityClass) -> str:
        """Explain the space complexity result."""
        recursive_funcs = self._find_recursive_functions()

        if recursive_funcs:
            return f"재귀 함수 ({', '.join(recursive_funcs)})로 인한 콜 스택 증가"

        if complexity == ComplexityClass.CONSTANT:
            return "상수 공간 - 입력 크기에 관계없이 고정된 메모리 사용"
        elif complexity == ComplexityClass.LINEAR:
            return "선형 공간 - 입력에 비례하는 메모리 사용"
        elif complexity == ComplexityClass.QUADRATIC:
            return "이차 공간 - 중첩 구조로 인한 메모리 증가"

        return "공간 복잡도 분석 결과"

    def _get_line(self, line_number: int) -> Optional[str]:
        """Get a specific line from the code."""
        if 1 <= line_number <= len(self.lines):
            return self.lines[line_number - 1].strip()
        return None


def analyze_complexity(code: str) -> tuple[ComplexityAnalysis, list[PerformanceIssue]]:
    """Analyze code complexity and return results with issues."""
    analyzer = ComplexityAnalyzer(code)
    result = analyzer.analyze()
    return result, analyzer.issues
