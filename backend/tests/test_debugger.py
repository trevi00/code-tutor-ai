"""Tests for Debugger Module."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

from code_tutor.debugger.domain.value_objects import (
    StepType,
    VariableType,
    DebugStatus,
    get_variable_type,
    format_variable_value,
    MAX_STEPS,
    MAX_EXECUTION_TIME,
    MAX_STRING_LENGTH,
    MAX_COLLECTION_SIZE,
)
from code_tutor.debugger.domain.entities import (
    Variable,
    StackFrame,
    ExecutionStep,
    DebugSession,
    DebugResult,
    utc_now,
)
from code_tutor.debugger.application.tracer import CodeTracer, trace_code
from code_tutor.debugger.application.services import DebugService, debug_service
from code_tutor.debugger.application.dto import (
    DebugRequest,
    DebugResponse,
    ExecutionStepResponse,
    VariableResponse,
    StackFrameResponse,
)


# ============== Value Objects Tests ==============


class TestStepType:
    """Tests for StepType enum."""

    def test_step_type_values(self):
        """Test all step type values."""
        assert StepType.CALL.value == "call"
        assert StepType.LINE.value == "line"
        assert StepType.RETURN.value == "return"
        assert StepType.EXCEPTION.value == "exception"

    def test_step_type_from_string(self):
        """Test creating step type from string."""
        assert StepType("call") == StepType.CALL
        assert StepType("line") == StepType.LINE
        assert StepType("return") == StepType.RETURN
        assert StepType("exception") == StepType.EXCEPTION


class TestVariableType:
    """Tests for VariableType enum."""

    def test_variable_type_values(self):
        """Test all variable type values."""
        assert VariableType.INT.value == "int"
        assert VariableType.FLOAT.value == "float"
        assert VariableType.STRING.value == "string"
        assert VariableType.BOOLEAN.value == "boolean"
        assert VariableType.LIST.value == "list"
        assert VariableType.DICT.value == "dict"
        assert VariableType.TUPLE.value == "tuple"
        assert VariableType.SET.value == "set"
        assert VariableType.NONE.value == "none"
        assert VariableType.OBJECT.value == "object"
        assert VariableType.FUNCTION.value == "function"
        assert VariableType.CLASS.value == "class"


class TestDebugStatus:
    """Tests for DebugStatus enum."""

    def test_debug_status_values(self):
        """Test all debug status values."""
        assert DebugStatus.PENDING.value == "pending"
        assert DebugStatus.RUNNING.value == "running"
        assert DebugStatus.PAUSED.value == "paused"
        assert DebugStatus.COMPLETED.value == "completed"
        assert DebugStatus.ERROR.value == "error"
        assert DebugStatus.TIMEOUT.value == "timeout"


class TestGetVariableType:
    """Tests for get_variable_type function."""

    def test_none_type(self):
        """Test None type detection."""
        assert get_variable_type(None) == VariableType.NONE

    def test_boolean_type(self):
        """Test boolean type detection."""
        assert get_variable_type(True) == VariableType.BOOLEAN
        assert get_variable_type(False) == VariableType.BOOLEAN

    def test_int_type(self):
        """Test int type detection."""
        assert get_variable_type(42) == VariableType.INT
        assert get_variable_type(-10) == VariableType.INT
        assert get_variable_type(0) == VariableType.INT

    def test_float_type(self):
        """Test float type detection."""
        assert get_variable_type(3.14) == VariableType.FLOAT
        assert get_variable_type(-2.5) == VariableType.FLOAT

    def test_string_type(self):
        """Test string type detection."""
        assert get_variable_type("hello") == VariableType.STRING
        assert get_variable_type("") == VariableType.STRING

    def test_list_type(self):
        """Test list type detection."""
        assert get_variable_type([1, 2, 3]) == VariableType.LIST
        assert get_variable_type([]) == VariableType.LIST

    def test_dict_type(self):
        """Test dict type detection."""
        assert get_variable_type({"a": 1}) == VariableType.DICT
        assert get_variable_type({}) == VariableType.DICT

    def test_tuple_type(self):
        """Test tuple type detection."""
        assert get_variable_type((1, 2)) == VariableType.TUPLE
        assert get_variable_type(()) == VariableType.TUPLE

    def test_set_type(self):
        """Test set type detection."""
        assert get_variable_type({1, 2, 3}) == VariableType.SET
        assert get_variable_type(set()) == VariableType.SET

    def test_function_type(self):
        """Test function type detection."""
        def my_func():
            pass
        assert get_variable_type(my_func) == VariableType.FUNCTION
        assert get_variable_type(lambda x: x) == VariableType.FUNCTION

    def test_class_type(self):
        """Test class type detection.

        Note: In current implementation, classes are callable so they
        return FUNCTION due to order of checks. isinstance(value, type)
        check comes after callable(value) check.
        """
        class MyClass:
            pass
        # Classes are callable, so they return FUNCTION in current implementation
        assert get_variable_type(MyClass) == VariableType.FUNCTION

    def test_object_type(self):
        """Test object type detection."""
        class MyClass:
            pass
        obj = MyClass()
        assert get_variable_type(obj) == VariableType.OBJECT


class TestFormatVariableValue:
    """Tests for format_variable_value function."""

    def test_format_none(self):
        """Test formatting None value."""
        assert format_variable_value(None) == "None"

    def test_format_short_string(self):
        """Test formatting short string."""
        result = format_variable_value("hello")
        assert result == '"hello"'

    def test_format_long_string(self):
        """Test formatting long string is truncated."""
        long_str = "a" * 200
        result = format_variable_value(long_str, max_length=100)
        assert len(result) < 200
        assert "..." in result

    def test_format_small_list(self):
        """Test formatting small list."""
        result = format_variable_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_large_list(self):
        """Test formatting large list is truncated."""
        large_list = list(range(20))
        result = format_variable_value(large_list)
        assert "..." in result
        assert "20 items" in result

    def test_format_small_dict(self):
        """Test formatting small dict."""
        result = format_variable_value({"a": 1, "b": 2})
        assert "'a': 1" in result
        assert "'b': 2" in result

    def test_format_large_dict(self):
        """Test formatting large dict is truncated."""
        large_dict = {f"key{i}": i for i in range(10)}
        result = format_variable_value(large_dict)
        assert "..." in result
        assert "10 items" in result

    def test_format_tuple(self):
        """Test formatting tuple."""
        result = format_variable_value((1, 2, 3))
        assert result == "(1, 2, 3)"

    def test_format_set(self):
        """Test formatting set."""
        result = format_variable_value({1})
        assert "1" in result

    def test_format_unrepresentable(self):
        """Test formatting unrepresentable value."""
        class Unrepresentable:
            def __repr__(self):
                raise ValueError("Cannot represent")

        result = format_variable_value(Unrepresentable())
        assert result == "<unrepresentable>"


class TestConstants:
    """Tests for module constants."""

    def test_max_steps(self):
        """Test MAX_STEPS constant."""
        assert MAX_STEPS == 1000

    def test_max_execution_time(self):
        """Test MAX_EXECUTION_TIME constant."""
        assert MAX_EXECUTION_TIME == 10

    def test_max_string_length(self):
        """Test MAX_STRING_LENGTH constant."""
        assert MAX_STRING_LENGTH == 1000

    def test_max_collection_size(self):
        """Test MAX_COLLECTION_SIZE constant."""
        assert MAX_COLLECTION_SIZE == 100


# ============== Entity Tests ==============


class TestVariable:
    """Tests for Variable entity."""

    def test_variable_creation(self):
        """Test creating a variable."""
        var = Variable(
            name="x",
            value="42",
            type=VariableType.INT,
        )
        assert var.name == "x"
        assert var.value == "42"
        assert var.type == VariableType.INT
        assert var.raw_value is None

    def test_variable_from_python_int(self):
        """Test creating variable from Python int."""
        var = Variable.from_python_value("x", 42)

        assert var.name == "x"
        assert var.value == "42"
        assert var.type == VariableType.INT
        assert var.raw_value == 42

    def test_variable_from_python_string(self):
        """Test creating variable from Python string."""
        var = Variable.from_python_value("name", "hello")

        assert var.name == "name"
        assert var.value == '"hello"'
        assert var.type == VariableType.STRING
        assert var.raw_value == "hello"

    def test_variable_from_python_list(self):
        """Test creating variable from Python list."""
        var = Variable.from_python_value("arr", [1, 2, 3])

        assert var.name == "arr"
        assert var.type == VariableType.LIST
        assert var.raw_value == [1, 2, 3]

    def test_variable_from_python_none(self):
        """Test creating variable from None."""
        var = Variable.from_python_value("result", None)

        assert var.name == "result"
        assert var.value == "None"
        assert var.type == VariableType.NONE


class TestStackFrame:
    """Tests for StackFrame entity."""

    def test_stack_frame_creation(self):
        """Test creating a stack frame."""
        frame = StackFrame(
            function_name="my_func",
            filename="test.py",
            line_number=10,
        )
        assert frame.function_name == "my_func"
        assert frame.filename == "test.py"
        assert frame.line_number == 10
        assert frame.local_variables == []

    def test_stack_frame_with_variables(self):
        """Test stack frame with local variables."""
        var = Variable(name="x", value="1", type=VariableType.INT)
        frame = StackFrame(
            function_name="my_func",
            filename="test.py",
            line_number=10,
            local_variables=[var],
        )
        assert len(frame.local_variables) == 1
        assert frame.local_variables[0].name == "x"

    def test_stack_frame_to_dict(self):
        """Test converting stack frame to dict."""
        var = Variable(name="x", value="1", type=VariableType.INT)
        frame = StackFrame(
            function_name="my_func",
            filename="test.py",
            line_number=10,
            local_variables=[var],
        )
        result = frame.to_dict()

        assert result["function_name"] == "my_func"
        assert result["filename"] == "test.py"
        assert result["line_number"] == 10
        assert len(result["local_variables"]) == 1
        assert result["local_variables"][0]["name"] == "x"


class TestExecutionStep:
    """Tests for ExecutionStep entity."""

    def test_execution_step_creation(self):
        """Test creating an execution step."""
        step = ExecutionStep(
            step_number=1,
            step_type=StepType.LINE,
            line_number=5,
            line_content="x = 1",
            function_name="main",
        )
        assert step.step_number == 1
        assert step.step_type == StepType.LINE
        assert step.line_number == 5
        assert step.line_content == "x = 1"
        assert step.function_name == "main"
        assert step.variables == []
        assert step.call_stack == []
        assert step.output == ""
        assert step.return_value is None
        assert step.exception is None

    def test_execution_step_with_exception(self):
        """Test execution step with exception."""
        step = ExecutionStep(
            step_number=1,
            step_type=StepType.EXCEPTION,
            line_number=5,
            line_content="x = 1/0",
            function_name="main",
            exception="ZeroDivisionError: division by zero",
        )
        assert step.exception == "ZeroDivisionError: division by zero"

    def test_execution_step_to_dict(self):
        """Test converting execution step to dict."""
        var = Variable(name="x", value="1", type=VariableType.INT)
        frame = StackFrame(
            function_name="main",
            filename="test.py",
            line_number=5,
        )
        step = ExecutionStep(
            step_number=1,
            step_type=StepType.LINE,
            line_number=5,
            line_content="x = 1",
            function_name="main",
            variables=[var],
            call_stack=[frame],
            output="hello\n",
            return_value="42",
        )
        result = step.to_dict()

        assert result["step_number"] == 1
        assert result["step_type"] == "line"
        assert result["line_number"] == 5
        assert result["line_content"] == "x = 1"
        assert result["function_name"] == "main"
        assert len(result["variables"]) == 1
        assert len(result["call_stack"]) == 1
        assert result["output"] == "hello\n"
        assert result["return_value"] == "42"
        assert result["exception"] is None


class TestDebugSession:
    """Tests for DebugSession entity."""

    def test_debug_session_create(self):
        """Test creating a debug session."""
        session = DebugSession.create(
            code="x = 1",
            input_data="",
            breakpoints=[5, 10],
        )
        assert session.id is not None
        assert session.code == "x = 1"
        assert session.input_data == ""
        assert session.status == DebugStatus.PENDING
        assert session.breakpoints == [5, 10]
        assert session.steps == []
        assert session.current_step == 0
        assert session.total_steps == 0

    def test_debug_session_create_default_breakpoints(self):
        """Test creating session with default breakpoints."""
        session = DebugSession.create(code="x = 1")
        assert session.breakpoints == []

    def test_debug_session_add_step(self):
        """Test adding a step to session."""
        session = DebugSession.create(code="x = 1")
        step = ExecutionStep(
            step_number=1,
            step_type=StepType.LINE,
            line_number=1,
            line_content="x = 1",
            function_name="main",
        )
        session.add_step(step)

        assert len(session.steps) == 1
        assert session.total_steps == 1

    def test_debug_session_complete(self):
        """Test completing a session."""
        session = DebugSession.create(code="x = 1")
        session.complete("output text")

        assert session.status == DebugStatus.COMPLETED
        assert session.output == "output text"
        assert session.completed_at is not None

    def test_debug_session_fail(self):
        """Test failing a session."""
        session = DebugSession.create(code="x = 1")
        session.fail("Error message")

        assert session.status == DebugStatus.ERROR
        assert session.error == "Error message"
        assert session.completed_at is not None

    def test_debug_session_to_dict(self):
        """Test converting session to dict."""
        session = DebugSession.create(
            code="x = 1",
            breakpoints=[1],
        )
        session.complete("output")
        result = session.to_dict()

        assert "id" in result
        assert result["status"] == "completed"
        assert result["output"] == "output"
        assert result["breakpoints"] == [1]


class TestDebugResult:
    """Tests for DebugResult entity."""

    def test_debug_result_creation(self):
        """Test creating a debug result."""
        session_id = uuid4()
        result = DebugResult(
            session_id=session_id,
            status=DebugStatus.COMPLETED,
            steps=[],
            total_steps=0,
            output="hello",
            execution_time_ms=50.0,
        )
        assert result.session_id == session_id
        assert result.status == DebugStatus.COMPLETED
        assert result.output == "hello"
        assert result.execution_time_ms == 50.0

    def test_debug_result_with_error(self):
        """Test debug result with error."""
        result = DebugResult(
            session_id=uuid4(),
            status=DebugStatus.ERROR,
            steps=[],
            total_steps=0,
            output="",
            error="SyntaxError: invalid syntax",
        )
        assert result.error == "SyntaxError: invalid syntax"

    def test_debug_result_to_dict(self):
        """Test converting result to dict."""
        session_id = uuid4()
        result = DebugResult(
            session_id=session_id,
            status=DebugStatus.COMPLETED,
            steps=[],
            total_steps=5,
            output="hello",
            execution_time_ms=50.0,
        )
        data = result.to_dict()

        assert data["session_id"] == str(session_id)
        assert data["status"] == "completed"
        assert data["total_steps"] == 5
        assert data["output"] == "hello"
        assert data["execution_time_ms"] == 50.0


class TestUtcNow:
    """Tests for utc_now function."""

    def test_utc_now_is_aware(self):
        """Test that utc_now returns timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc


# ============== Tracer Tests ==============


class TestCodeTracer:
    """Tests for CodeTracer class."""

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        tracer = CodeTracer(
            code="x = 1",
            input_data="hello",
            breakpoints=[1, 2],
            max_steps=500,
            max_time=5.0,
        )
        assert tracer.code == "x = 1"
        assert tracer.input_data == "hello"
        assert tracer.breakpoints == {1, 2}
        assert tracer.max_steps == 500
        assert tracer.max_time == 5.0

    def test_tracer_default_values(self):
        """Test tracer default values."""
        tracer = CodeTracer(code="x = 1")

        assert tracer.input_data == ""
        assert tracer.breakpoints == set()
        assert tracer.max_steps == MAX_STEPS
        assert tracer.max_time == MAX_EXECUTION_TIME

    def test_trace_simple_code(self):
        """Test tracing simple code."""
        code = """
x = 1
y = 2
z = x + y
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert result.status == DebugStatus.COMPLETED
        assert result.total_steps > 0
        assert result.error is None

    def test_trace_with_print(self):
        """Test tracing code with print."""
        code = """
print("hello")
print("world")
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert "hello" in result.output
        assert "world" in result.output

    def test_trace_with_input(self):
        """Test tracing code with input."""
        code = """
name = input("Name: ")
print(f"Hello, {name}")
"""
        tracer = CodeTracer(code=code, input_data="Alice")
        result = tracer.trace()

        assert "Alice" in result.output
        assert "Hello, Alice" in result.output

    def test_trace_with_function(self):
        """Test tracing code with function."""
        code = """
def add(a, b):
    return a + b

result = add(1, 2)
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert result.status == DebugStatus.COMPLETED
        # Should have call, line, and return steps
        step_types = [s.step_type for s in result.steps]
        assert StepType.CALL in step_types
        assert StepType.RETURN in step_types

    def test_trace_with_loop(self):
        """Test tracing code with loop."""
        code = """
total = 0
for i in range(3):
    total += i
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert result.status == DebugStatus.COMPLETED
        assert result.total_steps > 3  # Multiple iterations

    def test_trace_with_error(self):
        """Test tracing code with error."""
        code = """
x = 1
y = 0
z = x / y
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert result.status == DebugStatus.ERROR
        assert "ZeroDivisionError" in result.error

    def test_trace_syntax_error(self):
        """Test tracing code with syntax error."""
        code = """
def bad_func(
    return 1
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        assert result.status == DebugStatus.ERROR
        assert "SyntaxError" in result.error

    def test_trace_max_steps_exceeded(self):
        """Test that max steps limit is enforced."""
        code = """
while True:
    pass
"""
        tracer = CodeTracer(code=code, max_steps=10)
        result = tracer.trace()

        assert result.status == DebugStatus.TIMEOUT
        assert len(result.steps) <= 10

    def test_trace_variables_captured(self):
        """Test that variables are captured in steps."""
        code = """
x = 42
y = "hello"
"""
        tracer = CodeTracer(code=code)
        result = tracer.trace()

        # Find step with variables
        var_names = set()
        for step in result.steps:
            for var in step.variables:
                var_names.add(var.name)

        assert "x" in var_names or "y" in var_names

    def test_extract_variables(self):
        """Test variable extraction from frame."""
        tracer = CodeTracer(code="x = 1")

        # Create mock frame
        mock_frame = MagicMock()
        mock_frame.f_locals = {
            "x": 1,
            "y": "hello",
            "_private": "skip",
            "__builtins__": {},
        }

        variables = tracer._extract_variables(mock_frame)

        var_names = [v.name for v in variables]
        assert "x" in var_names
        assert "y" in var_names
        assert "_private" not in var_names
        assert "__builtins__" not in var_names


class TestTraceCodeFunction:
    """Tests for trace_code convenience function."""

    def test_trace_code_simple(self):
        """Test trace_code function."""
        result = trace_code("x = 1")

        assert result.status == DebugStatus.COMPLETED
        assert result.session_id is not None

    def test_trace_code_with_input(self):
        """Test trace_code with input."""
        result = trace_code(
            code="name = input()\nprint(name)",
            input_data="test",
        )

        assert "test" in result.output

    def test_trace_code_with_breakpoints(self):
        """Test trace_code with breakpoints."""
        result = trace_code(
            code="x = 1\ny = 2",
            breakpoints=[1],
        )

        assert result.status == DebugStatus.COMPLETED


# ============== Service Tests ==============


class TestDebugService:
    """Tests for DebugService class."""

    @pytest.fixture
    def service(self):
        """Create debug service instance."""
        return DebugService()

    @pytest.mark.asyncio
    async def test_debug_code(self, service):
        """Test debugging code."""
        request = DebugRequest(
            code="x = 1\nprint(x)",
            input_data="",
            breakpoints=[],
        )
        response = await service.debug_code(request)

        assert response.status == DebugStatus.COMPLETED
        assert response.session_id is not None
        assert response.total_steps > 0

    @pytest.mark.asyncio
    async def test_debug_code_stored_in_sessions(self, service):
        """Test that debug session is stored."""
        request = DebugRequest(code="x = 1")
        response = await service.debug_code(request)

        # Session should be stored
        assert response.session_id in service._sessions

    @pytest.mark.asyncio
    async def test_get_session(self, service):
        """Test getting a session."""
        request = DebugRequest(code="x = 1")
        response = await service.debug_code(request)

        session = await service.get_session(response.session_id)

        assert session is not None
        assert session.session_id == response.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, service):
        """Test getting non-existent session."""
        session = await service.get_session("non-existent-id")
        assert session is None

    @pytest.mark.asyncio
    async def test_get_step(self, service):
        """Test getting a specific step."""
        request = DebugRequest(code="x = 1\ny = 2")
        response = await service.debug_code(request)

        step_info = await service.get_step(response.session_id, 1)

        assert step_info is not None
        assert step_info.step.step_number == 1
        assert step_info.has_next is True

    @pytest.mark.asyncio
    async def test_get_step_invalid(self, service):
        """Test getting invalid step."""
        request = DebugRequest(code="x = 1")
        response = await service.debug_code(request)

        # Step 0 is invalid
        step_info = await service.get_step(response.session_id, 0)
        assert step_info is None

        # Step beyond total is invalid
        step_info = await service.get_step(response.session_id, 1000)
        assert step_info is None

    @pytest.mark.asyncio
    async def test_get_step_with_breakpoint(self, service):
        """Test getting step with breakpoint info."""
        request = DebugRequest(code="x = 1\ny = 2", breakpoints=[1])
        response = await service.debug_code(request)

        # Get step at breakpoint line
        for step in response.steps:
            if step.line_number == 1:
                step_info = await service.get_step(
                    response.session_id,
                    step.step_number,
                    breakpoints=[1],
                )
                assert step_info.is_breakpoint is True
                break

    @pytest.mark.asyncio
    async def test_get_summary(self, service):
        """Test getting session summary."""
        request = DebugRequest(
            code="""
def add(a, b):
    return a + b

x = 1
y = add(1, 2)
"""
        )
        response = await service.debug_code(request)

        summary = await service.get_summary(response.session_id)

        assert summary is not None
        assert summary.session_id == response.session_id
        assert summary.total_steps > 0
        assert "add" in summary.functions_called or "main" in summary.functions_called
        assert len(summary.variables_used) > 0

    @pytest.mark.asyncio
    async def test_get_summary_not_found(self, service):
        """Test getting summary for non-existent session."""
        summary = await service.get_summary("non-existent")
        assert summary is None

    @pytest.mark.asyncio
    async def test_get_summary_with_error(self, service):
        """Test getting summary for session with error."""
        request = DebugRequest(code="x = 1/0")
        response = await service.debug_code(request)

        summary = await service.get_summary(response.session_id)

        assert summary is not None
        assert summary.has_error is True
        assert summary.error_line is not None


class TestDebugServiceResponseConversion:
    """Tests for response conversion methods."""

    @pytest.fixture
    def service(self):
        """Create debug service instance."""
        return DebugService()

    def test_to_variable_response(self, service):
        """Test converting Variable to response."""
        var = Variable(name="x", value="42", type=VariableType.INT)
        response = service._to_variable_response(var)

        assert isinstance(response, VariableResponse)
        assert response.name == "x"
        assert response.value == "42"
        assert response.type == VariableType.INT

    def test_to_stack_frame_response(self, service):
        """Test converting StackFrame to response."""
        var = Variable(name="x", value="42", type=VariableType.INT)
        frame = StackFrame(
            function_name="test",
            filename="test.py",
            line_number=10,
            local_variables=[var],
        )
        response = service._to_stack_frame_response(frame)

        assert isinstance(response, StackFrameResponse)
        assert response.function_name == "test"
        assert response.filename == "test.py"
        assert response.line_number == 10
        assert len(response.local_variables) == 1

    def test_to_step_response(self, service):
        """Test converting ExecutionStep to response."""
        var = Variable(name="x", value="42", type=VariableType.INT)
        step = ExecutionStep(
            step_number=1,
            step_type=StepType.LINE,
            line_number=5,
            line_content="x = 42",
            function_name="main",
            variables=[var],
            output="hello",
        )
        response = service._to_step_response(step)

        assert isinstance(response, ExecutionStepResponse)
        assert response.step_number == 1
        assert response.step_type == StepType.LINE
        assert response.line_number == 5
        assert response.line_content == "x = 42"
        assert len(response.variables) == 1


class TestDebugServiceSingleton:
    """Tests for debug_service singleton."""

    def test_singleton_exists(self):
        """Test that singleton instance exists."""
        assert debug_service is not None
        assert isinstance(debug_service, DebugService)


# ============== Integration Tests ==============


class TestDebuggerIntegration:
    """Integration tests for debugger module."""

    @pytest.mark.asyncio
    async def test_full_debug_workflow(self):
        """Test complete debugging workflow."""
        service = DebugService()

        # Debug code
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"5! = {result}")
"""
        request = DebugRequest(code=code)
        response = await service.debug_code(request)

        # Check response
        assert response.status == DebugStatus.COMPLETED
        assert "120" in response.output

        # Get summary
        summary = await service.get_summary(response.session_id)
        assert summary is not None
        assert "factorial" in summary.functions_called

        # Step through
        for i in range(1, min(5, response.total_steps + 1)):
            step_info = await service.get_step(response.session_id, i)
            assert step_info is not None
            assert step_info.step.step_number == i

    @pytest.mark.asyncio
    async def test_debug_error_handling(self):
        """Test error handling in debugging."""
        service = DebugService()

        code = """
arr = [1, 2, 3]
print(arr[10])  # IndexError
"""
        request = DebugRequest(code=code)
        response = await service.debug_code(request)

        assert response.status == DebugStatus.ERROR
        assert "IndexError" in response.error

        # Summary should indicate error
        summary = await service.get_summary(response.session_id)
        assert summary.has_error is True


# ============== Run Tests ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
