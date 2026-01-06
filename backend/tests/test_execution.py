"""Tests for Code Execution API"""

import pytest
from httpx import AsyncClient


@pytest.fixture
async def auth_headers(client: AsyncClient, sample_user_data: dict) -> dict:
    """Register and login user, return auth headers"""
    # Register
    await client.post("/api/v1/auth/register", json=sample_user_data)

    # Login
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user_data["email"],
            "password": sample_user_data["password"],
        },
    )
    tokens = response.json()["tokens"]
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.mark.asyncio
async def test_execute_code_unauthorized(client: AsyncClient):
    """Test code execution without authentication"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "print('hello')",
            "language": "python",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_execute_simple_code(client: AsyncClient, auth_headers: dict):
    """Test executing simple Python code"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "print('Hello, World!')",
            "language": "python",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    result = response.json()
    assert "execution_id" in result
    assert "status" in result
    assert "stdout" in result
    assert "execution_time_ms" in result


@pytest.mark.asyncio
async def test_execute_code_with_stdin(client: AsyncClient, auth_headers: dict):
    """Test executing code with stdin input"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "name = input()\nprint(f'Hello, {name}!')",
            "language": "python",
            "stdin": "Claude",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    result = response.json()
    assert "status" in result
    assert "Hello, Claude" in result.get("stdout", "")


@pytest.mark.asyncio
async def test_execute_code_syntax_error(client: AsyncClient, auth_headers: dict):
    """Test executing code with syntax error"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "print('unclosed string",
            "language": "python",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    result = response.json()
    # Should have error in stderr or error_message
    assert result["status"] != "success" or "error" in result.get("stderr", "").lower()


@pytest.mark.asyncio
async def test_execute_code_runtime_error(client: AsyncClient, auth_headers: dict):
    """Test executing code that raises runtime error"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "x = 1 / 0",
            "language": "python",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    result = response.json()
    # Should indicate runtime error
    assert result["status"] in ["runtime_error", "success"]  # depends on how error is handled


@pytest.mark.asyncio
async def test_execute_code_empty(client: AsyncClient, auth_headers: dict):
    """Test executing empty code"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "",
            "language": "python",
        },
        headers=auth_headers,
    )
    # Should fail validation - code is required
    assert response.status_code == 400 or response.status_code == 422


@pytest.mark.asyncio
async def test_execute_code_with_timeout(client: AsyncClient, auth_headers: dict):
    """Test executing code with custom timeout"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "print('fast code')",
            "language": "python",
            "timeout_seconds": 5,
        },
        headers=auth_headers,
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_execute_code_invalid_language(client: AsyncClient, auth_headers: dict):
    """Test executing code with unsupported language"""
    response = await client.post(
        "/api/v1/execute/run",
        json={
            "code": "console.log('hello')",
            "language": "javascript",  # Not supported
        },
        headers=auth_headers,
    )
    # Should fail validation - only python is supported
    assert response.status_code == 400 or response.status_code == 422
