"""Tests for Problems API"""

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


@pytest.fixture
async def admin_headers(client: AsyncClient) -> dict:
    """Create admin user and return auth headers"""
    admin_data = {
        "email": "admin@example.com",
        "username": "admin",
        "password": "AdminPass123",
    }

    # Register
    await client.post("/api/v1/auth/register", json=admin_data)

    # Login
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": admin_data["email"],
            "password": admin_data["password"],
        },
    )
    tokens = response.json()["tokens"]

    # Note: In a real scenario, we'd need to set admin role in DB
    # For now, we'll test with regular user restrictions
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.mark.asyncio
async def test_list_problems_empty(client: AsyncClient, auth_headers: dict):
    """Test listing problems when none exist"""
    response = await client.get("/api/v1/problems", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["items"] == []
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_list_problems_unauthorized(client: AsyncClient):
    """Test listing problems without auth - should still work for public problems"""
    response = await client.get("/api/v1/problems")
    # Problems listing should be accessible without auth
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_get_problem_not_found(client: AsyncClient, auth_headers: dict):
    """Test getting a non-existent problem"""
    import uuid
    fake_id = str(uuid.uuid4())
    response = await client.get(f"/api/v1/problems/{fake_id}", headers=auth_headers)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_problem_forbidden(client: AsyncClient, auth_headers: dict, sample_problem_data: dict):
    """Test creating problem without admin role"""
    response = await client.post(
        "/api/v1/problems",
        json=sample_problem_data,
        headers=auth_headers,
    )
    # Regular user should not be able to create problems
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_problems_with_filters(client: AsyncClient, auth_headers: dict):
    """Test listing problems with category and difficulty filters"""
    # Test with filters (should return empty since no problems)
    response = await client.get(
        "/api/v1/problems?category=array&difficulty=easy",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data


@pytest.mark.asyncio
async def test_list_problems_pagination(client: AsyncClient, auth_headers: dict):
    """Test listing problems with pagination"""
    response = await client.get(
        "/api/v1/problems?page=1&size=10",
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
