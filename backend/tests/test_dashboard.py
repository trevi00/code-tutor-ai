"""Tests for Dashboard API"""

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
async def test_get_dashboard_unauthorized(client: AsyncClient):
    """Test dashboard access without authentication"""
    response = await client.get("/api/v1/dashboard")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_dashboard_empty(client: AsyncClient, auth_headers: dict):
    """Test getting dashboard for user with no activity"""
    response = await client.get("/api/v1/dashboard", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "data" in data

    dashboard = data["data"]
    assert "stats" in dashboard
    assert "category_progress" in dashboard
    assert "recent_submissions" in dashboard

    # Check stats structure
    stats = dashboard["stats"]
    assert stats["total_problems_attempted"] == 0
    assert stats["total_problems_solved"] == 0
    assert stats["total_submissions"] == 0
    assert stats["overall_success_rate"] == 0
    assert stats["easy_solved"] == 0
    assert stats["medium_solved"] == 0
    assert stats["hard_solved"] == 0

    # Check streak info
    assert "streak" in stats
    assert stats["streak"]["current_streak"] == 0
    assert stats["streak"]["longest_streak"] == 0

    # Check empty lists
    assert dashboard["category_progress"] == [] or isinstance(dashboard["category_progress"], list)
    assert dashboard["recent_submissions"] == []


@pytest.mark.asyncio
async def test_dashboard_response_structure(client: AsyncClient, auth_headers: dict):
    """Test dashboard response follows API standard format"""
    response = await client.get("/api/v1/dashboard", headers=auth_headers)
    assert response.status_code == 200

    data = response.json()

    # Check standard API response structure
    assert "success" in data
    assert "data" in data
    assert "meta" in data

    # Check meta structure
    meta = data["meta"]
    assert "request_id" in meta
    assert "timestamp" in meta
