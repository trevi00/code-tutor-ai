"""Authentication API tests"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_user(client: AsyncClient, sample_user_data: dict) -> None:
    """Test user registration"""
    response = await client.post("/api/v1/auth/register", json=sample_user_data)

    assert response.status_code == 201
    data = response.json()
    assert data["email"] == sample_user_data["email"]
    assert data["username"] == sample_user_data["username"]
    assert "id" in data


@pytest.mark.asyncio
async def test_register_duplicate_email(client: AsyncClient, sample_user_data: dict) -> None:
    """Test registration with duplicate email fails"""
    # First registration
    await client.post("/api/v1/auth/register", json=sample_user_data)

    # Second registration with same email
    response = await client.post("/api/v1/auth/register", json=sample_user_data)

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login(client: AsyncClient, sample_user_data: dict) -> None:
    """Test user login"""
    # Register first
    await client.post("/api/v1/auth/register", json=sample_user_data)

    # Login
    login_data = {
        "email": sample_user_data["email"],
        "password": sample_user_data["password"],
    }
    response = await client.post("/api/v1/auth/login", json=login_data)

    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "access_token" in data["tokens"]
    assert "refresh_token" in data["tokens"]


@pytest.mark.asyncio
async def test_login_wrong_password(client: AsyncClient, sample_user_data: dict) -> None:
    """Test login with wrong password fails"""
    # Register first
    await client.post("/api/v1/auth/register", json=sample_user_data)

    # Login with wrong password
    login_data = {
        "email": sample_user_data["email"],
        "password": "WrongPassword123",
    }
    response = await client.post("/api/v1/auth/login", json=login_data)

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_me(client: AsyncClient, sample_user_data: dict) -> None:
    """Test getting current user profile"""
    # Register and login
    await client.post("/api/v1/auth/register", json=sample_user_data)

    login_data = {
        "email": sample_user_data["email"],
        "password": sample_user_data["password"],
    }
    login_response = await client.post("/api/v1/auth/login", json=login_data)
    token = login_response.json()["tokens"]["access_token"]

    # Get profile
    response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == sample_user_data["email"]


@pytest.mark.asyncio
async def test_get_me_unauthorized(client: AsyncClient) -> None:
    """Test getting profile without auth fails"""
    response = await client.get("/api/v1/auth/me")

    assert response.status_code == 401  # No auth header
