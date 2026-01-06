"""Tests for Roadmap API Endpoints."""

import pytest
from httpx import AsyncClient
from uuid import uuid4


@pytest.fixture
async def auth_headers(client: AsyncClient, sample_user_data: dict) -> dict:
    """Register and login user, return auth headers."""
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


class TestRoadmapPathsAPI:
    """Tests for learning paths API endpoints."""

    @pytest.mark.asyncio
    async def test_list_paths_unauthenticated(self, client: AsyncClient):
        """Test listing paths without authentication."""
        response = await client.get("/api/v1/roadmap/paths")
        assert response.status_code == 200

        data = response.json()
        assert "items" in data
        assert "total" in data
        assert isinstance(data["items"], list)

    @pytest.mark.asyncio
    async def test_list_paths_authenticated(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test listing paths with authentication includes progress."""
        response = await client.get(
            "/api/v1/roadmap/paths",
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "items" in data

    @pytest.mark.asyncio
    async def test_get_path_not_found(self, client: AsyncClient):
        """Test getting non-existent path returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/roadmap/paths/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_path_by_level_not_found(self, client: AsyncClient):
        """Test getting path by invalid level."""
        response = await client.get("/api/v1/roadmap/paths/level/invalid")
        assert response.status_code == 400  # Invalid level error


class TestRoadmapModulesAPI:
    """Tests for modules API endpoints."""

    @pytest.mark.asyncio
    async def test_get_module_not_found(self, client: AsyncClient):
        """Test getting non-existent module returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/roadmap/modules/{fake_id}")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_module_lessons_empty(self, client: AsyncClient):
        """Test getting lessons for non-existent module."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/roadmap/modules/{fake_id}/lessons")
        # Returns empty list for non-existent module
        assert response.status_code == 200
        assert response.json() == []


class TestRoadmapLessonsAPI:
    """Tests for lessons API endpoints."""

    @pytest.mark.asyncio
    async def test_get_lesson_not_found(self, client: AsyncClient):
        """Test getting non-existent lesson returns 404."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/roadmap/lessons/{fake_id}")
        assert response.status_code == 404


class TestRoadmapProgressAPI:
    """Tests for progress API endpoints."""

    @pytest.mark.asyncio
    async def test_get_progress_unauthenticated(self, client: AsyncClient):
        """Test getting progress without authentication."""
        response = await client.get("/api/v1/roadmap/progress")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_progress_authenticated(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test getting progress with authentication."""
        response = await client.get(
            "/api/v1/roadmap/progress",
            headers=auth_headers,
        )
        assert response.status_code == 200

        data = response.json()
        assert "total_paths" in data
        assert "completed_paths" in data
        assert "total_lessons" in data
        assert "completed_lessons" in data

    @pytest.mark.asyncio
    async def test_get_path_progress_unauthenticated(self, client: AsyncClient):
        """Test getting path progress without authentication."""
        fake_id = uuid4()
        response = await client.get(f"/api/v1/roadmap/progress/paths/{fake_id}")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_start_path_unauthenticated(self, client: AsyncClient):
        """Test starting path without authentication."""
        fake_id = uuid4()
        response = await client.post(f"/api/v1/roadmap/paths/{fake_id}/start")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_start_path_not_found(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test starting non-existent path."""
        fake_id = uuid4()
        response = await client.post(
            f"/api/v1/roadmap/paths/{fake_id}/start",
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_complete_lesson_unauthenticated(self, client: AsyncClient):
        """Test completing lesson without authentication."""
        fake_id = uuid4()
        response = await client.post(
            f"/api/v1/roadmap/lessons/{fake_id}/complete",
            json={},
        )
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_complete_lesson_not_found(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test completing non-existent lesson."""
        fake_id = uuid4()
        response = await client.post(
            f"/api/v1/roadmap/lessons/{fake_id}/complete",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_next_lesson_unauthenticated(self, client: AsyncClient):
        """Test getting next lesson without authentication."""
        response = await client.get("/api/v1/roadmap/next-lesson")
        assert response.status_code == 401

    @pytest.mark.asyncio
    async def test_get_next_lesson_authenticated(
        self, client: AsyncClient, auth_headers: dict
    ):
        """Test getting next lesson with authentication."""
        response = await client.get(
            "/api/v1/roadmap/next-lesson",
            headers=auth_headers,
        )
        # Should return 200 with null or lesson data
        assert response.status_code == 200
