"""Tests for AI Tutor API"""

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
async def test_chat_unauthorized(client: AsyncClient):
    """Test chat access without authentication"""
    response = await client.post(
        "/api/v1/tutor/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_chat_new_conversation(client: AsyncClient, auth_headers: dict):
    """Test creating a new conversation with chat"""
    response = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "안녕하세요. 알고리즘 공부를 시작하려고 합니다.",
            "conversation_type": "general",
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    data = response.json()
    assert "conversation_id" in data
    assert "message" in data
    assert data["is_new_conversation"] is True

    # Check message structure
    message = data["message"]
    assert "id" in message
    assert message["role"] == "assistant"
    assert "content" in message
    assert len(message["content"]) > 0


@pytest.mark.asyncio
async def test_chat_continue_conversation(client: AsyncClient, auth_headers: dict):
    """Test continuing an existing conversation"""
    # Start conversation
    response1 = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "Python에서 리스트와 튜플의 차이점이 뭔가요?",
            "conversation_type": "concept",
        },
        headers=auth_headers,
    )
    assert response1.status_code == 200
    conversation_id = response1.json()["conversation_id"]

    # Continue conversation
    response2 = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "더 자세히 설명해주세요.",
            "conversation_id": conversation_id,
        },
        headers=auth_headers,
    )
    assert response2.status_code == 200

    data = response2.json()
    assert data["conversation_id"] == conversation_id
    assert data["is_new_conversation"] is False


@pytest.mark.asyncio
async def test_chat_with_code_context(client: AsyncClient, auth_headers: dict):
    """Test chat with code context for code review"""
    response = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "이 코드를 리뷰해주세요.",
            "conversation_type": "code_review",
            "code_context": {
                "code": "def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                "language": "python",
            },
        },
        headers=auth_headers,
    )
    assert response.status_code == 200

    data = response.json()
    assert "conversation_id" in data
    assert "message" in data


@pytest.mark.asyncio
async def test_list_conversations_empty(client: AsyncClient, auth_headers: dict):
    """Test listing conversations when none exist"""
    response = await client.get(
        "/api/v1/tutor/conversations",
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_list_conversations_after_chat(client: AsyncClient, auth_headers: dict):
    """Test listing conversations after creating one"""
    # Create conversation
    await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "알고리즘이란 무엇인가요?",
            "conversation_type": "concept",
        },
        headers=auth_headers,
    )

    # List conversations
    response = await client.get(
        "/api/v1/tutor/conversations",
        headers=auth_headers,
    )
    assert response.status_code == 200

    conversations = response.json()
    assert len(conversations) == 1
    assert "id" in conversations[0]
    assert "conversation_type" in conversations[0]
    assert conversations[0]["message_count"] >= 2  # user + assistant


@pytest.mark.asyncio
async def test_get_conversation_not_found(client: AsyncClient, auth_headers: dict):
    """Test getting a non-existent conversation"""
    import uuid
    fake_id = str(uuid.uuid4())
    response = await client.get(
        f"/api/v1/tutor/conversations/{fake_id}",
        headers=auth_headers,
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_get_conversation_detail(client: AsyncClient, auth_headers: dict):
    """Test getting conversation details with messages"""
    # Create conversation
    chat_response = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "배열 정렬에 대해 알려주세요.",
            "conversation_type": "concept",
        },
        headers=auth_headers,
    )
    conversation_id = chat_response.json()["conversation_id"]

    # Get conversation
    response = await client.get(
        f"/api/v1/tutor/conversations/{conversation_id}",
        headers=auth_headers,
    )
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == conversation_id
    assert "messages" in data
    assert len(data["messages"]) >= 2  # user + assistant
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_close_conversation(client: AsyncClient, auth_headers: dict):
    """Test closing a conversation"""
    # Create conversation
    chat_response = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "감사합니다!",
            "conversation_type": "general",
        },
        headers=auth_headers,
    )
    conversation_id = chat_response.json()["conversation_id"]

    # Close conversation
    response = await client.post(
        f"/api/v1/tutor/conversations/{conversation_id}/close",
        headers=auth_headers,
    )
    assert response.status_code == 200

    data = response.json()
    assert data["is_active"] is False


@pytest.mark.asyncio
async def test_chat_empty_message(client: AsyncClient, auth_headers: dict):
    """Test chat with empty message"""
    response = await client.post(
        "/api/v1/tutor/chat",
        json={
            "message": "",
            "conversation_type": "general",
        },
        headers=auth_headers,
    )
    # Should fail validation
    assert response.status_code == 400 or response.status_code == 422
