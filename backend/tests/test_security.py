"""Unit tests for Security Module"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from code_tutor.shared.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_jti,
    TokenPayload,
)
from code_tutor.shared.config import get_settings
from code_tutor.shared.exceptions import UnauthorizedError


class TestPasswordHashing:
    """Tests for password hashing"""

    def test_hash_password(self):
        """Test password hashing"""
        password = "SecurePassword123"
        hashed = hash_password(password)

        assert hashed != password
        assert len(hashed) > 0

    def test_verify_password_correct(self):
        """Test verifying correct password"""
        password = "SecurePassword123"
        hashed = hash_password(password)

        assert verify_password(password, hashed) is True

    def test_verify_password_incorrect(self):
        """Test verifying incorrect password"""
        password = "SecurePassword123"
        hashed = hash_password(password)

        assert verify_password("WrongPassword", hashed) is False

    def test_different_hashes_same_password(self):
        """Test that same password produces different hashes (salted)"""
        password = "SecurePassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Hashes should be different due to salt
        assert hash1 != hash2
        # But both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestJWTTokens:
    """Tests for JWT token handling"""

    def test_create_access_token(self):
        """Test creating access token"""
        data = {"sub": str(uuid4()), "role": "user"}
        token = create_access_token(data)

        assert token is not None
        assert len(token) > 0
        assert "." in token  # JWT format

    def test_decode_valid_token(self):
        """Test decoding valid token"""
        user_id = str(uuid4())
        data = {"sub": user_id, "role": "user"}
        token = create_access_token(data)

        decoded = decode_token(token)

        assert decoded is not None
        assert decoded["sub"] == user_id
        assert decoded["role"] == "user"

    def test_decode_invalid_token(self):
        """Test decoding invalid token raises error"""
        with pytest.raises(UnauthorizedError):
            decode_token("invalid.token.here")

    def test_token_contains_expiry(self):
        """Test that token contains expiry"""
        data = {"sub": str(uuid4())}
        token = create_access_token(data)
        decoded = decode_token(token)

        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded
        assert decoded["type"] == "access"

    def test_create_refresh_token(self):
        """Test creating refresh token"""
        data = {"sub": str(uuid4())}
        token = create_refresh_token(data)
        decoded = decode_token(token)

        assert decoded["type"] == "refresh"

    def test_get_token_jti(self):
        """Test getting JTI from token"""
        data = {"sub": str(uuid4())}
        token = create_access_token(data)
        jti = get_token_jti(token)

        assert jti is not None
        assert len(jti) > 0

    def test_get_token_jti_invalid_token(self):
        """Test getting JTI from invalid token"""
        jti = get_token_jti("invalid.token.here")
        assert jti == ""


class TestTokenPayload:
    """Tests for TokenPayload class"""

    def test_token_payload_creation(self):
        """Test creating token payload"""
        data = {"sub": str(uuid4())}
        token = create_access_token(data)
        decoded = decode_token(token)

        payload = TokenPayload(decoded)

        assert payload.user_id == data["sub"]
        assert payload.is_access_token is True
        assert payload.is_refresh_token is False

    def test_token_payload_refresh(self):
        """Test token payload for refresh token"""
        data = {"sub": str(uuid4())}
        token = create_refresh_token(data)
        decoded = decode_token(token)

        payload = TokenPayload(decoded)

        assert payload.is_access_token is False
        assert payload.is_refresh_token is True

    def test_token_payload_expiry(self):
        """Test token payload expiry check"""
        data = {"sub": str(uuid4())}
        token = create_access_token(data)
        decoded = decode_token(token)

        payload = TokenPayload(decoded)

        # Fresh token should not be expired
        assert payload.is_expired is False

    def test_token_payload_jti(self):
        """Test token payload JTI"""
        data = {"sub": str(uuid4())}
        token = create_access_token(data)
        decoded = decode_token(token)

        payload = TokenPayload(decoded)

        assert payload.jti is not None
        assert len(payload.jti) > 0
