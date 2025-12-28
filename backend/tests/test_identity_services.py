"""Unit tests for Identity Domain"""

import pytest
from uuid import uuid4

from code_tutor.identity.domain.value_objects import (
    Email,
    Password,
    Username,
    UserId,
    UserRole,
    HashedPassword,
)
from code_tutor.shared.exceptions import ValidationError


class TestEmail:
    """Tests for Email value object"""

    def test_valid_email(self):
        """Test creating valid email"""
        email = Email("test@example.com")
        assert str(email) == "test@example.com"

    def test_email_local_part(self):
        """Test email local part extraction"""
        email = Email("user@example.com")
        assert email.local_part == "user"

    def test_email_domain(self):
        """Test email domain extraction"""
        email = Email("user@example.com")
        assert email.domain == "example.com"

    def test_invalid_email_format(self):
        """Test invalid email format"""
        with pytest.raises(ValidationError):
            Email("invalid-email")

    def test_empty_email(self):
        """Test empty email"""
        with pytest.raises(ValidationError):
            Email("")

    def test_email_equality(self):
        """Test email equality"""
        email1 = Email("test@example.com")
        email2 = Email("test@example.com")
        assert email1 == email2

    def test_email_with_subdomain(self):
        """Test email with subdomain"""
        email = Email("user@mail.example.com")
        assert str(email) == "user@mail.example.com"
        assert email.domain == "mail.example.com"


class TestPassword:
    """Tests for Password value object"""

    def test_valid_password(self):
        """Test valid password creation"""
        password = Password("SecurePass123")
        assert password.value == "SecurePass123"

    def test_short_password(self):
        """Test password minimum length"""
        with pytest.raises(ValidationError):
            Password("short")

    def test_empty_password(self):
        """Test empty password"""
        with pytest.raises(ValidationError):
            Password("")

    def test_password_strength_valid(self):
        """Test valid strong password"""
        password = Password("SecurePass123")
        password.validate_strength()  # Should not raise

    def test_password_strength_no_uppercase(self):
        """Test password without uppercase"""
        password = Password("securepass123")
        with pytest.raises(ValidationError):
            password.validate_strength()

    def test_password_strength_no_lowercase(self):
        """Test password without lowercase"""
        password = Password("SECUREPASS123")
        with pytest.raises(ValidationError):
            password.validate_strength()

    def test_password_strength_no_digit(self):
        """Test password without digit"""
        password = Password("SecurePassword")
        with pytest.raises(ValidationError):
            password.validate_strength()


class TestUsername:
    """Tests for Username value object"""

    def test_valid_username(self):
        """Test valid username creation"""
        username = Username("testuser")
        assert str(username) == "testuser"

    def test_username_with_numbers(self):
        """Test username with numbers"""
        username = Username("user123")
        assert str(username) == "user123"

    def test_username_with_underscore(self):
        """Test username with underscore"""
        username = Username("test_user")
        assert str(username) == "test_user"

    def test_username_with_hyphen(self):
        """Test username with hyphen"""
        username = Username("test-user")
        assert str(username) == "test-user"

    def test_short_username(self):
        """Test username minimum length"""
        with pytest.raises(ValidationError):
            Username("ab")

    def test_empty_username(self):
        """Test empty username"""
        with pytest.raises(ValidationError):
            Username("")

    def test_username_with_special_chars(self):
        """Test username with invalid special characters"""
        with pytest.raises(ValidationError):
            Username("user@name")

    def test_long_username(self):
        """Test username maximum length"""
        with pytest.raises(ValidationError):
            Username("a" * 31)


class TestUserId:
    """Tests for UserId value object"""

    def test_generate_user_id(self):
        """Test generating new user ID"""
        user_id = UserId.generate()
        assert user_id.value is not None

    def test_user_id_from_string(self):
        """Test creating user ID from string"""
        uuid_str = str(uuid4())
        user_id = UserId.from_string(uuid_str)
        assert str(user_id) == uuid_str

    def test_invalid_user_id_string(self):
        """Test creating user ID from invalid string"""
        with pytest.raises(ValidationError):
            UserId.from_string("invalid-uuid")

    def test_user_id_equality(self):
        """Test user ID equality"""
        uuid_val = uuid4()
        user_id1 = UserId(value=uuid_val)
        user_id2 = UserId(value=uuid_val)
        assert user_id1 == user_id2


class TestUserRole:
    """Tests for UserRole enum"""

    def test_user_role_values(self):
        """Test user role enum values"""
        assert UserRole.STUDENT.value == "student"
        assert UserRole.ADMIN.value == "admin"

    def test_user_role_from_string(self):
        """Test creating role from string value"""
        role = UserRole("student")
        assert role == UserRole.STUDENT


class TestHashedPassword:
    """Tests for HashedPassword value object"""

    def test_hashed_password_never_exposed(self):
        """Test that hashed password never exposes hash"""
        hashed = HashedPassword(value="$2b$12$somehash")
        assert str(hashed) == "***"

    def test_hashed_password_value(self):
        """Test accessing hashed password value"""
        hash_value = "$2b$12$somehash"
        hashed = HashedPassword(value=hash_value)
        assert hashed.value == hash_value
