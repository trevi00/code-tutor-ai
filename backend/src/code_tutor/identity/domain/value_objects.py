"""Identity domain value objects"""

import re
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

from code_tutor.shared.domain.base import ValueObject
from code_tutor.shared.exceptions import ValidationError


class UserRole(str, Enum):
    """User roles in the system"""

    STUDENT = "student"
    ADMIN = "admin"


@dataclass(frozen=True)
class UserId(ValueObject):
    """User ID value object"""

    value: UUID

    @classmethod
    def generate(cls) -> "UserId":
        """Generate a new user ID"""
        return cls(value=uuid4())

    @classmethod
    def from_string(cls, value: str) -> "UserId":
        """Create UserId from string"""
        try:
            return cls(value=UUID(value))
        except ValueError:
            raise ValidationError(f"Invalid user ID format: {value}")

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Email(ValueObject):
    """Email value object with validation"""

    value: str

    EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def __post_init__(self) -> None:
        if not self.value:
            raise ValidationError("Email cannot be empty")
        if not self.EMAIL_REGEX.match(self.value):
            raise ValidationError(f"Invalid email format: {self.value}")

    @property
    def local_part(self) -> str:
        """Get the local part of the email (before @)"""
        return self.value.split("@")[0]

    @property
    def domain(self) -> str:
        """Get the domain part of the email (after @)"""
        return self.value.split("@")[1]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Username(ValueObject):
    """Username value object with validation"""

    value: str

    MIN_LENGTH = 3
    MAX_LENGTH = 30
    USERNAME_REGEX = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __post_init__(self) -> None:
        if not self.value:
            raise ValidationError("Username cannot be empty")
        if len(self.value) < self.MIN_LENGTH:
            raise ValidationError(
                f"Username must be at least {self.MIN_LENGTH} characters"
            )
        if len(self.value) > self.MAX_LENGTH:
            raise ValidationError(
                f"Username must be at most {self.MAX_LENGTH} characters"
            )
        if not self.USERNAME_REGEX.match(self.value):
            raise ValidationError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Password(ValueObject):
    """Password value object (for validation before hashing)"""

    value: str

    MIN_LENGTH = 8
    MAX_LENGTH = 128
    SPECIAL_CHARACTERS = "!@#$%^&*()_+-=[]{}|;:,.<>?"

    # Common weak passwords to block
    COMMON_PASSWORDS = frozenset(
        [
            "password",
            "123456",
            "12345678",
            "qwerty",
            "abc123",
            "password1",
            "password123",
            "admin",
            "letmein",
            "welcome",
            "monkey",
            "dragon",
            "master",
            "iloveyou",
            "sunshine",
            "princess",
            "football",
            "baseball",
            "shadow",
            "michael",
            "test1234",
            "test123",
            "qwerty123",
            "pass1234",
            "admin123",
        ]
    )

    def __post_init__(self) -> None:
        if not self.value:
            raise ValidationError("Password cannot be empty")
        if len(self.value) < self.MIN_LENGTH:
            raise ValidationError(
                f"Password must be at least {self.MIN_LENGTH} characters"
            )
        if len(self.value) > self.MAX_LENGTH:
            raise ValidationError(
                f"Password must be at most {self.MAX_LENGTH} characters"
            )

    def validate_strength(self) -> None:
        """Validate password strength

        Requirements:
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        - Not a common weak password
        """
        has_upper = any(c.isupper() for c in self.value)
        has_lower = any(c.islower() for c in self.value)
        has_digit = any(c.isdigit() for c in self.value)
        has_special = any(c in self.SPECIAL_CHARACTERS for c in self.value)

        if not has_upper:
            raise ValidationError("Password must contain at least one uppercase letter")
        if not has_lower:
            raise ValidationError("Password must contain at least one lowercase letter")
        if not has_digit:
            raise ValidationError("Password must contain at least one digit")
        if not has_special:
            raise ValidationError(
                f"Password must contain at least one special character ({self.SPECIAL_CHARACTERS})"
            )

        # Check against common passwords
        if self.value.lower() in self.COMMON_PASSWORDS:
            raise ValidationError(
                "Password is too common. Please choose a stronger password"
            )


@dataclass(frozen=True)
class HashedPassword(ValueObject):
    """Hashed password value object"""

    value: str

    def __str__(self) -> str:
        return "***"  # Never expose hash
