"""Identity domain entities"""

from datetime import datetime
from uuid import UUID

from code_tutor.identity.domain.value_objects import (
    Email,
    HashedPassword,
    UserId,
    UserRole,
    Username,
)
from code_tutor.identity.domain.events import (
    PasswordChanged,
    UserCreated,
    UserLoggedIn,
)
from code_tutor.shared.domain.base import AggregateRoot
from code_tutor.shared.security import hash_password, verify_password


class User(AggregateRoot):
    """User aggregate root"""

    def __init__(
        self,
        id: UUID | None = None,
        email: Email | None = None,
        username: Username | None = None,
        hashed_password: HashedPassword | None = None,
        role: UserRole = UserRole.STUDENT,
        is_active: bool = True,
        is_verified: bool = False,
        last_login_at: datetime | None = None,
        bio: str | None = None,
    ) -> None:
        super().__init__(id)
        self._email = email
        self._username = username
        self._hashed_password = hashed_password
        self._role = role
        self._is_active = is_active
        self._is_verified = is_verified
        self._last_login_at = last_login_at
        self._bio = bio

    @classmethod
    def create(
        cls,
        email: str,
        username: str,
        password: str,
        role: UserRole = UserRole.STUDENT,
    ) -> "User":
        """Factory method to create a new user"""
        from code_tutor.identity.domain.value_objects import Password

        # Validate inputs
        email_vo = Email(email)
        username_vo = Username(username)
        password_vo = Password(password)
        password_vo.validate_strength()

        # Hash password
        hashed = hash_password(password)
        hashed_password = HashedPassword(hashed)

        # Create user
        user = cls(
            email=email_vo,
            username=username_vo,
            hashed_password=hashed_password,
            role=role,
        )

        # Add domain event
        user.add_domain_event(
            UserCreated(
                user_id=user.id,
                email=email,
                username=username,
            )
        )

        return user

    # Properties
    @property
    def email(self) -> Email | None:
        return self._email

    @property
    def username(self) -> Username | None:
        return self._username

    @property
    def hashed_password(self) -> HashedPassword | None:
        return self._hashed_password

    @property
    def role(self) -> UserRole:
        return self._role

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def is_verified(self) -> bool:
        return self._is_verified

    @property
    def last_login_at(self) -> datetime | None:
        return self._last_login_at

    @property
    def bio(self) -> str | None:
        return self._bio

    # Behavior methods
    def verify_password(self, plain_password: str) -> bool:
        """Verify if the provided password matches"""
        if self._hashed_password is None:
            return False
        return verify_password(plain_password, self._hashed_password.value)

    def change_password(self, old_password: str, new_password: str) -> None:
        """Change user password"""
        from code_tutor.identity.domain.value_objects import Password

        if not self.verify_password(old_password):
            from code_tutor.shared.exceptions import ValidationError
            raise ValidationError("Current password is incorrect")

        # Validate new password
        password_vo = Password(new_password)
        password_vo.validate_strength()

        # Update password
        hashed = hash_password(new_password)
        self._hashed_password = HashedPassword(hashed)
        self._touch()

        self.add_domain_event(PasswordChanged(user_id=self.id))

    def record_login(self) -> None:
        """Record a successful login"""
        self._last_login_at = datetime.utcnow()
        self._touch()

        self.add_domain_event(UserLoggedIn(user_id=self.id))

    def activate(self) -> None:
        """Activate user account"""
        self._is_active = True
        self._touch()

    def deactivate(self) -> None:
        """Deactivate user account"""
        self._is_active = False
        self._touch()

    def verify_email(self) -> None:
        """Mark email as verified"""
        self._is_verified = True
        self._touch()

    def promote_to_admin(self) -> None:
        """Promote user to admin role"""
        self._role = UserRole.ADMIN
        self._touch()

    def update_profile(
        self,
        username: str | None = None,
        bio: str | None = None,
    ) -> None:
        """Update user profile"""
        if username is not None:
            self._username = Username(username)
        if bio is not None:
            self._bio = bio
        self._touch()
