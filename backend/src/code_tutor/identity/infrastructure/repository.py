"""Identity repository implementations"""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from code_tutor.identity.domain.entities import User
from code_tutor.identity.domain.repository import UserRepository
from code_tutor.identity.domain.value_objects import (
    Email,
    HashedPassword,
    Username,
)
from code_tutor.identity.infrastructure.models import UserModel


class SQLAlchemyUserRepository(UserRepository):
    """SQLAlchemy implementation of UserRepository"""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _to_entity(self, model: UserModel) -> User:
        """Convert SQLAlchemy model to domain entity"""
        user = User(
            id=model.id,
            email=Email(model.email),
            username=Username(model.username),
            hashed_password=HashedPassword(model.hashed_password),
            role=model.role,
            is_active=model.is_active,
            is_verified=model.is_verified,
            last_login_at=model.last_login_at,
            bio=model.bio,
        )
        # Restore timestamps
        user._created_at = model.created_at
        user._updated_at = model.updated_at
        return user

    def _to_model(self, entity: User) -> UserModel:
        """Convert domain entity to SQLAlchemy model"""
        return UserModel(
            id=entity.id,
            email=str(entity.email) if entity.email else "",
            username=str(entity.username) if entity.username else "",
            hashed_password=entity.hashed_password.value
            if entity.hashed_password
            else "",
            role=entity.role,
            is_active=entity.is_active,
            is_verified=entity.is_verified,
            last_login_at=entity.last_login_at,
            bio=entity.bio,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    async def get_by_id(self, id: UUID) -> User | None:
        """Get user by ID"""
        model = await self._session.get(UserModel, id)
        if model is None:
            return None
        return self._to_entity(model)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email"""
        stmt = select(UserModel).where(UserModel.email == email)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def get_by_username(self, username: str) -> User | None:
        """Get user by username"""
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def add(self, user: User) -> User:
        """Add a new user"""
        model = self._to_model(user)
        self._session.add(model)
        await self._session.flush()
        return self._to_entity(model)

    async def update(self, user: User) -> User:
        """Update an existing user"""
        model = self._to_model(user)
        merged = await self._session.merge(model)
        await self._session.flush()
        return self._to_entity(merged)

    async def delete(self, id: UUID) -> bool:
        """Delete user by ID"""
        model = await self._session.get(UserModel, id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.flush()
        return True

    async def exists_by_email(self, email: str) -> bool:
        """Check if user with email exists"""
        stmt = select(UserModel.id).where(UserModel.email == email)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def exists_by_username(self, username: str) -> bool:
        """Check if user with username exists"""
        stmt = select(UserModel.id).where(UserModel.username == username)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None
