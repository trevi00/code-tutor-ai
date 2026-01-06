"""Application configuration using Pydantic Settings"""

from functools import lru_cache
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(Exception):
    """Configuration validation error"""

    pass


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "Code Tutor AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/codetutor"
    )
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT Auth
    JWT_SECRET_KEY: str = "change-this-secret-key-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # LLM Configuration
    LLM_PROVIDER: Literal["ollama", "openai", "pattern"] = "ollama"
    LLM_MODEL_PATH: str = "yanolja/EEVE-Korean-2.8B-v1.0"
    LLM_DEVICE: str = "cuda"
    LLM_MAX_TOKENS: int = 512

    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    OLLAMA_TIMEOUT: int = 60

    # OpenAI Configuration (optional)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Docker Sandbox
    SANDBOX_TIMEOUT_SECONDS: int = 5
    SANDBOX_MEMORY_LIMIT_MB: int = 256
    SANDBOX_CPU_LIMIT: float = 0.5

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate critical settings in production environment"""
        if self.ENVIRONMENT == "production":
            # JWT secret must be changed from default
            if self.JWT_SECRET_KEY == "change-this-secret-key-in-production":
                raise ConfigurationError(
                    "JWT_SECRET_KEY must be set to a secure value in production. "
                    "Generate one with: openssl rand -hex 32"
                )

            # JWT secret should be at least 32 characters
            if len(self.JWT_SECRET_KEY) < 32:
                raise ConfigurationError(
                    "JWT_SECRET_KEY must be at least 32 characters in production"
                )

            # CORS wildcard not allowed in production
            if "*" in self.CORS_ORIGINS:
                raise ConfigurationError(
                    "Wildcard (*) CORS origin is not allowed in production. "
                    "Please specify explicit origins."
                )

            # Database URL should not be localhost in production
            if "localhost" in self.DATABASE_URL or "127.0.0.1" in self.DATABASE_URL:
                raise ConfigurationError(
                    "DATABASE_URL should not use localhost in production"
                )

        return self


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
