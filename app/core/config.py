# app/core/config.py
"""
Application Configuration Module.

Centralizes all environment-driven settings using Pydantic BaseSettings.
Supports multiple AI providers, database backends, and deployment environments.
All sensitive values are loaded from environment variables or .env file.

Usage:
    from app.core.config import settings
    print(settings.APP_NAME)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Enumerations
# -----------------------------------------------------------------------------

class AppEnvironment(str, Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AIProvider(str, Enum):
    """
    Supported AI/LLM providers.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Enterprise application settings loaded from environment variables.

    All fields have sensible defaults for local development.
    Production deployments MUST override SECRET_KEY and AI credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    APP_NAME: str = Field(
        default="ETL Order Service",
        description="Application display name",
    )

    APP_VERSION: str = Field(
        default="0.1.0",
        description="Semantic version string",
    )

    APP_ENV: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        description="Deployment environment",
    )

    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Minimum log level",
    )

    # -------------------------------------------------------------------------
    # Server
    # -------------------------------------------------------------------------

    HOST: str = Field(
        default="0.0.0.0",
        description="Bind host",
    )

    PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Bind port",
    )

    WORKERS: int = Field(
        default=1,
        ge=1,
        description="Uvicorn worker count",
    )

    RELOAD: bool = Field(
        default=False,
        description="Auto-reload in development",
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------

    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./data/orders.db",
        description="SQLAlchemy async database URL",
    )

    DATABASE_ECHO: bool = Field(
        default=False,
        description="Echo SQL statements",
    )

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------

    SECRET_KEY: str = Field(
        default="dev-secret-key-change-in-production",
        description="JWT signing secret",
    )

    ALLOWED_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:8000",
        description="Comma-separated CORS origins",
    )

    ALLOWED_HOSTS: str = Field(
        default="localhost,127.0.0.1",
        description="Comma-separated trusted hosts",
    )

    CSRF_SECRET_KEY: str = Field(
        default="dev-csrf-secret-change-in-production",
        description="CSRF signing secret",
    )

    # -------------------------------------------------------------------------
    # Rate Limiting
    # -------------------------------------------------------------------------

    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        ge=1,
    )

    RATE_LIMIT_AI_PER_MINUTE: int = Field(
        default=10,
        ge=1,
    )

    # -------------------------------------------------------------------------
    # AI / LLM
    # -------------------------------------------------------------------------

    AI_PROVIDER: AIProvider = Field(
        default=AIProvider.OPENAI,
    )

    AI_MODEL: str = Field(
        default="gpt-4o-mini",
    )

    AI_TEMPERATURE: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    AI_MAX_TOKENS: int = Field(
        default=1024,
        ge=1,
    )

    AI_TIMEOUT_SECONDS: int = Field(
        default=30,
        ge=1,
    )

    AI_MAX_RETRIES: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    # OpenAI

    OPENAI_API_KEY: Optional[str] = Field(default=None)

    OPENAI_ORG_ID: Optional[str] = Field(default=None)

    OPENAI_BASE_URL: str = Field(
        default="https://api.openai.com/v1",
    )

    # Anthropic

    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)

    # Azure OpenAI

    AZURE_OPENAI_API_KEY: Optional[str] = Field(default=None)

    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(default=None)

    AZURE_OPENAI_API_VERSION: str = Field(
        default="2024-02-01",
    )

    AZURE_OPENAI_DEPLOYMENT_NAME: Optional[str] = Field(default=None)

    # Ollama

    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
    )

    OLLAMA_MODEL: str = Field(
        default="llama3",
    )

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------

    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
    )

    EMBEDDING_DEVICE: str = Field(
        default="cpu",
    )

    FAISS_INDEX_PATH: str = Field(
        default="./data/faiss_index",
    )

    EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        ge=1,
    )

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    CACHE_TTL_SECONDS: int = Field(
        default=60,
        ge=1,
    )

    CACHE_MAX_SIZE: int = Field(
        default=1000,
        ge=1,
    )

    # -------------------------------------------------------------------------
    # ETL
    # -------------------------------------------------------------------------

    ETL_DATA_DIR: str = Field(
        default="./data",
    )

    ETL_BATCH_SIZE: int = Field(
        default=1000,
        ge=1,
    )

    EUR_TO_USD_RATE: float = Field(
        default=1.1,
        gt=0,
    )

    # -------------------------------------------------------------------------
    # Observability
    # -------------------------------------------------------------------------

    ENABLE_METRICS: bool = Field(default=True)

    METRICS_PATH: str = Field(default="/metrics")

    ENABLE_TRACING: bool = Field(default=False)

    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------

    @property
    def allowed_origins_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def allowed_hosts_list(self) -> list[str]:
        return [
            host.strip()
            for host in self.ALLOWED_HOSTS.split(",")
            if host.strip()
        ]

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == AppEnvironment.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.APP_ENV == AppEnvironment.DEVELOPMENT

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        if len(value) < 16:
            raise ValueError(
                "SECRET_KEY must be at least 16 characters long"
            )
        return value

    @model_validator(mode="after")
    def validate_ai_credentials(self) -> "Settings":
        """
        Warn about missing AI credentials without preventing startup.
        """

        import warnings

        if self.AI_PROVIDER == AIProvider.OPENAI:
            if not self.OPENAI_API_KEY:
                warnings.warn(
                    "AI_PROVIDER=openai but OPENAI_API_KEY is not set. "
                    "AI endpoints will return 503.",
                    UserWarning,
                    stacklevel=2,
                )

        elif self.AI_PROVIDER == AIProvider.ANTHROPIC:
            if not self.ANTHROPIC_API_KEY:
                warnings.warn(
                    "AI_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set. "
                    "AI endpoints will return 503.",
                    UserWarning,
                    stacklevel=2,
                )

        elif self.AI_PROVIDER == AIProvider.AZURE_OPENAI:
            if not self.AZURE_OPENAI_API_KEY:
                warnings.warn(
                    "AI_PROVIDER=azure_openai but "
                    "AZURE_OPENAI_API_KEY is not set.",
                    UserWarning,
                    stacklevel=2,
                )

            if not self.AZURE_OPENAI_ENDPOINT:
                warnings.warn(
                    "AI_PROVIDER=azure_openai but "
                    "AZURE_OPENAI_ENDPOINT is not set.",
                    UserWarning,
                    stacklevel=2,
                )

        return self

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        """
        Enforce production security requirements.
        """

        if self.is_production:

            if self.SECRET_KEY == "dev-secret-key-change-in-production":
                raise ValueError(
                    "Production deployment detected with default "
                    "SECRET_KEY. Configure a secure secret."
                )

            if self.DEBUG:
                raise ValueError(
                    "DEBUG must be False in production."
                )

        return self


# -----------------------------------------------------------------------------
# Singleton Factory
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return cached settings singleton.
    """
    return Settings()


# Convenience alias

settings: Settings = get_settings()