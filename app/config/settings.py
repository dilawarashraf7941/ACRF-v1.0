"""Centralized application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for ACRF, populated from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Adaptive Critic Routing Framework"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # LLM provider access (consumed via LiteLLM)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # ChromaDB
    chroma_persist_directory: str = Field(default=".chroma")

    # FastAPI
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
