"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    INTERNAL_SERVICE_TOKEN: str = "replace_me"


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()


settings = get_settings()
