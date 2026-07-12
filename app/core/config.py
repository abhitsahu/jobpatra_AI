"""Application configuration using pydantic-settings.

Sensitive values use SecretStr and have NO default — the application
will fail immediately on startup if they are not supplied via
environment variables or a .env file.  This prevents the service
from ever running with known/guessable credentials.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Non-sensitive values have safe defaults.
    Sensitive values (secrets, tokens, keys) have NO defaults and
    MUST be provided — pydantic will raise ValidationError at
    instantiation time if they are missing.
    """

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Non-sensitive — safe defaults
    APP_NAME: str = "JobPatra AI"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Sensitive — NO defaults, masked by SecretStr
    INTERNAL_API_KEY: SecretStr
    INTERNAL_SERVICE_TOKEN: SecretStr


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()


settings = get_settings()
