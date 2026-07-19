"""Application configuration using pydantic-settings.

Sensitive values use SecretStr and have NO default — the application
will fail immediately on startup if they are not supplied via
environment variables or a .env file.  This prevents the service
from ever running with known/guessable credentials.
"""

from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables from config/.env so they are available in os.environ for LiteLLM, LangSmith, etc.
load_dotenv("config/.env")


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
        extra="ignore",  # ignore unknown env vars (e.g. OPENAI_API_KEY set by LangChain tooling)
    )

    # Non-sensitive — safe defaults
    APP_NAME: str = "JobPatra AI"
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    DEBUG_ATS_PIPELINE: bool = False

    # Sensitive — NO defaults, masked by SecretStr
    INTERNAL_API_KEY: SecretStr
    INTERNAL_SERVICE_TOKEN: SecretStr

    # LiteLLM Routing Configuration
    LITELLM_ROUTER_CONFIG_PATH: str = "config/litellm_router.yaml"
    LITELLM_ROUTING_STRATEGY: str = "simple-shuffle"

    LITELLM_TIMEOUT: float = 10.0
    LITELLM_RETRY_COUNT: int = 2
    LITELLM_COOLDOWN_TIME: int = 30
    LITELLM_MAX_FAILURES: int = 3

    # LLM Provider API Keys
    GOOGLE_API_KEY: SecretStr | None = None
    GROQ_API_KEY: SecretStr | None = None

    # LangSmith observability (all optional)
    # LANGSMITH_API_KEY must be set to enable tracing; safe to omit in non-prod.
    LANGSMITH_API_KEY: SecretStr | None = None
    """Anthropic LangSmith API key. Required when LANGSMITH_TRACING_ENABLED=true."""
    LANGSMITH_PROJECT: str = "jobpatra-ai"
    """LangSmith project name. Traces are grouped under this project."""
    LANGSMITH_TRACING_ENABLED: bool = False
    """Master switch. When False the tracer is never created and no network calls are made."""


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()


settings = get_settings()
