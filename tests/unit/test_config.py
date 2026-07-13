"""Unit tests for the application configuration and secret management.

Each test instantiates Settings() directly (bypassing the lru_cache singleton)
and forces env_file to a nonexistent path so that the only source of values is
the environment monkeypatched by the test itself.
"""

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from app.core.config import Settings


class _NoFileSettings(Settings):
    """Settings subclass that disables .env file loading.

    This prevents the real config/.env from satisfying missing variables
    during tests that deliberately omit required secrets.
    """

    model_config = SettingsConfigDict(
        env_file=None,          # do NOT read any .env file
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


def test_settings_instantiation_with_secrets(monkeypatch):
    """Settings instantiates successfully when required secrets are provided."""
    monkeypatch.setenv("INTERNAL_API_KEY", "super_secret_api_key_123")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "super_secret_service_token_456")

    s = _NoFileSettings()

    assert isinstance(s.INTERNAL_API_KEY, SecretStr)
    assert isinstance(s.INTERNAL_SERVICE_TOKEN, SecretStr)
    assert s.INTERNAL_API_KEY.get_secret_value() == "super_secret_api_key_123"
    assert s.INTERNAL_SERVICE_TOKEN.get_secret_value() == "super_secret_service_token_456"

    # SecretStr must mask the value in str() / repr()
    assert "super_secret_api_key_123" not in str(s.INTERNAL_API_KEY)
    assert "**********" in str(s.INTERNAL_API_KEY)


def test_settings_fails_when_api_key_missing(monkeypatch):
    """Settings raises ValidationError when INTERNAL_API_KEY is absent."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "some_token")

    with pytest.raises(ValidationError) as exc_info:
        _NoFileSettings()

    assert "INTERNAL_API_KEY" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


def test_settings_fails_when_service_token_missing(monkeypatch):
    """Settings raises ValidationError when INTERNAL_SERVICE_TOKEN is absent."""
    monkeypatch.setenv("INTERNAL_API_KEY", "some_key")
    monkeypatch.delenv("INTERNAL_SERVICE_TOKEN", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        _NoFileSettings()

    assert "INTERNAL_SERVICE_TOKEN" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)
