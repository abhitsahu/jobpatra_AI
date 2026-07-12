"""Unit tests for the application configuration and secret management."""

import pytest
from pydantic import SecretStr, ValidationError

from app.core.config import Settings


def test_settings_instantiation_with_secrets(monkeypatch):
    """Verify Settings instantiates successfully when required secrets are provided."""
    monkeypatch.setenv("INTERNAL_API_KEY", "super_secret_api_key_123")
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "super_secret_service_token_456")

    # Instantiate Settings directly
    settings = Settings()

    # Verify types are SecretStr
    assert isinstance(settings.INTERNAL_API_KEY, SecretStr)
    assert isinstance(settings.INTERNAL_SERVICE_TOKEN, SecretStr)

    # Verify they can be read using get_secret_value()
    assert settings.INTERNAL_API_KEY.get_secret_value() == "super_secret_api_key_123"
    assert settings.INTERNAL_SERVICE_TOKEN.get_secret_value() == "super_secret_service_token_456"

    # Verify representation masks the secret value
    assert "super_secret_api_key_123" not in str(settings.INTERNAL_API_KEY)
    assert "**********" in str(settings.INTERNAL_API_KEY)


def test_settings_fails_when_api_key_missing(monkeypatch):
    """Verify Settings raises ValidationError when INTERNAL_API_KEY is missing."""
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "some_token")

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "INTERNAL_API_KEY" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)


def test_settings_fails_when_service_token_missing(monkeypatch):
    """Verify Settings raises ValidationError when INTERNAL_SERVICE_TOKEN is missing."""
    monkeypatch.setenv("INTERNAL_API_KEY", "some_key")
    monkeypatch.delenv("INTERNAL_SERVICE_TOKEN", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "INTERNAL_SERVICE_TOKEN" in str(exc_info.value)
    assert "Field required" in str(exc_info.value)
