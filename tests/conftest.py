"""Shared pytest configuration and fixtures."""

import os

# Set fallback environment variables for test execution so that Settings validation passes during import
os.environ["INTERNAL_API_KEY"] = "test_api_key_123"
os.environ["INTERNAL_SERVICE_TOKEN"] = "test_service_token_123"
