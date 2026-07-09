from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def mock_client():
    """A mock AsyncPublicApiClient that works as an async context manager."""
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.fixture
def patch_get_client(mock_client):
    """Patch _get_client to return mock_client."""
    with patch("publicdotcom_mcp_server.server._get_client", return_value=mock_client):
        yield mock_client


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    """Ensure required env vars are set for all tests."""
    monkeypatch.setenv("PUBLIC_COM_SECRET", "test-secret-key")
    monkeypatch.setenv("PUBLIC_COM_ACCOUNT_ID", "test-account-123")
