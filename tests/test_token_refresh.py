"""
Tests for token lifecycle behaviour across MCP tool calls.

Two properties are verified:

1. Token reuse — a single access token is minted for multiple tool calls
   made within the token's validity window. Before the client-caching fix
   a fresh token was created (and immediately discarded) on every call.

2. Automatic refresh — when a token expires the SDK transparently mints a
   new one on the next call, without surfacing an error to the caller.

These tests patch AsyncApiKeyAuthProvider._create_personal_access_token so
they do not need a network connection and do not depend on real token TTLs.
Expiry is simulated by backdating _access_token_expires_at directly.
"""

import time
from unittest.mock import patch

import pytest

import publicdotcom_mcp_server.server as srv
from public_api_sdk.async_auth_provider import AsyncApiKeyAuthProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_token_creator(store: list[str]):
    """Return a replacement for _create_personal_access_token that records calls."""

    async def _create(self_provider):
        token = f"token-{len(store) + 1}"
        store.append(token)
        self_provider._access_token = token
        self_provider._access_token_expires_at = time.time() + 600  # 10 min
        self_provider.api_client.set_auth_header(token)

    return _create


def _provider(secret: str) -> AsyncApiKeyAuthProvider:
    """Return the auth provider from the cached client for the given secret."""
    return srv._clients[secret].auth_manager.auth_provider


# ---------------------------------------------------------------------------
# Token reuse — one mint across multiple _get_client() calls
# ---------------------------------------------------------------------------

class TestTokenReuse:
    def setup_method(self):
        srv._clients.clear()

    async def test_token_minted_once_for_multiple_calls(self, monkeypatch):
        """_create_personal_access_token should be called exactly once regardless
        of how many times _get_client() is entered with the same secret."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "reuse-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            for _ in range(5):
                async with srv._get_client() as client:
                    await client.auth_manager.refresh_token_if_needed()

        assert len(minted) == 1, (
            f"Expected 1 token creation, got {len(minted)}. "
            "Each call is creating a new token instead of reusing the cached one."
        )

    async def test_same_token_used_across_calls(self, monkeypatch):
        """The access token set on the HTTP client should be the same object
        throughout the validity window."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "same-token-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()
            token_after_first_call = _provider("same-token-secret")._access_token

            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()
            token_after_second_call = _provider("same-token-secret")._access_token

        assert token_after_first_call == token_after_second_call


# ---------------------------------------------------------------------------
# Automatic refresh — new token minted after expiry
# ---------------------------------------------------------------------------

class TestTokenRefreshAfterExpiry:
    def setup_method(self):
        srv._clients.clear()

    async def test_expired_token_triggers_refresh(self, monkeypatch):
        """After the token expiry timestamp passes, the next call should
        automatically mint a replacement token without raising an error."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "refresh-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            # First call: mint initial token
            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()

            assert len(minted) == 1

            # Simulate expiry by backdating the expiry timestamp
            _provider("refresh-secret")._access_token_expires_at = time.time() - 1

            # Second call: provider detects expiry and mints a new token
            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()

        assert len(minted) == 2, (
            "Expected a second token to be minted after simulated expiry."
        )

    async def test_new_token_differs_from_expired_token(self, monkeypatch):
        """The refreshed token should be a different value from the expired one."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "new-token-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()
            first_token = _provider("new-token-secret")._access_token

            _provider("new-token-secret")._access_token_expires_at = time.time() - 1

            async with srv._get_client() as client:
                await client.auth_manager.refresh_token_if_needed()
            second_token = _provider("new-token-secret")._access_token

        assert first_token != second_token

    async def test_no_refresh_within_validity_window(self, monkeypatch):
        """Token must NOT be refreshed when it is still within the validity window,
        even across multiple calls."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "no-refresh-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            for _ in range(3):
                async with srv._get_client() as client:
                    await client.auth_manager.refresh_token_if_needed()

            # Manually confirm the token is not expired yet
            provider = _provider("no-refresh-secret")
            assert provider._access_token_expires_at > time.time()

        assert len(minted) == 1

    async def test_multiple_expiry_cycles(self, monkeypatch):
        """Token should be refreshed each time it expires, not just the first time."""
        monkeypatch.setenv("PUBLIC_COM_SECRET", "cycle-secret")
        minted: list[str] = []

        with patch.object(AsyncApiKeyAuthProvider, "_create_personal_access_token",
                          new=_make_token_creator(minted)):
            for expected_count in range(1, 4):
                async with srv._get_client() as client:
                    await client.auth_manager.refresh_token_if_needed()

                assert len(minted) == expected_count

                # Expire the token before the next iteration
                _provider("cycle-secret")._access_token_expires_at = time.time() - 1
