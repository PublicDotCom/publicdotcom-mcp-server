"""Tests for per-request auth (ContextVars + ApiKeyMiddleware)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from starlette.testclient import TestClient
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route

import publicdotcom_mcp_server.server as srv


# ---------------------------------------------------------------------------
# _get_client — ContextVar resolution
# ---------------------------------------------------------------------------

class TestGetClientAuthResolution:
    def setup_method(self):
        # Each test gets a clean cache so patching AsyncPublicApiClient is effective.
        srv._clients.clear()

    async def test_uses_contextvar_api_key_over_env(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "env-key")
        token = srv._api_key.set("header-key")
        try:
            with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
                async with srv._get_client():
                    pass
                call_kwargs = MockClient.call_args.kwargs
                assert call_kwargs["auth_config"].api_secret_key == "header-key"
        finally:
            srv._api_key.reset(token)

    async def test_falls_back_to_env_when_contextvar_empty(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "env-key")
        with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
            async with srv._get_client():
                pass
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["auth_config"].api_secret_key == "env-key"

    async def test_raises_when_no_key_available(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_COM_SECRET", raising=False)
        with pytest.raises(RuntimeError, match="No API key found"):
            async with srv._get_client():
                pass

    async def test_uses_contextvar_account_id_over_env(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "key")
        monkeypatch.setenv("PUBLIC_COM_ACCOUNT_ID", "env-account")
        token = srv._account_id.set("header-account")
        try:
            with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
                async with srv._get_client():
                    pass
                call_kwargs = MockClient.call_args.kwargs
                assert call_kwargs["config"].default_account_number == "header-account"
        finally:
            srv._account_id.reset(token)

    async def test_falls_back_to_env_account_when_contextvar_empty(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "key")
        monkeypatch.setenv("PUBLIC_COM_ACCOUNT_ID", "env-account")
        with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
            async with srv._get_client():
                pass
            call_kwargs = MockClient.call_args.kwargs
            assert call_kwargs["config"].default_account_number == "env-account"

    async def test_explicit_account_id_arg_takes_priority(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "key")
        ctx_token = srv._account_id.set("ctx-account")
        try:
            with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
                async with srv._get_client(account_id="arg-account"):
                    pass
                call_kwargs = MockClient.call_args.kwargs
                assert call_kwargs["config"].default_account_number == "arg-account"
        finally:
            srv._account_id.reset(ctx_token)

    async def test_same_secret_reuses_cached_client(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "same-key")
        with patch("publicdotcom_mcp_server.server.AsyncPublicApiClient") as MockClient:
            async with srv._get_client():
                pass
            async with srv._get_client():
                pass
            # Constructor called exactly once despite two _get_client() calls
            assert MockClient.call_count == 1


# ---------------------------------------------------------------------------
# ApiKeyMiddleware — header extraction
# ---------------------------------------------------------------------------

def make_test_app(captured: dict):
    """Build a minimal Starlette app wrapped in ApiKeyMiddleware."""

    async def endpoint(request):
        captured["api_key"] = srv._api_key.get()
        captured["account_id"] = srv._account_id.get()
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route("/", endpoint)])
    return srv.ApiKeyMiddleware(inner)


class TestApiKeyMiddleware:
    def test_extracts_bearer_token(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_COM_SECRET", raising=False)
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/", headers={"Authorization": "Bearer my-api-key"})
        assert captured["api_key"] == "my-api-key"

    def test_extracts_account_id_header(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_COM_ACCOUNT_ID", raising=False)
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/", headers={
            "Authorization": "Bearer key",
            "X-Account-Id": "acct-456",
        })
        assert captured["account_id"] == "acct-456"

    def test_falls_back_to_env_when_no_auth_header(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "env-fallback-key")
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/")
        assert captured["api_key"] == "env-fallback-key"

    def test_header_takes_priority_over_env(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "env-key")
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/", headers={"Authorization": "Bearer header-key"})
        assert captured["api_key"] == "header-key"

    def test_empty_when_no_header_and_no_env(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_COM_SECRET", raising=False)
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/")
        assert captured["api_key"] == ""

    def test_ignores_non_bearer_auth_scheme(self, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_SECRET", "env-key")
        captured = {}
        client = TestClient(make_test_app(captured))
        client.get("/", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        # Non-Bearer scheme → falls back to env
        assert captured["api_key"] == "env-key"

    def test_contextvar_reset_after_request(self, monkeypatch):
        """ContextVar should not leak between requests."""
        monkeypatch.delenv("PUBLIC_COM_SECRET", raising=False)
        captured = {}
        client = TestClient(make_test_app(captured))

        client.get("/", headers={"Authorization": "Bearer request-1-key"})
        assert captured["api_key"] == "request-1-key"

        client.get("/")  # no auth header
        assert captured["api_key"] == ""
