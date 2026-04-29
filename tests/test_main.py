"""Tests for server startup and transport selection."""

from unittest.mock import patch

import publicdotcom_mcp_server.server as srv


class TestMain:
    def test_defaults_to_stdio_transport(self, monkeypatch):
        monkeypatch.delenv("MCP_TRANSPORT", raising=False)

        with patch.object(srv.mcp, "run") as mock_mcp_run, patch(
            "uvicorn.run"
        ) as mock_uvicorn_run:
            srv.main()

        mock_mcp_run.assert_called_once_with(transport="stdio")
        mock_uvicorn_run.assert_not_called()

    def test_uses_streamable_http_when_requested(self, monkeypatch):
        monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
        monkeypatch.setenv("HOST", "127.0.0.1")
        monkeypatch.setenv("PORT", "9001")

        with patch.object(srv.mcp, "run") as mock_mcp_run, patch(
            "uvicorn.run"
        ) as mock_uvicorn_run:
            srv.main()

        mock_mcp_run.assert_not_called()
        mock_uvicorn_run.assert_called_once()

        call_args = mock_uvicorn_run.call_args
        assert call_args.kwargs["host"] == "127.0.0.1"
        assert call_args.kwargs["port"] == 9001
