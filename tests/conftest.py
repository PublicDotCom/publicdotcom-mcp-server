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


@pytest.fixture
def original_order(patch_get_client):
    """The existing order cancel_and_replace_order looks up to preflight its replacement."""
    from decimal import Decimal
    from unittest.mock import MagicMock

    from public_api_sdk import InstrumentType, OrderInstrument
    from public_api_sdk.models import OrderSide

    order = MagicMock()
    order.instrument = OrderInstrument(symbol="AAPL", type=InstrumentType.EQUITY)
    order.side = OrderSide.BUY
    order.open_close_indicator = None
    order.quantity = Decimal("10")
    order.notional_value = None
    patch_get_client.get_order = AsyncMock(return_value=order)
    return order
