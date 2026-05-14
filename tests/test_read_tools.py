"""Tests for read-only MCP tools."""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from publicdotcom_mcp_server.server import (
    check_setup,
    get_accounts,
    get_all_instruments,
    get_historic_bars,
    get_history,
    get_instrument,
    get_option_chain,
    get_option_expirations,
    get_option_greeks,
    get_order,
    get_orders,
    get_portfolio,
    get_quotes,
)


def _make_model(data: dict) -> MagicMock:
    """Create a mock Pydantic-like model with model_dump()."""
    m = MagicMock()
    m.model_dump.return_value = data
    return m


class TestGetAccounts:
    async def test_returns_serialized_accounts(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_accounts = AsyncMock(return_value=_make_model({"accounts": []}))

        result = await get_accounts()
        assert '"accounts"' in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_accounts = AsyncMock(side_effect=Exception("network timeout"))

        result = await get_accounts()
        assert "Error" in result
        assert "network timeout" in result

    async def test_missing_secret_returns_error(self, monkeypatch, patch_get_client):
        # When _get_client raises (e.g., missing secret), tool returns error string
        from unittest.mock import patch as mock_patch
        with mock_patch(
            "publicdotcom_mcp_server.server._get_client",
            side_effect=RuntimeError("PUBLIC_COM_SECRET is not set"),
        ):
            result = await get_accounts()
        assert "Error" in result
        assert "PUBLIC_COM_SECRET" in result


class TestGetPortfolio:
    async def test_returns_serialized_portfolio(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_portfolio = AsyncMock(
            return_value=_make_model({"equity": "10000.00"})
        )

        result = await get_portfolio()
        assert '"equity"' in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_portfolio = AsyncMock(side_effect=Exception("API error"))

        result = await get_portfolio()
        assert "Error" in result


class TestGetOrders:
    async def test_returns_only_open_orders(self, patch_get_client):
        mock_client = patch_get_client
        mock_order = _make_model({"order_id": "abc-123", "status": "OPEN"})
        mock_portfolio = MagicMock()
        mock_portfolio.orders = [mock_order]
        mock_client.get_portfolio = AsyncMock(return_value=mock_portfolio)

        result = await get_orders()
        data = json.loads(result)
        assert len(data) == 1
        assert data[0]["order_id"] == "abc-123"

    async def test_returns_empty_list_when_no_orders(self, patch_get_client):
        mock_client = patch_get_client
        mock_portfolio = MagicMock()
        mock_portfolio.orders = None
        mock_client.get_portfolio = AsyncMock(return_value=mock_portfolio)

        result = await get_orders()
        assert result == "[]"

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_portfolio = AsyncMock(side_effect=Exception("timeout"))

        result = await get_orders()
        assert "Error" in result


class TestGetOrder:
    async def test_returns_order_details(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_order = AsyncMock(
            return_value=_make_model({"order_id": "order-uuid", "status": "FILLED"})
        )

        result = await get_order(order_id="order-uuid")
        assert '"order_id"' in result
        assert "order-uuid" in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_order = AsyncMock(side_effect=Exception("not found"))

        result = await get_order(order_id="bad-id")
        assert "Error" in result


class TestGetQuotes:
    async def test_returns_quotes(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_quotes = AsyncMock(
            return_value=_make_model({"quotes": [{"symbol": "AAPL", "last": "200.00"}]})
        )

        result = await get_quotes(symbols=["AAPL"])
        assert "AAPL" in result

    async def test_invalid_instrument_type_returns_error(self, patch_get_client):
        result = await get_quotes(symbols=["AAPL"], instrument_type="INVALID")
        assert "Error" in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_quotes = AsyncMock(side_effect=Exception("rate limited"))

        result = await get_quotes(symbols=["AAPL"])
        assert "Error" in result


class TestGetHistoricBars:
    async def test_returns_bars(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_bars = AsyncMock(
            return_value=_make_model({"bars": [{"close": "200.00"}]})
        )

        result = await get_historic_bars(symbol="AAPL", period="DAY")
        assert '"close"' in result

    async def test_defaults_to_equity(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_bars = AsyncMock(return_value=_make_model({"bars": []}))

        await get_historic_bars(symbol="AAPL", period="DAY")
        from public_api_sdk import InstrumentType
        assert mock_client.get_bars.await_args.kwargs["instrument_type"] == InstrumentType.EQUITY

    async def test_passes_aggregation_and_instrument_type(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_bars = AsyncMock(return_value=_make_model({"bars": []}))

        await get_historic_bars(
            symbol="BTC",
            period="WEEK",
            instrument_type="CRYPTO",
            aggregation="ONE_HOUR",
        )
        from public_api_sdk import BarAggregation, BarPeriod, InstrumentType
        kwargs = mock_client.get_bars.await_args.kwargs
        assert kwargs["instrument_type"] == InstrumentType.CRYPTO
        assert kwargs["aggregation"] == BarAggregation.ONE_HOUR
        assert kwargs["period"] == BarPeriod.WEEK

    async def test_invalid_period_returns_error(self, patch_get_client):
        result = await get_historic_bars(symbol="AAPL", period="NOPE")
        assert "Error" in result
        assert "period" in result.lower()

    async def test_invalid_instrument_type_returns_error(self, patch_get_client):
        result = await get_historic_bars(
            symbol="AAPL", period="DAY", instrument_type="INVALID"
        )
        assert "Error" in result

    async def test_invalid_aggregation_returns_error(self, patch_get_client):
        result = await get_historic_bars(
            symbol="AAPL", period="DAY", aggregation="WHENEVER"
        )
        assert "Error" in result
        assert "aggregation" in result.lower()

    async def test_since_purchase_requires_purchase_date(self, patch_get_client):
        result = await get_historic_bars(symbol="AAPL", period="SINCE_PURCHASE")
        assert "Error" in result
        assert "purchase_date" in result

    async def test_since_purchase_with_date_succeeds(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_bars = AsyncMock(return_value=_make_model({"bars": []}))

        await get_historic_bars(
            symbol="AAPL", period="SINCE_PURCHASE", purchase_date="2025-01-01"
        )
        assert mock_client.get_bars.await_args.kwargs["purchase_date"] == "2025-01-01"

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_bars = AsyncMock(side_effect=Exception("upstream 500"))

        result = await get_historic_bars(symbol="AAPL", period="DAY")
        assert "Error" in result
        assert "upstream 500" in result


class TestGetHistory:
    async def test_returns_history(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_history = AsyncMock(
            return_value=_make_model({"events": []})
        )

        result = await get_history()
        assert '"events"' in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_history = AsyncMock(side_effect=Exception("server error"))

        result = await get_history()
        assert "Error" in result


class TestGetInstrument:
    async def test_returns_instrument_details(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_instrument = AsyncMock(
            return_value=_make_model({"symbol": "AAPL", "tradeable": True})
        )

        result = await get_instrument(symbol="AAPL")
        assert "AAPL" in result

    async def test_invalid_instrument_type_returns_error(self, patch_get_client):
        result = await get_instrument(symbol="AAPL", instrument_type="INVALID")
        assert "Error" in result


class TestGetOptionGreeks:
    async def test_returns_greeks(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_greeks = AsyncMock(
            return_value=_make_model({"greeks": []})
        )

        result = await get_option_greeks(osi_symbols=["AAPL260320C00280000"])
        assert '"greeks"' in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_greeks = AsyncMock(side_effect=Exception("bad symbol"))

        result = await get_option_greeks(osi_symbols=["INVALID"])
        assert "Error" in result


class TestCheckSetup:
    async def test_missing_secret_returns_error_without_api_call(self, monkeypatch):
        monkeypatch.delenv("PUBLIC_COM_SECRET", raising=False)
        result = await check_setup()
        assert "PUBLIC_COM_SECRET" in result
        assert "❌" in result

    async def test_successful_auth_returns_account_list(self, patch_get_client):
        mock_client = patch_get_client
        mock_account = MagicMock()
        mock_account.account_id = "acct-123"
        mock_account.account_type.value = "BROKERAGE"
        mock_accounts = MagicMock()
        mock_accounts.accounts = [mock_account]
        mock_client.get_accounts = AsyncMock(return_value=mock_accounts)

        result = await check_setup()
        assert "✅" in result
        assert "acct-123" in result
        assert "BROKERAGE" in result

    async def test_api_error_returns_auth_failed_message(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_accounts = AsyncMock(side_effect=Exception("invalid key"))

        result = await check_setup()
        assert "❌" in result
        assert "Authentication failed" in result
        assert "invalid key" in result


class TestGetAllInstruments:
    async def test_returns_instruments(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_all_instruments = AsyncMock(
            return_value=_make_model({"instruments": []})
        )

        result = await get_all_instruments()
        assert '"instruments"' in result

    async def test_invalid_type_filter_returns_error(self, patch_get_client):
        result = await get_all_instruments(type_filter=["INVALID"])
        assert "Error" in result

    async def test_invalid_trading_filter_returns_error(self, patch_get_client):
        result = await get_all_instruments(trading_filter=["INVALID_TRADING_STATUS"])
        assert "Error" in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_all_instruments = AsyncMock(side_effect=Exception("timeout"))

        result = await get_all_instruments()
        assert "Error" in result


class TestGetOptionExpirations:
    async def test_returns_expirations(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_expirations = AsyncMock(
            return_value=_make_model({"expiration_dates": ["2026-03-21", "2026-04-17"]})
        )

        result = await get_option_expirations(symbol="AAPL")
        assert "expiration_dates" in result

    async def test_invalid_instrument_type_returns_error(self, patch_get_client):
        result = await get_option_expirations(symbol="AAPL", instrument_type="INVALID")
        assert "Error" in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_expirations = AsyncMock(side_effect=Exception("not found"))

        result = await get_option_expirations(symbol="AAPL")
        assert "Error" in result


class TestGetOptionChain:
    async def test_returns_option_chain(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_chain = AsyncMock(
            return_value=_make_model({"calls": [], "puts": []})
        )

        result = await get_option_chain(symbol="AAPL", expiration_date="2026-03-21")
        assert "calls" in result
        assert "puts" in result

    async def test_invalid_instrument_type_returns_error(self, patch_get_client):
        result = await get_option_chain(
            symbol="AAPL", expiration_date="2026-03-21", instrument_type="INVALID"
        )
        assert "Error" in result

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.get_option_chain = AsyncMock(side_effect=Exception("no chain"))

        result = await get_option_chain(symbol="AAPL", expiration_date="2026-03-21")
        assert "Error" in result
