"""Tests for the order-history tools search_orders and get_order (SDK 0.1.26).

search_orders queries /order/search; get_order now returns the SDK's OrderV2 (the
order plus its fills). The SDK's deprecated get_order_v2 alias is deliberately not
exposed as a tool.

The MCP layer only checks that the search arguments cohere — enum values parse,
symbol specs are well-formed, timestamps are ISO 8601. The 30-day window and the
500-order cap are the API's, and the field semantics live in the SDK's
OrderSearchRequest, so both halves are exercised here with the real SDK models.
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from public_api_sdk.models import (
    InstrumentType,
    OpenCloseIndicator,
    OrderMarketSession,
    OrderSearchRequest,
    OrderSide,
    OrderStatus,
    OrderV2,
)

from publicdotcom_mcp_server.server import (
    _build_order_search_request,
    _parse_symbol_spec,
    _parse_timestamp,
    _serialize,
    get_order,
    search_orders,
)


def _search_request(**overrides):
    kwargs = {
        "status": None,
        "side": None,
        "symbols": None,
        "security_type": None,
        "open_close_indicator": None,
        "created_after": None,
        "created_before": None,
    }
    kwargs.update(overrides)
    return _build_order_search_request(**kwargs)


def _order_v2_payload(**overrides) -> dict:
    """A filled AAPL limit order as get-order / order-search return it, with one trade."""
    payload = {
        "orderId": "order-uuid-1",
        "instrument": {"symbol": "AAPL", "type": "EQUITY"},
        "type": "LIMIT",
        "side": "BUY",
        "status": "FILLED",
        "createdAt": "2026-09-20T14:30:00Z",
        "quantity": "10",
        "limitPrice": "200.00",
        "filledQuantity": "10",
        "averagePrice": "199.50",
        "closedAt": "2026-09-20T14:31:00Z",
        "filledAt": "2026-09-20T14:31:00Z",
        "lastModified": "2026-09-20T14:31:00Z",
        "equityMarketSession": "REGULAR",
        "expiration": {"timeInForce": "DAY"},
        "trades": [
            {
                "instrument": {"symbol": "AAPL", "type": "EQUITY"},
                "quantity": "10",
                "price": "199.50",
                "side": "BUY",
                "tradeId": "trade-uuid-1",
                "timestamp": "2026-09-20T14:31:00Z",
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestParseTimestamp:
    def test_offset_form_is_accepted(self):
        parsed = _parse_timestamp("created_after", "2026-09-01T09:00:00-05:00")
        assert parsed.utcoffset() == timedelta(hours=-5)

    def test_trailing_z_means_utc_on_every_python(self):
        # datetime.fromisoformat only learned 'Z' in 3.11; CI also runs 3.10.
        parsed = _parse_timestamp("created_after", "2026-09-01T00:00:00Z")
        assert parsed == datetime(2026, 9, 1, tzinfo=timezone.utc)

    def test_garbage_names_the_field(self):
        with pytest.raises(ValueError, match="created_before must be an ISO 8601"):
            _parse_timestamp("created_before", "last tuesday")


class TestParseSymbolSpec:
    def test_bare_symbol_defaults_to_equity(self):
        inst = _parse_symbol_spec("AAPL")
        assert (inst.symbol, inst.type) == ("AAPL", InstrumentType.EQUITY)

    def test_explicit_type_is_parsed_case_insensitively(self):
        inst = _parse_symbol_spec("SPY260313P00670000:option")
        assert (inst.symbol, inst.type) == ("SPY260313P00670000", InstrumentType.OPTION)

    def test_unknown_type_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Invalid instrument type 'STOCK'") as exc:
            _parse_symbol_spec("AAPL:STOCK")
        assert "EQUITY" in str(exc.value)

    @pytest.mark.parametrize("spec", ["", ":EQUITY", "AAPL:", "AAPL:EQUITY:BUY"])
    def test_malformed_spec_is_rejected(self, spec):
        with pytest.raises(ValueError, match="Invalid symbol spec"):
            _parse_symbol_spec(spec)


class TestBuildOrderSearchRequest:
    def test_no_filters_yields_none_so_the_sdk_sends_its_empty_request(self):
        assert _search_request() is None
        # The SDK substitutes OrderSearchRequest(), which serializes to {}.
        assert OrderSearchRequest().model_dump(by_alias=True, exclude_none=True) == {}

    def test_every_filter_maps_to_the_sdk_field(self):
        req = _search_request(
            status="filled",
            side="buy",
            symbols=["AAPL", "BTC:CRYPTO"],
            security_type="equity",
            open_close_indicator="open",
            created_after="2026-09-01T00:00:00Z",
            created_before="2026-09-15T00:00:00Z",
        )
        assert req.status is OrderStatus.FILLED
        assert req.side is OrderSide.BUY
        assert [(i.symbol, i.type) for i in req.instruments] == [
            ("AAPL", InstrumentType.EQUITY),
            ("BTC", InstrumentType.CRYPTO),
        ]
        assert req.security_type is InstrumentType.EQUITY
        assert req.open_close_indicator is OpenCloseIndicator.OPEN
        assert req.created_after == datetime(2026, 9, 1, tzinfo=timezone.utc)
        assert req.created_before == datetime(2026, 9, 15, tzinfo=timezone.utc)

    def test_request_serializes_to_the_api_field_names(self):
        payload = _search_request(
            status="CANCELLED",
            symbols=["AAPL"],
            open_close_indicator="CLOSE",
            created_after="2026-09-01T00:00:00+00:00",
        ).model_dump(by_alias=True, exclude_none=True, mode="json")
        assert payload == {
            "status": "CANCELLED",
            "instruments": [{"symbol": "AAPL", "type": "EQUITY"}],
            "openCloseIndicator": "CLOSE",
            "createdAfter": "2026-09-01T00:00:00+00:00",
        }

    def test_invalid_status_lists_the_real_statuses_but_not_unknown(self):
        with pytest.raises(ValueError, match="Invalid status: 'OPEN'") as exc:
            _search_request(status="OPEN")
        message = str(exc.value)
        for valid in ("NEW", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"):
            assert valid in message
        assert "UNKNOWN" not in message

    def test_unknown_status_is_refused(self):
        # UNKNOWN is the SDK's client-side fallback, never an API value — and
        # OrderStatus("anything") silently becomes UNKNOWN, so it must not slip
        # through as a filter.
        with pytest.raises(ValueError, match="Invalid status: 'UNKNOWN'"):
            _search_request(status="UNKNOWN")

    def test_invalid_side_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid side: 'LONG'. Expected BUY or SELL"):
            _search_request(side="LONG")

    def test_invalid_open_close_indicator_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid open_close_indicator: 'BOTH'"):
            _search_request(open_close_indicator="BOTH")

    def test_invalid_security_type_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid instrument type 'STOCK'"):
            _search_request(security_type="STOCK")

    def test_event_contract_security_type_and_symbol_spec_are_accepted(self):
        req = _search_request(
            security_type="eventcontract",
            symbols=["KALSHI.KXBALANCESHEET-EO26-6.6.Y-EVENTCONTRACT:EVENTCONTRACT"],
        )
        assert req.security_type is InstrumentType.EVENTCONTRACT
        assert req.instruments[0].type is InstrumentType.EVENTCONTRACT
        assert req.instruments[0].symbol == "KALSHI.KXBALANCESHEET-EO26-6.6.Y-EVENTCONTRACT"

    def test_empty_symbols_list_applies_no_instrument_filter(self):
        assert _search_request(symbols=[]) is None


class TestSearchOrdersTool:
    """search_orders end-to-end with the client mocked, asserting the request sent."""

    @staticmethod
    def _mock_search(patch_get_client, orders=None):
        patch_get_client.search_orders = AsyncMock(return_value=orders or [])
        return patch_get_client

    async def test_unfiltered_call_sends_no_request_and_forwards_the_account(
        self, patch_get_client
    ):
        mock_client = self._mock_search(patch_get_client)

        result = await search_orders(account_id="acct-9")

        assert json.loads(result) == []
        kwargs = mock_client.search_orders.call_args.kwargs
        assert kwargs == {"order_search_request": None, "account_id": "acct-9"}

    async def test_filters_reach_the_sdk_as_an_order_search_request(self, patch_get_client):
        mock_client = self._mock_search(patch_get_client)

        await search_orders(status="filled", side="SELL", symbols=["TSLA"])

        req = mock_client.search_orders.call_args.kwargs["order_search_request"]
        assert isinstance(req, OrderSearchRequest)
        assert req.status is OrderStatus.FILLED
        assert req.side is OrderSide.SELL
        assert req.instruments[0].symbol == "TSLA"

    async def test_results_are_serialized_in_the_v2_shape(self, patch_get_client):
        order = OrderV2.model_validate(_order_v2_payload())
        self._mock_search(patch_get_client, orders=[order])

        data = json.loads(await search_orders())

        assert len(data) == 1
        got = data[0]
        assert got["orderId"] == "order-uuid-1"
        assert got["status"] == "FILLED"
        assert got["equityMarketSession"] == "REGULAR"
        assert got["filledAt"] == "2026-09-20T14:31:00+00:00"
        assert got["lastModified"] == "2026-09-20T14:31:00+00:00"
        assert "replacedAt" not in got  # exclude_none keeps the payload lean
        assert got["trades"] == [
            {
                "instrument": {"symbol": "AAPL", "type": "EQUITY"},
                "quantity": "10",
                "price": "199.50",
                "side": "BUY",
                "tradeId": "trade-uuid-1",
                "timestamp": "2026-09-20T14:31:00+00:00",
            }
        ]

    async def test_bad_argument_is_reported_not_raised(self, patch_get_client):
        mock_client = self._mock_search(patch_get_client)

        result = await search_orders(status="OPEN")

        assert result.startswith("Error:")
        assert "Invalid status: 'OPEN'" in result
        mock_client.search_orders.assert_not_called()

    async def test_api_error_returns_error_string(self, patch_get_client):
        patch_get_client.search_orders = AsyncMock(side_effect=Exception("account not found"))

        result = await search_orders()

        assert result.startswith("Error:")
        assert "account not found" in result


class TestGetOrderTool:
    async def test_returns_the_order_with_its_fills(self, patch_get_client):
        order = OrderV2.model_validate(_order_v2_payload())
        patch_get_client.get_order = AsyncMock(return_value=order)

        data = json.loads(await get_order("order-uuid-1", account_id="acct-9"))

        assert patch_get_client.get_order.call_args.kwargs == {
            "order_id": "order-uuid-1",
            "account_id": "acct-9",
        }
        assert data["orderId"] == "order-uuid-1"
        assert data["averagePrice"] == "199.50"
        assert data["equityMarketSession"] == "REGULAR"
        assert data["filledAt"] == "2026-09-20T14:31:00+00:00"
        assert data["trades"][0]["tradeId"] == "trade-uuid-1"

    async def test_bracket_id_survives_serialization(self, patch_get_client):
        order = OrderV2.model_validate(_order_v2_payload(bracketId="entry-order-uuid"))
        patch_get_client.get_order = AsyncMock(return_value=order)

        data = json.loads(await get_order("order-uuid-1"))

        assert data["bracketId"] == "entry-order-uuid"

    async def test_does_not_call_the_deprecated_v2_alias(self, patch_get_client):
        patch_get_client.get_order = AsyncMock(
            return_value=OrderV2.model_validate(_order_v2_payload())
        )

        await get_order("order-uuid-1")

        patch_get_client.get_order_v2.assert_not_called()

    async def test_not_found_returns_error_string(self, patch_get_client):
        patch_get_client.get_order = AsyncMock(side_effect=Exception("Order not found"))

        result = await get_order("missing-order")

        assert result.startswith("Error:")
        assert "Order not found" in result


def test_get_order_v2_is_not_exposed_as_a_tool():
    import publicdotcom_mcp_server.server as server

    assert not hasattr(server, "get_order_v2")


class TestOrderV2Model:
    """The SDK contract the tools rely on, pinned so a future SDK bump fails loudly."""

    def test_v2_is_a_superset_of_the_v1_order(self):
        from public_api_sdk.models import Order

        order = OrderV2.model_validate(_order_v2_payload())
        assert isinstance(order, Order)
        assert order.trades[0].price == Decimal("199.50")
        assert order.equity_market_session is OrderMarketSession.REGULAR

    def test_response_session_vocabulary_is_not_the_request_one(self):
        # The API reports REGULAR / REST_OF_DAY / TWENTY_FOUR_HOURS on orders but
        # accepts CORE / EXTENDED / TWENTY_FOUR_HOURS when placing them. Mapping one
        # through the other would break, so the tools never do.
        from public_api_sdk.models import EquityMarketSession

        assert {s.value for s in OrderMarketSession} >= {"REGULAR", "REST_OF_DAY"}
        assert {s.value for s in EquityMarketSession} == {"CORE", "EXTENDED", "TWENTY_FOUR_HOURS"}
        assert "REGULAR" not in {s.value for s in EquityMarketSession}

    def test_unrecognised_session_falls_back_to_unknown_instead_of_failing(self):
        order = OrderV2.model_validate(_order_v2_payload(equityMarketSession="PRE_MARKET"))
        assert order.equity_market_session is OrderMarketSession.UNKNOWN
        assert json.loads(_serialize(order))["equityMarketSession"] == "UNKNOWN"
