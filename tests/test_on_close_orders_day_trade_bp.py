"""Tests for the SDK 0.1.28 sync: on-close order types, day-trade buying power,
trade venue MIC, event-outcome floor strike, and the SDK's query-param and
error-text fixes.

The MCP layer owns one rule here: a LIMIT_ON_CLOSE order needs limit_price, like
LIMIT and STOP_LIMIT. Which order types may carry which prices, and which entry
types a bracket allows, stay in the SDK's request models, so those halves are
exercised with real SDK models and, where the outgoing request matters, a real
AsyncPublicApiClient over a mocked transport.
"""

import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from public_api_sdk import (
    ApiKeyAuthConfig,
    AsyncPublicApiClient,
    AsyncPublicApiClientConfiguration,
)
from public_api_sdk.models import (
    CancelAndReplaceRequest,
    EventDetails,
    OrderRequest,
    OrderType,
    OrderV2,
    Portfolio,
    PreflightRequest,
)

from publicdotcom_mcp_server.server import (
    _validate_order_params,
    cancel_and_replace_order,
    flatten_and_go_short,
    get_all_instruments,
    get_event_details,
    get_order,
    get_portfolio,
    mcp,
    place_order,
    place_short_order,
    preflight_order,
    preflight_short_order,
    search_bonds,
    search_orders,
)

ORDER_ID = "00000000-0000-4000-a000-000000000001"
BASE_URL = "https://gateway.test"


def _get_tool(name: str):
    return next(t for t in mcp._tool_manager._tools.values() if t.fn.__name__ == name)


def _new_order(order_id: str = "returned-id") -> MagicMock:
    new_order = MagicMock()
    new_order.order_id = order_id
    return new_order


def _body(req) -> dict:
    return req.model_dump(by_alias=True, exclude_none=True, mode="json")


def _validate(**overrides) -> None:
    kwargs = {
        "quantity": "10",
        "amount": None,
        "order_type": "MARKET",
        "limit_price": None,
        "stop_price": None,
    }
    kwargs.update(overrides)
    _validate_order_params(**kwargs)


@pytest.fixture
def sdk_client():
    """A real SDK client (so its request builders and validators run), patched in."""
    client = AsyncPublicApiClient(
        auth_config=ApiKeyAuthConfig(api_secret_key="test-secret-key"),
        config=AsyncPublicApiClientConfiguration(default_account_number="acct-1"),
    )
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
        yield client


def _transport_client(status: int = 200, body=None):
    """A real SDK client whose HTTP transport records each request and replies with body."""
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(status, json=body if body is not None else {})

    client = AsyncPublicApiClient(
        auth_config=ApiKeyAuthConfig(api_secret_key="test-secret-key"),
        config=AsyncPublicApiClientConfiguration(
            default_account_number="acct-1", base_url=BASE_URL
        ),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client.auth_manager.refresh_token_if_needed = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client, captured


# ---------------------------------------------------------------------------
# On-close order types
# ---------------------------------------------------------------------------


class TestOnCloseValidation:
    def test_limit_on_close_requires_limit_price(self):
        with pytest.raises(ValueError, match="limit_price is required for LIMIT_ON_CLOSE"):
            _validate(order_type="LIMIT_ON_CLOSE")

    def test_limit_on_close_requirement_is_case_insensitive(self):
        with pytest.raises(ValueError, match="limit_price is required"):
            _validate(order_type="limit_on_close")

    def test_limit_on_close_with_limit_price_passes(self):
        _validate(order_type="LIMIT_ON_CLOSE", limit_price="227.50")

    def test_market_on_close_needs_no_price(self):
        _validate(order_type="MARKET_ON_CLOSE")

    @pytest.mark.parametrize(
        "tool,kwargs",
        [
            (place_order, {"symbol": "AAPL", "instrument_type": "EQUITY", "order_side": "BUY"}),
            (preflight_order, {"symbol": "AAPL", "instrument_type": "EQUITY", "order_side": "BUY"}),
            (cancel_and_replace_order, {"order_id": ORDER_ID}),
            (place_short_order, {"symbol": "AAPL"}),
            (preflight_short_order, {"symbol": "AAPL"}),
            (flatten_and_go_short, {"symbol": "AAPL", "short_quantity": "10"}),
        ],
        ids=lambda v: getattr(v, "__name__", ""),
    )
    async def test_every_order_tool_rejects_limit_on_close_without_limit_price(
        self, patch_get_client, tool, kwargs
    ):
        if "short_quantity" not in kwargs:
            kwargs = {**kwargs, "quantity": "10"}

        result = await tool(order_type="LIMIT_ON_CLOSE", **kwargs)

        assert "limit_price is required for LIMIT_ON_CLOSE" in result
        assert patch_get_client.method_calls == []


class TestOnCloseRequests:
    async def test_place_market_on_close(self, patch_get_client):
        patch_get_client.place_order = AsyncMock(return_value=_new_order())

        data = json.loads(
            await place_order(
                symbol="AAPL",
                instrument_type="EQUITY",
                order_side="BUY",
                order_type="market_on_close",
                quantity="10",
            )
        )

        assert data["status"] == "submitted"
        req = patch_get_client.place_order.call_args.kwargs["order_request"]
        assert isinstance(req, OrderRequest)
        assert req.order_type is OrderType.MARKET_ON_CLOSE
        body = _body(req)
        assert body["orderType"] == "MARKET_ON_CLOSE"
        assert "limitPrice" not in body

    async def test_place_limit_on_close_sends_the_limit_price(self, patch_get_client):
        patch_get_client.place_order = AsyncMock(return_value=_new_order())

        await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="SELL",
            order_type="LIMIT_ON_CLOSE",
            quantity="10",
            limit_price="227.50",
        )

        body = _body(patch_get_client.place_order.call_args.kwargs["order_request"])
        assert body["orderType"] == "LIMIT_ON_CLOSE"
        assert body["limitPrice"] == "227.50"
        assert body["orderSide"] == "SELL"

    async def test_preflight_limit_on_close(self, patch_get_client):
        patch_get_client.perform_preflight_calculation = AsyncMock(
            return_value=MagicMock(model_dump=MagicMock(return_value={"orderValue": "2275"}))
        )

        result = await preflight_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="LIMIT_ON_CLOSE",
            quantity="10",
            limit_price="227.50",
        )

        assert json.loads(result) == {"orderValue": "2275"}
        req = patch_get_client.perform_preflight_calculation.call_args.kwargs[
            "preflight_request"
        ]
        assert isinstance(req, PreflightRequest)
        assert _body(req)["orderType"] == "LIMIT_ON_CLOSE"
        assert _body(req)["limitPrice"] == "227.50"

    async def test_cancel_and_replace_to_limit_on_close(self, patch_get_client):
        patch_get_client.cancel_and_replace_order = AsyncMock(return_value=_new_order())

        data = json.loads(
            await cancel_and_replace_order(
                order_id=ORDER_ID,
                order_type="LIMIT_ON_CLOSE",
                quantity="10",
                limit_price="227.50",
            )
        )

        assert data["status"] == "submitted"
        req = patch_get_client.cancel_and_replace_order.call_args.kwargs["request"]
        assert isinstance(req, CancelAndReplaceRequest)
        body = _body(req)
        assert body["orderType"] == "LIMIT_ON_CLOSE"
        assert body["limitPrice"] == "227.50"

    async def test_cancel_and_replace_to_market_on_close(self, patch_get_client):
        patch_get_client.cancel_and_replace_order = AsyncMock(return_value=_new_order())

        await cancel_and_replace_order(
            order_id=ORDER_ID, order_type="MARKET_ON_CLOSE", quantity="10"
        )

        req = patch_get_client.cancel_and_replace_order.call_args.kwargs["request"]
        assert req.order_type is OrderType.MARKET_ON_CLOSE

    async def test_place_short_order_passes_the_on_close_type(self, patch_get_client):
        patch_get_client.place_short_order = AsyncMock(return_value=_new_order())

        await place_short_order(
            symbol="AAPL", quantity="10", order_type="LIMIT_ON_CLOSE", limit_price="230"
        )

        kwargs = patch_get_client.place_short_order.call_args.kwargs
        assert kwargs["order_type"] is OrderType.LIMIT_ON_CLOSE
        assert kwargs["limit_price"] == Decimal("230")


class TestOnCloseRulesOwnedBySdk:
    """Rules the MCP layer leaves to the SDK's request models."""

    async def test_market_on_close_with_limit_price_is_rejected(self, patch_get_client):
        data = json.loads(
            await place_order(
                symbol="AAPL",
                instrument_type="EQUITY",
                order_side="BUY",
                order_type="MARKET_ON_CLOSE",
                quantity="10",
                limit_price="227.50",
            )
        )

        assert data["status"] == "error"
        assert "LIMIT_ON_CLOSE" in data["message"]
        assert "not MARKET_ON_CLOSE" in data["message"]
        patch_get_client.place_order.assert_not_called()

    async def test_preflight_market_on_close_with_limit_price_is_rejected(
        self, patch_get_client
    ):
        result = await preflight_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="MARKET_ON_CLOSE",
            quantity="10",
            limit_price="227.50",
        )

        assert result.startswith("Error:")
        assert "not MARKET_ON_CLOSE" in result
        patch_get_client.perform_preflight_calculation.assert_not_called()

    async def test_on_close_stop_price_is_rejected(self, patch_get_client):
        data = json.loads(
            await place_order(
                symbol="AAPL",
                instrument_type="EQUITY",
                order_side="BUY",
                order_type="LIMIT_ON_CLOSE",
                quantity="10",
                limit_price="227.50",
                stop_price="220",
            )
        )

        assert data["status"] == "error"
        assert "stop_price" in data["message"]
        patch_get_client.place_order.assert_not_called()

    @pytest.mark.parametrize("entry_type", ["MARKET_ON_CLOSE", "LIMIT_ON_CLOSE"])
    async def test_bracket_entry_cannot_be_on_close(self, patch_get_client, entry_type):
        data = json.loads(
            await place_order(
                symbol="AAPL",
                instrument_type="EQUITY",
                order_side="BUY",
                order_type=entry_type,
                quantity="10",
                limit_price="200" if entry_type == "LIMIT_ON_CLOSE" else None,
                equity_market_session="CORE",
                order_class="BRACKET",
                take_profit_limit_price="215",
                stop_loss_stop_price="190",
            )
        )

        assert data["status"] == "error"
        assert "entry order type" in data["message"]
        patch_get_client.place_order.assert_not_called()

    async def test_short_order_built_by_sdk_carries_the_on_close_type(self, sdk_client):
        sdk_client.place_order = AsyncMock(return_value=_new_order())

        await place_short_order(
            symbol="AAPL", quantity="5", order_type="LIMIT_ON_CLOSE", limit_price="230"
        )

        body = _body(sdk_client.place_order.call_args.args[0])
        assert body["orderType"] == "LIMIT_ON_CLOSE"
        assert body["limitPrice"] == "230.00"  # the SDK quantizes equity prices
        assert body["orderSide"] == "SELL"
        assert body["openCloseIndicator"] == "OPEN"

    async def test_short_preflight_market_on_close_with_limit_price_is_rejected(
        self, sdk_client
    ):
        sdk_client.perform_preflight_calculation = AsyncMock()

        result = await preflight_short_order(
            symbol="AAPL", quantity="5", order_type="MARKET_ON_CLOSE", limit_price="230"
        )

        assert result.startswith("Error:")
        assert "not MARKET_ON_CLOSE" in result
        sdk_client.perform_preflight_calculation.assert_not_called()


class TestOnCloseDocs:
    @pytest.mark.parametrize(
        "name",
        [
            "place_order",
            "preflight_order",
            "cancel_and_replace_order",
            "place_short_order",
            "preflight_short_order",
            "flatten_and_go_short",
        ],
    )
    def test_order_type_docs_list_the_on_close_types(self, name):
        description = _get_tool(name).description
        assert "MARKET_ON_CLOSE" in description
        assert "LIMIT_ON_CLOSE" in description


# ---------------------------------------------------------------------------
# New response fields
# ---------------------------------------------------------------------------


def _portfolio(buying_power: dict) -> Portfolio:
    return Portfolio.model_validate(
        {
            "accountId": "acct-1",
            "accountType": "BROKERAGE",
            "buyingPower": buying_power,
            "equity": [],
            "positions": [],
            "orders": [],
        }
    )


_BUYING_POWER = {
    "cashOnlyBuyingPower": "10000.00",
    "buyingPower": "20000.00",
    "optionsBuyingPower": "5000.00",
}


class TestDayTradeBuyingPower:
    async def test_day_trade_buying_power_is_shown(self, patch_get_client):
        patch_get_client.get_portfolio = AsyncMock(
            return_value=_portfolio({**_BUYING_POWER, "dayTradeBuyingPower": "80000.00"})
        )

        data = json.loads(await get_portfolio())

        assert data["buyingPower"]["dayTradeBuyingPower"] == "80000.00"
        assert data["buyingPower"]["buyingPower"] == "20000.00"

    async def test_missing_day_trade_buying_power_is_omitted(self, patch_get_client):
        patch_get_client.get_portfolio = AsyncMock(return_value=_portfolio(_BUYING_POWER))

        data = json.loads(await get_portfolio())

        assert "dayTradeBuyingPower" not in data["buyingPower"]

    def test_get_portfolio_docs_mention_it(self):
        assert "dayTradeBuyingPower" in _get_tool("get_portfolio").description


def _order_v2(mic: str | None) -> OrderV2:
    trade = {
        "instrument": {"symbol": "AAPL", "type": "EQUITY"},
        "quantity": "10",
        "price": "199.50",
        "side": "BUY",
        "tradeId": "trade-uuid-1",
        "timestamp": "2026-10-28T14:31:00Z",
    }
    if mic is not None:
        trade["marketIdentificationCode"] = mic
    return OrderV2.model_validate(
        {
            "orderId": "order-uuid-1",
            "instrument": {"symbol": "AAPL", "type": "EQUITY"},
            "type": "MARKET_ON_CLOSE",
            "side": "BUY",
            "status": "FILLED",
            "createdAt": "2026-10-28T14:30:00Z",
            "quantity": "10",
            "filledQuantity": "10",
            "averagePrice": "199.50",
            "expiration": {"timeInForce": "DAY"},
            "trades": [trade],
        }
    )


class TestTradeVenueMic:
    async def test_get_order_shows_the_trade_mic(self, patch_get_client):
        patch_get_client.get_order = AsyncMock(return_value=_order_v2("XNAS"))

        data = json.loads(await get_order("order-uuid-1"))

        assert data["type"] == "MARKET_ON_CLOSE"
        assert data["trades"][0]["marketIdentificationCode"] == "XNAS"

    async def test_search_orders_shows_the_trade_mic(self, patch_get_client):
        patch_get_client.search_orders = AsyncMock(return_value=[_order_v2("ARCX")])

        data = json.loads(await search_orders())

        assert data[0]["trades"][0]["marketIdentificationCode"] == "ARCX"

    async def test_missing_mic_is_omitted(self, patch_get_client):
        patch_get_client.get_order = AsyncMock(return_value=_order_v2(None))

        data = json.loads(await get_order("order-uuid-1"))

        assert "marketIdentificationCode" not in data["trades"][0]

    @pytest.mark.parametrize("name", ["get_order", "search_orders"])
    def test_docs_mention_it(self, name):
        assert "marketIdentificationCode" in _get_tool(name).description


class TestFloorStrike:
    def _details(self, **outcome_extra) -> EventDetails:
        return EventDetails.model_validate(
            {
                "eventSymbol": "KALSHI.KXBALANCESHEET-EO26",
                "outcomes": [
                    {
                        "outcomeId": "KXBALANCESHEET-EO26-6.6",
                        "title": "ABOVE $6.6 TRILLION",
                        "state": "STATE_OPEN",
                        "contracts": [],
                        **outcome_extra,
                    }
                ],
            }
        )

    async def test_floor_strike_is_shown(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(
            return_value=self._details(floorStrike="6.6")
        )

        data = json.loads(await get_event_details("KALSHI.KXBALANCESHEET-EO26"))

        assert data["outcomes"][0]["floorStrike"] == "6.6"
        # The SDK's derived shortcuts are still added alongside it.
        assert "tradable" in data["outcomes"][0]

    async def test_missing_floor_strike_is_omitted(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(return_value=self._details())

        data = json.loads(await get_event_details("KALSHI.KXBALANCESHEET-EO26"))

        assert "floorStrike" not in data["outcomes"][0]

    def test_docs_mention_it(self):
        assert "floorStrike" in _get_tool("get_event_details").description


# ---------------------------------------------------------------------------
# SDK fixes the tools now rely on
# ---------------------------------------------------------------------------


class TestEnumFiltersReachTheGateway:
    """SDK 0.1.27's async client sent enum filters as e.g. 'InstrumentType.CRYPTO',
    which the gateway rejected with HTTP 400. Through the tools, they now go out as
    their values."""

    async def test_get_all_instruments_filters(self):
        client, captured = _transport_client(body={"instruments": []})
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            result = await get_all_instruments(
                type_filter=["crypto", "EQUITY"], trading_filter=["buy_and_sell"]
            )

        assert json.loads(result) == {"instruments": []}
        params = captured[0].url.params
        assert params.get_list("typeFilter") == ["CRYPTO", "EQUITY"]
        assert params.get_list("tradingFilter") == ["BUY_AND_SELL"]
        assert "InstrumentType" not in str(captured[0].url)

    async def test_search_bonds_filters(self):
        client, captured = _transport_client(
            body={"content": [], "pageNumber": 0, "pageSize": 0, "totalElements": 0}
        )
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            await search_bonds(
                sort_direction="asc",
                bond_type=["treasury"],
                bond_status=["OUTSTANDING"],
                rating_category="investment_grade",
                coupon_frequency=["SEMI_ANNUAL"],
            )

        params = captured[0].url.params
        assert params["sortDirection"] == "ASC"
        assert params.get_list("bondType") == ["TREASURY"]
        assert params.get_list("bondStatus") == ["OUTSTANDING"]
        assert params["ratingCategory"] == "INVESTMENT_GRADE"
        assert params.get_list("couponFrequency") == ["SEMI_ANNUAL"]
        assert "BondType" not in str(captured[0].url)


class TestGatewayErrorText:
    """API errors now carry the gateway's own text instead of 'Unknown error'."""

    @pytest.mark.parametrize(
        "body,expected",
        [
            ({"error": "typeFilter must be one of EQUITY, CRYPTO"}, "typeFilter must be one of"),
            ({"detail": "Account not enabled for bonds"}, "Account not enabled for bonds"),
        ],
    )
    async def test_read_tool_reports_the_gateway_text(self, body, expected):
        client, _ = _transport_client(status=400, body=body)
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            result = await get_all_instruments(type_filter=["CRYPTO"])

        assert result.startswith("Error:")
        assert expected in result
        assert "Unknown error" not in result

    async def test_order_tool_reports_the_gateway_text(self):
        client, _ = _transport_client(status=400, body={"error": "Market is closed"})
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            data = json.loads(
                await place_order(
                    symbol="AAPL",
                    instrument_type="EQUITY",
                    order_side="BUY",
                    order_type="MARKET_ON_CLOSE",
                    quantity="1",
                )
            )

        assert data["status"] == "error"
        assert "Market is closed" in data["message"]
        assert "Unknown error" not in data["message"]
