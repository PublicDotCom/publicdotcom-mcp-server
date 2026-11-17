"""Tests for the order gate: every order-creating tool preflights its exact order,
checks it against the operator's limits and asks the user to confirm before
anything reaches the trading API.

The scenarios mirror the external report that prompted the gate: a sell-to-open
of 1,000 option contracts and a two-leg spread with both legs sold, each placed
without a preflight or a confirmation.
"""

import inspect
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types
from mcp.shared.memory import create_connected_server_and_client_session
from public_api_sdk import InstrumentType, OrderInstrument
from public_api_sdk.models import (
    MultilegOrderRequest,
    OrderRequest,
    PreflightMultiLegRequest,
    PreflightRequest,
    PreflightResponse,
    WaitTimeoutError,
)

from publicdotcom_mcp_server import server
from publicdotcom_mcp_server.server import (
    _apply_read_only_mode,
    buy_event_contract,
    cancel_and_replace_order,
    flatten_and_go_short,
    mcp,
    place_call_credit_spread,
    place_multileg_order,
    place_order,
    place_short_order,
)

CALL_260 = "AAPL261120C00260000"
CALL_270 = "AAPL261120C00270000"

SELL_TO_OPEN_1000 = {
    "symbol": CALL_260,
    "instrument_type": "OPTION",
    "order_side": "SELL",
    "order_type": "LIMIT",
    "limit_price": "3.00",
    "open_close_indicator": "OPEN",
    "time_in_force": "DAY",
    "quantity": "1000",
}

BOTH_LEGS_SOLD = {
    "legs": [
        {"symbol": CALL_260, "type": "OPTION", "side": "SELL",
         "open_close_indicator": "OPEN", "ratio_quantity": 1},
        {"symbol": CALL_270, "type": "OPTION", "side": "SELL",
         "open_close_indicator": "OPEN", "ratio_quantity": 1},
    ],
    "quantity": 1,
    "limit_price": "-4.00",
}


def _preflight(**fields) -> PreflightResponse:
    return PreflightResponse.model_validate(
        {"instrument": {"symbol": CALL_260, "type": "OPTION"}, "orderValue": "300", **fields}
    )


def _new_order(order_id: str = "returned-id") -> MagicMock:
    order = MagicMock()
    order.order_id = order_id
    return order


def _elicit_ctx(action: str = "accept", confirm: bool = True, error: Exception | None = None):
    """A stand-in Context whose client supports elicitation."""
    ctx = MagicMock()
    ctx.session.client_params.capabilities.elicitation = types.ElicitationCapability()
    if error is not None:
        ctx.elicit = AsyncMock(side_effect=error)
    else:
        data = SimpleNamespace(confirm=confirm) if action == "accept" else None
        ctx.elicit = AsyncMock(return_value=SimpleNamespace(action=action, data=data))
    return ctx


@pytest.fixture
def client(patch_get_client):
    patch_get_client.perform_preflight_calculation = AsyncMock(return_value=_preflight())
    patch_get_client.perform_multi_leg_preflight_calculation = AsyncMock(
        return_value=_preflight()
    )
    patch_get_client.place_order = AsyncMock(return_value=_new_order())
    patch_get_client.place_multileg_order = AsyncMock(return_value=_new_order())
    return patch_get_client


@pytest.fixture(autouse=True)
def no_limits(monkeypatch):
    for name in (
        "PUBLIC_COM_MAX_ORDER_QUANTITY",
        "PUBLIC_COM_MAX_ORDER_VALUE",
        "PUBLIC_COM_ORDER_CONFIRMATION",
        "PUBLIC_COM_READ_ONLY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestPreflightRunsFirst:
    async def test_place_order_preflights_the_exact_order_it_submits(self, client):
        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "submitted"
        assert data["preflight"] == {"orderValue": "300"}
        pf = client.perform_preflight_calculation.call_args.kwargs["preflight_request"]
        placed = client.place_order.call_args.kwargs["order_request"]
        assert isinstance(pf, PreflightRequest)
        assert isinstance(placed, OrderRequest)
        pf_body = pf.model_dump(mode="json", exclude_none=True)
        placed_body = placed.model_dump(mode="json", exclude_none=True)
        assert {k: v for k, v in placed_body.items() if k != "order_id"} == pf_body

    async def test_preflight_failure_stops_the_order(self, client):
        client.perform_preflight_calculation.side_effect = Exception("insufficient buying power")

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "preflight_failed"
        assert data["submitted"] is False
        assert "insufficient buying power" in data["message"]
        assert "No order was sent" in data["message"]
        client.place_order.assert_not_called()

    async def test_invalid_order_fails_before_any_preflight(self, client):
        data = json.loads(await place_order(**{**SELL_TO_OPEN_1000, "quantity": "lots"}))

        assert data["status"] == "error"
        client.perform_preflight_calculation.assert_not_called()

    async def test_multileg_preflights_the_same_legs(self, client):
        data = json.loads(await place_multileg_order(**BOTH_LEGS_SOLD))

        assert data["status"] == "submitted"
        pf = client.perform_multi_leg_preflight_calculation.call_args.kwargs["preflight_request"]
        placed = client.place_multileg_order.call_args.kwargs["order_request"]
        assert isinstance(pf, PreflightMultiLegRequest)
        assert isinstance(placed, MultilegOrderRequest)
        assert pf.legs == placed.legs
        assert pf.quantity == placed.quantity
        assert pf.limit_price == placed.limit_price

    async def test_multileg_preflight_failure_stops_the_order(self, client):
        client.perform_multi_leg_preflight_calculation.side_effect = Exception("not approved")

        data = json.loads(await place_multileg_order(**BOTH_LEGS_SOLD))

        assert data["status"] == "preflight_failed"
        client.place_multileg_order.assert_not_called()

    async def test_spread_preflights_without_the_order_id(self, patch_get_client):
        patch_get_client.preflight_call_credit_spread = AsyncMock(return_value=_preflight())
        patch_get_client.place_call_credit_spread = AsyncMock(return_value=_new_order())

        data = json.loads(
            await place_call_credit_spread(CALL_260, CALL_270, quantity=2, limit_price="1.20")
        )

        assert data["status"] == "submitted"
        pf = patch_get_client.preflight_call_credit_spread.call_args.kwargs
        placed = patch_get_client.place_call_credit_spread.call_args.kwargs
        assert "order_id" not in pf
        assert {k: v for k, v in placed.items() if k != "order_id"} == pf
        assert len(placed["order_id"]) == 36

    async def test_short_order_preflight_failure_stops_the_order(self, patch_get_client):
        patch_get_client.preflight_short_order = AsyncMock(side_effect=Exception("hard to borrow"))
        patch_get_client.place_short_order = AsyncMock()

        data = json.loads(await place_short_order(symbol="GME", quantity="100"))

        assert data["status"] == "preflight_failed"
        patch_get_client.place_short_order.assert_not_called()

    async def test_event_contract_preflight_failure_stops_the_buy(self, patch_get_client):
        patch_get_client.preflight_buy_event_contract = AsyncMock(side_effect=Exception("closed"))
        patch_get_client.buy_event_contract = AsyncMock()

        data = json.loads(
            await buy_event_contract("KALSHI.KXBTC15M-26OCT201200-00.Y", "5", limit_price="0.40")
        )

        assert data["status"] == "preflight_failed"
        patch_get_client.buy_event_contract.assert_not_called()

    async def test_price_only_replace_preflights_the_original_size(
        self, client, original_order
    ):
        client.cancel_and_replace_order = AsyncMock(return_value=_new_order())

        data = json.loads(
            await cancel_and_replace_order(
                order_id="00000000-0000-4000-a000-000000000001",
                order_type="LIMIT",
                limit_price="150",
            )
        )

        assert data["status"] == "submitted"
        pf = client.perform_preflight_calculation.call_args.kwargs["preflight_request"]
        assert pf.instrument == original_order.instrument
        assert pf.order_side == original_order.side
        assert pf.quantity == Decimal("10")
        assert pf.limit_price == Decimal("150")

    async def test_replace_is_not_sent_when_its_preflight_fails(self, client, original_order):
        client.perform_preflight_calculation.side_effect = Exception("rejected")
        client.cancel_and_replace_order = AsyncMock()

        data = json.loads(
            await cancel_and_replace_order(
                order_id="00000000-0000-4000-a000-000000000001",
                order_type="LIMIT",
                quantity="5000",
                limit_price="150",
            )
        )

        assert data["status"] == "preflight_failed"
        client.cancel_and_replace_order.assert_not_called()


def _holding(quantity: str) -> SimpleNamespace:
    if Decimal(quantity) == 0:
        return SimpleNamespace(positions=[])
    return SimpleNamespace(
        positions=[
            SimpleNamespace(
                instrument=OrderInstrument(symbol="GME", type=InstrumentType.EQUITY),
                quantity=Decimal(quantity),
            )
        ]
    )


class TestFlattenWorkflow:
    """flatten_and_go_short gates each of its two orders just before it is sent:
    the flatten up front (with the one confirmation for the whole workflow), and
    the short only once the flatten has filled and the account is flat."""

    @pytest.fixture
    def wf(self, patch_get_client):
        c = patch_get_client
        calls: list[str] = []
        c.calls = calls

        flatten = MagicMock()
        flatten.order_id = "flatten-id"
        flatten.wait_for_fill = AsyncMock(side_effect=lambda **_: calls.append("wait_for_fill"))
        short = MagicMock()
        short.order_id = "short-id"

        holdings = iter([_holding("40"), _holding("0")])
        c.get_portfolio = AsyncMock(side_effect=lambda **_: calls.append("portfolio") or next(holdings))
        c.perform_preflight_calculation = AsyncMock(
            side_effect=lambda **_: calls.append("preflight_flatten")
            or _preflight(orderValue="50000")
        )
        c.preflight_short_order = AsyncMock(
            side_effect=lambda **_: calls.append("preflight_short")
            or _preflight(orderValue="2000", buyingPowerRequirement="3000")
        )
        c.place_order = AsyncMock(side_effect=lambda **_: calls.append("place_flatten") or flatten)
        c.place_short_order = AsyncMock(side_effect=lambda **_: calls.append("place_short") or short)
        c.flatten_order = flatten
        return c

    async def _run(self, ctx=None, **overrides):
        args = {"symbol": "gme", "short_quantity": "100", **overrides}
        return json.loads(await flatten_and_go_short(**args, ctx=ctx))

    async def test_each_order_is_preflighted_just_before_it_is_sent(self, wf):
        ctx = _elicit_ctx()
        ctx.elicit.side_effect = lambda **_: wf.calls.append("confirm") or SimpleNamespace(
            action="accept", data=SimpleNamespace(confirm=True)
        )

        data = await self._run(ctx)

        assert wf.calls == [
            "portfolio", "preflight_flatten", "confirm", "place_flatten", "wait_for_fill",
            "portfolio", "preflight_short", "place_short",
        ]
        assert data["status"] == "submitted"
        assert data["flatten_order_id"] == "flatten-id"
        assert data["short_order_id"] == "short-id"
        assert data["confirmation"] == "user_confirmed"
        assert data["flatten_preflight"] == {"orderValue": "50000"}
        assert data["preflight"]["buyingPowerRequirement"] == "3000"

    async def test_one_prompt_describes_both_orders(self, wf):
        ctx = _elicit_ctx()

        await self._run(ctx)

        ctx.elicit.assert_called_once()
        message = ctx.elicit.call_args.kwargs["message"]
        assert "SELL 40 GME at MARKET" in message
        assert "SELL SHORT 100 GME" in message
        assert "left with no position" in message

    async def test_the_flatten_closes_the_whole_long_at_market(self, wf):
        await self._run(order_type="LIMIT", limit_price="20")

        pf = wf.perform_preflight_calculation.call_args.kwargs["preflight_request"]
        placed = wf.place_order.call_args.kwargs["order_request"]
        for req in (pf, placed):
            assert req.quantity == Decimal("40")
            assert req.order_type.value == "MARKET"
            assert req.order_side.value == "SELL"
            assert req.open_close_indicator.value == "CLOSE"
        short = wf.place_short_order.call_args.kwargs
        assert short["order_type"].value == "LIMIT"
        assert short["limit_price"] == Decimal("20")
        assert {k: v for k, v in short.items() if k != "order_id"} == (
            wf.preflight_short_order.call_args.kwargs
        )

    async def test_declining_sends_nothing(self, wf):
        data = await self._run(_elicit_ctx("decline"))

        assert data["status"] == "not_confirmed"
        assert data["submitted"] is False
        wf.place_order.assert_not_called()
        wf.place_short_order.assert_not_called()

    async def test_a_failed_flatten_preflight_sends_nothing(self, wf):
        wf.perform_preflight_calculation.side_effect = Exception("market closed")

        data = await self._run()

        assert data["status"] == "preflight_failed"
        wf.place_order.assert_not_called()

    async def test_too_large_a_short_is_refused_before_anything_is_sold(self, wf, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "50")

        data = await self._run()

        assert data["status"] == "blocked_by_limit"
        wf.place_order.assert_not_called()

    async def test_the_value_limit_never_stops_the_flatten(self, wf, monkeypatch):
        # Selling the $50,000 long is over the limit, but only the short opens risk.
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "5000")

        data = await self._run()

        assert data["status"] == "submitted"
        wf.place_short_order.assert_called_once()

    async def test_a_short_over_the_value_limit_leaves_the_account_flat(self, wf, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "2500")

        data = await self._run()

        assert data["status"] == "flattened_not_shorted"
        assert data["short_blocked_reason"] == "blocked_by_limit"
        assert data["flatten_order_id"] == "flatten-id"
        assert data["short_order_id"] is None
        assert data["short_preflight"]["buyingPowerRequirement"] == "3000"
        assert "no GME position" in data["message"]
        wf.place_short_order.assert_not_called()

    async def test_a_failed_short_preflight_leaves_the_account_flat(self, wf):
        wf.preflight_short_order.side_effect = Exception("hard to borrow")

        data = await self._run()

        assert data["status"] == "flattened_not_shorted"
        assert data["short_blocked_reason"] == "preflight_failed"
        assert "hard to borrow" in data["message"]
        wf.place_short_order.assert_not_called()

    async def test_an_unfilled_flatten_stops_before_the_short(self, wf):
        wf.flatten_order.wait_for_fill.side_effect = WaitTimeoutError("timed out")

        data = await self._run(flatten_timeout=5)

        assert data["status"] == "flatten_not_filled"
        assert data["flatten_order_id"] == "flatten-id"
        assert "cancel_order" in data["message"]
        wf.preflight_short_order.assert_not_called()
        wf.place_short_order.assert_not_called()

    async def test_a_position_left_after_the_flatten_stops_the_short(self, wf):
        holdings = iter([_holding("40"), _holding("15")])
        wf.get_portfolio.side_effect = lambda **_: next(holdings)

        data = await self._run()

        assert data["status"] == "flatten_incomplete"
        assert "15 shares" in data["message"]
        wf.place_short_order.assert_not_called()

    async def test_an_error_after_the_flatten_reports_the_flatten(self, wf):
        wf.place_short_order.side_effect = Exception("gateway timeout")

        data = await self._run()

        assert data["status"] == "error"
        assert data["flatten_order_id"] == "flatten-id"
        assert "gateway timeout" in data["message"]

    async def test_with_no_position_it_is_a_gated_short(self, wf):
        wf.get_portfolio.side_effect = None
        wf.get_portfolio.return_value = _holding("0")
        ctx = _elicit_ctx()

        data = await self._run(ctx)

        assert data["status"] == "submitted"
        assert data["flatten_order_id"] is None
        wf.place_order.assert_not_called()
        ctx.elicit.assert_called_once()
        assert "SELL SHORT 100 GME" in ctx.elicit.call_args.kwargs["message"]

    async def test_with_no_position_a_failed_preflight_sends_nothing(self, wf):
        wf.get_portfolio.side_effect = None
        wf.get_portfolio.return_value = _holding("0")
        wf.preflight_short_order.side_effect = Exception("hard to borrow")

        data = await self._run()

        assert data["status"] == "preflight_failed"
        wf.place_short_order.assert_not_called()

    async def test_the_sdk_composite_is_not_used(self, wf):
        await self._run()

        wf.flatten_and_go_short.assert_not_called()


class TestLimits:
    async def test_quantity_limit_blocks_the_reported_sell_to_open(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "10")

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "blocked_by_limit"
        assert data["submitted"] is False
        assert "PUBLIC_COM_MAX_ORDER_QUANTITY" in data["message"]
        client.perform_preflight_calculation.assert_not_called()
        client.place_order.assert_not_called()

    async def test_quantity_limit_counts_contracts_on_the_largest_leg(
        self, client, monkeypatch
    ):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "5")
        order = {
            **BOTH_LEGS_SOLD,
            "quantity": 3,
            "legs": [{**BOTH_LEGS_SOLD["legs"][0], "ratio_quantity": 2}, BOTH_LEGS_SOLD["legs"][1]],
        }

        data = json.loads(await place_multileg_order(**order))

        assert data["status"] == "blocked_by_limit"
        assert "6" in data["message"]
        client.place_multileg_order.assert_not_called()

    async def test_value_limit_uses_the_buying_power_requirement(self, client, monkeypatch):
        # Selling to open collects only $300 of premium, but the preflight's
        # buying power requirement shows the real exposure.
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "10000")
        client.perform_preflight_calculation.return_value = _preflight(
            buyingPowerRequirement="2600000"
        )

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "blocked_by_limit"
        assert "PUBLIC_COM_MAX_ORDER_VALUE" in data["message"]
        assert data["preflight"]["buyingPowerRequirement"] == "2600000"
        client.place_order.assert_not_called()

    async def test_value_limit_uses_the_magnitude_of_a_credit(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "100")
        client.perform_multi_leg_preflight_calculation.return_value = _preflight(
            orderValue="-400"
        )

        data = json.loads(await place_multileg_order(**BOTH_LEGS_SOLD))

        assert data["status"] == "blocked_by_limit"
        client.place_multileg_order.assert_not_called()

    async def test_order_within_limits_is_placed(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "1000")
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "500")

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "submitted"
        client.place_order.assert_called_once()

    async def test_value_limit_fails_closed_without_an_estimate(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_VALUE", "500")
        client.perform_preflight_calculation.return_value = MagicMock(spec=[])

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "blocked_by_limit"
        assert "cannot be checked" in data["message"]
        client.place_order.assert_not_called()

    @pytest.mark.parametrize(
        "name,value",
        [
            ("PUBLIC_COM_MAX_ORDER_QUANTITY", "ten"),
            ("PUBLIC_COM_MAX_ORDER_VALUE", "-5"),
            ("PUBLIC_COM_MAX_ORDER_VALUE", "NaN"),
            ("PUBLIC_COM_ORDER_CONFIRMATION", "sometimes"),
        ],
    )
    async def test_a_malformed_setting_blocks_orders(self, client, monkeypatch, name, value):
        monkeypatch.setenv(name, value)

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "configuration_error"
        assert name in data["message"]
        client.place_order.assert_not_called()


class TestConfirmation:
    async def test_confirmed_order_is_placed(self, client):
        ctx = _elicit_ctx()

        data = json.loads(await place_order(**SELL_TO_OPEN_1000, ctx=ctx))

        assert data["status"] == "submitted"
        assert data["confirmation"] == "user_confirmed"
        message = ctx.elicit.call_args.kwargs["message"]
        assert "SELL 1000 AAPL261120C00260000 (OPTION, OPEN)" in message
        assert "LIMIT, limit 3.00, DAY" in message
        assert "orderValue: 300" in message
        client.place_order.assert_called_once()

    async def test_the_prompt_comes_after_the_preflight(self, client):
        calls: list[str] = []
        client.perform_preflight_calculation.side_effect = (
            lambda **_: calls.append("preflight") or _preflight()
        )
        ctx = _elicit_ctx()
        ctx.elicit.side_effect = lambda **_: calls.append("elicit") or SimpleNamespace(
            action="accept", data=SimpleNamespace(confirm=True)
        )
        client.place_order.side_effect = lambda **_: calls.append("place") or _new_order()

        await place_order(**SELL_TO_OPEN_1000, ctx=ctx)

        assert calls == ["preflight", "elicit", "place"]

    @pytest.mark.parametrize(
        "action,confirm", [("decline", True), ("cancel", True), ("accept", False)]
    )
    async def test_unconfirmed_order_is_not_placed(self, client, action, confirm):
        data = json.loads(
            await place_order(**SELL_TO_OPEN_1000, ctx=_elicit_ctx(action, confirm))
        )

        assert data["status"] == "not_confirmed"
        assert data["submitted"] is False
        assert data["preflight"] == {"orderValue": "300"}
        client.place_order.assert_not_called()

    async def test_a_failed_prompt_does_not_place_the_order(self, client):
        ctx = _elicit_ctx(error=RuntimeError("client went away"))

        data = json.loads(await place_multileg_order(**BOTH_LEGS_SOLD, ctx=ctx))

        assert data["status"] == "confirmation_failed"
        client.place_multileg_order.assert_not_called()

    async def test_without_elicitation_auto_relies_on_client_approval(self, client):
        ctx = MagicMock()
        ctx.session.client_params.capabilities.elicitation = None

        data = json.loads(await place_order(**SELL_TO_OPEN_1000, ctx=ctx))

        assert data["status"] == "submitted"
        assert data["confirmation"] == "client_tool_approval"
        ctx.elicit.assert_not_called()

    async def test_required_mode_refuses_clients_without_elicitation(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_ORDER_CONFIRMATION", "required")

        data = json.loads(await place_order(**SELL_TO_OPEN_1000))

        assert data["status"] == "confirmation_unavailable"
        client.place_order.assert_not_called()

    async def test_off_mode_skips_the_prompt_but_still_preflights(self, client, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_ORDER_CONFIRMATION", "off")
        ctx = _elicit_ctx()

        data = json.loads(await place_order(**SELL_TO_OPEN_1000, ctx=ctx))

        assert data["status"] == "submitted"
        assert data["confirmation"] == "disabled"
        ctx.elicit.assert_not_called()
        client.perform_preflight_calculation.assert_called_once()


class TestOverMcp:
    """Drive the tools through a real MCP client session, so Context injection and
    the client's elicitation capability are exercised end to end."""

    async def _call(self, client, elicitation_callback=None):
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            # The low-level server, which every supported mcp release accepts here.
            async with create_connected_server_and_client_session(
                mcp._mcp_server, elicitation_callback=elicitation_callback
            ) as session:
                result = await session.call_tool("place_order", SELL_TO_OPEN_1000)
        return json.loads(result.content[0].text)

    async def test_ctx_is_not_part_of_the_tool_schema(self):
        tool = next(t for t in mcp._tool_manager.list_tools() if t.name == "place_order")
        assert "ctx" not in tool.parameters["properties"]

    async def test_user_sees_the_order_and_confirms(self, client):
        seen: list[types.ElicitRequestParams] = []

        async def confirm(context, params):
            seen.append(params)
            return types.ElicitResult(action="accept", content={"confirm": True})

        data = await self._call(client, confirm)

        assert data["status"] == "submitted"
        assert data["confirmation"] == "user_confirmed"
        assert CALL_260 in seen[0].message
        client.place_order.assert_called_once()

    async def test_user_declines(self, client):
        async def decline(context, params):
            return types.ElicitResult(action="decline")

        data = await self._call(client, decline)

        assert data["status"] == "not_confirmed"
        client.place_order.assert_not_called()


# Tools that open, enlarge or replace an order must take a Context so they can
# run the gate. cancel_order only removes an order, so it is exempt.
_UNGATED_WRITE_TOOLS = {"cancel_order"}


def test_every_order_tool_takes_a_context():
    write_tools = [
        t for t in mcp._tool_manager.list_tools()
        if not (t.annotations and t.annotations.readOnlyHint is True)
    ]
    assert write_tools, "expected write tools to be registered"
    for tool in write_tools:
        if tool.name in _UNGATED_WRITE_TOOLS:
            continue
        assert "ctx" in inspect.signature(tool.fn).parameters, (
            f"{tool.name} places orders but does not take a Context for the order gate"
        )


class TestReadOnlyMode:
    @pytest.fixture
    def tools(self, monkeypatch):
        # Work on a copy so removing tools here leaves the shared server intact.
        monkeypatch.setattr(mcp._tool_manager, "_tools", dict(mcp._tool_manager._tools))
        return mcp._tool_manager._tools

    def test_off_by_default(self, tools):
        before = set(tools)
        assert _apply_read_only_mode() == []
        assert set(tools) == before

    def test_removes_every_write_tool(self, tools, monkeypatch):
        monkeypatch.setenv("PUBLIC_COM_READ_ONLY", "true")

        removed = _apply_read_only_mode()

        assert {"place_order", "place_multileg_order", "cancel_order",
                "buy_event_contract", "flatten_and_go_short"} <= set(removed)
        assert all(t.annotations.readOnlyHint is True for t in tools.values())
        assert "preflight_order" in tools
        assert "get_portfolio" in tools


def test_policy_is_read_per_call(monkeypatch):
    monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "5")
    assert server._load_order_policy().max_quantity == Decimal("5")
    monkeypatch.setenv("PUBLIC_COM_MAX_ORDER_QUANTITY", "7")
    assert server._load_order_policy().max_quantity == Decimal("7")
