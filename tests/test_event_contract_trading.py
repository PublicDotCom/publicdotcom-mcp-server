"""Tests for the recurring-market event-contract tools (SDK 0.1.27).

get_current_event, wait_for_settlement, preflight_buy_event_contract and
buy_event_contract wrap the SDK's helpers; get_event_summary / get_event_details
gained the SDK's derived shortcuts. Response fixtures are real SDK models; the
client is mocked.
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from public_api_sdk.models import (
    EventDetails,
    EventFrequency,
    EventSummary,
    EventSummaryPage,
    PreflightResponse,
    WaitTimeoutError,
)

from publicdotcom_mcp_server.server import (
    _parse_event_frequency,
    buy_event_contract,
    get_current_event,
    get_event_details,
    get_event_summary,
    mcp,
    preflight_buy_event_contract,
    wait_for_settlement,
)

EVENT_SYMBOL = "KALSHI.KXBTC15M-26OCT201200"
YES = "KALSHI.KXBTC15M-26OCT201200-00.Y"
NO = "KALSHI.KXBTC15M-26OCT201200-00.N"


def _get_tool(name: str):
    return next(t for t in mcp._tool_manager._tools.values() if t.fn.__name__ == name)


def _outcome(**overrides) -> dict:
    outcome = {
        "outcomeId": "KXBTC15M-26OCT201200-00",
        "title": "Up",
        "state": "STATE_OPEN",
        "trading": "BUY_AND_SELL",
        "timeline": {
            "closeTime": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        },
        "contracts": [
            {"symbol": YES, "predictedOutcome": "YES", "bid": "0.52", "ask": "0.553"},
            {"symbol": NO, "predictedOutcome": "NO", "bid": "0.44", "ask": "0.47"},
        ],
    }
    outcome.update(overrides)
    return outcome


def _details(*outcomes: dict) -> EventDetails:
    return EventDetails.model_validate(
        {"eventSymbol": EVENT_SYMBOL, "outcomes": list(outcomes) or [_outcome()]}
    )


def _summary(event_symbol: str = EVENT_SYMBOL) -> EventSummary:
    return EventSummary.model_validate(
        {"eventSymbol": event_symbol, "title": "BTC Up or Down", "resolved": False}
    )


def _preflight() -> PreflightResponse:
    return PreflightResponse.model_validate(
        {
            "instrument": {"symbol": YES, "type": "EVENTCONTRACT"},
            "orderValue": "0.55",
            "estimatedCost": "0.58",
            "estimatedCommission": "0.02",
        }
    )


class TestParseEventFrequency:
    def test_parses_case_insensitively(self):
        assert _parse_event_frequency("one_hour") is EventFrequency.ONE_HOUR

    @pytest.mark.parametrize("raw", ["ALL", "ONCE", "UNKNOWN", "HOURLY"])
    def test_refuses_values_that_name_no_recurring_series(self, raw):
        with pytest.raises(ValueError, match="Invalid frequency") as exc:
            _parse_event_frequency(raw)
        assert "FIFTEEN_MINUTES" in str(exc.value)


class TestDerivedShortcuts:
    async def test_summary_events_carry_their_series(self, patch_get_client):
        patch_get_client.get_event_summary = AsyncMock(
            return_value=EventSummaryPage(content=[_summary(), _summary("NOT-AN-EVENT")])
        )

        data = json.loads(await get_event_summary())

        assert data["content"][0]["series"] == "KXBTC15M"
        assert "series" not in data["content"][1]

    async def test_details_carry_series_and_per_outcome_shortcuts(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(return_value=_details())

        data = json.loads(await get_event_details(EVENT_SYMBOL))

        assert data["series"] == "KXBTC15M"
        outcome = data["outcomes"][0]
        assert outcome["tradable"] is True
        assert 0 < outcome["secondsLeft"] <= 300
        assert "winner" not in outcome
        assert outcome["contracts"][0]["ask"] == "0.553"

    async def test_closed_outcome_is_not_tradable_even_if_state_is_open(self, patch_get_client):
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        patch_get_client.get_event_details = AsyncMock(
            return_value=_details(_outcome(timeline={"closeTime": past}))
        )

        outcome = json.loads(await get_event_details(EVENT_SYMBOL))["outcomes"][0]

        assert outcome["tradable"] is False
        assert outcome["secondsLeft"] == 0

    async def test_settled_outcome_names_the_winner(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(
            return_value=_details(
                _outcome(state="STATE_SETTLED", settledOutcome="SETTLED_OUTCOME_NO")
            )
        )

        outcome = json.loads(await get_event_details(EVENT_SYMBOL))["outcomes"][0]

        assert outcome["winner"] == NO


class TestGetCurrentEvent:
    async def test_checks_once_by_default(self, patch_get_client):
        patch_get_client.get_current_event = AsyncMock(return_value=_summary())
        patch_get_client.wait_for_current_event = AsyncMock()

        data = json.loads(await get_current_event("kxbtc15m", category="Crypto"))

        patch_get_client.get_current_event.assert_awaited_once_with(
            "kxbtc15m", EventFrequency.FIFTEEN_MINUTES, category="Crypto", subcategory=None
        )
        patch_get_client.wait_for_current_event.assert_not_called()
        assert data["eventSymbol"] == EVENT_SYMBOL
        assert data["series"] == "KXBTC15M"

    async def test_no_open_event_is_reported_not_errored(self, patch_get_client):
        patch_get_client.get_current_event = AsyncMock(return_value=None)

        data = json.loads(await get_current_event("KXBTC15M", frequency="ONE_HOUR"))

        assert data["status"] == "no_open_event"
        assert "ONE_HOUR" in data["message"]

    async def test_wait_seconds_polls_with_a_bounded_timeout(self, patch_get_client):
        patch_get_client.wait_for_current_event = AsyncMock(return_value=_summary())

        data = json.loads(await get_current_event("KXBTC15M", wait_seconds=90))

        kwargs = patch_get_client.wait_for_current_event.call_args.kwargs
        assert kwargs["timeout"] == 90
        assert kwargs["poll_interval"] == 10.0
        assert data["eventSymbol"] == EVENT_SYMBOL

    async def test_wait_timeout_reports_no_open_event(self, patch_get_client):
        patch_get_client.wait_for_current_event = AsyncMock(
            side_effect=WaitTimeoutError("No open KXBTC15M event within 30s")
        )

        data = json.loads(await get_current_event("KXBTC15M", wait_seconds=30))

        assert data["status"] == "no_open_event"
        assert "within 30s" in data["message"]

    @pytest.mark.parametrize("wait", [-1, 121])
    async def test_out_of_range_wait_is_refused_without_calling_the_api(
        self, patch_get_client, wait
    ):
        patch_get_client.get_current_event = AsyncMock()
        patch_get_client.wait_for_current_event = AsyncMock()

        result = await get_current_event("KXBTC15M", wait_seconds=wait)

        assert result.startswith("Error:")
        assert "between 0 and 120" in result
        patch_get_client.get_current_event.assert_not_called()
        patch_get_client.wait_for_current_event.assert_not_called()

    async def test_empty_series_error_is_reported(self, patch_get_client):
        patch_get_client.get_current_event = AsyncMock(
            side_effect=ValueError("series is required")
        )

        result = await get_current_event(" ")

        assert result == "Error: series is required"


class TestWaitForSettlement:
    async def test_settled_outcome_reports_winner_and_won(self, patch_get_client):
        settled = _details(
            _outcome(state="STATE_SETTLED", settledOutcome="SETTLED_OUTCOME_YES")
        ).outcomes[0]
        patch_get_client.wait_for_settlement = AsyncMock(return_value=settled)

        data = json.loads(
            await wait_for_settlement(EVENT_SYMBOL.lower(), contract_symbol=NO.lower())
        )

        args = patch_get_client.wait_for_settlement.call_args
        assert args.args == (EVENT_SYMBOL, None)
        assert args.kwargs == {"timeout": 60, "poll_interval": 10.0}
        assert data["status"] == "settled"
        assert data["settledOutcome"] == "SETTLED_OUTCOME_YES"
        assert data["winner"] == YES
        assert data["won"] is False

    async def test_won_is_omitted_without_a_contract(self, patch_get_client):
        settled = _details(
            _outcome(state="STATE_SETTLED", settledOutcome="SETTLED_OUTCOME_YES")
        ).outcomes[0]
        patch_get_client.wait_for_settlement = AsyncMock(return_value=settled)

        data = json.loads(await wait_for_settlement(EVENT_SYMBOL, "KXBTC15M-26OCT201200-00"))

        assert patch_get_client.wait_for_settlement.call_args.args[1] == (
            "KXBTC15M-26OCT201200-00"
        )
        assert "won" not in data

    async def test_timeout_is_pending_not_an_error(self, patch_get_client):
        patch_get_client.wait_for_settlement = AsyncMock(
            side_effect=WaitTimeoutError("did not settle within 0s (state STATE_CLOSED)")
        )

        data = json.loads(await wait_for_settlement(EVENT_SYMBOL, timeout_seconds=0))

        assert data["status"] == "pending"
        assert "STATE_CLOSED" in data["message"]
        assert patch_get_client.wait_for_settlement.call_args.kwargs["timeout"] == 0

    async def test_multi_outcome_event_without_outcome_id_is_an_error(self, patch_get_client):
        patch_get_client.wait_for_settlement = AsyncMock(
            side_effect=ValueError(f"{EVENT_SYMBOL} has 2 outcomes; pass outcome_id")
        )

        result = await wait_for_settlement(EVENT_SYMBOL)

        assert result.startswith("Error:")
        assert "pass outcome_id" in result

    async def test_out_of_range_timeout_is_refused(self, patch_get_client):
        patch_get_client.wait_for_settlement = AsyncMock()

        result = await wait_for_settlement(EVENT_SYMBOL, timeout_seconds=600)

        assert "between 0 and 120" in result
        patch_get_client.wait_for_settlement.assert_not_called()


class TestPreflightBuyEventContract:
    def test_is_read_only(self):
        tool = _get_tool("preflight_buy_event_contract")
        assert tool.annotations.readOnlyHint is True

    async def test_explicit_limit_price_skips_the_details_lookup(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock()
        patch_get_client.preflight_buy_event_contract = AsyncMock(return_value=_preflight())

        data = json.loads(
            await preflight_buy_event_contract(f" {YES.lower()} ", "2", limit_price="0.55")
        )

        patch_get_client.get_event_details.assert_not_called()
        patch_get_client.preflight_buy_event_contract.assert_awaited_once_with(
            YES, Decimal("2"), limit_price=Decimal("0.55"), account_id=None
        )
        assert data["contractSymbol"] == YES
        assert data["limitPrice"] == "0.55"
        assert data["preflight"]["estimatedCost"] == "0.58"

    async def test_omitted_limit_price_uses_the_ask_rounded_up(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(return_value=_details())
        patch_get_client.preflight_buy_event_contract = AsyncMock(return_value=_preflight())

        data = json.loads(
            await preflight_buy_event_contract(YES, event_symbol=EVENT_SYMBOL.lower())
        )

        assert patch_get_client.get_event_details.call_args.kwargs == {
            "event_symbol": EVENT_SYMBOL,
            "include_all_outcomes": True,
        }
        # ask 0.553 → 0.56: rounding down would price the order below the ask.
        assert patch_get_client.preflight_buy_event_contract.call_args.kwargs[
            "limit_price"
        ] == Decimal("0.56")
        assert data["limitPrice"] == "0.56"

    async def test_no_limit_price_and_no_event_symbol_is_refused(self, patch_get_client):
        patch_get_client.preflight_buy_event_contract = AsyncMock()

        result = await preflight_buy_event_contract(YES)

        assert result.startswith("Error:")
        assert "event_symbol" in result
        patch_get_client.preflight_buy_event_contract.assert_not_called()

    async def test_contract_missing_from_the_event_is_refused(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(return_value=_details())
        patch_get_client.preflight_buy_event_contract = AsyncMock()

        result = await preflight_buy_event_contract(
            "KALSHI.OTHER-1.Y", event_symbol=EVENT_SYMBOL
        )

        assert "has no contract 'KALSHI.OTHER-1.Y'" in result
        patch_get_client.preflight_buy_event_contract.assert_not_called()

    async def test_contract_without_an_ask_needs_a_limit_price(self, patch_get_client):
        no_ask = _outcome(contracts=[{"symbol": YES, "predictedOutcome": "YES"}])
        patch_get_client.get_event_details = AsyncMock(return_value=_details(no_ask))

        result = await preflight_buy_event_contract(YES, event_symbol=EVENT_SYMBOL)

        assert "has no ask; pass limit_price" in result

    async def test_sdk_price_validation_error_is_reported(self, patch_get_client):
        patch_get_client.preflight_buy_event_contract = AsyncMock(
            side_effect=ValueError("limit_price must be in whole cents, got 0.555")
        )

        result = await preflight_buy_event_contract(YES, limit_price="0.555")

        assert "whole cents" in result


class TestBuyEventContract:
    def test_annotations_match_place_order(self):
        buy = _get_tool("buy_event_contract").annotations
        place = _get_tool("place_order").annotations
        assert buy.model_dump(exclude={"title"}) == place.model_dump(exclude={"title"})
        assert buy.readOnlyHint is False
        assert buy.idempotentHint is False

    async def test_places_a_limit_buy_with_a_generated_order_id(self, patch_get_client):
        new_order = AsyncMock()
        new_order.order_id = "returned-id"
        patch_get_client.buy_event_contract = AsyncMock(return_value=new_order)

        data = json.loads(await buy_event_contract(NO, "3", limit_price="0.47"))

        call = patch_get_client.buy_event_contract.call_args
        assert call.args == (NO, Decimal("3"))
        assert call.kwargs["limit_price"] == Decimal("0.47")
        assert len(call.kwargs["order_id"]) == 36
        assert data == {
            "order_id": "returned-id",
            "status": "submitted",
            "contract_symbol": NO,
            "quantity": "3",
            "limit_price": "0.47",
            "message": data["message"],
        }

    async def test_omitted_limit_price_buys_at_the_ask(self, patch_get_client):
        new_order = AsyncMock()
        new_order.order_id = "returned-id"
        patch_get_client.get_event_details = AsyncMock(return_value=_details())
        patch_get_client.buy_event_contract = AsyncMock(return_value=new_order)

        data = json.loads(await buy_event_contract(NO, event_symbol=EVENT_SYMBOL))

        assert patch_get_client.buy_event_contract.call_args.kwargs["limit_price"] == (
            Decimal("0.47")
        )
        assert data["limit_price"] == "0.47"

    async def test_failure_returns_the_order_id_to_check(self, patch_get_client):
        patch_get_client.buy_event_contract = AsyncMock(side_effect=Exception("rejected"))

        data = json.loads(await buy_event_contract(YES, limit_price="0.55"))

        assert data["status"] == "error"
        assert "rejected" in data["message"]
        assert data["order_id"] in data["message"]

    async def test_unresolvable_price_never_places_an_order(self, patch_get_client):
        patch_get_client.buy_event_contract = AsyncMock()

        data = json.loads(await buy_event_contract(YES))

        assert data["status"] == "error"
        patch_get_client.buy_event_contract.assert_not_called()


class TestRequestsSentByTheSdk:
    """Run the tools through a real SDK client to check the request it builds."""

    @pytest.fixture
    def sdk_client(self):
        from unittest.mock import patch

        from public_api_sdk import (
            ApiKeyAuthConfig,
            AsyncPublicApiClient,
            AsyncPublicApiClientConfiguration,
        )

        client = AsyncPublicApiClient(
            auth_config=ApiKeyAuthConfig(api_secret_key="test-secret-key"),
            config=AsyncPublicApiClientConfiguration(default_account_number="acct-1"),
        )
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        with patch("publicdotcom_mcp_server.server._get_client", return_value=client):
            yield client

    async def test_preflight_body_is_a_day_limit_buy_with_two_decimal_quantity(
        self, sdk_client
    ):
        sdk_client.perform_preflight_calculation = AsyncMock(return_value=_preflight())

        await preflight_buy_event_contract(YES, "2", limit_price="0.55")

        req = sdk_client.perform_preflight_calculation.call_args.args[0]
        assert req.model_dump(by_alias=True, exclude_none=True, mode="json") == {
            "instrument": {"symbol": YES, "type": "EVENTCONTRACT"},
            "orderSide": "BUY",
            "orderType": "LIMIT",
            "expiration": {"timeInForce": "DAY"},
            "quantity": "2.00",
            "limitPrice": "0.55",
        }

    async def test_order_body_carries_the_tool_order_id(self, sdk_client):
        new_order = AsyncMock()
        new_order.order_id = "returned-id"
        sdk_client.place_order = AsyncMock(return_value=new_order)

        await buy_event_contract(NO, "1", limit_price="0.47")

        req = sdk_client.place_order.call_args.args[0]
        body = req.model_dump(by_alias=True, exclude_none=True, mode="json")
        assert body["instrument"] == {"symbol": NO, "type": "EVENTCONTRACT"}
        assert body["quantity"] == "1.00"
        assert body["limitPrice"] == "0.47"
        assert len(body["orderId"]) == 36
