"""Tests for bracket orders on place_order (SDK 0.1.23).

The MCP layer only checks that the bracket arguments cohere with each other; the
API's own rules (which class needs which leg, instrument type, whole-share
quantity, market session, entry order type) live in the SDK's OrderRequest. Both
halves are covered here so a future change to either surface fails loudly.
"""
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from public_api_sdk.models import (
    InstrumentType,
    Order,
    OrderClass,
    OrderInstrument,
    OrderSide,
    OrderStatus,
    OrderType,
)

from publicdotcom_mcp_server.server import (
    _build_bracket_kwargs,
    _serialize,
    get_order,
    place_order,
)


def _bracket_kwargs(**overrides):
    kwargs = {
        "order_class": None,
        "take_profit_limit_price": None,
        "stop_loss_stop_price": None,
        "stop_loss_limit_price": None,
    }
    kwargs.update(overrides)
    return _build_bracket_kwargs(**kwargs)


class TestBuildBracketKwargs:
    def test_no_bracket_args_yields_no_kwargs(self):
        assert _bracket_kwargs() == {}

    def test_order_class_is_parsed(self):
        assert _bracket_kwargs(order_class="BRACKET")["order_class"] is OrderClass.BRACKET

    def test_order_class_is_case_insensitive(self):
        assert _bracket_kwargs(order_class="oco")["order_class"] is OrderClass.OCO

    def test_invalid_order_class_lists_the_valid_ones(self):
        with pytest.raises(ValueError) as exc:
            _bracket_kwargs(order_class="TRAILING")
        message = str(exc.value)
        assert "TRAILING" in message
        for valid in ("SIMPLE", "BRACKET", "OCO", "OTO"):
            assert valid in message

    def test_take_profit_leg_is_built(self):
        kwargs = _bracket_kwargs(order_class="BRACKET", take_profit_limit_price="215.50")
        assert kwargs["take_profit"].limit_price == Decimal("215.50")
        assert "stop_loss" not in kwargs

    def test_stop_loss_without_limit_is_a_plain_stop(self):
        kwargs = _bracket_kwargs(order_class="BRACKET", stop_loss_stop_price="190.00")
        assert kwargs["stop_loss"].stop_price == Decimal("190.00")
        assert kwargs["stop_loss"].limit_price is None

    def test_stop_loss_with_limit_carries_both_prices(self):
        kwargs = _bracket_kwargs(
            order_class="BRACKET",
            stop_loss_stop_price="190.00",
            stop_loss_limit_price="189.00",
        )
        assert kwargs["stop_loss"].stop_price == Decimal("190.00")
        assert kwargs["stop_loss"].limit_price == Decimal("189.00")

    def test_stop_loss_limit_without_stop_is_rejected(self):
        # Without this check the limit price would be silently dropped, since
        # there would be no StopLoss to attach it to.
        with pytest.raises(ValueError, match="stop_loss_limit_price requires"):
            _bracket_kwargs(order_class="BRACKET", stop_loss_limit_price="189.00")

    @pytest.mark.parametrize(
        "field",
        ["take_profit_limit_price", "stop_loss_stop_price", "stop_loss_limit_price"],
    )
    def test_non_numeric_price_is_rejected(self, field):
        args = {field: "high"}
        if field == "stop_loss_limit_price":
            args["stop_loss_stop_price"] = "190.00"
        with pytest.raises(ValueError, match="numeric string"):
            _bracket_kwargs(order_class="BRACKET", **args)

    def test_zero_take_profit_price_is_rejected_by_the_sdk(self):
        with pytest.raises(ValueError, match="greater than 0"):
            _bracket_kwargs(order_class="BRACKET", take_profit_limit_price="0")


class TestPlaceBracketOrder:
    """place_order end-to-end with the client mocked, asserting the OrderRequest."""

    ENTRY = {
        "symbol": "AAPL",
        "instrument_type": "EQUITY",
        "order_side": "BUY",
        "order_type": "LIMIT",
        "quantity": "10",
        "limit_price": "200.00",
    }

    @staticmethod
    def _mock_placement(patch_get_client):
        result = MagicMock()
        result.order_id = "entry-order-uuid"
        patch_get_client.place_order = AsyncMock(return_value=result)
        return patch_get_client

    def _sent_request(self, mock_client):
        return mock_client.place_order.call_args.kwargs["order_request"]

    async def test_bracket_with_both_legs_is_submitted(self, patch_get_client):
        mock_client = self._mock_placement(patch_get_client)

        result = await place_order(
            **self.ENTRY,
            order_class="BRACKET",
            take_profit_limit_price="215.50",
            stop_loss_stop_price="190.00",
        )

        assert json.loads(result)["status"] == "submitted"
        req = self._sent_request(mock_client)
        assert req.order_class is OrderClass.BRACKET
        assert req.take_profit.limit_price == Decimal("215.50")
        assert req.stop_loss.stop_price == Decimal("190.00")

    async def test_bracket_with_only_take_profit_is_allowed(self, patch_get_client):
        # The spec says a bracket uses takeProfit "and" stopLoss without marking
        # either required, so one leg is accepted rather than rejected here.
        mock_client = self._mock_placement(patch_get_client)

        result = await place_order(
            **self.ENTRY, order_class="BRACKET", take_profit_limit_price="215.50"
        )

        assert json.loads(result)["status"] == "submitted"
        assert self._sent_request(mock_client).stop_loss is None

    async def test_bracket_with_only_stop_loss_is_allowed(self, patch_get_client):
        mock_client = self._mock_placement(patch_get_client)

        result = await place_order(
            **self.ENTRY, order_class="BRACKET", stop_loss_stop_price="190.00"
        )

        assert json.loads(result)["status"] == "submitted"
        assert self._sent_request(mock_client).take_profit is None

    async def test_simple_order_sends_no_bracket_fields(self, patch_get_client):
        mock_client = self._mock_placement(patch_get_client)

        await place_order(**self.ENTRY)

        req = self._sent_request(mock_client)
        assert req.order_class is None
        assert req.take_profit is None
        assert req.stop_loss is None

    async def test_request_serializes_to_the_api_field_names(self, patch_get_client):
        mock_client = self._mock_placement(patch_get_client)

        await place_order(
            **self.ENTRY,
            order_class="BRACKET",
            take_profit_limit_price="215.5",
            stop_loss_stop_price="190",
            stop_loss_limit_price="189.456",
        )

        payload = self._sent_request(mock_client).model_dump(
            by_alias=True, exclude_none=True
        )
        assert payload["orderClass"] == "BRACKET"
        # Prices go out as 2dp strings, like the other price fields.
        assert payload["takeProfit"] == {"limitPrice": "215.50"}
        assert payload["stopLoss"] == {"stopPrice": "190.00", "limitPrice": "189.46"}


class TestBracketValidationErrors:
    """Every rule below is enforced by the SDK; place_order must surface it."""

    ENTRY = TestPlaceBracketOrder.ENTRY

    async def test_exit_legs_without_a_bracket_class_are_rejected(self, patch_get_client):
        result = await place_order(**self.ENTRY, take_profit_limit_price="215.50")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "order_class" in data["message"]

    async def test_bracket_class_without_any_exit_leg_is_rejected(self, patch_get_client):
        result = await place_order(**self.ENTRY, order_class="BRACKET")
        data = json.loads(result)
        assert data["status"] == "error"
        assert "at least one" in data["message"]

    async def test_crypto_bracket_is_rejected(self, patch_get_client):
        entry = {**self.ENTRY, "symbol": "BTC", "instrument_type": "CRYPTO"}
        result = await place_order(
            **entry, order_class="BRACKET", take_profit_limit_price="215.50"
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "EQUITY and OPTION" in data["message"]

    async def test_notional_bracket_is_rejected(self, patch_get_client):
        entry = {k: v for k, v in self.ENTRY.items() if k != "quantity"}
        result = await place_order(
            **entry,
            amount="1000",
            order_class="BRACKET",
            take_profit_limit_price="215.50",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "amount" in data["message"]

    async def test_fractional_quantity_bracket_is_rejected(self, patch_get_client):
        entry = {**self.ENTRY, "quantity": "1.5"}
        result = await place_order(
            **entry, order_class="BRACKET", take_profit_limit_price="215.50"
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "whole-share" in data["message"]

    async def test_extended_session_bracket_is_rejected(self, patch_get_client):
        result = await place_order(
            **self.ENTRY,
            equity_market_session="EXTENDED",
            order_class="BRACKET",
            take_profit_limit_price="215.50",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "CORE" in data["message"]

    async def test_stop_entry_bracket_is_rejected(self, patch_get_client):
        entry = {**self.ENTRY, "order_type": "STOP_LIMIT", "stop_price": "195.00"}
        result = await place_order(
            **entry, order_class="BRACKET", take_profit_limit_price="215.50"
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "entry order type" in data["message"]

    async def test_oco_requires_a_limit_entry(self, patch_get_client):
        entry = {k: v for k, v in self.ENTRY.items() if k != "limit_price"}
        entry["order_type"] = "MARKET"
        result = await place_order(
            **entry, order_class="OCO", take_profit_limit_price="215.50"
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "must be LIMIT" in data["message"]

    async def test_market_entry_is_allowed_for_bracket(self, patch_get_client):
        # The MARKET restriction applies to OCO only — BRACKET must still accept it.
        result_model = MagicMock()
        result_model.order_id = "entry-order-uuid"
        patch_get_client.place_order = AsyncMock(return_value=result_model)

        entry = {k: v for k, v in self.ENTRY.items() if k != "limit_price"}
        entry["order_type"] = "MARKET"
        result = await place_order(
            **entry, order_class="BRACKET", take_profit_limit_price="215.50"
        )
        assert json.loads(result)["status"] == "submitted"

    async def test_error_response_keeps_the_order_id(self, patch_get_client):
        result = await place_order(**self.ENTRY, order_class="BRACKET")
        data = json.loads(result)
        assert data["order_id"]
        assert "get_order" in data["message"]


class TestBracketIdOnReads:
    def _order(self, **overrides) -> Order:
        fields = {
            "order_id": "leg-order-uuid",
            "instrument": OrderInstrument(symbol="AAPL", type=InstrumentType.EQUITY),
            "type": OrderType.LIMIT,
            "side": OrderSide.SELL,
            "status": OrderStatus.NEW,
        }
        fields.update(overrides)
        return Order(**fields)

    def test_bracket_id_survives_serialization(self):
        payload = json.loads(_serialize(self._order(bracket_id="entry-order-uuid")))
        assert payload["bracketId"] == "entry-order-uuid"

    def test_non_bracket_order_omits_bracket_id(self):
        # _serialize drops None fields, so a standalone order stays unchanged.
        assert "bracketId" not in json.loads(_serialize(self._order()))

    async def test_get_order_exposes_bracket_id(self, patch_get_client):
        patch_get_client.get_order = AsyncMock(
            return_value=self._order(bracket_id="entry-order-uuid")
        )

        payload = json.loads(await get_order(order_id="leg-order-uuid"))
        assert payload["bracketId"] == "entry-order-uuid"
