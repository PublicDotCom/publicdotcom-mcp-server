"""Tests for write (order placement) MCP tools."""
import json
from unittest.mock import AsyncMock, MagicMock

from publicdotcom_mcp_server.server import (
    cancel_and_replace_order,
    cancel_order,
    mcp,
    place_multileg_order,
    place_order,
    preflight_multileg_order,
)


def _get_tool(name: str):
    """Retrieve a registered MCP tool by function name."""
    return next(t for t in mcp._tool_manager._tools.values() if t.fn.__name__ == name)


class TestToolAnnotations:
    """Verify MCP tool annotations are correct for all write tools."""

    def test_place_order_idempotent_hint_is_false(self):
        tool = _get_tool("place_order")
        assert tool.annotations.idempotentHint is False

    def test_place_multileg_order_idempotent_hint_is_false(self):
        tool = _get_tool("place_multileg_order")
        assert tool.annotations.idempotentHint is False

    def test_cancel_and_replace_order_idempotent_hint_is_false(self):
        tool = _get_tool("cancel_and_replace_order")
        assert tool.annotations.idempotentHint is False

    def test_cancel_order_idempotent_hint_is_true(self):
        # cancel_order IS idempotent — cancelling twice is safe
        tool = _get_tool("cancel_order")
        assert tool.annotations.idempotentHint is True

    def test_place_order_open_world_hint_is_true(self):
        tool = _get_tool("place_order")
        assert tool.annotations.openWorldHint is True

    def test_place_order_destructive_hint_is_false(self):
        # Placing an order adds state rather than destroying it, so it is not
        # marked destructive (matches the hosted server's annotations).
        tool = _get_tool("place_order")
        assert tool.annotations.destructiveHint is False

    def test_read_only_tools_open_world_hint_is_true(self):
        for name in [
            "get_accounts", "get_portfolio", "get_orders", "get_order",
            "get_history", "get_quotes", "get_instrument", "get_all_instruments",
            "get_option_expirations", "get_option_chain", "get_option_greeks",
            "preflight_order", "preflight_multileg_order",
        ]:
            tool = _get_tool(name)
            assert tool.annotations.openWorldHint is True, (
                f"{name} should have openWorldHint=True"
            )


class TestPlaceOrder:
    async def test_successful_order_returns_submitted_json(self, patch_get_client):
        mock_client = patch_get_client
        mock_result = MagicMock()
        mock_result.order_id = "returned-order-uuid"
        mock_client.place_order = AsyncMock(return_value=mock_result)

        result = await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="MARKET",
            quantity="1",
        )
        data = json.loads(result)
        assert data["status"] == "submitted"
        assert data["order_id"] == "returned-order-uuid"
        assert "get_order" in data["message"]

    async def test_api_error_returns_error_json_with_order_id(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.place_order = AsyncMock(side_effect=Exception("API unavailable"))

        result = await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="MARKET",
            quantity="1",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "order_id" in data  # CRITICAL: preserved for follow-up check
        assert "API unavailable" in data["message"]
        assert "get_order" in data["message"]  # user told how to verify

    async def test_validation_error_for_limit_without_price(self, patch_get_client):
        result = await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="LIMIT",
            quantity="1",
            # limit_price intentionally omitted
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "limit_price" in data["message"]

    async def test_validation_error_for_quantity_and_amount(self, patch_get_client):
        result = await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="MARKET",
            quantity="5",
            amount="1000",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "mutually exclusive" in data["message"]

    async def test_validation_error_for_option_without_open_close(self, patch_get_client):
        result = await place_order(
            symbol="AAPL260320C00280000",
            instrument_type="OPTION",
            order_side="BUY",
            order_type="LIMIT",
            quantity="1",
            limit_price="2.50",
            # open_close_indicator intentionally omitted
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "open_close_indicator" in data["message"]

    async def test_non_numeric_quantity_returns_error(self, patch_get_client):
        result = await place_order(
            symbol="AAPL",
            instrument_type="EQUITY",
            order_side="BUY",
            order_type="MARKET",
            quantity="ten",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "numeric string" in data["message"]

    async def test_unique_order_id_per_call(self, patch_get_client):
        """Each call must generate a different order_id (not idempotent)."""
        mock_client = patch_get_client
        mock_client.place_order = AsyncMock(side_effect=Exception("fail"))

        result1 = await place_order(
            symbol="AAPL", instrument_type="EQUITY",
            order_side="BUY", order_type="MARKET", quantity="1",
        )
        result2 = await place_order(
            symbol="AAPL", instrument_type="EQUITY",
            order_side="BUY", order_type="MARKET", quantity="1",
        )
        id1 = json.loads(result1)["order_id"]
        id2 = json.loads(result2)["order_id"]
        assert id1 != id2, "Each call must generate a unique order_id"


class TestPlaceMultilegOrder:
    SAMPLE_LEGS = [
        {"symbol": "AAPL260320C00280000", "type": "OPTION", "side": "BUY", "open_close_indicator": "OPEN"},
        {"symbol": "AAPL260320C00290000", "type": "OPTION", "side": "SELL", "open_close_indicator": "OPEN"},
    ]

    async def test_successful_multileg_order(self, patch_get_client):
        mock_client = patch_get_client
        mock_result = MagicMock()
        mock_result.order_id = "multileg-order-uuid"
        mock_client.place_multileg_order = AsyncMock(return_value=mock_result)

        result = await place_multileg_order(
            legs=self.SAMPLE_LEGS,
            quantity=1,
            limit_price="1.50",
        )
        data = json.loads(result)
        assert data["status"] == "submitted"
        assert data["order_id"] == "multileg-order-uuid"

    async def test_api_error_preserves_order_id(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.place_multileg_order = AsyncMock(side_effect=Exception("API down"))

        result = await place_multileg_order(
            legs=self.SAMPLE_LEGS,
            quantity=1,
            limit_price="1.50",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "order_id" in data
        assert "get_order" in data["message"]

    async def test_non_numeric_limit_price_returns_error(self, patch_get_client):
        result = await place_multileg_order(
            legs=self.SAMPLE_LEGS,
            quantity=1,
            limit_price="cheap",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "numeric string" in data["message"]

    async def test_unique_order_id_per_call(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.place_multileg_order = AsyncMock(side_effect=Exception("fail"))

        result1 = await place_multileg_order(legs=self.SAMPLE_LEGS, quantity=1, limit_price="1.50")
        result2 = await place_multileg_order(legs=self.SAMPLE_LEGS, quantity=1, limit_price="1.50")
        id1 = json.loads(result1)["order_id"]
        id2 = json.loads(result2)["order_id"]
        assert id1 != id2


class TestPreflightMultilegOrder:
    SAMPLE_LEGS = [
        {"symbol": "AAPL260320C00280000", "type": "OPTION", "side": "BUY", "open_close_indicator": "OPEN"},
        {"symbol": "AAPL260320C00290000", "type": "OPTION", "side": "SELL", "open_close_indicator": "OPEN"},
    ]

    async def test_successful_preflight_builds_typed_legs(self, patch_get_client):
        mock_client = patch_get_client
        mock_result = MagicMock()
        mock_result.model_dump.return_value = {"estimated_cost": "150.00"}
        mock_client.perform_multi_leg_preflight_calculation = AsyncMock(
            return_value=mock_result
        )

        result = await preflight_multileg_order(
            legs=self.SAMPLE_LEGS,
            limit_price="1.50",
            quantity=1,
        )
        assert "estimated_cost" in result
        # dict legs are coerced through the typed OrderLeg model into the request
        req = mock_client.perform_multi_leg_preflight_calculation.await_args.kwargs[
            "preflight_request"
        ]
        assert len(req.legs) == 2

    async def test_non_numeric_limit_price_returns_error(self, patch_get_client):
        result = await preflight_multileg_order(
            legs=self.SAMPLE_LEGS,
            limit_price="cheap",
        )
        assert "Error" in result
        assert "numeric string" in result

    async def test_gtd_requires_expiration_time(self, patch_get_client):
        result = await preflight_multileg_order(
            legs=self.SAMPLE_LEGS,
            limit_price="1.50",
            time_in_force="GTD",
        )
        assert "Error" in result
        assert "expiration_time" in result


class TestCancelOrder:
    async def test_successful_cancel_returns_cancel_requested(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.cancel_order = AsyncMock(return_value=None)

        result = await cancel_order(order_id="order-uuid-123")
        data = json.loads(result)
        assert data["status"] == "cancel_requested"
        assert data["order_id"] == "order-uuid-123"
        assert "get_order" in data["message"]

    async def test_api_error_returns_error_string(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.cancel_order = AsyncMock(side_effect=Exception("order not found"))

        result = await cancel_order(order_id="order-uuid-123")
        assert "Error" in result
        assert "order not found" in result


class TestCancelAndReplaceOrder:
    async def test_successful_cancel_and_replace(self, patch_get_client):
        mock_client = patch_get_client
        mock_result = MagicMock()
        mock_result.order_id = "new-order-uuid"
        mock_client.cancel_and_replace_order = AsyncMock(return_value=mock_result)

        result = await cancel_and_replace_order(
            order_id="00000000-0000-4000-a000-000000000001",
            order_type="LIMIT",
            limit_price="155.00",
        )
        data = json.loads(result)
        assert data["status"] == "submitted"
        assert data["new_order_id"] == "new-order-uuid"
        assert data["cancelled_order_id"] == "00000000-0000-4000-a000-000000000001"

    async def test_api_error_returns_error_json_with_original_order_id(self, patch_get_client):
        mock_client = patch_get_client
        mock_client.cancel_and_replace_order = AsyncMock(side_effect=Exception("rejected"))

        result = await cancel_and_replace_order(
            order_id="00000000-0000-4000-a000-000000000001",
            order_type="LIMIT",
            limit_price="155.00",
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert data["original_order_id"] == "00000000-0000-4000-a000-000000000001"
        assert "rejected" in data["message"]

    async def test_limit_price_required_for_limit_order(self, patch_get_client):
        result = await cancel_and_replace_order(
            order_id="00000000-0000-4000-a000-000000000001",
            order_type="LIMIT",
            # limit_price omitted
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "limit_price" in data["message"]

    async def test_gtd_requires_expiration_time(self, patch_get_client):
        result = await cancel_and_replace_order(
            order_id="00000000-0000-4000-a000-000000000001",
            order_type="MARKET",
            time_in_force="GTD",
            # expiration_time omitted
        )
        data = json.loads(result)
        assert data["status"] == "error"
        assert "expiration_time" in data["message"]
