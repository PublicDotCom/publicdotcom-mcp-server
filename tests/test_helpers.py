"""Unit tests for server helper functions."""
import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from publicdotcom_mcp_server.server import (
    OrderLeg,
    _build_leg_request,
    _parse_instrument_type,
    _serialize,
    _validate_order_params,
)


class TestParseInstrumentType:
    def test_valid_equity_uppercase(self):
        result = _parse_instrument_type("EQUITY")
        assert result.value == "EQUITY"

    def test_valid_crypto_lowercase(self):
        result = _parse_instrument_type("crypto")
        assert result.value == "CRYPTO"

    def test_valid_option_mixed_case(self):
        result = _parse_instrument_type("Option")
        assert result.value == "OPTION"

    def test_invalid_raises_with_valid_types_listed(self):
        with pytest.raises(ValueError, match="Invalid instrument type"):
            _parse_instrument_type("STONKS")

    def test_invalid_lists_valid_options(self):
        with pytest.raises(ValueError, match="Valid types"):
            _parse_instrument_type("BONDS")


class TestValidateOrderParams:
    def test_quantity_and_amount_mutual_exclusion(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _validate_order_params(
                quantity="10",
                amount="500",
                order_type="MARKET",
                limit_price=None,
                stop_price=None,
            )

    def test_limit_price_required_for_limit_order(self):
        with pytest.raises(ValueError, match="limit_price is required"):
            _validate_order_params(
                quantity="10",
                amount=None,
                order_type="LIMIT",
                limit_price=None,
                stop_price=None,
            )

    def test_limit_price_required_for_stop_limit_order(self):
        with pytest.raises(ValueError, match="limit_price is required"):
            _validate_order_params(
                quantity="1",
                amount=None,
                order_type="STOP_LIMIT",
                limit_price=None,
                stop_price="145.00",
            )

    def test_stop_price_required_for_stop_order(self):
        with pytest.raises(ValueError, match="stop_price is required"):
            _validate_order_params(
                quantity="10",
                amount=None,
                order_type="STOP",
                limit_price=None,
                stop_price=None,
            )

    def test_stop_price_required_for_stop_limit_order(self):
        with pytest.raises(ValueError, match="stop_price is required"):
            _validate_order_params(
                quantity="1",
                amount=None,
                order_type="STOP_LIMIT",
                limit_price="150.00",
                stop_price=None,
            )

    def test_open_close_required_for_option(self):
        with pytest.raises(ValueError, match="open_close_indicator"):
            _validate_order_params(
                quantity="1",
                amount=None,
                order_type="LIMIT",
                limit_price="1.50",
                stop_price=None,
                instrument_type="OPTION",
                open_close_indicator=None,
            )

    def test_expiration_time_required_for_gtd(self):
        with pytest.raises(ValueError, match="expiration_time"):
            _validate_order_params(
                quantity="1",
                amount=None,
                order_type="MARKET",
                limit_price=None,
                stop_price=None,
                time_in_force="GTD",
                expiration_time=None,
            )

    def test_non_numeric_quantity_raises(self):
        with pytest.raises(ValueError, match="numeric string"):
            _validate_order_params(
                quantity="ten",
                amount=None,
                order_type="MARKET",
                limit_price=None,
                stop_price=None,
            )

    def test_non_numeric_limit_price_raises(self):
        with pytest.raises(ValueError, match="numeric string"):
            _validate_order_params(
                quantity="1",
                amount=None,
                order_type="LIMIT",
                limit_price="cheap",
                stop_price=None,
            )

    def test_valid_market_order_passes(self):
        # Should not raise
        _validate_order_params(
            quantity="10",
            amount=None,
            order_type="MARKET",
            limit_price=None,
            stop_price=None,
        )

    def test_valid_limit_order_passes(self):
        _validate_order_params(
            quantity="5",
            amount=None,
            order_type="LIMIT",
            limit_price="150.00",
            stop_price=None,
        )

    def test_valid_option_order_with_open_close_passes(self):
        _validate_order_params(
            quantity="1",
            amount=None,
            order_type="LIMIT",
            limit_price="2.50",
            stop_price=None,
            instrument_type="OPTION",
            open_close_indicator="OPEN",
        )

    def test_valid_gtd_with_expiration_passes(self):
        _validate_order_params(
            quantity="1",
            amount=None,
            order_type="MARKET",
            limit_price=None,
            stop_price=None,
            time_in_force="GTD",
            expiration_time="2026-03-15T16:00:00-05:00",
        )

    def test_neither_quantity_nor_amount_passes(self):
        # Both None is allowed (API determines appropriate default)
        _validate_order_params(
            quantity=None,
            amount=None,
            order_type="MARKET",
            limit_price=None,
            stop_price=None,
        )


class TestSerialize:
    def test_serializes_decimal_as_string(self):
        class FakeModel:
            def model_dump(self, **kwargs):
                return {"price": Decimal("123.45")}

        result = _serialize(FakeModel())
        data = json.loads(result)
        assert data["price"] == "123.45"

    def test_serializes_list_of_models(self):
        class FakeModel:
            def model_dump(self, **kwargs):
                return {"id": "abc"}

        result = _serialize([FakeModel(), FakeModel()])
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["id"] == "abc"

    def test_serializes_plain_dict(self):
        result = _serialize({"key": "value"})
        data = json.loads(result)
        assert data["key"] == "value"


class TestOrderLeg:
    def test_ratio_quantity_defaults_to_one(self):
        leg = OrderLeg(symbol="AAPL260320C00280000", type="OPTION", side="BUY")
        assert leg.ratio_quantity == 1
        assert leg.open_close_indicator is None

    def test_model_validate_accepts_dict(self):
        leg = OrderLeg.model_validate(
            {
                "symbol": "AAPL260320C00280000",
                "type": "OPTION",
                "side": "SELL",
                "open_close_indicator": "OPEN",
                "ratio_quantity": 2,
            }
        )
        assert leg.side == "SELL"
        assert leg.open_close_indicator == "OPEN"
        assert leg.ratio_quantity == 2

    def test_invalid_side_rejected(self):
        with pytest.raises(ValidationError):
            OrderLeg(symbol="AAPL", type="EQUITY", side="HOLD")


class TestBuildLegRequest:
    def test_builds_option_leg_with_open_close(self):
        from public_api_sdk.models import (
            LegInstrumentType,
            OpenCloseIndicator,
            OrderSide,
        )

        leg = OrderLeg(
            symbol="AAPL260320C00280000",
            type="OPTION",
            side="BUY",
            open_close_indicator="OPEN",
            ratio_quantity=3,
        )
        req = _build_leg_request(leg)
        assert req.instrument.symbol == "AAPL260320C00280000"
        assert req.instrument.type == LegInstrumentType.OPTION
        assert req.side == OrderSide.BUY
        assert req.open_close_indicator == OpenCloseIndicator.OPEN
        assert req.ratio_quantity == 3

    def test_equity_leg_omits_open_close_indicator(self):
        leg = OrderLeg(symbol="AAPL", type="EQUITY", side="SELL")
        req = _build_leg_request(leg)
        assert req.open_close_indicator is None
