"""
Parametrized tests confirming place_order and preflight_order share
identical validation behaviour via _validate_order_params().
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from publicdotcom_mcp_server.server import place_order, preflight_order

BASE_KWARGS = {
    "symbol": "AAPL",
    "instrument_type": "EQUITY",
    "order_side": "BUY",
    "order_type": "MARKET",
}

SHARED_VALIDATION_CASES = [
    (
        "quantity and amount both provided",
        {"quantity": "10", "amount": "500"},
        "mutually exclusive",
    ),
    (
        "LIMIT order without limit_price",
        {"quantity": "1", "order_type": "LIMIT"},
        "limit_price",
    ),
    (
        "STOP order without stop_price",
        {"quantity": "1", "order_type": "STOP"},
        "stop_price",
    ),
    (
        "OPTION without open_close_indicator",
        {
            "quantity": "1",
            "instrument_type": "OPTION",
            "order_type": "LIMIT",
            "limit_price": "2.50",
        },
        "open_close_indicator",
    ),
    (
        "GTD without expiration_time",
        {"quantity": "1", "time_in_force": "GTD"},
        "expiration_time",
    ),
    (
        "non-numeric quantity",
        {"quantity": "bad"},
        "numeric string",
    ),
]


@pytest.mark.parametrize("description,overrides,error_fragment", SHARED_VALIDATION_CASES)
async def test_place_order_validation(patch_get_client, description, overrides, error_fragment):
    """place_order should return an error JSON for each invalid input."""
    kwargs = {**BASE_KWARGS, **overrides}
    result = await place_order(**kwargs)
    data = json.loads(result)
    assert data["status"] == "error", f"Expected error status for: {description}"
    assert error_fragment.lower() in data["message"].lower(), (
        f"Expected '{error_fragment}' in error message for: {description}\n"
        f"Got: {data['message']}"
    )


@pytest.mark.parametrize("description,overrides,error_fragment", SHARED_VALIDATION_CASES)
async def test_preflight_order_validation(patch_get_client, description, overrides, error_fragment):
    """preflight_order should return the same validation errors as place_order."""
    kwargs = {**BASE_KWARGS, **overrides}
    result = await preflight_order(**kwargs)
    assert error_fragment.lower() in result.lower(), (
        f"Expected '{error_fragment}' in error response for: {description}\n"
        f"Got: {result}"
    )
    # preflight errors are plain strings (not JSON) — confirm it's not accidentally
    # returning a success JSON payload
    assert not result.strip().startswith("{"), (
        f"Expected plain error string for: {description}, got JSON: {result}"
    )
