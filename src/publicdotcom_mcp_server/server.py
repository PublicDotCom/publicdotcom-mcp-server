"""
Public.com MCP Server

Exposes Public.com brokerage functionality as MCP tools for use with
any MCP-compatible AI client (Claude, etc.).

Requires:
  - PUBLIC_COM_SECRET environment variable (API key) — can also be passed
    per-request via the Authorization: Bearer <key> HTTP header.
  - PUBLIC_COM_ACCOUNT_ID environment variable (optional default account)
"""

import json
import logging
import os
from contextvars import ContextVar
from decimal import Decimal
from typing import Any, Optional
from uuid import uuid4

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# SDK import — installed via `pip install publicdotcom-py`
# ---------------------------------------------------------------------------
from public_api_sdk import (
    ApiKeyAuthConfig,
    AsyncPublicApiClient,
    AsyncPublicApiClientConfiguration,
    InstrumentType,
    OrderInstrument,
)
from public_api_sdk.models import (
    CancelAndReplaceRequest,
    HistoryRequest,
    InstrumentsRequest,
    LegInstrument,
    LegInstrumentType,
    MultilegOrderRequest,
    OpenCloseIndicator,
    OptionChainRequest,
    OptionExpirationsRequest,
    OrderExpirationRequest,
    OrderLegRequest,
    OrderRequest,
    OrderSide,
    OrderType,
    PreflightMultiLegRequest,
    PreflightRequest,
    TimeInForce,
    Trading,
    EquityMarketSession,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-request context (populated by ApiKeyMiddleware for HTTP transport)
# ---------------------------------------------------------------------------
_api_key: ContextVar[str] = ContextVar("api_key", default="")
_account_id: ContextVar[str] = ContextVar("account_id", default="")

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "Public.com",
    instructions=(
        "MCP server for the Public.com Trading API. Provides tools to view "
        "portfolio, get quotes, place/cancel orders, view history, look up "
        "instruments, and work with options — all through a Public.com "
        "brokerage account. Requires a PUBLIC_COM_SECRET environment variable."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_client(account_id: Optional[str] = None) -> AsyncPublicApiClient:
    """Create an async SDK client.

    The API key is resolved in priority order:
      1. Per-request Authorization header (HTTP transport)
      2. PUBLIC_COM_SECRET environment variable (stdio / fallback)
    """
    secret = _api_key.get() or os.environ.get("PUBLIC_COM_SECRET")
    if not secret:
        raise RuntimeError(
            "No API key found. Set the PUBLIC_COM_SECRET environment variable "
            "or pass 'Authorization: Bearer <key>' in the request header. "
            "Get your key at https://public.com/settings/v2/api"
        )
    acct = account_id or _account_id.get() or os.environ.get("PUBLIC_COM_ACCOUNT_ID")
    return AsyncPublicApiClient(
        auth_config=ApiKeyAuthConfig(api_secret_key=secret),
        config=AsyncPublicApiClientConfiguration(
            default_account_number=acct,
        ),
    )


def _serialize(obj: Any) -> str:
    """Serialize a Pydantic model (or list of models) to a JSON string."""

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return str(o)
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if hasattr(o, "model_dump"):
            return o.model_dump(by_alias=True, exclude_none=True)
        if hasattr(o, "value"):
            return o.value
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    if hasattr(obj, "model_dump"):
        data = obj.model_dump(by_alias=True, exclude_none=True)
    elif isinstance(obj, list):
        data = [
            item.model_dump(by_alias=True, exclude_none=True)
            if hasattr(item, "model_dump")
            else item
            for item in obj
        ]
    else:
        data = obj

    return json.dumps(data, indent=2, default=_default)


def _parse_instrument_type(type_str: str) -> InstrumentType:
    """Parse a string to InstrumentType enum."""
    try:
        return InstrumentType(type_str.upper())
    except ValueError:
        valid = [t.value for t in InstrumentType]
        raise ValueError(
            f"Invalid instrument type '{type_str}'. Valid types: {valid}"
        )


def _validate_order_params(
    *,
    quantity: Optional[str],
    amount: Optional[str],
    order_type: str,
    limit_price: Optional[str],
    stop_price: Optional[str],
    instrument_type: Optional[str] = None,
    open_close_indicator: Optional[str] = None,
    time_in_force: Optional[str] = None,
    expiration_time: Optional[str] = None,
) -> None:
    """Validate order parameters and raise ValueError with a clear message on violation."""
    if quantity is not None and amount is not None:
        raise ValueError("quantity and amount are mutually exclusive — provide one, not both")
    if order_type.upper() in ("LIMIT", "STOP_LIMIT") and limit_price is None:
        raise ValueError(f"limit_price is required for {order_type} orders")
    if order_type.upper() in ("STOP", "STOP_LIMIT") and stop_price is None:
        raise ValueError(f"stop_price is required for {order_type} orders")
    if instrument_type and instrument_type.upper() == "OPTION" and open_close_indicator is None:
        raise ValueError("open_close_indicator (OPEN or CLOSE) is required for OPTION orders")
    if time_in_force and time_in_force.upper() == "GTD" and expiration_time is None:
        raise ValueError("expiration_time is required when time_in_force is GTD")
    for field_name, raw_value in [
        ("quantity", quantity),
        ("amount", amount),
        ("limit_price", limit_price),
        ("stop_price", stop_price),
    ]:
        if raw_value is not None:
            try:
                Decimal(raw_value)
            except Exception:
                raise ValueError(f"{field_name} must be a numeric string, got: {raw_value!r}")


# ========================================================================
# READ-ONLY TOOLS
# ========================================================================


@mcp.tool(
    annotations={
        "title": "Check Setup",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def check_setup() -> str:
    """
    Verify that the Public.com API credentials are configured correctly.

    Checks the PUBLIC_COM_SECRET environment variable and attempts to
    authenticate. Run this first to confirm connectivity.
    """
    secret = os.environ.get("PUBLIC_COM_SECRET")
    if not secret:
        return (
            "❌ PUBLIC_COM_SECRET is not set.\n"
            "Set it with: export PUBLIC_COM_SECRET=your_api_key\n"
            "Get your key at https://public.com/settings/v2/api"
        )
    try:
        async with _get_client() as client:
            accounts = await client.get_accounts()
            acct_list = [
                f"  - {a.account_id} ({a.account_type.value})"
                for a in accounts.accounts
            ]
            default = os.environ.get("PUBLIC_COM_ACCOUNT_ID", "(not set)")
            return (
                "✅ Authenticated successfully.\n"
                f"Accounts found:\n" + "\n".join(acct_list) + "\n"
                f"Default account ID: {default}"
            )
    except Exception as e:
        return f"❌ Authentication failed: {e}"


@mcp.tool(
    annotations={
        "title": "Get Accounts",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_accounts() -> str:
    """
    List all brokerage accounts associated with the API key.

    Returns account IDs and types (BROKERAGE, HIGH_YIELD, etc.).
    """
    try:
        async with _get_client() as client:
            accounts = await client.get_accounts()
            return _serialize(accounts)
    except Exception as e:
        logger.error("get_accounts failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Portfolio",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_portfolio(account_id: Optional[str] = None) -> str:
    """
    Get a snapshot of the account portfolio.

    Returns positions, equity breakdown, buying power, and open orders.
    Only non-IRA accounts are supported.

    Args:
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        async with _get_client(account_id) as client:
            portfolio = await client.get_portfolio(account_id=account_id)
            return _serialize(portfolio)
    except Exception as e:
        logger.error("get_portfolio failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Orders",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_orders(account_id: Optional[str] = None) -> str:
    """
    Get all open/active orders on the account.

    Fetches the account portfolio and returns only the orders list.
    Returns order details including symbol, side, type, status, quantity,
    and prices.

    Args:
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        async with _get_client(account_id) as client:
            portfolio = await client.get_portfolio(account_id=account_id)
            orders = portfolio.orders or []
            return _serialize(orders)
    except Exception as e:
        logger.error("get_orders failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Order",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_order(order_id: str, account_id: Optional[str] = None) -> str:
    """
    Get the status and details of a specific order.

    Note: Order placement is asynchronous. This may return an error if
    the order has not yet been indexed.

    Args:
        order_id: The UUID of the order to look up.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        async with _get_client(account_id) as client:
            order = await client.get_order(order_id=order_id, account_id=account_id)
            return _serialize(order)
    except Exception as e:
        logger.error("get_order failed (order_id=%s): %s", order_id, e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get History",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_history(
    account_id: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    page_size: Optional[int] = None,
    next_token: Optional[str] = None,
) -> str:
    """
    Retrieve account transaction history.

    Returns trades, money movements (deposits, withdrawals, dividends),
    and position adjustments (splits, mergers).

    Args:
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
        start: Start timestamp in ISO 8601 format (e.g. 2025-01-15T09:00:00-05:00).
        end: End timestamp in ISO 8601 format.
        page_size: Max number of records to return.
        next_token: Pagination token for the next page.
    """
    from datetime import datetime as dt

    req_kwargs: dict[str, Any] = {}
    if start:
        req_kwargs["start"] = dt.fromisoformat(start)
    if end:
        req_kwargs["end"] = dt.fromisoformat(end)
    if page_size is not None:
        req_kwargs["page_size"] = page_size
    if next_token:
        req_kwargs["next_token"] = next_token

    history_request = HistoryRequest(**req_kwargs) if req_kwargs else None

    try:
        async with _get_client(account_id) as client:
            history = await client.get_history(
                history_request=history_request, account_id=account_id
            )
            return _serialize(history)
    except Exception as e:
        logger.error("get_history failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Quotes",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_quotes(
    symbols: list[str],
    instrument_type: str = "EQUITY",
    account_id: Optional[str] = None,
) -> str:
    """
    Get real-time quotes for one or more symbols.

    Returns last price, bid, ask, volume, and other market data.

    Args:
        symbols: List of ticker symbols (e.g. ["AAPL", "GOOGL"]).
        instrument_type: Type for all symbols — EQUITY, CRYPTO, or OPTION.
            Default is EQUITY. For mixed types, call this tool multiple times.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        itype = _parse_instrument_type(instrument_type)
        instruments = [OrderInstrument(symbol=s, type=itype) for s in symbols]
        async with _get_client(account_id) as client:
            quotes = await client.get_quotes(instruments=instruments, account_id=account_id)
            return _serialize(quotes)
    except Exception as e:
        logger.error("get_quotes failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Instrument",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_instrument(symbol: str, instrument_type: str = "EQUITY") -> str:
    """
    Get details about a specific tradeable instrument.

    Returns trading status, fractional trading availability, and option
    trading capabilities.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        instrument_type: EQUITY, CRYPTO, OPTION, etc. Default is EQUITY.
    """
    try:
        itype = _parse_instrument_type(instrument_type)
        async with _get_client() as client:
            instrument = await client.get_instrument(
                symbol=symbol, instrument_type=itype
            )
            return _serialize(instrument)
    except Exception as e:
        logger.error("get_instrument failed (symbol=%s): %s", symbol, e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get All Instruments",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_all_instruments(
    type_filter: Optional[list[str]] = None,
    trading_filter: Optional[list[str]] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    List all available tradeable instruments with optional filters.

    Args:
        type_filter: Filter by instrument types (e.g. ["EQUITY", "CRYPTO"]).
            Valid: EQUITY, CRYPTO, OPTION, ALT, BOND, INDEX, TREASURY.
        trading_filter: Filter by trading status (e.g. ["BUY_AND_SELL"]).
            Valid: BUY_AND_SELL, LIQUIDATION_ONLY, DISABLED.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        req_kwargs: dict[str, Any] = {}
        if type_filter:
            req_kwargs["type_filter"] = [_parse_instrument_type(t) for t in type_filter]
        if trading_filter:
            req_kwargs["trading_filter"] = [Trading(t.upper()) for t in trading_filter]

        req = InstrumentsRequest(**req_kwargs) if req_kwargs else None
        async with _get_client(account_id) as client:
            instruments = await client.get_all_instruments(
                instruments_request=req, account_id=account_id
            )
            return _serialize(instruments)
    except Exception as e:
        logger.error("get_all_instruments failed: %s", e, exc_info=True)
        return f"Error: {e}"


# ========================================================================
# OPTIONS — READ-ONLY
# ========================================================================


@mcp.tool(
    annotations={
        "title": "Get Option Expirations",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_option_expirations(
    symbol: str,
    instrument_type: str = "EQUITY",
    account_id: Optional[str] = None,
) -> str:
    """
    Get available option expiration dates for a symbol.

    Args:
        symbol: Underlying ticker symbol (e.g. "AAPL").
        instrument_type: EQUITY or UNDERLYING_SECURITY_FOR_INDEX_OPTION.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        itype = _parse_instrument_type(instrument_type)
        req = OptionExpirationsRequest(
            instrument=OrderInstrument(symbol=symbol, type=itype)
        )
        async with _get_client(account_id) as client:
            result = await client.get_option_expirations(
                expirations_request=req, account_id=account_id
            )
            return _serialize(result)
    except Exception as e:
        logger.error("get_option_expirations failed (symbol=%s): %s", symbol, e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Option Chain",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_option_chain(
    symbol: str,
    expiration_date: str,
    instrument_type: str = "EQUITY",
    account_id: Optional[str] = None,
) -> str:
    """
    Get the full option chain (calls and puts) for a symbol and expiration.

    Args:
        symbol: Underlying ticker symbol (e.g. "AAPL").
        expiration_date: Expiration date in YYYY-MM-DD format.
        instrument_type: EQUITY or UNDERLYING_SECURITY_FOR_INDEX_OPTION.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        itype = _parse_instrument_type(instrument_type)
        req = OptionChainRequest(
            instrument=OrderInstrument(symbol=symbol, type=itype),
            expiration_date=expiration_date,
        )
        async with _get_client(account_id) as client:
            result = await client.get_option_chain(
                option_chain_request=req, account_id=account_id
            )
            return _serialize(result)
    except Exception as e:
        logger.error("get_option_chain failed (symbol=%s): %s", symbol, e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Get Option Greeks",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def get_option_greeks(
    osi_symbols: list[str],
    account_id: Optional[str] = None,
) -> str:
    """
    Get option Greeks (delta, gamma, theta, vega, rho, IV) for option symbols.

    Args:
        osi_symbols: List of OSI-normalized option symbols
            (e.g. ["AAPL260320C00280000"]).
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        async with _get_client(account_id) as client:
            result = await client.get_option_greeks(
                osi_symbols=osi_symbols, account_id=account_id
            )
            return _serialize(result)
    except Exception as e:
        logger.error("get_option_greeks failed: %s", e, exc_info=True)
        return f"Error: {e}"


# ========================================================================
# PREFLIGHT (cost estimation) — READ-ONLY
# ========================================================================


@mcp.tool(
    annotations={
        "title": "Preflight Single-Leg Order",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def preflight_order(
    symbol: str,
    instrument_type: str,
    order_side: str,
    order_type: str,
    time_in_force: str = "DAY",
    quantity: Optional[str] = None,
    amount: Optional[str] = None,
    limit_price: Optional[str] = None,
    stop_price: Optional[str] = None,
    open_close_indicator: Optional[str] = None,
    expiration_time: Optional[str] = None,
    equity_market_session: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Estimate costs and impact of a potential single-leg trade before placing it.

    Returns estimated commission, regulatory fees, order value, buying power
    requirements, and margin impact. Does NOT place an order.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        instrument_type: EQUITY, OPTION, or CRYPTO.
        order_side: BUY or SELL.
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT.
        time_in_force: DAY or GTD. Default is DAY.
        quantity: Number of shares/contracts (mutually exclusive with amount).
        amount: Dollar amount (mutually exclusive with quantity).
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price: Required for STOP and STOP_LIMIT orders.
        open_close_indicator: For options only — OPEN or CLOSE.
        expiration_time: Required when time_in_force is GTD. ISO 8601 format.
        equity_market_session: CORE or EXTENDED. For equity orders only.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    from datetime import datetime as dt

    try:
        _validate_order_params(
            quantity=quantity,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            instrument_type=instrument_type,
            open_close_indicator=open_close_indicator,
            time_in_force=time_in_force,
            expiration_time=expiration_time,
        )

        exp_kwargs: dict[str, Any] = {"time_in_force": TimeInForce(time_in_force.upper())}
        if expiration_time:
            exp_kwargs["expiration_time"] = dt.fromisoformat(expiration_time)

        req_kwargs: dict[str, Any] = {
            "instrument": OrderInstrument(
                symbol=symbol, type=_parse_instrument_type(instrument_type)
            ),
            "order_side": OrderSide(order_side.upper()),
            "order_type": OrderType(order_type.upper()),
            "expiration": OrderExpirationRequest(**exp_kwargs),
        }
        if quantity is not None:
            req_kwargs["quantity"] = Decimal(quantity)
        if amount is not None:
            req_kwargs["amount"] = Decimal(amount)
        if limit_price is not None:
            req_kwargs["limit_price"] = Decimal(limit_price)
        if stop_price is not None:
            req_kwargs["stop_price"] = Decimal(stop_price)
        if open_close_indicator:
            req_kwargs["open_close_indicator"] = OpenCloseIndicator(
                open_close_indicator.upper()
            )
        if equity_market_session:
            req_kwargs["equity_market_session"] = EquityMarketSession(
                equity_market_session.upper()
            )

        req = PreflightRequest(**req_kwargs)
        async with _get_client(account_id) as client:
            result = await client.perform_preflight_calculation(
                preflight_request=req, account_id=account_id
            )
            return _serialize(result)
    except Exception as e:
        logger.error("preflight_order failed: %s", e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Preflight Multi-Leg Order",
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
)
async def preflight_multileg_order(
    legs: list[dict],
    limit_price: str,
    time_in_force: str = "DAY",
    quantity: Optional[int] = None,
    expiration_time: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Estimate costs for a multi-leg (options strategy) trade before placing it.

    Does NOT place an order.

    Args:
        legs: List of leg objects. Each leg must have:
            - symbol (str): The option/equity symbol (e.g. "SPY260313P00670000")
            - type (str): EQUITY or OPTION
            - side (str): BUY or SELL
            - open_close_indicator (str, optional): OPEN or CLOSE (required for options)
            - ratio_quantity (int, optional): Ratio between legs (default 1)
          Example: [{"symbol": "SPY260313P00670000", "type": "OPTION", "side": "SELL",
                     "open_close_indicator": "OPEN", "ratio_quantity": 1},
                    {"symbol": "SPY260313P00665000", "type": "OPTION", "side": "BUY",
                     "open_close_indicator": "OPEN", "ratio_quantity": 1}]
        limit_price: The limit price for the spread.
        time_in_force: DAY or GTD. Default is DAY.
        quantity: Number of spreads. Must be > 0.
        expiration_time: Required when time_in_force is GTD. ISO 8601 format.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    from datetime import datetime as dt

    try:
        if time_in_force.upper() == "GTD" and expiration_time is None:
            raise ValueError("expiration_time is required when time_in_force is GTD")
        try:
            Decimal(limit_price)
        except Exception:
            raise ValueError(f"limit_price must be a numeric string, got: {limit_price!r}")

        exp_kwargs: dict[str, Any] = {"time_in_force": TimeInForce(time_in_force.upper())}
        if expiration_time:
            exp_kwargs["expiration_time"] = dt.fromisoformat(expiration_time)

        leg_requests = []
        for leg in legs:
            # Support both flat format {"symbol": ..., "type": ...} and
            # nested format {"instrument": {"symbol": ..., "type": ...}, ...}
            instrument = leg.get("instrument") or {}
            symbol = instrument.get("symbol") or leg["symbol"]
            inst_type = instrument.get("type") or leg["type"]
            leg_kwargs: dict[str, Any] = {
                "instrument": LegInstrument(
                    symbol=symbol,
                    type=LegInstrumentType(inst_type.upper()),
                ),
                "side": OrderSide(leg["side"].upper()),
            }
            if leg.get("open_close_indicator"):
                leg_kwargs["open_close_indicator"] = OpenCloseIndicator(
                    leg["open_close_indicator"].upper()
                )
            if leg.get("ratio_quantity"):
                leg_kwargs["ratio_quantity"] = leg["ratio_quantity"]
            leg_requests.append(OrderLegRequest(**leg_kwargs))

        req_kwargs: dict[str, Any] = {
            "order_type": OrderType.LIMIT,
            "expiration": OrderExpirationRequest(**exp_kwargs),
            "limit_price": Decimal(limit_price),
            "legs": leg_requests,
        }
        if quantity is not None:
            req_kwargs["quantity"] = quantity

        req = PreflightMultiLegRequest(**req_kwargs)
        async with _get_client(account_id) as client:
            result = await client.perform_multi_leg_preflight_calculation(
                preflight_request=req, account_id=account_id
            )
            return _serialize(result)
    except Exception as e:
        logger.error("preflight_multileg_order failed: %s", e, exc_info=True)
        return f"Error: {e}"


# ========================================================================
# WRITE TOOLS (order placement, cancellation)
# ========================================================================


@mcp.tool(
    annotations={
        "title": "Place Order",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def place_order(
    symbol: str,
    instrument_type: str,
    order_side: str,
    order_type: str,
    time_in_force: str = "DAY",
    quantity: Optional[str] = None,
    amount: Optional[str] = None,
    limit_price: Optional[str] = None,
    stop_price: Optional[str] = None,
    open_close_indicator: Optional[str] = None,
    expiration_time: Optional[str] = None,
    equity_market_session: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Place a single-leg order (buy/sell stocks, crypto, or options).

    ⚠️ This executes a real trade. Consider running preflight_order first.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        instrument_type: EQUITY, OPTION, or CRYPTO.
        order_side: BUY or SELL.
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT.
        time_in_force: DAY or GTD. Default is DAY.
        quantity: Number of shares/contracts (mutually exclusive with amount).
        amount: Dollar amount (mutually exclusive with quantity).
        limit_price: Required for LIMIT and STOP_LIMIT orders.
        stop_price: Required for STOP and STOP_LIMIT orders.
        open_close_indicator: For options only — OPEN or CLOSE.
        expiration_time: Required when time_in_force is GTD. ISO 8601 format.
        equity_market_session: CORE or EXTENDED. For equity orders only.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    from datetime import datetime as dt

    order_id = str(uuid4())

    try:
        _validate_order_params(
            quantity=quantity,
            amount=amount,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            instrument_type=instrument_type,
            open_close_indicator=open_close_indicator,
            time_in_force=time_in_force,
            expiration_time=expiration_time,
        )

        exp_kwargs: dict[str, Any] = {"time_in_force": TimeInForce(time_in_force.upper())}
        if expiration_time:
            exp_kwargs["expiration_time"] = dt.fromisoformat(expiration_time)

        req_kwargs: dict[str, Any] = {
            "order_id": order_id,
            "instrument": OrderInstrument(
                symbol=symbol, type=_parse_instrument_type(instrument_type)
            ),
            "order_side": OrderSide(order_side.upper()),
            "order_type": OrderType(order_type.upper()),
            "expiration": OrderExpirationRequest(**exp_kwargs),
        }
        if quantity is not None:
            req_kwargs["quantity"] = Decimal(quantity)
        if amount is not None:
            req_kwargs["amount"] = Decimal(amount)
        if limit_price is not None:
            req_kwargs["limit_price"] = Decimal(limit_price)
        if stop_price is not None:
            req_kwargs["stop_price"] = Decimal(stop_price)
        if open_close_indicator:
            req_kwargs["open_close_indicator"] = OpenCloseIndicator(
                open_close_indicator.upper()
            )
        if equity_market_session:
            req_kwargs["equity_market_session"] = EquityMarketSession(
                equity_market_session.upper()
            )

        req = OrderRequest(**req_kwargs)
        logger.info(
            "Placing order: order_id=%s symbol=%s side=%s type=%s qty=%s amount=%s",
            order_id, symbol, order_side, order_type, quantity, amount,
        )
        async with _get_client(account_id) as client:
            new_order = await client.place_order(
                order_request=req, account_id=account_id
            )
        logger.info("Order accepted: order_id=%s returned_order_id=%s", order_id, new_order.order_id)
        return json.dumps(
            {
                "order_id": new_order.order_id,
                "status": "submitted",
                "message": (
                    "Order submitted. Placement is asynchronous — use "
                    "get_order to confirm status."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error("place_order failed (order_id=%s): %s", order_id, e, exc_info=True)
        return json.dumps(
            {
                "order_id": order_id,
                "status": "error",
                "message": (
                    f"Order submission failed: {e}. "
                    f"Use get_order({order_id!r}) to check if the order was accepted."
                ),
            },
            indent=2,
        )


@mcp.tool(
    annotations={
        "title": "Place Multi-Leg Order",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def place_multileg_order(
    legs: list[dict],
    quantity: int,
    limit_price: str,
    time_in_force: str = "DAY",
    expiration_time: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Place a multi-leg order (options strategies: spreads, straddles, etc.).

    ⚠️ This executes a real trade. Consider running preflight_multileg_order first.

    Args:
        legs: List of leg objects. Each leg must have:
            - symbol (str): The option/equity symbol (e.g. "SPY260313P00670000")
            - type (str): EQUITY or OPTION
            - side (str): BUY or SELL
            - open_close_indicator (str, optional): OPEN or CLOSE (required for options)
            - ratio_quantity (int, optional): Ratio between legs (default 1)
          Example: [{"symbol": "SPY260313P00670000", "type": "OPTION", "side": "SELL",
                     "open_close_indicator": "OPEN", "ratio_quantity": 1},
                    {"symbol": "SPY260313P00665000", "type": "OPTION", "side": "BUY",
                     "open_close_indicator": "OPEN", "ratio_quantity": 1}]
        quantity: Number of spreads. Must be > 0.
        limit_price: Limit price. Positive for debit, negative for credit.
        time_in_force: DAY or GTD. Default is DAY.
        expiration_time: Required when time_in_force is GTD. ISO 8601 format.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    from datetime import datetime as dt

    order_id = str(uuid4())

    try:
        if time_in_force.upper() == "GTD" and expiration_time is None:
            raise ValueError("expiration_time is required when time_in_force is GTD")
        try:
            Decimal(limit_price)
        except Exception:
            raise ValueError(f"limit_price must be a numeric string, got: {limit_price!r}")

        exp_kwargs: dict[str, Any] = {"time_in_force": TimeInForce(time_in_force.upper())}
        if expiration_time:
            exp_kwargs["expiration_time"] = dt.fromisoformat(expiration_time)

        leg_requests = []
        for leg in legs:
            # Support both flat format {"symbol": ..., "type": ...} and
            # nested format {"instrument": {"symbol": ..., "type": ...}, ...}
            instrument = leg.get("instrument") or {}
            symbol = instrument.get("symbol") or leg["symbol"]
            inst_type = instrument.get("type") or leg["type"]
            leg_kwargs: dict[str, Any] = {
                "instrument": LegInstrument(
                    symbol=symbol,
                    type=LegInstrumentType(inst_type.upper()),
                ),
                "side": OrderSide(leg["side"].upper()),
            }
            if leg.get("open_close_indicator"):
                leg_kwargs["open_close_indicator"] = OpenCloseIndicator(
                    leg["open_close_indicator"].upper()
                )
            if leg.get("ratio_quantity"):
                leg_kwargs["ratio_quantity"] = leg["ratio_quantity"]
            leg_requests.append(OrderLegRequest(**leg_kwargs))

        req = MultilegOrderRequest(
            order_id=order_id,
            quantity=quantity,
            type=OrderType.LIMIT,
            limit_price=Decimal(limit_price),
            expiration=OrderExpirationRequest(**exp_kwargs),
            legs=leg_requests,
        )
        logger.info(
            "Placing multileg order: order_id=%s legs=%d qty=%d limit=%s",
            order_id, len(legs), quantity, limit_price,
        )
        async with _get_client(account_id) as client:
            new_order = await client.place_multileg_order(
                order_request=req, account_id=account_id
            )
        logger.info("Multileg order accepted: order_id=%s returned_order_id=%s", order_id, new_order.order_id)
        return json.dumps(
            {
                "order_id": new_order.order_id,
                "status": "submitted",
                "message": (
                    "Multi-leg order submitted. Placement is asynchronous — "
                    "use get_order to confirm status."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error("place_multileg_order failed (order_id=%s): %s", order_id, e, exc_info=True)
        return json.dumps(
            {
                "order_id": order_id,
                "status": "error",
                "message": (
                    f"Multi-leg order submission failed: {e}. "
                    f"Use get_order({order_id!r}) to check if the order was accepted."
                ),
            },
            indent=2,
        )


@mcp.tool(
    annotations={
        "title": "Cancel Order",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def cancel_order(
    order_id: str,
    account_id: Optional[str] = None,
) -> str:
    """
    Cancel an existing order.

    Note: While most cancellations are processed immediately during market
    hours, this is not guaranteed. Use get_order to confirm cancellation.

    Args:
        order_id: The UUID of the order to cancel.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    try:
        logger.info("Cancelling order: order_id=%s account_id=%s", order_id, account_id)
        async with _get_client(account_id) as client:
            await client.cancel_order(order_id=order_id, account_id=account_id)
        logger.info("Cancel submitted: order_id=%s", order_id)
        return json.dumps(
            {
                "order_id": order_id,
                "status": "cancel_requested",
                "message": (
                    "Cancellation submitted. Use get_order to confirm "
                    "the order was cancelled."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error("cancel_order failed (order_id=%s): %s", order_id, e, exc_info=True)
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "title": "Cancel and Replace Order",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": True,
    },
)
async def cancel_and_replace_order(
    order_id: str,
    order_type: str,
    time_in_force: str = "DAY",
    quantity: Optional[str] = None,
    limit_price: Optional[str] = None,
    stop_price: Optional[str] = None,
    expiration_time: Optional[str] = None,
    account_id: Optional[str] = None,
) -> str:
    """
    Atomically cancel an existing order and replace it with new parameters.

    ⚠️ This modifies an existing order.

    Args:
        order_id: UUID of the existing order to cancel and replace.
        order_type: MARKET, LIMIT, STOP, or STOP_LIMIT for the replacement.
        time_in_force: DAY or GTD. Default is DAY.
        quantity: New quantity for the replacement order.
        limit_price: New limit price (for LIMIT/STOP_LIMIT orders).
        stop_price: New stop price (for STOP/STOP_LIMIT orders).
        expiration_time: Required when time_in_force is GTD. ISO 8601 format.
        account_id: Account ID. Optional if PUBLIC_COM_ACCOUNT_ID is set.
    """
    from datetime import datetime as dt

    request_id = str(uuid4())

    try:
        _validate_order_params(
            quantity=quantity,
            amount=None,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            expiration_time=expiration_time,
        )

        exp_kwargs: dict[str, Any] = {"time_in_force": TimeInForce(time_in_force.upper())}
        if expiration_time:
            exp_kwargs["expiration_time"] = dt.fromisoformat(expiration_time)

        req_kwargs: dict[str, Any] = {
            "order_id": order_id,
            "request_id": request_id,
            "order_type": OrderType(order_type.upper()),
            "expiration": OrderExpirationRequest(**exp_kwargs),
        }
        if quantity is not None:
            req_kwargs["quantity"] = Decimal(quantity)
        if limit_price is not None:
            req_kwargs["limit_price"] = Decimal(limit_price)
        if stop_price is not None:
            req_kwargs["stop_price"] = Decimal(stop_price)

        req = CancelAndReplaceRequest(**req_kwargs)
        logger.info(
            "Cancel-replace: original_order_id=%s request_id=%s type=%s",
            order_id, request_id, order_type,
        )
        async with _get_client(account_id) as client:
            new_order = await client.cancel_and_replace_order(
                request=req, account_id=account_id
            )
        logger.info(
            "Cancel-replace accepted: old_order_id=%s new_order_id=%s",
            order_id, new_order.order_id,
        )
        return json.dumps(
            {
                "new_order_id": new_order.order_id,
                "cancelled_order_id": order_id,
                "status": "submitted",
                "message": (
                    "Cancel-and-replace submitted. Use get_order to "
                    "confirm the new order status."
                ),
            },
            indent=2,
        )
    except Exception as e:
        logger.error(
            "cancel_and_replace_order failed (original_order_id=%s request_id=%s): %s",
            order_id, request_id, e, exc_info=True,
        )
        return json.dumps(
            {
                "original_order_id": order_id,
                "request_id": request_id,
                "status": "error",
                "message": (
                    f"Cancel-and-replace failed: {e}. "
                    f"Use get_order({order_id!r}) to check the original order status."
                ),
            },
            indent=2,
        )


# ========================================================================
# ASGI middleware — extracts API key from Authorization header
# ========================================================================

class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Populate per-request ContextVars from HTTP headers.

    Clients should send:
        Authorization: Bearer <PUBLIC_COM_SECRET>

    Optionally also:
        X-Account-Id: <account_number>

    Falls back to environment variables when headers are absent (useful for
    single-tenant deployments where the key is baked into the environment).
    """

    async def dispatch(self, request, call_next):
        auth = request.headers.get("Authorization", "")
        key = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
        acct = request.headers.get("X-Account-Id", "")

        key_token = _api_key.set(key or os.environ.get("PUBLIC_COM_SECRET", ""))
        acct_token = _account_id.set(acct or os.environ.get("PUBLIC_COM_ACCOUNT_ID", ""))
        try:
            return await call_next(request)
        finally:
            _api_key.reset(key_token)
            _account_id.reset(acct_token)


# ========================================================================
# Entry point
# ========================================================================

def main():
    """Run the MCP server.

    Transport is selected via the MCP_TRANSPORT environment variable:
      - "streamable-http" (default) — for hosted/remote deployments
      - "stdio"                     — for local Claude Desktop use

    HTTP server binds to HOST (default 0.0.0.0) and PORT (default 8000).
    """
    import uvicorn

    transport = os.environ.get("MCP_TRANSPORT", "streamable-http")

    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", "8000"))
        app = ApiKeyMiddleware(mcp.streamable_http_app())
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
