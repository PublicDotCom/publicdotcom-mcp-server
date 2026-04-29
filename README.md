# Public.com MCP Server

[![License](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple?style=flat-square)](https://modelcontextprotocol.io)

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that connects AI assistants to your [Public.com](https://public.com) brokerage account. Trade stocks, options, and crypto — get quotes, manage orders, and view your portfolio — all through natural language.

> **Disclaimer:** For illustrative and informational purposes only. Not investment advice or recommendations. Use at your own risk.

## Tools

### Read-Only

| Tool | Description |
|------|-------------|
| `check_setup` | Verify API credentials and connectivity |
| `get_accounts` | List all brokerage accounts |
| `get_portfolio` | View positions, equity, buying power, open orders |
| `get_orders` | List active/open orders |
| `get_order` | Get status of a specific order |
| `get_history` | Transaction history (trades, deposits, dividends, etc.) |
| `get_quotes` | Real-time quotes for stocks, crypto, options |
| `get_instrument` | Details about a specific tradeable instrument |
| `get_all_instruments` | List all available instruments with filters |
| `get_option_expirations` | Available expiration dates for options |
| `get_option_chain` | Full option chain (calls + puts) for a symbol |
| `get_option_greeks` | Greeks (delta, gamma, theta, vega, rho, IV) for multiple options |
| `get_option_greek` | Greeks for a single option symbol |
| `preflight_order` | Estimate costs/impact before placing a single-leg order |
| `preflight_multileg_order` | Estimate costs for multi-leg options strategies |
| `preflight_short_order` | Estimate costs before placing a short-sale order |
| `preflight_call_credit_spread` | Estimate costs for a Bear Call Spread |
| `preflight_call_debit_spread` | Estimate costs for a Bull Call Spread |
| `preflight_put_credit_spread` | Estimate costs for a Bull Put Spread |
| `preflight_put_debit_spread` | Estimate costs for a Bear Put Spread |

### Write (Destructive)

| Tool | Description |
|------|-------------|
| `place_order` | Place a single-leg order (stocks, crypto, options) |
| `place_multileg_order` | Place multi-leg orders (spreads, straddles, etc.) |
| `place_call_credit_spread` | Place a Bear Call Spread |
| `place_call_debit_spread` | Place a Bull Call Spread |
| `place_put_credit_spread` | Place a Bull Put Spread |
| `place_put_debit_spread` | Place a Bear Put Spread |
| `place_short_order` | Place an equity short-sale order |
| `flatten_and_go_short` | Sell an existing long position then go short (experimental) |
| `cancel_order` | Cancel an existing order |
| `cancel_and_replace_order` | Atomically cancel and replace an order |

## Prerequisites

- **Python 3.10+**
- **Public.com account** — [Sign up](https://public.com/signup)
- **Public.com API key** — [Get one here](https://public.com/settings/v2/api)

## Installation

```bash
pip install publicdotcom-mcp-server
```

Or install from source:

```bash
git clone https://github.com/publicdotcom/publicdotcom-mcp-server.git
cd publicdotcom-mcp-server
pip install .
```

## Configuration

Set your API credentials as environment variables:

```bash
# Required
export PUBLIC_COM_SECRET=your_api_secret_key

# Optional — sets a default account so you don't need to specify it each time
export PUBLIC_COM_ACCOUNT_ID=your_account_id
```

## Usage

### Claude Desktop

Add this to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "public-com": {
      "command": "publicdotcom-mcp-server",
      "env": {
        "PUBLIC_COM_SECRET": "your_api_secret_key",
        "PUBLIC_COM_ACCOUNT_ID": "your_account_id"
      }
    }
  }
}
```

### Claude Desktop (using uvx)

If you prefer using `uvx` (no pre-install needed):

```json
{
  "mcpServers": {
    "public-com": {
      "command": "uvx",
      "args": ["publicdotcom-mcp-server"],
      "env": {
        "PUBLIC_COM_SECRET": "your_api_secret_key",
        "PUBLIC_COM_ACCOUNT_ID": "your_account_id"
      }
    }
  }
}
```

### Running Directly

```bash
# stdio transport (default — for Claude Desktop, Claude Code, etc.)
publicdotcom-mcp-server

# Or run as a Python module
python -m publicdotcom_mcp_server
```

### Hosted / Remote Deployment

For remote deployments (behind a reverse proxy or load balancer), switch to the streamable-HTTP transport:

```bash
export MCP_TRANSPORT=streamable-http
export PUBLIC_COM_SECRET=your_api_secret_key
export PORT=8000  # optional, defaults to 8000
export HOST=0.0.0.0  # optional, defaults to 0.0.0.0
publicdotcom-mcp-server
```

In this mode the server listens for MCP requests at `POST /mcp`. Clients authenticate per-request via an `Authorization: Bearer <key>` header, which takes priority over the `PUBLIC_COM_SECRET` environment variable — useful for multi-tenant deployments.

### Testing with MCP Inspector

```bash
npx @modelcontextprotocol/inspector publicdotcom-mcp-server
```

## Development

```bash
# Clone and install in development mode
git clone https://github.com/publicdotcom/publicdotcom-mcp-server.git
cd publicdotcom-mcp-server
pip install -e ".[dev]"

# Run tests
pytest

# Run the server locally
python -m publicdotcom_mcp_server
```

## How It Works

This server wraps the [`publicdotcom-py`](https://pypi.org/project/publicdotcom-py/) Python SDK, exposing each API operation as an MCP tool. The MCP protocol allows AI clients to discover and call these tools through a standardized interface.

```
AI Client (Claude, etc.)
    ↕ MCP Protocol (stdio)
Public.com MCP Server
    ↕ HTTPS
Public.com Trading API
```

All tools include proper [MCP tool annotations](https://modelcontextprotocol.io/docs/concepts/tools#tool-annotations):
- Read-only tools are marked with `readOnlyHint: true`
- Write tools are marked with `destructiveHint: true`

## License

[Apache 2.0](LICENSE)
