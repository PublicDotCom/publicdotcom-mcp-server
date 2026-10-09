"""Tests for the event-contract tools (SDK 0.1.25 bars, SDK 0.1.26 discovery).

The tools are thin: they parse their arguments into the SDK's request models and
serialize the SDK's response models. Request building is exercised against the
real SDK models; the client is mocked.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from public_api_sdk.models import (
    EventCategoriesResponse,
    EventContractBarPeriod,
    EventContractChartsResponse,
    EventDetails,
    EventFrequency,
    EventOutcomeState,
    EventSortingMode,
    EventSummaryPage,
    EventSummaryRequest,
)

from publicdotcom_mcp_server.server import (
    _build_event_summary_request,
    get_event_categories,
    get_event_contract_bars,
    get_event_details,
    get_event_summary,
    get_price_history,
    get_quotes,
)

EVENT_SYMBOL = "KALSHI.KXBALANCESHEET-EO26"
EVENT_ID = "KALSHI.KXBALANCESHEET-EO26-EVENT"
CONTRACT_SYMBOL = "KALSHI.KXBALANCESHEET-EO26-6.6.Y-EVENTCONTRACT"


def _summary_request(**overrides) -> EventSummaryRequest:
    kwargs = {
        "sorting_mode": "VOLUME",
        "category": None,
        "subcategory": None,
        "event_symbols": None,
        "frequencies": None,
        "resolution_time_start": None,
        "resolution_time_end": None,
        "display_resolved_events": None,
        "created_within_days": None,
        "next_token": None,
    }
    kwargs.update(overrides)
    return _build_event_summary_request(**kwargs)


def _body(req: EventSummaryRequest) -> dict:
    """The JSON body the SDK sends for this request."""
    return req.model_dump(by_alias=True, exclude_none=True)


class TestBuildEventSummaryRequest:
    def test_defaults_send_only_the_sorting_mode(self):
        assert _body(_summary_request()) == {"sortingMode": "VOLUME"}

    def test_sorting_mode_is_case_insensitive(self):
        assert _summary_request(sorting_mode="recently_added").sorting_mode is (
            EventSortingMode.RECENTLY_ADDED
        )

    def test_invalid_sorting_mode_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="Invalid sorting_mode: 'POPULAR'") as exc:
            _summary_request(sorting_mode="POPULAR")
        assert "EXPIRATION" in str(exc.value)

    def test_top_level_fields_map_to_the_api_names(self):
        body = _body(
            _summary_request(
                sorting_mode="EXPIRATION",
                category="Economics",
                subcategory="Fed",
                display_resolved_events=False,
                created_within_days=7,
                next_token="tok-2",
            )
        )
        assert body == {
            "sortingMode": "EXPIRATION",
            "category": "Economics",
            "subcategory": "Fed",
            "displayResolvedEvents": False,
            "createdWithinDays": 7,
            "nextToken": "tok-2",
        }

    def test_symbols_only_fills_frequencies_with_all(self):
        # The API requires both lists whenever the filters block is sent.
        body = _body(_summary_request(event_symbols=["kalshi.kxbalancesheet-eo26"]))
        assert body["filters"] == {"eventSymbols": [EVENT_SYMBOL], "frequencies": ["ALL"]}

    def test_frequencies_only_sends_an_empty_symbol_list(self):
        body = _body(_summary_request(frequencies=["one_day", "ONE_WEEK"]))
        assert body["filters"] == {"eventSymbols": [], "frequencies": ["ONE_DAY", "ONE_WEEK"]}

    def test_comma_separated_symbols_are_split_and_blanks_dropped(self):
        req = _summary_request(event_symbols=["A-1, B-2", " ", "c-3"])
        assert req.filters.event_symbols == ["A-1", "B-2", "C-3"]

    def test_resolution_window_alone_still_sends_the_required_lists(self):
        req = _summary_request(
            resolution_time_start="2026-12-01T00:00:00Z",
            resolution_time_end="2026-12-31T23:59:59+00:00",
        )
        assert req.filters.event_symbols == []
        assert req.filters.frequencies == [EventFrequency.ALL]
        assert req.filters.resolution_time_start == datetime(2026, 12, 1, tzinfo=timezone.utc)
        assert _body(req)["filters"]["resolutionTimeEnd"] == "2026-12-31T23:59:59+00:00"

    def test_invalid_frequency_lists_the_real_values_but_not_unknown(self):
        # EventFrequency("anything") silently becomes UNKNOWN, so it must be
        # checked explicitly rather than trusted to raise.
        with pytest.raises(ValueError, match="Invalid frequency: 'HOURLY'") as exc:
            _summary_request(frequencies=["HOURLY"])
        assert "ONE_HOUR" in str(exc.value)
        assert "UNKNOWN" not in str(exc.value)

    def test_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="Invalid frequency: 'UNKNOWN'"):
            _summary_request(frequencies=["UNKNOWN"])

    def test_bad_resolution_timestamp_names_the_field(self):
        with pytest.raises(ValueError, match="resolution_time_start must be an ISO 8601"):
            _summary_request(resolution_time_start="next friday")


class TestGetEventCategories:
    async def test_returns_serialized_categories(self, patch_get_client):
        patch_get_client.get_event_categories = AsyncMock(
            return_value=EventCategoriesResponse.model_validate(
                {
                    "categories": [
                        {
                            "category": "Economics",
                            "subcategories": ["Fed", "GDP"],
                            "eventFrequency": {"show": True, "frequencies": ["ALL", "ONE_DAY"]},
                        }
                    ]
                }
            )
        )

        data = json.loads(await get_event_categories())

        assert data["categories"][0]["category"] == "Economics"
        assert data["categories"][0]["subcategories"] == ["Fed", "GDP"]

    async def test_api_error_returns_error_string(self, patch_get_client):
        patch_get_client.get_event_categories = AsyncMock(side_effect=Exception("boom"))

        result = await get_event_categories()

        assert result.startswith("Error:")
        assert "boom" in result


class TestGetEventSummary:
    async def test_default_call_sends_a_volume_sorted_request(self, patch_get_client):
        patch_get_client.get_event_summary = AsyncMock(
            return_value=EventSummaryPage.model_validate({"content": []})
        )

        await get_event_summary()

        req = patch_get_client.get_event_summary.call_args.kwargs["event_summary_request"]
        assert isinstance(req, EventSummaryRequest)
        assert _body(req) == {"sortingMode": "VOLUME"}

    async def test_page_and_next_token_are_serialized(self, patch_get_client):
        patch_get_client.get_event_summary = AsyncMock(
            return_value=EventSummaryPage.model_validate(
                {
                    "content": [
                        {
                            "eventSymbol": EVENT_SYMBOL,
                            "title": "Fed balance sheet end of 2026",
                            "volume": "12345",
                            "resolved": False,
                        }
                    ],
                    "nextToken": "tok-2",
                }
            )
        )

        data = json.loads(await get_event_summary(next_token="tok-1"))

        req = patch_get_client.get_event_summary.call_args.kwargs["event_summary_request"]
        assert req.next_token == "tok-1"
        assert data["nextToken"] == "tok-2"
        assert data["content"][0]["eventSymbol"] == EVENT_SYMBOL

    async def test_bad_argument_is_reported_not_raised(self, patch_get_client):
        patch_get_client.get_event_summary = AsyncMock()

        result = await get_event_summary(frequencies=["HOURLY"])

        assert result.startswith("Error:")
        assert "Invalid frequency" in result
        patch_get_client.get_event_summary.assert_not_called()

    async def test_api_error_returns_error_string(self, patch_get_client):
        patch_get_client.get_event_summary = AsyncMock(side_effect=Exception("400 bad body"))

        result = await get_event_summary()

        assert result.startswith("Error:")
        assert "400 bad body" in result


class TestGetEventDetails:
    async def test_forwards_symbol_and_outcome_flag(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(
            return_value=EventDetails.model_validate(
                {
                    "eventSymbol": EVENT_SYMBOL,
                    "outcomeCount": 12,
                    "outcomes": [
                        {
                            "outcomeId": "o-1",
                            "title": "ABOVE $6.6 TRILLION",
                            "state": "STATE_OPEN",
                            "contracts": [
                                {
                                    "symbol": "KALSHI.KXBALANCESHEET-EO26-6.6.Y",
                                    "predictedOutcome": "YES",
                                    "bid": "0.41",
                                    "ask": "0.43",
                                }
                            ],
                        }
                    ],
                }
            )
        )

        data = json.loads(await get_event_details(" kalshi.kxbalancesheet-eo26 ", False))

        assert patch_get_client.get_event_details.call_args.kwargs == {
            "event_symbol": EVENT_SYMBOL,
            "include_all_outcomes": False,
        }
        assert data["outcomeCount"] == 12
        assert data["outcomes"][0]["state"] == EventOutcomeState.STATE_OPEN.value
        assert data["outcomes"][0]["contracts"][0]["ask"] == "0.43"

    async def test_all_outcomes_is_the_default(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(
            return_value=EventDetails.model_validate({"eventSymbol": EVENT_SYMBOL})
        )

        await get_event_details(EVENT_SYMBOL)

        assert patch_get_client.get_event_details.call_args.kwargs["include_all_outcomes"] is True

    async def test_not_found_returns_error_string(self, patch_get_client):
        patch_get_client.get_event_details = AsyncMock(
            side_effect=Exception("No event found (code 7004)")
        )

        result = await get_event_details("KALSHI.NOPE")

        assert result.startswith("Error:")
        assert "7004" in result


class TestGetEventContractBars:
    async def test_forwards_ids_and_parsed_period(self, patch_get_client):
        patch_get_client.get_event_contract_bars = AsyncMock(
            return_value=EventContractChartsResponse.model_validate(
                {
                    "period": "WEEK",
                    "charts": [
                        {
                            "symbol": CONTRACT_SYMBOL,
                            "bars": [
                                {
                                    "timestamp": "2026-10-01T00:00:00Z",
                                    "open": "0.40",
                                    "high": "0.45",
                                    "low": "0.39",
                                    "close": "0.42",
                                    "value": "0.42",
                                    "volume": "1500",
                                }
                            ],
                        }
                    ],
                }
            )
        )

        data = json.loads(
            await get_event_contract_bars(
                event_id="kalshi.kxbalancesheet-eo26-event",
                period="week",
                symbols=[" " + CONTRACT_SYMBOL.lower()],
            )
        )

        kwargs = patch_get_client.get_event_contract_bars.call_args.kwargs
        assert kwargs == {
            "event_id": EVENT_ID,
            "period": EventContractBarPeriod.WEEK,
            "symbols": [CONTRACT_SYMBOL],
        }
        assert data["charts"][0]["symbol"] == CONTRACT_SYMBOL

    async def test_invalid_period_is_reported_without_calling_the_api(self, patch_get_client):
        patch_get_client.get_event_contract_bars = AsyncMock()

        result = await get_event_contract_bars(EVENT_ID, "YEAR", [CONTRACT_SYMBOL])

        assert result.startswith("Error:")
        patch_get_client.get_event_contract_bars.assert_not_called()

    async def test_sdk_symbol_limit_error_is_reported(self, patch_get_client):
        patch_get_client.get_event_contract_bars = AsyncMock(
            side_effect=ValueError("At most 8 event contract symbols are allowed per request, got 9")
        )

        result = await get_event_contract_bars(EVENT_ID, "DAY", [CONTRACT_SYMBOL] * 9)

        assert result.startswith("Error:")
        assert "At most 8" in result


class TestEventContractInstrumentType:
    async def test_quotes_accept_eventcontract(self, patch_get_client):
        patch_get_client.get_quotes = AsyncMock(return_value=[])

        await get_quotes(symbols=[CONTRACT_SYMBOL], instrument_type="eventcontract")

        instruments = patch_get_client.get_quotes.call_args.kwargs["instruments"]
        assert instruments[0].type.value == "EVENTCONTRACT"

    async def test_price_history_accepts_eventcontract(self, patch_get_client):
        patch_get_client.get_bars = AsyncMock(return_value={"bars": []})

        await get_price_history(
            symbol=CONTRACT_SYMBOL, period="WEEK", instrument_type="EVENTCONTRACT"
        )

        assert (
            patch_get_client.get_bars.call_args.kwargs["instrument_type"].value
            == "EVENTCONTRACT"
        )
