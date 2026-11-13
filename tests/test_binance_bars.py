"""Offline tests of the closed-bar cache and of the bridge to the real Quant Scouts parser.

``test_real_*`` start the real repository code (no network); they are skipped where the
Quant Scouts checkout or its environment is absent.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from kairos_core.contracts import ClosedBarEventV1
from support import MINUTE, BarFeed, kline, parse_pages

from kairos_autotrader import binance_bars
from kairos_autotrader.binance_bars import (
    OVERLAP_BARS,
    PAGE_BARS,
    BarsError,
    ClosedBarStore,
    QuantConfig,
    http_fetcher,
)

T0 = 1_794_000_000_000 // MINUTE * MINUTE
HISTORY = 1_200 * MINUTE


def update(store, feed, now_ms, symbols=("SOLUSDT",), history_ms=HISTORY, parser=parse_pages):
    return asyncio.run(
        store.update(QuantConfig(), list(symbols), now_ms, history_ms, fetch=feed.fetch, parser=parser)
    )


def test_first_update_downloads_the_window_in_pages_and_later_ones_only_the_tail(tmp_path):
    feed = BarFeed(last_open_ms=T0 + 10 * MINUTE)
    store = ClosedBarStore(tmp_path)
    bars, problems = update(store, feed, T0 + 30_000)  # the minute that opened at T0 is still forming
    rows = bars["SOLUSDT"]
    assert problems == {} and len(rows) == 1_200 and binance_bars.contiguous(rows)
    assert (rows[0][0], rows[-1][0]) == (T0 - HISTORY, T0 - MINUTE)
    assert rows[0][1:9] == [
        80.0,
        80.2,
        79.8,
        80.0 + (rows[0][0] // MINUTE % 7) * 0.01,
        5.0,
        400.0,
        2.5,
        200.0,
    ]
    assert len(rows[0][9]) == 64, "every bar keeps the identity the parser gave it"
    spans = [(start, end) for _, start, end in feed.asked]
    assert [(end - start) // MINUTE for start, end in spans] == [PAGE_BARS, PAGE_BARS, 1_200 - 2 * PAGE_BARS]
    assert spans[0][0] == T0 - HISTORY and spans[-1][1] == T0

    feed.asked.clear()
    restarted = ClosedBarStore(tmp_path)  # a new process reads the cache from disk
    later, problems = update(restarted, feed, T0 + 3 * MINUTE + 5_000)
    assert problems == {} and len(later["SOLUSDT"]) == 1_200
    assert feed.asked == [("SOLUSDT", T0 - OVERLAP_BARS * MINUTE, T0 + 3 * MINUTE)]
    assert later["SOLUSDT"][0][0] == T0 - HISTORY + 3 * MINUTE, "the window moves with the clock"
    assert json.loads((tmp_path / "SOLUSDT.1m.json").read_text()) == later["SOLUSDT"]


def test_an_incomplete_page_keeps_the_old_cache_and_names_the_symbol(tmp_path):
    feed = BarFeed(last_open_ms=T0 - MINUTE)
    store = ClosedBarStore(tmp_path)
    assert update(store, feed, T0)[1] == {}
    before = (tmp_path / "SOLUSDT.1m.json").read_text()

    feed.last_open_ms = T0 + 3 * MINUTE  # the venue has not published the newest minute yet
    bars, problems = update(store, feed, T0 + 5 * MINUTE)
    assert bars == {} and "page is incomplete" in problems["SOLUSDT"]
    assert (tmp_path / "SOLUSDT.1m.json").read_text() == before

    feed.last_open_ms, feed.drop = T0 + 10 * MINUTE, T0 + 2 * MINUTE  # a hole inside the page
    assert "page is incomplete" in update(store, feed, T0 + 5 * MINUTE)[1]["SOLUSDT"]
    feed.drop = None
    bars, problems = update(store, feed, T0 + 5 * MINUTE)
    assert problems == {} and bars["SOLUSDT"][-1][0] == T0 + 4 * MINUTE


def test_a_revised_closed_bar_is_a_conflict_that_drops_the_cache(tmp_path):
    feed = BarFeed(last_open_ms=T0 + 20 * MINUTE)
    store = ClosedBarStore(tmp_path)
    update(store, feed, T0)
    feed.revised = {T0 - 2 * MINUTE}
    bars, problems = update(store, feed, T0 + 5 * MINUTE)
    assert bars == {} and "changed a closed bar" in problems["SOLUSDT"]
    assert json.loads((tmp_path / "SOLUSDT.1m.json").read_text()) == []
    feed.asked.clear()
    bars, problems = update(store, feed, T0 + 5 * MINUTE)
    assert problems == {} and len(bars["SOLUSDT"]) == 1_200
    assert feed.asked[0][1] == T0 + 5 * MINUTE - HISTORY, "the whole window is downloaded again"


def test_one_failing_symbol_does_not_stop_the_others(tmp_path):
    class Feed(BarFeed):
        async def fetch(self, symbol, start_ms, end_ms):
            if symbol == "XRPUSDT":
                raise BarsError("Binance answered HTTP 451")
            return await super().fetch(symbol, start_ms, end_ms)

    def parser(config, pages):
        answers = parse_pages(config, pages)
        return [
            {**a, "bars": None, "error": "RecoveryError: boolean bar values are invalid"}
            if a["symbol"] == "ETHUSDT"
            else a
            for a in answers
        ]

    store = ClosedBarStore(tmp_path)
    bars, problems = update(
        store, Feed(last_open_ms=T0), T0, symbols=("SOLUSDT", "XRPUSDT", "ETHUSDT"), parser=parser
    )
    assert list(bars) == ["SOLUSDT"]
    assert "HTTP 451" in problems["XRPUSDT"] and "boolean bar values" in problems["ETHUSDT"]

    def down(config, pages):
        raise BarsError("Quant Scouts: worker failed")

    bars, problems = update(ClosedBarStore(tmp_path / "b"), BarFeed(last_open_ms=T0), T0, parser=down)
    assert bars == {} and problems == {"SOLUSDT": "Quant Scouts: worker failed"}


def test_a_cache_that_cannot_be_written_gives_no_bars_for_that_symbol(tmp_path, monkeypatch):
    store = ClosedBarStore(tmp_path)

    def full_disk(self, symbol, bars):
        raise OSError("No space left on device")

    monkeypatch.setattr(ClosedBarStore, "_save", full_disk)
    bars, problems = update(store, BarFeed(last_open_ms=T0), T0)
    assert bars == {} and problems == {"SOLUSDT": "closed-bar cache could not be written: OSError"}
    monkeypatch.undo()
    bars, problems = update(store, BarFeed(last_open_ms=T0), T0)
    assert problems == {} and len(bars["SOLUSDT"]) == 1_200


def test_large_downloads_are_parsed_in_several_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(binance_bars, "WORKER_BARS", 1_000)
    calls = []

    def parser(config, pages):
        calls.append(sum((p["end"] - p["start"]) // MINUTE for p in pages))
        return parse_pages(config, pages)

    bars, problems = update(
        ClosedBarStore(tmp_path), BarFeed(last_open_ms=T0), T0, symbols=("SOLUSDT", "ETHUSDT"), parser=parser
    )
    assert problems == {} and len(bars) == 2
    assert sum(calls) == 2_400 and len(calls) == 3 and max(calls) <= 1_000


def test_a_damaged_cache_file_is_rebuilt(tmp_path):
    (tmp_path / "SOLUSDT.1m.json").write_text("{not json")
    (tmp_path / "ETHUSDT.1m.json").write_text(json.dumps([[T0 - 2 * MINUTE] + [1.0] * 8 + ["a" * 64]] * 2))
    bars, problems = update(
        ClosedBarStore(tmp_path), BarFeed(last_open_ms=T0), T0, symbols=("SOLUSDT", "ETHUSDT")
    )
    assert problems == {} and all(len(rows) == 1_200 for rows in bars.values())


def test_the_public_endpoint_is_asked_for_exactly_one_page_of_closed_minutes():
    seen = {}

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def text(self):
            return json.dumps([kline(T0)])

    class Session:
        def get(self, url, *, params, timeout):
            seen.update(url=url, params=params)
            return Response()

    rows = asyncio.run(http_fetcher(Session())("SOLUSDT", T0, T0 + 7 * MINUTE))
    assert rows == [kline(T0)]
    assert seen["url"] == "https://fapi.binance.com/fapi/v1/klines"
    assert seen["params"] == {
        "symbol": "SOLUSDT",
        "interval": "1m",
        "startTime": str(T0),
        "endTime": str(T0 + 7 * MINUTE - 1),
        "limit": "7",
    }

    Response.status = 451
    with pytest.raises(BarsError, match="HTTP 451"):
        asyncio.run(http_fetcher(Session())("SOLUSDT", T0, T0 + MINUTE))


def test_real_quant_scouts_parser_gives_every_kline_its_contract_identity():
    config = QuantConfig()
    try:
        config.check()
    except BarsError:
        pytest.skip("Quant Scouts checkout or environment is not present")
    rows = [kline(T0 + i * MINUTE) for i in range(30)]
    good = {"symbol": "SOLUSDT", "start": T0, "end": T0 + 30 * MINUTE, "rows": rows}
    short = {"symbol": "SOLUSDT", "start": T0, "end": T0 + 30 * MINUTE, "rows": rows[:-1]}
    holed = {
        "symbol": "SOLUSDT",
        "start": T0,
        "end": T0 + 30 * MINUTE,
        "rows": rows[:9] + rows[10:] + rows[:1],
    }
    answers = binance_bars._run_parser(config, [good, short, holed])
    assert "error" not in answers[0] and "incomplete" in answers[1]["error"] and "error" in answers[2]
    for row, bar in zip(rows, answers[0]["bars"], strict=True):
        contract = ClosedBarEventV1(
            source="test",
            symbol="SOLUSDT",
            open_time_ms=row[0],
            close_time_ms=row[6],
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            base_volume=float(row[5]),
            quote_volume=float(row[7]),
            taker_buy_base_volume=float(row[9]),
            taker_buy_quote_volume=float(row[10]),
        )
        assert bar == [
            contract.open_time_ms,
            contract.open,
            contract.high,
            contract.low,
            contract.close,
            contract.base_volume,
            contract.quote_volume,
            contract.taker_buy_base_volume,
            contract.taker_buy_quote_volume,
            contract.bar_sha256,
        ]
