"""Offline tests of the bridge to the real Kairos Strategy Engine.

Only the ``test_real_engine_...`` tests start the engine itself; they are skipped where
the engine checkout or its environment is absent. ``KAIROS_TEST_ENGINE_SOURCE`` points
them at another engine source directory than the sibling checkout.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest
from kairos_core.contracts import ClosedBarEventV1
from support import CONFIG_SHA256, engine_state, strict_intent

from kairos_autotrader import system_cli, system_engine
from kairos_autotrader.evedex import Candle, Evedex
from kairos_autotrader.system_engine import (
    HOURLY_TREND_HISTORY_MS,
    MINUTE_MS,
    EngineConfig,
    EngineError,
    MinuteStore,
    collect,
    records_from,
    sleeve_record,
)

T0 = 1_794_000_000_000 // MINUTE_MS * MINUTE_MS


def intent(**over) -> dict:
    """A research ``SleeveIntent`` as ``system_engine_worker.py`` serialises it."""
    base = {
        "intent_id": "abc",
        "sleeve_id": "trend_pullback_reclaim_v1",
        "symbol": "SOLUSD",
        "side": "SHORT",
        "decision_ts_ms": T0 + 5 * MINUTE_MS - 1,
        "entry_eligible_ts_ms": T0 + 5 * MINUTE_MS,
        "entry_expires_ts_ms": T0 + 10 * MINUTE_MS,
        "reference_price": 83.0,
        "signal_strength": 0.61234,
        "gross_reward_bps": 80.0,
        "stop_price": 83.5,
        "target_price": 82.0,
        "max_holding_ms": 3 * 3_600_000,
        "trailing_activation_price": None,
        "trailing_distance": None,
        "metadata": {},
    }
    base.update(over)
    return base


def minute_rows(count: int, start: int = T0, price: float = 100.0) -> list[list[float]]:
    return [
        [start + i * MINUTE_MS, price, price + 0.2, price - 0.2, price, 10.0, 1000.0] for i in range(count)
    ]


def test_sleeve_record_keeps_side_exits_and_hold_time():
    r = sleeve_record(intent(), note="engine status rejected")
    assert (r["symbol"], r["side"], r["setup"]) == ("SOLUSD", "SHORT", "trend_pullback_reclaim_v1")
    assert (r["ref_price"], r["stop"], r["take"], r["timeout_minutes"]) == (83.0, 83.5, 82.0, 180)
    assert r["signal_time"] == T0 + 5 * MINUTE_MS, "the engine's inclusive close becomes the next millisecond"
    assert r["id"] == f"SOLUSD-SHORT-trend_pullback_reclaim_v1-{T0 + 5 * MINUTE_MS - 1}"
    assert r["atr"] == pytest.approx(0.5) and r["score"] == 0.6123 and r["note"] == "engine status rejected"
    assert r["strategy_id"] == "trend_pullback_reclaim_v1"
    trailing = sleeve_record(intent(trailing_distance=0.2, trailing_activation_price=82.5))
    assert trailing["note"] == "engine trailing stop not applied"

    live = sleeve_record(
        intent(
            sleeve_id=system_engine.HOURLY_TREND_ID,
            symbol="SOLUSDT",
            metadata={"setup": "BREAKOUT", "atr": "0.40000000000000002"},
        ),
        instrument="SOLUSD",
    )
    assert (live["symbol"], live["setup"], live["strategy_id"]) == (
        "SOLUSD",
        "BREAKOUT",
        system_engine.HOURLY_TREND_ID,
    )
    assert live["atr"] == pytest.approx(0.4) and live["id"].startswith("SOLUSD-SHORT-BREAKOUT-")


@pytest.mark.parametrize("bad", [{"side": "FLAT"}, {"stop_price": 82.5}, {"side": "LONG"}])
def test_inconsistent_sleeve_intent_is_refused(bad):
    with pytest.raises(EngineError):
        sleeve_record(intent(**bad))


def test_records_are_filtered_by_time_and_sleeve_errors_are_reported():
    old = intent(decision_ts_ms=T0 - 1)
    results = {
        "trend_pullback_reclaim_v1": {"revision": "1", "status": "rejected", "intents": [old, intent()]},
        "range_mean_reversion_v1": {"error": "ValueError: boom"},
        "right_tail_trend_v1": {"revision": "1", "status": "rejected", "intents": [intent(stop_price=1.0)]},
    }
    found, problems = records_from(results, since_ms=T0)
    assert [r["signal_time"] for r in found] == [T0 + 5 * MINUTE_MS]
    assert "rev 1" in found[0]["note"] and "status rejected" in found[0]["note"]
    assert len(problems) == 2 and "boom" in problems[0] and "bad intent" in problems[1]


class FakeMinuteApi:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def candle_rows(self, symbol, group, start_ms, end_ms):
        assert group == "1m"
        self.calls.append((start_ms, end_ms))
        return [r for r in self.rows if start_ms <= r[0] < end_ms]


def test_minute_cache_fetches_only_the_missing_tail_and_drops_the_forming_bar(tmp_path):
    rows = minute_rows(200)
    api = FakeMinuteApi(rows)
    store = MinuteStore(tmp_path)
    now = T0 + 100 * MINUTE_MS + 30_000  # bar 100 is still forming
    first = asyncio.run(store.update(api, "SOLUSD", now, days=1))
    assert [r[0] for r in first] == [T0 + i * MINUTE_MS for i in range(100)]
    later = now + 50 * MINUTE_MS
    second = asyncio.run(MinuteStore(tmp_path).update(api, "SOLUSD", later, days=1))
    assert api.calls[1][0] == T0 + 100 * MINUTE_MS, "a restart continues from the cached bars"
    assert len(second) == 150 and system_engine.gaps(second) == 0
    assert json.loads((tmp_path / "SOLUSD.1m.json").read_text())[-1][0] == second[-1][0]


def test_collect_runs_every_instrument_and_reports_gaps_and_failures(tmp_path):
    holes = [r for r in minute_rows(60) if r[0] != T0 + 30 * MINUTE_MS]

    class Api(FakeMinuteApi):
        async def candle_rows(self, symbol, group, start_ms, end_ms):
            if symbol == "XRPUSD":
                raise system_engine.EvedexError("venue down")
            return await super().candle_rows(symbol, group, start_ms, end_ms)

    def runner(config, request):
        assert request["strategies"] == list(config.sleeves) and len(request["rows"][0]) == 7
        return {
            "trend_pullback_reclaim_v1": {
                "revision": "1",
                "status": "rejected",
                "intents": [intent(symbol=request["symbol"])],
            }
        }

    found, problems, bars = asyncio.run(
        collect(
            Api(holes),
            MinuteStore(tmp_path),
            EngineConfig(),
            ["SOLUSD", "XRPUSD"],
            T0 + 61 * MINUTE_MS,
            runner=runner,
        )
    )
    assert [r["symbol"] for r in found] == ["SOLUSD"] and len(bars["SOLUSD"]) == 59
    assert any("1 gaps" in p for p in problems) and any("XRPUSD" in p and "venue down" in p for p in problems)


def test_outcomes_use_only_the_bars_after_the_signal():
    candles = [Candle(T0 + i * MINUTE_MS, 83.0, 83.1, 82.9, 83.0, 1.0) for i in range(400)]
    candles[2] = Candle(candles[2].t, 83.0, 84.0, 82.9, 83.0, 1.0)  # a spike before the signal must not count
    candles[20] = Candle(candles[20].t, 83.0, 83.1, 81.9, 82.0, 1.0)  # the target is reached here
    done = sleeve_record(intent())
    open_one = sleeve_record(intent(decision_ts_ms=T0 + 399 * MINUTE_MS - 1))
    rows = system_cli.evaluate([done, open_one], candles, cost_fraction=0.0013)
    assert len(rows) == 1 and rows[0]["reason"] == "TAKE" and rows[0]["setup"] == "trend_pullback_reclaim_v1"
    assert rows[0]["r"] == pytest.approx(2.0 - 0.0013 * 83.0 / 0.5, abs=1e-3)
    assert rows[0]["cost_r"] == pytest.approx(0.2158, abs=1e-3) and rows[0]["stop_pct"] == pytest.approx(
        0.602, abs=1e-3
    )


def test_minute_history_is_paged_and_reordered_from_the_venue_layout():
    api = Evedex(None)
    asked = []

    async def request(method, url, **kwargs):
        asked.append(kwargs["params"])
        # venue layout: time, open, close, high, low, quote volume, base volume
        return [[T0, "100", "101", "102", "99", "5000", "50"]] if len(asked) == 1 else []

    api._request = request
    rows = asyncio.run(api.candle_rows("SOLUSD", "1m", T0 - 1500 * MINUTE_MS, T0 + MINUTE_MS))
    assert len(asked) == 2 and all(p["group"] == "1m" for p in asked)
    assert rows == [[T0, 100.0, 102.0, 99.0, 101.0, 50.0, 5000.0]]


def shadow_args(tmp_path) -> argparse.Namespace:
    return argparse.Namespace(runtime_dir=tmp_path / "shadow", config=None, interval=300, max_cycles=1)


def test_shadow_records_each_new_intent_once_and_refuses_a_trading_directory(tmp_path, monkeypatch):
    offered = [sleeve_record(intent(), note="engine status rejected")]

    async def fake_collect(api, store, engine, symbols, now_ms, *, since_ms=0, runner=None):
        return list(offered), [], {}

    monkeypatch.setattr(system_cli, "collect", fake_collect)
    monkeypatch.setattr(system_cli, "engine_revision", lambda source: "rev-test")
    monkeypatch.setattr(EngineConfig, "check", lambda self: None)
    args = shadow_args(tmp_path)
    assert asyncio.run(system_cli.cmd_shadow(args)) == 0
    assert asyncio.run(system_cli.cmd_shadow(args)) == 0
    records = [json.loads(line) for line in (args.runtime_dir / "journal.jsonl").read_text().splitlines()]
    scans = [r for r in records if r["event"] == "scan"]
    assert (
        len(scans) == 1
        and scans[0]["source"] == "strategy_engine"
        and scans[0]["engine_revision"] == "rev-test"
    )
    assert scans[0]["candidates"] == offered, "the report reads the record as it was written"
    state = json.loads((args.runtime_dir / "state.json").read_text())
    assert state["mode"] == "engine-shadow" and list(state["seen"]) == [offered[0]["id"]]

    live = tmp_path / "live"
    live.mkdir()
    (live / "state.json").write_text(json.dumps({"mode": "live"}))
    with pytest.raises(system_cli.ConfigError):
        asyncio.run(
            system_cli.cmd_shadow(
                argparse.Namespace(runtime_dir=live, config=None, interval=300, max_cycles=1)
            )
        )


def test_real_engine_worker_runs_registered_sleeves():
    config = EngineConfig()
    if not config.python.exists() or not (config.source / "kairos_strategy").exists():
        pytest.skip("Strategy Engine checkout or environment is not present")
    rows, price = [], 100.0
    for i in range(3 * 1440):  # three days of a slow zigzag
        step = 0.02 if (i // 180) % 2 == 0 else -0.015
        nxt = price + step
        rows.append(
            [T0 + i * MINUTE_MS, price, max(price, nxt) + 0.01, min(price, nxt) - 0.01, nxt, 5.0, 5.0 * nxt]
        )
        price = nxt
    results = system_engine._run_worker(
        config, {"symbol": "TESTUSD", "strategies": list(config.sleeves), "rows": rows}
    )
    assert set(results) == set(config.sleeves)
    for sleeve, result in results.items():
        assert "error" not in result, (sleeve, result)
        assert result["status"] and isinstance(result["intents"], list)
        for raw in result["intents"]:
            sleeve_record(raw)


def test_minute_cache_fetches_the_older_part_when_a_longer_window_is_asked(tmp_path):
    day = 1440
    rows = minute_rows(5 * day)
    api = FakeMinuteApi(rows)
    now = T0 + 5 * day * MINUTE_MS
    short = asyncio.run(MinuteStore(tmp_path).update(api, "SOLUSD", now, days=2))
    assert len(short) == 2 * day
    longer = asyncio.run(MinuteStore(tmp_path).update(api, "SOLUSD", now, days=4))
    assert len(longer) == 4 * day and system_engine.gaps(longer) == 0
    assert api.calls[-1] == (now - 4 * day * MINUTE_MS, now - 2 * day * MINUTE_MS), (
        "only the missing head is fetched"
    )


# ---- the live rules: kairos_strategy.hourly_trend ---------------------------------------

QUARTER_MS, HOUR_MS = 15 * MINUTE_MS, 60 * MINUTE_MS
B0 = T0 // HOUR_MS * HOUR_MS  # an hour boundary
IDENTITY = {
    "strategy_id": system_engine.HOURLY_TREND_ID,
    "revision": "1",
    "strategy_code_sha256": system_engine.HOURLY_TREND_CODE_SHA256,
    "config_sha256": CONFIG_SHA256,
}


def cached_bars(first_ms: int, count: int, symbol: str = "SOLUSDT") -> list[list]:
    """Cache rows with stand-in identities: enough for everything but the real engine."""
    return [
        [
            first_ms + i * MINUTE_MS,
            80.0,
            80.2,
            79.8,
            80.0,
            5.0,
            400.0,
            2.5,
            200.0,
            hashlib.sha256(f"{symbol}:{first_ms + i * MINUTE_MS}".encode()).hexdigest(),
        ]
        for i in range(count)
    ]


def test_engine_answer_is_refused_unless_it_is_the_pinned_source():
    assert system_engine.checked_hourly_answer({"identity": IDENTITY, "results": {}}) == (IDENTITY, {})
    for bad in (
        {"identity": {**IDENTITY, "strategy_code_sha256": "0" * 64}, "results": {}},
        {"identity": {**IDENTITY, "strategy_id": "other_v1"}, "results": {}},
        {"identity": IDENTITY},
        {"results": {}},
    ):
        with pytest.raises(EngineError):
            system_engine.checked_hourly_answer(bad)


def test_scan_spreads_instruments_over_engine_processes_and_needs_the_whole_window():
    boundary = B0 + HOURLY_TREND_HISTORY_MS + QUARTER_MS
    full = HOURLY_TREND_HISTORY_MS // MINUTE_MS
    series = {
        "SOLUSD": cached_bars(boundary - HOURLY_TREND_HISTORY_MS - 30 * MINUTE_MS, full + 32, "SOLUSDT"),
        "ETHUSD": cached_bars(boundary - HOURLY_TREND_HISTORY_MS, full, "ETHUSDT"),
        "XRPUSD": cached_bars(boundary - HOURLY_TREND_HISTORY_MS, full - 1, "XRPUSDT"),  # last minute absent
        "ZECUSD": cached_bars(boundary - HOURLY_TREND_HISTORY_MS + MINUTE_MS, full - 1, "ZECUSDT"),
        "LINKUSD": cached_bars(boundary - HOURLY_TREND_HISTORY_MS, full, "LINKUSDT"),
    }
    requests = []

    def runner(config, request):
        requests.append(request)
        results = {}
        for symbol, rows in request["series"].items():
            decision = int(rows[-1][0]) + MINUTE_MS - 1
            if symbol == "LINKUSDT":
                results[symbol] = {"error": "ClosedBarSequenceError: closed-bar gap"}
                continue
            raw = strict_intent(
                symbol=symbol, decision_ts_ms=decision, bar_hashes=tuple(r[9] for r in rows)
            ).model_dump(mode="json")
            results[symbol] = {
                "decision_ts_ms": decision,
                "state": engine_state(symbol, decision),
                "intent": raw if symbol == "SOLUSDT" else None,
            }
        return {"identity": IDENTITY, "results": results}

    found = asyncio.run(
        system_engine.hourly_trend_scan(
            EngineConfig(), series, boundary, reward_risk=1.5, timeout_minutes=720, runner=runner, processes=2
        )
    )
    assert len(requests) == 2 and all(r["op"] == "hourly_trend_bars" for r in requests)
    sent = {symbol: rows for request in requests for symbol, rows in request["series"].items()}
    assert set(sent) == {"SOLUSDT", "ETHUSDT", "LINKUSDT"}, "requests are keyed by the Binance symbol"
    assert len(sent["SOLUSDT"]) == full and sent["SOLUSDT"][-1][0] == boundary - MINUTE_MS
    assert sorted(found.features) == ["ETHUSD", "SOLUSD"] and list(found.intents) == ["SOLUSD"]
    assert found.features["SOLUSD"].symbol == "SOLUSD" and found.intents["SOLUSD"].symbol == "SOLUSDT"
    assert sorted(found.skipped) == [
        "LINKUSD: ClosedBarSequenceError: closed-bar gap",
        "XRPUSD: no fresh closed bar",
        "ZECUSD: closed bars do not reach back the whole history window",
    ]

    def other_clock(config, request):
        answer = runner(config, request)
        for result in answer["results"].values():
            if "decision_ts_ms" in result:
                result["decision_ts_ms"] -= QUARTER_MS
        return answer

    late = asyncio.run(
        system_engine.hourly_trend_scan(
            EngineConfig(),
            {"ETHUSD": series["ETHUSD"]},
            boundary,
            reward_risk=1.5,
            timeout_minutes=720,
            runner=other_clock,
        )
    )
    assert late.features == {} and late.skipped == ["ETHUSD: the engine decided on another clock"]


def test_history_of_the_live_rules_reports_bad_instruments_and_keeps_the_rest():
    asked = []
    good = intent(
        sleeve_id=system_engine.HOURLY_TREND_ID,
        symbol="SOLUSDT",
        metadata={"setup": "BREAKOUT", "atr": "0.4"},
    )

    def runner(config, request):
        asked.append(request)
        return {
            "identity": IDENTITY,
            "results": {
                "SOLUSDT": {"intents": [good, {**good, "stop_price": 1.0}]},
                "XRPUSDT": {"error": "ValueError: boom"},
            },
        }

    bars = cached_bars(B0, 30)
    found, problems = system_engine.hourly_trend_history(
        EngineConfig(), {"SOLUSD": bars, "XRPUSD": bars}, reward_risk=1.5, timeout_minutes=720, runner=runner
    )
    assert asked[0]["op"] == "hourly_trend_history" and set(asked[0]["series"]) == {"SOLUSDT", "XRPUSDT"}
    assert asked[0]["config"] == {"target_reward_to_risk": 1.5, "max_hold_minutes": 720}
    assert [(r["symbol"], r["setup"]) for r in found] == [("SOLUSD", "BREAKOUT")]
    assert (
        len(problems) == 2 and any("boom" in p for p in problems) and any("bad intent" in p for p in problems)
    )


def real_engine() -> EngineConfig:
    source = os.environ.get("KAIROS_TEST_ENGINE_SOURCE")
    config = EngineConfig(source=Path(source)) if source else EngineConfig()
    try:
        config.check_hourly_trend()
    except EngineError:
        pytest.skip("Strategy Engine checkout with the strict hourly_trend adapter is not present")
    return config


def test_real_engine_gives_a_strict_intent_on_bars_with_contract_identities():
    config = real_engine()
    full = HOURLY_TREND_HISTORY_MS // MINUTE_MS
    boundary = B0 + HOURLY_TREND_HISTORY_MS + QUARTER_MS
    first = boundary - HOURLY_TREND_HISTORY_MS
    bars, price = [], 100.0
    for i in range(full):  # three weeks of a slow climb with pullbacks
        step = 0.012 if (i // 240) % 3 else -0.015
        nxt = price + step
        values = [price, max(price, nxt) + 0.01, min(price, nxt) - 0.01, nxt, 5.0, 5.0 * nxt, 2.0, 2.0 * nxt]
        bar = ClosedBarEventV1(
            source="test",
            symbol="TESTUSDT",
            open_time_ms=first + i * MINUTE_MS,
            close_time_ms=first + (i + 1) * MINUTE_MS - 1,
            open=values[0],
            high=values[1],
            low=values[2],
            close=values[3],
            base_volume=values[4],
            quote_volume=values[5],
            taker_buy_base_volume=values[6],
            taker_buy_quote_volume=values[7],
        )
        bars.append([bar.open_time_ms, *values, bar.bar_sha256])
        price = nxt
    found = asyncio.run(
        system_engine.hourly_trend_scan(
            config, {"TESTUSD": bars}, boundary, reward_risk=1.5, timeout_minutes=720
        )
    )
    assert found.identity["strategy_code_sha256"] == system_engine.HOURLY_TREND_CODE_SHA256, (
        "the pin in system_engine.py must be updated together with the engine source"
    )
    assert found.skipped == [] and list(found.features) == ["TESTUSD"]
    features = found.features["TESTUSD"]
    assert features.trend in ("UP", "DOWN", "FLAT") and features.price == pytest.approx(bars[-1][4])
    assert set(found.context["TESTUSD"]) == {
        "change_1h_pct",
        "change_7d_pct",
        "position_in_7d_range",
        "last_bar_volume_vs_20bar_avg",
        "consecutive_up_bars",
        "consecutive_down_bars",
    }
    for strict in found.intents.values():
        assert strict.symbol == "TESTUSDT" and strict.decision_ts_ms == boundary - 1
        assert len(strict.provenance.input_bar_sha256s) == full

    history, problems = system_engine.hourly_trend_history(
        config, {"TESTUSD": bars}, reward_risk=2.0, timeout_minutes=240
    )
    assert problems == [] and history, "the synthetic climb must produce candidates"
    for record in history:
        assert record["symbol"] == "TESTUSD" and record["timeout_minutes"] == 240
        assert abs(record["take"] - record["ref_price"]) == pytest.approx(
            2.0 * abs(record["ref_price"] - record["stop"])
        )
    newest = [r for r in history if r["signal_time"] == boundary]
    assert [(r["side"], r["ref_price"], r["stop"]) for r in newest] == [
        (i.side.value, i.reference_price, i.exit_plan.stop_price) for i in found.intents.values()
    ], "the strict intent is the research candidate of the same decision"

    forged = [list(row) for row in bars]
    forged[-3][4] += 0.01  # a close that no longer matches its recorded identity
    refused = asyncio.run(
        system_engine.hourly_trend_scan(
            config, {"TESTUSD": forged}, boundary, reward_risk=1.5, timeout_minutes=720
        )
    )
    assert refused.features == {} and "does not hash to its recorded identity" in refused.skipped[0]


# ---- the daily right-tail rule ------------------------------------------------------

RT_IDENTITY = {
    "strategy_id": system_engine.RIGHT_TAIL_ID,
    "revision": "1",
    "status": "forward_frozen",
    "strategy_code_sha256": system_engine.RIGHT_TAIL_CODE_SHA256,
    "config_sha256": CONFIG_SHA256,
}


def test_the_daily_rule_decides_once_a_day_at_one_utc():
    day = 86_400_000
    first = (B0 // day + 1) * day
    assert system_engine.right_tail_due(first + HOUR_MS)
    assert not system_engine.right_tail_due(first)
    assert not system_engine.right_tail_due(first + HOUR_MS + QUARTER_MS)
    assert not system_engine.right_tail_due(first + 2 * HOUR_MS)
    every_four = [h for h in range(24) if system_engine.right_tail_due(first + h * HOUR_MS, 4)]
    assert every_four == [1, 5, 9, 13, 17, 21]
    with pytest.raises(ValueError):
        system_engine.right_tail_due(first + HOUR_MS, 5)


def test_right_tail_scan_trades_only_the_pinned_rule_on_the_whole_window():
    history = system_engine.RIGHT_TAIL_HISTORY_MS
    full = history // MINUTE_MS
    boundary = (B0 // 86_400_000 + 40) * 86_400_000 + HOUR_MS
    series = {
        "BTCUSD": cached_bars(boundary - history, full, "BTCUSDT"),
        "ETHUSD": cached_bars(boundary - history + MINUTE_MS, full - 1, "ETHUSDT"),  # too short
        "XRPUSD": cached_bars(boundary - history, full, "XRPUSDT"),
    }
    requests = []

    def runner(config, request):
        requests.append(request)
        results = {}
        for symbol, rows in request["series"].items():
            raw = strict_intent(
                symbol=symbol,
                decision_ts_ms=boundary - 1,
                side="SHORT",
                setup=system_engine.RIGHT_TAIL_SETUP,
                ref_price=100.0,
                stop=104.0,
                take=84.0,
                timeout_minutes=72 * 60,
                bar_hashes=tuple(r[9] for r in rows),
                code_sha256=system_engine.RIGHT_TAIL_CODE_SHA256,
                strategy_id=system_engine.RIGHT_TAIL_ID,
            ).model_dump(mode="json")
            results[symbol] = {"decision_ts_ms": boundary - 1, "intent": raw if symbol == "BTCUSDT" else None}
        return {"identity": RT_IDENTITY, "results": results}

    intents, skipped, identity = asyncio.run(
        system_engine.right_tail_scan(
            EngineConfig(), series, boundary, runner=runner, processes=2, price_increments={"BTCUSD": "0.1"}
        )
    )
    assert all(
        r["op"] == "registered_bars" and r["strategy_id"] == system_engine.RIGHT_TAIL_ID for r in requests
    )
    assert {r["decision_ts_ms"] for r in requests} == {boundary - 1}
    assert {r["config"]["decision_interval_hours"] for r in requests} == {24}
    assert {s for r in requests for s in r["price_increments"]} == {"BTCUSDT"}
    assert list(intents) == ["BTCUSD"] and intents["BTCUSD"].exit_plan.max_holding_ms == 72 * HOUR_MS
    assert skipped == ["ETHUSD: closed bars do not reach back the daily rule's history"]
    assert identity["strategy_code_sha256"] == system_engine.RIGHT_TAIL_CODE_SHA256

    def other_source(config, request):
        answer = runner(config, request)
        return {**answer, "identity": {**answer["identity"], "strategy_code_sha256": "0" * 64}}

    with pytest.raises(EngineError):
        asyncio.run(system_engine.right_tail_scan(EngineConfig(), series, boundary, runner=other_source))

    def wrong_bars(config, request):
        answer = runner(config, request)
        for result in answer["results"].values():
            if result.get("intent"):
                result["intent"]["provenance"]["input_bar_sha256s"] = ["f" * 64]
        return answer

    refused, skipped, _ = asyncio.run(
        system_engine.right_tail_scan(EngineConfig(), series, boundary, runner=wrong_bars)
    )
    # A changed bar list no longer matches the intent identity: refused before it is traded.
    assert refused == {} and any(s.startswith("BTCUSD: bad engine output") for s in skipped), skipped


def test_the_worker_puts_both_barriers_on_the_tick_without_shrinking_the_risk():
    from decimal import Decimal

    from kairos_autotrader.system_engine_worker import _on_tick

    tick = Decimal("0.01")
    # long: stop below the entry rounds down (away), target above rounds down (toward the entry)
    assert _on_tick(81.537, tick, up=False) == 81.53
    assert _on_tick(90.019, tick, up=False) == 90.01
    # short: stop above rounds up (away), target below rounds up (toward the entry)
    assert _on_tick(104.001, tick, up=True) == 104.01
    assert _on_tick(84.001, tick, up=True) == 84.01
    assert _on_tick(84.001, None, up=True) == 84.001


def test_the_venue_tick_keeps_the_engine_intent_valid():
    """Run in the engine's own Python: the adjusted intent must pass its model's checks."""
    import subprocess

    config = EngineConfig()
    if not config.python.exists() or not (config.source / "kairos_strategy").exists():
        pytest.skip("Strategy Engine checkout or environment is not present")
    worker_dir = Path(system_engine.__file__).parent
    script = f"""
import sys
sys.path[:0] = [{str(config.source)!r}, {str(worker_dir)!r}]
from decimal import Decimal
from kairos_core.enums import Side
from kairos_strategy.models import ExitPlan, SleeveIntent
from system_engine_worker import _venue_ready
raw = SleeveIntent(
    sleeve_id="regime_aligned_right_tail_v1", symbol="1000PEPEUSDT", side=Side.SHORT,
    decision_ts_ms=1_000_000 * 3_600_000 - 1, entry_eligible_ts_ms=1_000_000 * 3_600_000,
    entry_expires_ts_ms=1_000_001 * 3_600_000 - 1, reference_price=0.0037731, signal_strength=0.6,
    gross_reward_bps=abs(0.0029333 - 0.0037731) / 0.0037731 * 10_000,
    exit_plan=ExitPlan(stop_price=0.00398314, target_price=0.0029333, max_holding_ms=72 * 3_600_000),
    metadata=(("atr", "0.0001"), ("strategy_version", "regime_aligned_right_tail_v1")),
)
ready = _venue_ready(raw, Decimal("0.000001"), "RIGHT_TAIL_TREND")
plan = ready.exit_plan
assert plan.stop_price == 0.003984 and plan.target_price == 0.002934, plan
assert dict(ready.metadata)["setup"] == "RIGHT_TAIL_TREND"
print("ok")
"""
    answer = subprocess.run([str(config.python), "-c", script], capture_output=True, text=True, timeout=120)
    assert answer.returncode == 0 and answer.stdout.strip().endswith("ok"), answer.stderr[-800:]


FAKE_SERVE_WORKER = """
import json, sys, time
MARKER = "@@KAIROS-WORKER@@"
assert sys.argv[2:] == ["--serve"], sys.argv


def say(value):
    sys.stdout.write("a library log line\\n" + MARKER + json.dumps(value) + "\\n")
    sys.stdout.flush()


say({"ready": True})
count = 0
for line in sys.stdin:
    request = json.loads(line)
    count += 1
    if request.get("op") == "die":
        sys.exit(3)
    if request.get("op") == "hang":
        time.sleep(60)
    say({"count": count, "op": request.get("op")})
"""


def test_engine_pool_keeps_each_worker_and_restarts_a_failed_one(tmp_path, monkeypatch):
    import sys

    script = tmp_path / "worker.py"
    script.write_text(FAKE_SERVE_WORKER, encoding="utf-8")
    monkeypatch.setattr(system_engine, "WORKER", script)
    pool = system_engine.EnginePool(
        EngineConfig(python=Path(sys.executable), source=tmp_path, timeout_s=5), 2
    )

    async def run():
        try:
            assert await pool.ask(0, {"op": "a"}) == {"count": 1, "op": "a"}
            assert await pool.ask(0, {"op": "b"}) == {"count": 2, "op": "b"}  # the same process
            assert await pool.ask(1, {"op": "c"}) == {"count": 1, "op": "c"}  # its own process
            assert await pool.ask(2, {"op": "d"}) == {"count": 3, "op": "d"}  # index wraps to 0
            with pytest.raises(EngineError, match="exited"):
                await pool.ask(0, {"op": "die"})
            assert await pool.ask(0, {"op": "e"}) == {"count": 1, "op": "e"}  # a fresh one
            with pytest.raises(EngineError, match="TimeoutError"):
                await pool.ask(1, {"op": "hang"})
            assert await pool.ask(1, {"op": "f"}) == {"count": 1, "op": "f"}
        finally:
            await pool.stop()

    asyncio.run(run())


def right_tail_answer(request: dict) -> dict:
    return {
        "identity": RT_IDENTITY,
        "results": {
            symbol: {"decision_ts_ms": request["decision_ts_ms"], "intent": None}
            for symbol in request["series"]
        },
    }


class FakePool:
    size = 3

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[int, list[str]]] = []

    async def ask(self, index: int, request: dict) -> dict:
        self.calls.append((index, sorted(request["series"])))
        if self.fail:
            raise EngineError("Strategy Engine: the worker exited")
        return right_tail_answer(request)


def test_right_tail_scan_sends_each_instrument_to_its_own_pool_worker():
    history = system_engine.RIGHT_TAIL_HISTORY_MS
    full = history // MINUTE_MS
    boundary = (B0 // 86_400_000 + 40) * 86_400_000 + HOUR_MS

    def series(*short):
        return {
            name: cached_bars(boundary - history + (MINUTE_MS if name in short else 0), full, name + "T")
            for name in ("AAVEUSD", "BTCUSD", "ETHUSD", "XRPUSD")
        }

    def one_shot(config, request):
        raise AssertionError("the pool answered")

    pool = FakePool()
    asyncio.run(system_engine.right_tail_scan(EngineConfig(), series(), boundary, runner=one_shot, pool=pool))
    assert sorted(pool.calls) == [(0, ["AAVEUSDT", "XRPUSDT"]), (1, ["BTCUSDT"]), (2, ["ETHUSDT"])]
    # An instrument without the whole window does not move the others to another worker.
    pool = FakePool()
    _, skipped, _ = asyncio.run(
        system_engine.right_tail_scan(EngineConfig(), series("BTCUSD"), boundary, runner=one_shot, pool=pool)
    )
    assert sorted(pool.calls) == [(0, ["AAVEUSDT", "XRPUSDT"]), (2, ["ETHUSDT"])]
    assert skipped == ["BTCUSD: closed bars do not reach back the daily rule's history"]


def test_right_tail_scan_falls_back_to_a_one_shot_worker_when_the_pool_fails():
    history = system_engine.RIGHT_TAIL_HISTORY_MS
    full = history // MINUTE_MS
    boundary = (B0 // 86_400_000 + 40) * 86_400_000 + HOUR_MS
    series = {name: cached_bars(boundary - history, full, name + "T") for name in ("BTCUSD", "ETHUSD")}
    asked = []

    def one_shot(config, request):
        asked.append(sorted(request["series"]))
        return right_tail_answer(request)

    pool = FakePool(fail=True)
    intents, skipped, identity = asyncio.run(
        system_engine.right_tail_scan(EngineConfig(), series, boundary, runner=one_shot, pool=pool)
    )
    assert sorted(pool.calls) == [(0, ["BTCUSDT"]), (1, ["ETHUSDT"])]
    assert sorted(asked) == [["BTCUSDT"], ["ETHUSDT"]]
    assert (intents, skipped, identity) == ({}, [], RT_IDENTITY)


def test_real_engine_worker_reuses_only_bars_it_has_checked():
    """Run in the engine's own Python: an unchanged row reuses its bar, a changed one is rechecked."""
    import subprocess

    config = EngineConfig()
    if not config.python.exists() or not (config.source / "kairos_strategy").exists():
        pytest.skip("Strategy Engine checkout or environment is not present")
    worker_dir = Path(system_engine.__file__).parent
    script = f"""
import sys
sys.path[:0] = [{str(config.source)!r}, {str(worker_dir)!r}]
from kairos_strategy.candles import Candle
from kairos_strategy.runtime import candle_to_closed_bar
import system_engine_worker as w

T = {T0}
def row(i, close=100.5):
    c = Candle(symbol="BTCUSDT", timeframe="1m", open_time_ms=T + i * 60_000,
               close_time_ms=T + i * 60_000 + 59_999, open=100.0, high=101.0, low=99.0, close=close,
               volume=1.0, quote_volume=100.0, taker_buy_volume=0.5, taker_buy_quote_volume=50.0)
    sha = candle_to_closed_bar(c).bar_sha256
    return [c.open_time_ms, 100.0, 101.0, 99.0, close, 1.0, 100.0, 0.5, 50.0, sha]

rows = [row(i) for i in range(6)]
first, candles = w._checked_bars("BTCUSDT", rows[:5], Candle)
again, _ = w._checked_bars("BTCUSDT", rows[1:], Candle)
assert all(a is b for a, b in zip(again[:4], first[1:])), "unchanged rows must reuse their bars"
assert again[4].bar_sha256 == rows[5][9] and len(w._CHECKED["BTCUSDT"]) == 5
changed = [list(r) for r in rows[1:]]
changed[2][4] = 100.6  # the venue's value moved, the recorded identity did not
try:
    w._checked_bars("BTCUSDT", changed, Candle)
except ValueError as exc:
    assert "does not hash" in str(exc), exc
else:
    raise SystemExit("a changed row was not rechecked")
print("ok")
"""
    answer = subprocess.run([str(config.python), "-c", script], capture_output=True, text=True, timeout=120)
    assert answer.returncode == 0 and answer.stdout.strip().endswith("ok"), answer.stderr[-800:]


def test_real_engine_worker_serves_until_its_input_closes():
    config = EngineConfig()
    if not config.python.exists() or not (config.source / "kairos_strategy").exists():
        pytest.skip("Strategy Engine checkout or environment is not present")
    pool = system_engine.EnginePool(config, 1)

    async def run():
        try:
            first = await pool.ask(0, {"op": "no_such_op"})
            second = await pool.ask(0, ["not", "an", "object"])
            assert first["error"].startswith("KeyError") and second["error"].startswith("ValueError")
            process = pool._processes[0]
            assert process is not None and process.returncode is None  # errors do not end the worker
            assert process.stdin is not None
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), timeout=30) == 0
        finally:
            await pool.stop()

    asyncio.run(run())


def load_worker_module():
    """``system_engine_worker`` imports only the standard library at module level."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("held_worker", Path(system_engine.WORKER))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_held_window_refers_to_the_held_rows_and_sends_the_newest_again():
    rows = cached_bars(T0, 100, "BTCUSDT")
    window = system_engine.held_window(rows, rows[89][0])
    assert window is not None
    # Everything after the held rows, and the last ten held minutes, which Binance may still settle.
    assert window["tail"] == rows[80:] and window["count"] == 100 and window["first_open_ms"] == T0
    assert window["head_sha256"] == system_engine.held_digest(rows[:80])
    assert system_engine.held_window(rows, rows[0][0] - 3 * HOUR_MS)["tail"] == rows, "nothing held in range"
    assert system_engine.held_window(rows[:50] + rows[51:], rows[89][0]) is None, "a gap is sent in full"
    assert system_engine.held_window([], T0) is None


def test_the_worker_rebuilds_a_held_window_or_asks_for_it_in_full():
    worker = load_worker_module()
    rows = cached_bars(T0, 100, "BTCUSDT")
    worker._CHECKED["BTCUSDT"] = {int(row[0]): (tuple(row), None, None) for row in rows[:90]}
    request = {
        "op": "registered_bars",
        "series": {
            "BTCUSDT": system_engine.held_window(rows, rows[89][0]),
            "ETHUSDT": cached_bars(T0, 3, "ETHUSDT"),
        },
    }
    expanded = worker._expand(request)
    assert expanded["series"]["BTCUSDT"] == rows
    assert expanded["series"]["ETHUSDT"] == cached_bars(T0, 3, "ETHUSDT")
    full = worker._expand({"op": "x", "series": {"A": rows}})
    assert full["series"]["A"] is rows, "full rows pass as they are"
    changed = dict(worker._CHECKED["BTCUSDT"])
    changed[T0] = (tuple([*rows[0][:9], "another identity"]), None, None)
    worker._CHECKED["BTCUSDT"] = changed
    with pytest.raises(worker._MissingBars):
        worker._expand(request)  # the held rows are not the ones the runner refers to
    del changed[T0 + 5 * MINUTE_MS]
    worker._CHECKED["BTCUSDT"] = {k: v for k, v in changed.items() if k != T0}
    with pytest.raises(worker._MissingBars):
        worker._expand(request)


FAKE_HOLDING_WORKER = """
import hashlib, json, sys
MARKER = "@@KAIROS-WORKER@@"
held = {}


def say(value):
    sys.stdout.write(MARKER + json.dumps(value) + "\\n")
    sys.stdout.flush()


say({"ready": True})
for line in sys.stdin:
    request = json.loads(line)
    if request.get("forget"):
        held.clear()
    if request.get("op") == "die":
        sys.exit(3)
    sent, full = {}, {}
    for symbol, value in request["series"].items():
        if isinstance(value, dict):
            known = held.get(symbol, {})
            first, size = value["first_open_ms"], value["count"] - len(value["tail"])
            head = [known.get(first + i * 60000) for i in range(size)]
            digest = hashlib.sha256("".join(str(r[9]) for r in head if r).encode()).hexdigest()
            if None in head or digest != value["head_sha256"]:
                say({"missing_bars": True, "symbol": symbol})
                break
            full[symbol] = head + value["tail"]
            sent[symbol] = len(value["tail"])
        else:
            full[symbol] = value
            sent[symbol] = "full"
    else:
        for symbol, rows in full.items():
            held[symbol] = {row[0]: row for row in rows}
        say({"results": {s: {"rows": len(r), "last": r[-1][0]} for s, r in full.items()}, "sent": sent})
"""


def test_engine_pool_sends_only_new_bars_to_a_worker_that_holds_the_rest(tmp_path, monkeypatch):
    import sys

    script = tmp_path / "worker.py"
    script.write_text(FAKE_HOLDING_WORKER, encoding="utf-8")
    monkeypatch.setattr(system_engine, "WORKER", script)
    config = EngineConfig(python=Path(sys.executable), source=tmp_path, timeout_s=10)
    pool = system_engine.EnginePool(config, 1)
    rows = cached_bars(T0, 600, "BTCUSDT")

    def request(first, last):
        return {"op": "registered_bars", "series": {"BTCUSDT": rows[first:last]}}

    async def run():
        try:
            first = await pool.ask(0, request(0, 300))
            assert first["sent"] == {"BTCUSDT": "full"} and first["results"]["BTCUSDT"]["rows"] == 300
            later = await pool.ask(0, request(60, 360))  # an hour later: 60 new minutes, 10 sent again
            assert later["sent"] == {"BTCUSDT": 70} and later["results"]["BTCUSDT"]["last"] == rows[359][0]
            again = await pool.ask(0, {**request(120, 420), "forget": True})  # lost them: all again
            assert again["sent"] == {"BTCUSDT": "full"} and again["results"]["BTCUSDT"]["rows"] == 300
            with pytest.raises(EngineError):
                await pool.ask(0, {"op": "die", "series": {}})
            fresh = await pool.ask(0, request(180, 480))  # a restarted worker holds nothing
            assert fresh["sent"] == {"BTCUSDT": "full"}
            other = await pool.ask(0, {"op": "hourly_trend_history", "series": {"BTCUSDT": rows[200:500]}})
            assert other["sent"] == {"BTCUSDT": "full"}, "only the ops whose bars a worker keeps"
        finally:
            await pool.stop()

    asyncio.run(run())


def test_real_engine_worker_hashes_a_window_as_the_engine_does():
    """Run in the engine's own Python: the cached window hash equals the engine's own."""
    import subprocess

    config = EngineConfig()
    if not config.python.exists() or not (config.source / "kairos_strategy").exists():
        pytest.skip("Strategy Engine checkout or environment is not present")
    worker_dir = Path(system_engine.__file__).parent
    script = f"""
import sys
sys.path[:0] = [{str(config.source)!r}, {str(worker_dir)!r}]
from kairos_strategy import runtime
from kairos_strategy.candles import Candle
import system_engine_worker as w

original = runtime._window_sha256
assert w._install_window_hash(Candle) and runtime._window_sha256 is w._window_sha256
T = {T0}
bars = [
    runtime.candle_to_closed_bar(Candle(
        symbol="SOLUSDT", timeframe="1m", open_time_ms=T + i * 60_000, close_time_ms=T + i * 60_000 + 59_999,
        open=80.0 + i / 7, high=81.0 + i / 7, low=79.0 + i / 11, close=80.5 + i / 11, volume=1.0 + i,
        quote_volume=80.0 * (1 + i), taker_buy_volume=0.5, taker_buy_quote_volume=40.0))
    for i in range(300)
]
assert w._window_sha256(bars[:200]) == original(bars[:200])
assert w._window_sha256(bars[60:260]) == original(bars[60:260]), "the reused bytes of a later window"
assert set(w._WINDOW_BYTES["SOLUSDT"]) == {{bar.bar_sha256 for bar in bars[60:260]}}, "only the latest window"
held = {{"first_open_ms": T, "count": 2, "head_sha256": "", "tail": []}}
answer = w._answer({{"op": "registered_bars", "series": {{"XUSDT": held}}}})
assert answer == {{"missing_bars": True, "symbol": "XUSDT"}}, answer
print("ok")
"""
    answer = subprocess.run([str(config.python), "-c", script], capture_output=True, text=True, timeout=120)
    assert answer.returncode == 0 and answer.stdout.strip().endswith("ok"), answer.stderr[-800:]


def test_hourly_scan_uses_its_pool_workers_and_falls_back_when_one_fails():
    boundary = (B0 // HOUR_MS + 600) * HOUR_MS
    full = HOURLY_TREND_HISTORY_MS // MINUTE_MS
    names = ("AAVEUSD", "BTCUSD", "ETHUSD", "XRPUSD")
    series = {name: cached_bars(boundary - HOURLY_TREND_HISTORY_MS, full, name + "T") for name in names}

    def answer(request):
        results = {}
        for symbol, rows in request["series"].items():
            decision = int(rows[-1][0]) + MINUTE_MS - 1
            state = engine_state(symbol, decision)
            results[symbol] = {"decision_ts_ms": decision, "state": state, "intent": None}
        return {"identity": IDENTITY, "results": results}

    class Pool:
        size = 3

        def __init__(self, broken=()):
            self.broken, self.calls = set(broken), []

        async def ask(self, index, request):
            assert request["op"] == "hourly_trend_bars"
            self.calls.append((index, sorted(request["series"])))
            if index in self.broken:
                raise EngineError("Strategy Engine: the worker exited")
            return answer(request)

    one_shot = []

    def runner(config, request):
        one_shot.append(sorted(request["series"]))
        return answer(request)

    def scan(pool):
        return asyncio.run(
            system_engine.hourly_trend_scan(
                EngineConfig(),
                series,
                boundary,
                reward_risk=1.5,
                timeout_minutes=720,
                runner=runner,
                pool=pool,
            )
        )

    pool = Pool()
    found = scan(pool)
    assert sorted(pool.calls) == [(0, ["AAVEUSDT", "XRPUSDT"]), (1, ["BTCUSDT"]), (2, ["ETHUSDT"])]
    assert one_shot == [] and sorted(found.features) == sorted(names) and found.skipped == []
    # A failed worker's instruments are decided by a one-shot worker; the others stay pooled.
    pool = Pool(broken={1})
    found = scan(pool)
    assert one_shot == [["BTCUSDT"]] and sorted(found.features) == sorted(names)
