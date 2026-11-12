"""Offline tests of the bridge to the real Kairos Strategy Engine.

Only the two ``test_real_engine_...`` tests start the engine itself; they are skipped where
the engine checkout or its environment is absent.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest

from kairos_autotrader import candidates, system_cli, system_engine
from kairos_autotrader.evedex import Candle, Evedex
from kairos_autotrader.system_engine import (
    MINUTE_MS,
    EngineConfig,
    EngineError,
    MinuteStore,
    candidates_from,
    collect,
    intent_to_candidate,
)

T0 = 1_794_000_000_000 // MINUTE_MS * MINUTE_MS


def intent(**over) -> dict:
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


def test_intent_keeps_side_exits_and_hold_time():
    c = intent_to_candidate(intent(), revision="1", status="rejected")
    assert (c.symbol, c.side, c.setup) == ("SOLUSD", "SHORT", "trend_pullback_reclaim_v1")
    assert (c.ref_price, c.stop, c.take, c.timeout_minutes) == (83.0, 83.5, 82.0, 180)
    assert c.signal_time == T0 + 5 * MINUTE_MS, "the engine's inclusive close becomes the next millisecond"
    assert c.id == f"SOLUSD-SHORT-trend_pullback_reclaim_v1-{T0 + 5 * MINUTE_MS - 1}"
    assert c.atr == pytest.approx(0.5) and c.score == 0.6123 and "status rejected" in c.rationale
    trailing = intent_to_candidate(
        intent(trailing_distance=0.2, trailing_activation_price=82.5), revision="1", status="rejected"
    )
    assert "trailing stop not applied" in trailing.rationale


@pytest.mark.parametrize("bad", [{"side": "FLAT"}, {"stop_price": 82.5}, {"side": "LONG"}])
def test_inconsistent_intent_is_refused(bad):
    with pytest.raises(EngineError):
        intent_to_candidate(intent(**bad), revision="1", status="rejected")


def test_candidates_are_filtered_by_time_and_sleeve_errors_are_reported():
    old = intent(decision_ts_ms=T0 - 1)
    results = {
        "trend_pullback_reclaim_v1": {"revision": "1", "status": "rejected", "intents": [old, intent()]},
        "range_mean_reversion_v1": {"error": "ValueError: boom"},
        "right_tail_trend_v1": {"revision": "1", "status": "rejected", "intents": [intent(stop_price=1.0)]},
    }
    found, problems = candidates_from(results, since_ms=T0)
    assert [c.signal_time for c in found] == [T0 + 5 * MINUTE_MS]
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
    assert [c.symbol for c in found] == ["SOLUSD"] and len(bars["SOLUSD"]) == 59
    assert any("1 gaps" in p for p in problems) and any("XRPUSD" in p and "venue down" in p for p in problems)


def test_outcomes_use_only_the_bars_after_the_signal():
    candles = [Candle(T0 + i * MINUTE_MS, 83.0, 83.1, 82.9, 83.0, 1.0) for i in range(400)]
    candles[2] = Candle(candles[2].t, 83.0, 84.0, 82.9, 83.0, 1.0)  # a spike before the signal must not count
    candles[20] = Candle(candles[20].t, 83.0, 83.1, 81.9, 82.0, 1.0)  # the target is reached here
    done = intent_to_candidate(intent(), revision="1", status="rejected")
    open_one = intent_to_candidate(
        intent(decision_ts_ms=T0 + 399 * MINUTE_MS - 1), revision="1", status="rejected"
    )
    rows = system_cli.evaluate([done, open_one], candles, cost_fraction=0.0013)
    assert len(rows) == 1 and rows[0]["reason"] == "TAKE"
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
    offered = [intent_to_candidate(intent(), revision="1", status="rejected")]

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
    assert candidates.Candidate(**scans[0]["candidates"][0]) == offered[0], (
        "the report can rebuild the candidate"
    )
    state = json.loads((args.runtime_dir / "state.json").read_text())
    assert state["mode"] == "engine-shadow" and list(state["seen"]) == [offered[0].id]

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
            intent_to_candidate(raw, revision=result["revision"], status=result["status"])


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


def hourly_intent(**over) -> dict:
    base = intent(
        sleeve_id=system_engine.HOURLY_TREND_ID,
        decision_ts_ms=B0 + QUARTER_MS - 1,
        max_holding_ms=720 * MINUTE_MS,
        metadata={"setup": "BREAKOUT", "atr": "0.40000000000000002", "trend": "DOWN"},
    )
    base.update(over)
    return base


def test_quiet_minutes_are_bridged_only_inside_a_quarter_hour_that_traded():
    rows = minute_rows(60, start=B0)
    kept = [r for i, r in enumerate(rows) if i not in (3, 4) and not 15 <= i < 30]
    kept[2][4] = 101.5  # the close before the hole
    filled, added = system_engine.fill_quiet_minutes(kept)
    assert added == 2 and [r[0] for r in filled[:6]] == [B0 + i * MINUTE_MS for i in range(6)]
    assert filled[3] == [B0 + 3 * MINUTE_MS, 101.5, 101.5, 101.5, 101.5, 0.0, 0.0]
    assert not any(B0 + 15 * MINUTE_MS <= r[0] < B0 + 30 * MINUTE_MS for r in filled), (
        "a quarter hour without a single minute stays missing"
    )
    assert system_engine.fill_quiet_minutes(rows) == (rows, 0)
    assert system_engine.fill_quiet_minutes([]) == ([], 0)
    polled = [list(r) for r in rows]
    polled[20] = [polled[20][0], 100.3, 100.4, 100.2, 100.25, 0.0, 0.0]  # a live-polled quiet minute
    normalised, added = system_engine.fill_quiet_minutes(polled)
    assert added == 1 and normalised[20] == [polled[20][0], 100.0, 100.0, 100.0, 100.0, 0.0, 0.0], (
        "a zero-volume row is replaced, so the series does not depend on how it was collected"
    )

    window, _ = system_engine.decision_window(minute_rows(600, start=B0), B0 + 300 * MINUTE_MS)
    assert window[0][0] == B0 and window[-1][0] == B0 + 299 * MINUTE_MS


def test_hourly_intent_keeps_the_setup_name_prices_and_bar_times():
    c = system_engine.hourly_intent_to_candidate(hourly_intent())
    assert c.id == f"SOLUSD-SHORT-BREAKOUT-{B0 + QUARTER_MS - HOUR_MS}"
    assert (c.bar_time, c.signal_time) == (B0 + QUARTER_MS - HOUR_MS, B0 + QUARTER_MS)
    assert (c.ref_price, c.stop, c.take, c.timeout_minutes) == (83.0, 83.5, 82.0, 720)
    assert c.atr == pytest.approx(0.4) and c.score == 0.6123 and "20-bar low" in c.rationale
    for bad in (
        {"side": "LONG"},
        {"stop_price": 82.5},
        {"metadata": {"setup": "SOMETHING_ELSE", "atr": "0.4"}},
        {"metadata": {"setup": "BREAKOUT"}},
    ):
        with pytest.raises((EngineError, KeyError)):
            system_engine.hourly_intent_to_candidate(hourly_intent(**bad))


def test_engine_answer_is_refused_unless_it_is_the_pinned_source():
    identity = {
        "strategy_id": system_engine.HOURLY_TREND_ID,
        "revision": "1",
        "strategy_code_sha256": system_engine.HOURLY_TREND_CODE_SHA256,
        "config_sha256": "c" * 64,
    }
    assert system_engine.checked_hourly_answer({"identity": identity, "results": {}}) == (identity, {})
    for bad in (
        {"identity": {**identity, "strategy_code_sha256": "0" * 64}, "results": {}},
        {"identity": {**identity, "strategy_id": "other_v1"}, "results": {}},
        {"identity": identity},
        {"results": {}},
    ):
        with pytest.raises(EngineError):
            system_engine.checked_hourly_answer(bad)


def test_history_of_the_live_rules_reports_bad_instruments_and_keeps_the_rest():
    asked = []

    def runner(config, request):
        asked.append(request)
        return {
            "identity": {
                "strategy_id": system_engine.HOURLY_TREND_ID,
                "strategy_code_sha256": system_engine.HOURLY_TREND_CODE_SHA256,
            },
            "results": {
                "SOLUSD": {"intents": [hourly_intent(), hourly_intent(stop_price=1.0)]},
                "XRPUSD": {"error": "ValueError: boom"},
            },
        }

    rows = [r for r in minute_rows(30, start=B0) if r[0] != B0 + 7 * MINUTE_MS]
    found, problems = system_engine.hourly_trend_history(
        EngineConfig(), {"SOLUSD": rows, "XRPUSD": rows}, reward_risk=1.5, timeout_minutes=720, runner=runner
    )
    assert asked[0]["all"] is True and len(asked[0]["series"]["SOLUSD"]) == 30, "the hole is bridged"
    assert [c.setup for c in found] == ["BREAKOUT"]
    assert (
        len(problems) == 2 and any("boom" in p for p in problems) and any("bad intent" in p for p in problems)
    )


def test_real_engine_decides_on_the_newest_quarter_hour_with_the_pinned_rules():
    config = EngineConfig()
    try:
        config.check_hourly_trend()
    except EngineError:
        pytest.skip("Strategy Engine checkout with kairos_strategy.hourly_trend is not present")
    rows, price = [], 100.0
    boundary = B0 + 12 * 1440 * MINUTE_MS + QUARTER_MS
    for i in range(12 * 1440 + 15):  # twelve days of a slow climb with pullbacks
        step = 0.012 if (i // 240) % 3 else -0.015
        nxt = price + step
        rows.append(
            [B0 + i * MINUTE_MS, price, max(price, nxt) + 0.01, min(price, nxt) - 0.01, nxt, 5.0, 5.0 * nxt]
        )
        price = nxt
    window, added = system_engine.decision_window(rows, boundary)
    assert added == 0 and window[-1][0] == boundary - MINUTE_MS
    found = asyncio.run(
        system_engine.hourly_trend_scan(
            config, {"TESTUSD": window}, boundary, reward_risk=1.5, timeout_minutes=720
        )
    )
    assert found.identity["strategy_code_sha256"] == system_engine.HOURLY_TREND_CODE_SHA256, (
        "the pin in system_engine.py must be updated together with the engine source"
    )
    assert found.skipped == [] and list(found.features) == ["TESTUSD"]
    features = found.features["TESTUSD"]
    assert features.trend in ("UP", "DOWN", "FLAT") and features.price == pytest.approx(rows[-1][4])
    assert set(found.context["TESTUSD"]) == {
        "change_1h_pct",
        "change_7d_pct",
        "position_in_7d_range",
        "last_bar_volume_vs_20bar_avg",
        "consecutive_up_bars",
        "consecutive_down_bars",
    }
    for c in found.candidates:
        assert c.signal_time == boundary and c.setup in ("PULLBACK_RECLAIM", "BREAKOUT")

    late = asyncio.run(
        system_engine.hourly_trend_scan(
            config, {"TESTUSD": window[:-1]}, boundary, reward_risk=1.5, timeout_minutes=720
        )
    )
    assert late.features == {} and late.skipped == ["TESTUSD: no fresh closed bar"]

    history, problems = system_engine.hourly_trend_history(
        config, {"TESTUSD": rows}, reward_risk=2.0, timeout_minutes=240
    )
    assert problems == [] and history, "the synthetic climb must produce candidates"
    for c in history:
        assert c.timeout_minutes == 240
        assert abs(c.take - c.ref_price) == pytest.approx(2.0 * abs(c.ref_price - c.stop))
