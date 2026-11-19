"""Early exits when the Macro regime turns against an open position, and their measurement."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from decimal import Decimal

from kairos_autotrader.engine import REGIME_EXIT_MIN_SAMPLES, Config, Store, regime_exit_reason
from kairos_autotrader.signals import Candle, held_outcome
from kairos_autotrader.system_rules import MacroDesk, RulesConfig
from tests.test_autotrader import FakeVenue, enter, journal_of, only, paper_trader
from tests.test_execution_session import _real_review
from tests.test_system_rules import MacroWorker, refresh

HOUR = 3_600_000


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


class Macro:
    """A Macro desk stand-in holding one allocation."""

    def __init__(self, regime: str, produced_ms: int) -> None:
        self.value = {"regime": regime, "produced_at": iso(produced_ms)}

    def allocation(self, now_ms=None):
        return self.value


def test_only_a_newer_opposing_regime_closes_early():
    entered = 1_794_400_000_000
    bear_later = {"regime": "BEAR", "produced_at": iso(entered + HOUR)}
    assert "BEAR opposes the open LONG" in regime_exit_reason("LONG", "BULL", bear_later, entered)
    assert regime_exit_reason("SHORT", "BULL", bear_later, entered) is None, "BEAR does not oppose a SHORT"
    assert regime_exit_reason("LONG", "BEAR", bear_later, entered) is None, "entered in the same regime"
    chop = {"regime": "CHOP", "produced_at": iso(entered + HOUR)}
    assert regime_exit_reason("LONG", "BULL", chop, entered) is None, "a range is not an opposing trend"
    stale = {"regime": "BEAR", "produced_at": iso(entered - HOUR)}
    assert regime_exit_reason("LONG", "UNKNOWN", stale, entered) is None, "judged before the entry"
    assert regime_exit_reason("LONG", "BULL", {"regime": "BEAR", "produced_at": "?"}, entered) is None
    assert regime_exit_reason("LONG", "BULL", None, entered) is None
    bull = {"regime": "BULL", "produced_at": iso(entered + HOUR)}
    assert "BULL opposes the open SHORT" in regime_exit_reason("SHORT", "CHOP", bull, entered)


def bars(start: int, prices: list[tuple[float, float, float]]) -> list[Candle]:
    return [Candle(start + i * 60_000, c, h, low, c, 0.0) for i, (h, low, c) in enumerate(prices)]


def test_held_outcome_follows_the_plan_on_the_initial_risk():
    plan = {
        "side": "LONG",
        "entry": "100",
        "stop": "95",
        "take": "110",
        "risk_price": "5",
        "start_ms": 0,
        "deadline_ms": 600_000,
    }
    assert held_outcome(plan, bars(0, [(101, 99, 100), (111, 100, 110)]), cost_fraction=0) == 2.0
    assert held_outcome(plan, bars(0, [(101, 94, 96)]), cost_fraction=0) == -1.0
    assert held_outcome(plan, bars(0, [(101, 99, 100)] * 10 + [(103, 101, 102)]), cost_fraction=0) == 0.4
    assert held_outcome(plan, bars(0, [(101, 99, 100)]), cost_fraction=0) is None, "not decided yet"
    breakeven = {**plan, "stop": "100"}
    assert held_outcome(breakeven, bars(0, [(101, 99.5, 100)]), cost_fraction=0) == 0.0
    assert held_outcome(plan, bars(0, [(111, 99, 100)]), cost_fraction=0.001) == 1.98
    short = {**plan, "side": "SHORT", "stop": "105", "take": "90"}
    assert held_outcome(short, bars(0, [(106, 99, 104)]), cost_fraction=0) == -1.0


def test_macro_refresh_interval_is_configurable(tmp_path):
    calls = []

    async def extract(session, *, api_key, model, system, user, effort, timeout_s):
        calls.append(model)
        return {"regime": "BEAR"}, {"input_tokens": 1}, ""

    desk = MacroDesk(
        Store(tmp_path / "rt"),
        RulesConfig(),
        ("PULLBACK_RECLAIM",),
        runner=MacroWorker(),
        extract=extract,
        refresh_ms=6 * HOUR,
    )
    now = 1_794_400_000_000
    assert refresh(desk, now) is not None and len(calls) == 1
    assert refresh(desk, now + 5 * HOUR) is None and len(calls) == 1
    assert refresh(desk, now + 6 * HOUR) is not None and len(calls) == 2
    assert Config(macro_refresh_hours=6).validate() is None
    for wrong in (0.5, 25):
        try:
            Config(macro_refresh_hours=wrong).validate()
        except ValueError:
            continue
        raise AssertionError(f"macro_refresh_hours={wrong} was accepted")


def test_paper_position_closes_early_and_is_measured_against_holding(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, cooldown_minutes=0))
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    trade["regime"] = "BULL"
    trader.macro = Macro("BEAR", int(trade["decided_at_ms"]) + 1)
    asyncio.run(trader.manage(trade, int(time.time() * 1000)))
    assert only(trader) is None
    closed = [e for e in journal_of(trader) if e["event"] == "trade_closed"][-1]
    assert closed["reason"] == "REGIME_EXIT" and "BEAR opposes the open LONG" in closed["early_exit"]
    (plan,) = trader.store.state["early_exit_pending"]
    assert plan["exit_r"] == closed["r_multiple"] and plan["signal_symbol"] == trade["signal_symbol"]

    # The bars after the exit reach the take: holding would have made more.
    start, deadline = int(plan["start_ms"]), int(plan["deadline_ms"])
    rows = [[start + i * 60_000, 82.4, 82.5, 82.3, 82.4, 0, 1, 0, 0, ""] for i in range(3)]
    trader._measure_early_exits({"SOLUSD": rows})
    assert trader.store.state["early_exit_pending"], "not measurable before the plan's deadline"
    rows.append([start + 3 * 60_000, 82.4, 84.0, 82.3, 83.9, 0, 1, 0, 0, ""])
    rows.append([deadline, 83.9, 83.9, 83.9, 83.9, 0, 1, 0, 0, ""])
    trader._measure_early_exits({"SOLUSD": rows})
    measured = [e for e in journal_of(trader) if e["event"] == "early_exit_measured"][-1]
    assert Decimal(measured["held_r"]) > Decimal("1") and Decimal(measured["saved_r"]) < 0
    stats = trader.store.state["early_exit_stats"]
    assert stats["n"] == 1 and stats["saved_r"] < 0 and not trader.store.state["early_exit_pending"]


def test_early_exits_switch_themselves_off_when_they_cost_more_than_holding(tmp_path):
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, cooldown_minutes=0))
    asyncio.run(trader.start())
    trader.macro = Macro("BEAR", int(time.time() * 1000))
    assert trader._regime_exit("LONG", "BULL", 0) is not None
    trader.store.state["early_exit_stats"] = {"n": REGIME_EXIT_MIN_SAMPLES - 1, "saved_r": -3.0}
    assert trader._regime_exit("LONG", "BULL", 0) is not None, "too few measurements to judge"
    trader.store.state["early_exit_stats"] = {"n": REGIME_EXIT_MIN_SAMPLES, "saved_r": 1.5}
    assert trader._regime_exit("LONG", "BULL", 0) is not None, "they help: they stay on"
    trader.store.state["early_exit_stats"] = {"n": REGIME_EXIT_MIN_SAMPLES, "saved_r": -0.5}
    assert trader._regime_exit("LONG", "BULL", 0) is None
    assert trader._regime_exit("LONG", "BULL", 0) is None
    disabled = [e for e in journal_of(trader) if e["event"] == "regime_exit_disabled"]
    assert len(disabled) == 1 and disabled[0]["measured"] == REGIME_EXIT_MIN_SAMPLES
    trader.config.regime_exit = False
    trader.store.state["early_exit_stats"] = {}
    assert trader._regime_exit("LONG", "BULL", 0) is None, "the owner can switch them off"
