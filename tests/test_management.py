"""Break-even, trailing stops and partial exits: measured first, applied only when proven."""

from __future__ import annotations

import asyncio
import dataclasses
import random
import time
from decimal import Decimal

import pytest

from kairos_autotrader import brain
from kairos_autotrader.engine import Book, Config, management_step
from kairos_autotrader.management import (
    MAX_POLICY_AGE_MS,
    PLAN,
    POLICY_FILE,
    Management,
    PartialTarget,
    choose_policy,
    compare_policies,
    load_policy,
    managed_outcome,
    managed_stop,
    policies_for,
    write_policy,
)
from kairos_autotrader.signals import Candle, outcome
from tests.test_autotrader import (
    SOL,
    FakeVenue,
    candidate,
    enter,
    journal_of,
    live_trader,
    only,
    paper_trader,
)
from tests.test_execution_session import _real_review

TRAIL = Management("trail", trail_start_r=1.0, trail_distance_r=0.5)
HALF = Management("half", partial_at_r=0.75, partial_fraction=0.5, breakeven_at_r=0.75)
LADDER = Management(
    "quarters",
    partial_targets=(PartialTarget(0.5, 0.25), PartialTarget(1.0, 0.25)),
    breakeven_at_r=0.5,
)


def record(side="LONG", entry=100.0, stop=95.0, take=107.5, start=0, minutes=60, **over):
    return {
        "id": over.pop("id", "s"),
        "symbol": over.pop("symbol", "SOLUSD"),
        "side": side,
        "ref_price": entry,
        "stop": stop,
        "take": take,
        "signal_time": start,
        "timeout_minutes": minutes,
        **over,
    }


def path(start: int, closes: list[float], spread: float = 0.2) -> list[Candle]:
    return [Candle(start + i * 60_000, c, c + spread, c - spread, c, 0.0) for i, c in enumerate(closes)]


def test_without_management_the_result_is_exactly_the_plan():
    rng = random.Random(7)
    for _ in range(300):
        side = rng.choice(["LONG", "SHORT"])
        sign = 1 if side == "LONG" else -1
        rec = record(side=side, stop=100 - sign * 5, take=100 + sign * 7.5, minutes=rng.randint(5, 50))
        price, closes = 100.0, []
        for _ in range(60):
            price += rng.uniform(-1.5, 1.5)
            closes.append(price)
        bars = path(0, closes, spread=rng.uniform(0.1, 1.0))
        assert (
            managed_outcome(rec, bars, cost_fraction=0.0013)["r"]
            == outcome(rec, bars, cost_fraction=0.0013)["r"]
        )


def test_trailing_stop_keeps_part_of_a_reversed_gain():
    bars = path(0, [101, 103, 106, 106.5, 104, 101, 97, 94])
    assert managed_outcome(record(), bars, cost_fraction=0)["r"] == -1.0, "the plan rides it back to the stop"
    managed = managed_outcome(record(), bars, cost_fraction=0, policy=TRAIL)
    assert managed["reason"] == "MANAGED_STOP" and managed["r"] == 0.84, "stopped 0.5 R under the best price"


def test_a_stop_moves_only_toward_less_risk():
    assert managed_stop(TRAIL, "LONG", 100, 95, 104) == 95, "not yet at the trailing start"
    assert managed_stop(TRAIL, "LONG", 100, 95, 106) == 103.5
    assert managed_stop(TRAIL, "SHORT", 100, 105, 94) == 96.5
    assert managed_stop(Management("be", breakeven_at_r=1.0), "SHORT", 100, 105, 99) == 105


def test_partial_exit_then_break_even():
    bars = path(0, [101, 104, 102, 99.7, 98])
    result = managed_outcome(record(), bars, cost_fraction=0, policy=HALF)
    assert result["reason"] == "MANAGED_STOP" and result["r"] == 0.375, "half at +0.75 R, half at entry"


@pytest.mark.parametrize("side,sign", [("LONG", 1), ("SHORT", -1)])
def test_replay_partial_targets_and_stop_share_one_original_position(side, sign):
    rec = record(side=side, stop=100 - sign * 5, take=100 + sign * 7.5)
    bars = path(0, [100 + sign * n for n in (3, 5.5, 8)])
    result = managed_outcome(rec, bars, cost_fraction=0, policy=LADDER)
    assert (result["reason"], result["r"]) == ("TAKE", 0.25 * 0.5 + 0.25 + 0.5 * 1.5)
    reversed_bars = path(0, [100 + sign * n for n in (3, 5.5, -1)])
    stopped = managed_outcome(rec, reversed_bars, cost_fraction=0, policy=LADDER)
    assert (stopped["reason"], stopped["r"]) == ("MANAGED_STOP", 0.375)
    assert any(p.partial_targets == LADDER.partial_targets for p in policies_for(1.5))


@pytest.mark.parametrize(
    "levels",
    [
        [{"at_r": 1, "fraction": 0.5}, {"at_r": 2, "fraction": 0.5}],
        [{"at_r": 1, "fraction": 0.25}, {"at_r": 1, "fraction": 0.25}],
        [{"at_r": float("nan"), "fraction": 0.25}],
    ],
)
def test_invalid_ladders_cannot_be_admitted(levels):
    with pytest.raises(ValueError):
        Management.from_dict({"name": "invalid", "partial_targets": levels})


def test_a_policy_is_chosen_only_with_proof_over_many_days():
    records, candles = [], []
    for day in range(40):
        start = day * 86_400_000
        records.append(record(id=f"d{day}", start=start))
        candles += path(start, [101, 103, 106, 106.5, 104, 101, 97, 94])
    results = compare_policies(records, {"SOLUSD": candles}, 0.0, (TRAIL, PLAN))
    trail, plan = results
    assert trail["signals"] == 40 and trail["clusters"] == 40 and trail["gain_r"] == 1.84
    assert plan["gain_r"] == 0 and plan["gain_lower_r"] == 0
    assert choose_policy(results, min_signals=40)["policy"]["name"] == "trail"
    assert choose_policy(results) is None, "fewer signals than the minimum"
    mixed = records + [record(id=f"u{day}", start=day * 86_400_000 + 3_600_000) for day in range(40)]
    up = []
    for day in range(40):
        up += path(day * 86_400_000 + 3_600_000, [101, 103, 106, 107.6])
    noisy = compare_policies(mixed, {"SOLUSD": sorted(candles + up, key=lambda c: c.t)}, 0.0, (TRAIL,))
    assert noisy[0]["clusters"] == 40, "signals of one instrument and day form one cluster"


def test_policy_grid_follows_the_take_distance():
    names = [policy.name for policy in policies_for(1.5)]
    assert "breakeven_at_0.5r" in names and "trail_from_1.0r_by_0.5r" in names
    for policy in policies_for(3.0):
        for level in (policy.breakeven_at_r, policy.trail_start_r, policy.partial_at_r):
            assert level is None or level < 3.0


def test_measured_policy_file_is_used_only_while_fresh_and_for_the_same_take(tmp_path):
    now = 1_794_400_000_000
    result = {"policy": TRAIL.as_dict(), "signals": 300, "gain_lower_r": 0.02}
    write_policy(tmp_path, [result], now_ms=now, reward_risk=1.5)
    policy, reason = load_policy(tmp_path, now_ms=now, reward_risk=1.5)
    assert policy == TRAIL and reason == "measured"
    assert load_policy(tmp_path, now_ms=now, reward_risk=2.0)[0] is None
    assert load_policy(tmp_path, now_ms=now + MAX_POLICY_AGE_MS + 1, reward_risk=1.5)[0] is None
    write_policy(tmp_path, [{**result, "gain_lower_r": -0.01}], now_ms=now, reward_risk=1.5)
    assert load_policy(tmp_path, now_ms=now, reward_risk=1.5) == (None, "no policy beat the plan with proof")
    (tmp_path / POLICY_FILE).write_text("{", encoding="utf-8")
    assert load_policy(tmp_path, now_ms=now, reward_risk=1.5)[0] is None
    with pytest.raises(ValueError):
        Management.from_dict({"name": "bad", "trail_start_r": 1.0})


def managed_trade(**over):
    return {
        "side": "LONG",
        "entry_price": "100",
        "stop": "95",
        "initial_stop": "95",
        "quantity": "1.0",
        "management": TRAIL.as_dict(),
        **over,
    }


def test_management_step_is_on_tick_away_from_the_market_and_never_looser():
    tick = dataclasses.replace(SOL, price_step=Decimal("0.01"))
    trade = managed_trade()
    assert management_step(trade, Book(0, Decimal("104"), Decimal("104.01")), tick) == (None, None)
    stop, part = management_step(trade, Book(0, Decimal("106.337"), Decimal("106.35")), tick)
    assert stop == Decimal("103.83") and part is None, "rounded down onto the tick for a long"
    trade["stop"] = str(stop)
    assert management_step(trade, Book(0, Decimal("105"), Decimal("105.01")), tick) == (None, None)
    short = managed_trade(side="SHORT", stop="105", initial_stop="105")
    stop, _ = management_step(short, Book(0, Decimal("93.99"), Decimal("94.003")), tick)
    assert stop == Decimal("96.51"), "rounded up onto the tick for a short"
    halves = managed_trade(management=HALF.as_dict(), quantity="0.3")
    _, part = management_step(halves, Book(0, Decimal("103.8"), Decimal("103.81")), SOL)
    assert part == Decimal("0.1"), "0.15 rounds down to the 0.1 step and leaves the minimum lot"
    small = managed_trade(management=HALF.as_dict(), quantity="0.1")
    assert management_step(small, Book(0, Decimal("103.8"), Decimal("103.81")), SOL)[1] is None
    assert small["partial_done"].startswith("skipped")


def test_ladder_coalesces_small_lots_and_recovers_a_fill_before_the_runner_saved():
    trade = managed_trade(management=LADDER.as_dict(), quantity="0.2", initial_quantity="0.2")
    first = Book(0, Decimal("103"), Decimal("103.01"))
    second = Book(0, Decimal("106"), Decimal("106.01"))
    assert management_step(trade, first, SOL)[1] is None
    assert management_step(trade, second, SOL)[1] == Decimal("0.1")
    # Startup restored a confirmed reduction from Execution, without a runner-side target flag.
    trade["quantity"] = "0.1"
    assert management_step(trade, second, SOL)[1] is None


def _write_policy(trader, policy: Management) -> None:
    write_policy(
        trader.store.directory,
        [{"policy": policy.as_dict(), "signals": 500, "gain_lower_r": 0.05}],
        now_ms=int(time.time() * 1000),
        reward_risk=trader.config.reward_risk,
    )


@pytest.mark.parametrize("qualified", [True, False])
def test_live_runner_hands_selected_targets_to_one_execution_owner(tmp_path, monkeypatch, qualified):
    _real_review(monkeypatch)
    trader = live_trader(tmp_path, FakeVenue())
    instrument = dataclasses.replace(SOL, min_quantity=Decimal(".01"), quantity_step=Decimal(".01"))
    trader.instruments["SOLUSD"] = instrument
    asyncio.run(trader.start())
    _write_policy(trader, LADDER)
    monkeypatch.setattr(trader.broker, "native_ladder_qualified_for", lambda symbol: qualified, raising=False)
    broker_enter = trader.broker.enter
    selected = []

    async def submit(*args, partial_targets=None, **kwargs):
        selected.extend(partial_targets or [])
        fill = await broker_enter(*args, **kwargs)
        return {**fill, "native_partial_targets": bool(partial_targets)}

    monkeypatch.setattr(trader.broker, "enter", submit)
    enter(trader)
    trade = only(trader)
    assert trade["native_partial_targets"] is qualified
    assert selected == (
        [{"at_r": 0.5, "fraction": 0.25}, {"at_r": 1.0, "fraction": 0.25}] if qualified else []
    )
    entry, stop = Decimal(trade["entry_price"]), Decimal(trade["initial_stop"])
    bid = entry + (entry - stop) * Decimal(".6")
    moved, partial = management_step(trade, Book(0, bid, bid + SOL.price_step), instrument)
    assert moved is not None and moved >= entry - SOL.price_step
    assert (partial is None) is qualified


@pytest.mark.parametrize("reason", ["STOP", "TAKE"])
def test_live_runner_accounts_for_bracket_completed_during_entry(tmp_path, monkeypatch, reason):
    _real_review(monkeypatch)
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    broker_enter = trader.broker.enter

    async def fast_bracket(*args, **kwargs):
        fill = await broker_enter(*args, **kwargs)
        venue.venue_exit("-0.15" if reason == "STOP" else ".27")
        trader.broker.receipt["exit_origin"] = "system"
        return {**fill, "closed_at_entry": reason, "exit_price": "82.1"}

    monkeypatch.setattr(trader.broker, "enter", fast_bracket)
    enter(trader)
    assert only(trader) is None and not venue.close_bodies and len(venue.market_bodies) == 1
    assert trader.store.state["round_trips"] == 1
    (closed,) = [row for row in journal_of(trader) if row["event"] == "trade_closed"]
    assert (closed["reason"], closed["exit_origin"], closed["exit_price"]) == (reason, "system", "82.1")


def test_paper_trade_trails_its_stop_and_exits_above_entry(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    venue = FakeVenue()
    trader = paper_trader(tmp_path, venue, Config(use_llm=False, cooldown_minutes=0))
    _write_policy(trader, Management("trail", trail_start_r=0.5, trail_distance_r=0.5))
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    assert trade["management"]["name"] == "trail" and trade["initial_stop"] == trade["stop"]
    entry, stop0 = Decimal(trade["entry_price"]), Decimal(trade["stop"])
    risk = entry - stop0
    venue.bid, venue.ask = entry + risk * Decimal("1.2"), entry + risk * Decimal("1.2") + Decimal("0.02")
    asyncio.run(trader.manage(trade, int(time.time() * 1000)))
    moved = Decimal(only(trader)["stop"])
    assert moved > entry, "trailing 0.5 R under the best price, above the entry"
    venue.bid, venue.ask = moved + Decimal("0.05"), moved + Decimal("0.07")
    asyncio.run(trader.manage(only(trader), int(time.time() * 1000)))
    assert Decimal(only(trader)["stop"]) == moved, "a falling price never loosens the stop"
    venue.bid, venue.ask = moved - Decimal("0.01"), moved + Decimal("0.01")
    asyncio.run(trader.manage(only(trader), int(time.time() * 1000)))
    assert only(trader) is None
    closed = [e for e in journal_of(trader) if e["event"] == "trade_closed"][-1]
    assert closed["reason"] == "STOP" and Decimal(closed["r_multiple"]) > 0
    assert [e for e in journal_of(trader) if e["event"] == "stop_moved"]


def test_paper_partial_exit_counts_in_the_trade_result(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    venue = FakeVenue()
    trader = paper_trader(tmp_path, venue, Config(use_llm=False, cooldown_minutes=0))
    _write_policy(trader, HALF)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    quantity, entry, stop0 = Decimal(trade["quantity"]), Decimal(trade["entry_price"]), Decimal(trade["stop"])
    risk = entry - stop0
    venue.bid, venue.ask = entry + risk * Decimal("0.8"), entry + risk * Decimal("0.8") + Decimal("0.02")
    asyncio.run(trader.manage(trade, int(time.time() * 1000)))
    trade = only(trader)
    assert trade["partial_done"] == "done" and Decimal(trade["quantity"]) < quantity
    assert Decimal(trade["stop"]) >= entry - SOL.price_step, "the rest is at break-even"
    venue.bid, venue.ask = Decimal(trade["stop"]) - Decimal("0.01"), Decimal(trade["stop"])
    asyncio.run(trader.manage(trade, int(time.time() * 1000)))
    closed = [e for e in journal_of(trader) if e["event"] == "trade_closed"][-1]
    assert Decimal(closed["r_multiple"]) > 0, "the partial profit is part of the result"


@pytest.mark.parametrize("side,sign", [("LONG", 1), ("SHORT", -1)])
def test_paper_targets_survive_restart_and_leave_a_protected_remainder(tmp_path, monkeypatch, side, sign):
    _real_review(monkeypatch)
    venue = FakeVenue()
    cfg = Config(use_llm=False, cooldown_minutes=0)
    instrument = dataclasses.replace(SOL, min_quantity=Decimal(".01"), quantity_step=Decimal(".01"))

    def runner():
        trader = paper_trader(tmp_path, venue, cfg)
        trader.instruments["SOLUSD"] = instrument
        asyncio.run(trader.start())
        return trader

    trader = runner()
    _write_policy(trader, LADDER)
    stop, take = (81.5, 83.62) if side == "LONG" else (83.2, 81.08)
    offer = candidate(side=side, stop=stop, take=take)
    asyncio.run(trader.try_enter(offer, brain.control_decision([offer])))
    trade = only(trader)
    entry, initial = Decimal(trade["entry_price"]), Decimal(trade["initial_stop"])
    size = Decimal(trade["quantity"])
    assert size == Decimal(".2")

    def poll(at_r):
        touch = entry + sign * Decimal(at_r) * abs(entry - initial)
        venue.bid, venue.ask = (
            (touch, touch + SOL.price_step) if sign == 1 else (touch - SOL.price_step, touch)
        )
        asyncio.run(trader.manage(only(trader), int(time.time() * 1000)))

    poll("0.6")
    assert Decimal(only(trader)["quantity"]) == size * Decimal(".75")
    assert abs(Decimal(only(trader)["stop"]) - entry) <= SOL.price_step
    trader = runner()
    poll("0.6")
    poll("1.1")
    poll("1.1")
    assert Decimal(only(trader)["quantity"]) == size / 2
    poll("2")
    assert only(trader) is None
    log = journal_of(trader)
    exits = [Decimal(row["quantity"]) for row in log if row["event"] == "partial_exit"]
    assert exits == [size / 4, size / 4]
    (closed,) = [row for row in log if row["event"] == "trade_closed"]
    assert closed["reason"] == "TAKE" and Decimal(closed["r_multiple"]) > 0


def test_without_a_proven_policy_nothing_is_managed(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, cooldown_minutes=0))
    asyncio.run(trader.start())
    enter(trader)
    assert only(trader)["management"] is None
    events = [e for e in journal_of(trader) if e["event"] == "management_policy"]
    assert events and events[-1]["reason"] == "no management measurement yet"
    off = paper_trader(tmp_path / "off", FakeVenue(), Config(use_llm=False, management="off"))
    _write_policy(off, TRAIL)
    assert off.management_policy(int(time.time() * 1000)) is None


def test_adding_to_an_open_position_stays_refused_with_the_measured_verdict(tmp_path, monkeypatch):
    from kairos_autotrader.management import scale_in_status

    now = 1_794_400_000_000
    assert scale_in_status(tmp_path, now_ms=now, reward_risk=1.5) == "adding to it is not measured yet"
    write_policy(
        tmp_path,
        [],
        now_ms=now,
        reward_risk=1.5,
        scale_in=[{"min_open_gain_r": 0.5, "signals": 124, "proven": False}],
    )
    assert scale_in_status(tmp_path, now_ms=now, reward_risk=1.5) == "adding to it is not proven profitable"
    assert "current plan" in scale_in_status(tmp_path, now_ms=now, reward_risk=2.0)
    write_policy(
        tmp_path,
        [],
        now_ms=now,
        reward_risk=1.5,
        scale_in=[{"min_open_gain_r": 1.0, "proven": True}, {"min_open_gain_r": 0.5, "proven": True}],
    )
    assert "from +0.5 R is proven" in scale_in_status(tmp_path, now_ms=now, reward_risk=1.5)

    _real_review(monkeypatch)
    trader = paper_trader(tmp_path / "rt", FakeVenue(), Config(use_llm=False, cooldown_minutes=0))
    asyncio.run(trader.start())
    enter(trader)
    enter(trader)
    skipped = [e for e in journal_of(trader) if e["event"] == "entry_skipped"][-1]
    assert (
        skipped["reason"]
        == "there is already a position in this instrument; adding to it is not measured yet"
    )


def test_scan_cadence_takes_one_position_per_instrument_with_cooldown():
    from kairos_autotrader.management import scan_cadence_summary

    day = 86_400_000
    records, candles = [], []
    for d in range(3):
        start = d * day
        candles += path(start, [101, 103, 106, 108, 108, 108, 108, 108, 108, 108])
        records += [
            record(id=f"a{d}", start=start, minutes=60),
            record(id=f"b{d}", start=start + 60_000, minutes=60),
            record(id=f"c{d}", start=start + 9 * 60_000, minutes=60),
        ]
    summary = scan_cadence_summary(records, {"SOLUSD": candles}, 0.0, cooldown_minutes=30, days=3)
    assert summary["every_signal"]["signals"] == 9
    assert summary["taken"]["signals"] == 3, "one at a time, nothing during the cooldown"
    assert summary["taken"]["mean_r"] == 1.5
