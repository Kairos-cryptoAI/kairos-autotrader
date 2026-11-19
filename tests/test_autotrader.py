"""Offline tests. The venue here is a fake: nothing in this file touches the network."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest
from support import (
    HOUR,
    MINUTE,
    QUARTER,
    BarFeed,
    fake_engine,
    fake_parser,
    fake_router,
    routed,
    strict_intent,
)

from kairos_autotrader import brain, routing, signals, system_engine
from kairos_autotrader.engine import (
    SCAN_AFTER_CLOSE_MS,
    Config,
    ConfigError,
    LiveBroker,
    NeedsOperator,
    PaperBroker,
    Store,
    Trader,
    software_exit,
)
from kairos_autotrader.evedex import (
    Book,
    Candle,
    Instrument,
    OrderSigner,
    Position,
    Unknown,
    eth_number,
    new_order_id,
    plain,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "sdk_parity.json").read_text())
TEST_KEY = (
    "0x" + "11" * 32
)  # the project's synthetic test key; the fixture was signed with it by the official SDK
LINK = Instrument("LINKUSD", Decimal("0.1"), Decimal("0.1"), Decimal("0.001"), Decimal("5"), True)
SOL = Instrument("SOLUSD", Decimal("0.1"), Decimal("0.1"), Decimal("0.01"), Decimal("5"), True)
BTC = Instrument("BTCUSD", Decimal("0.001"), Decimal("0.001"), Decimal("0.1"), Decimal("5"), True)
FEE = Decimal("0.00045")

# Entries are decided by the Risk Manager; here a stand-in approves with a loss-at-stop size.
pytestmark = pytest.mark.usefixtures("risk_approves")


def now_ms() -> int:
    return int(time.time() * 1000)


# ---- signing: byte-for-byte parity with the official SDK -----------------------


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["chain"])
def test_signatures_match_official_sdk(case):
    signer = OrderSigner(TEST_KEY, int(case["chain"]))
    assert signer.address == FIXTURE["address"]
    market = signer.market_order(
        order_id=case["market"]["id"], instrument="SOLUSD", side="BUY", cash=Decimal("16.46")
    )
    assert market == case["market"]
    close = signer.close_order(order_id=case["close"]["id"], instrument="SOLUSD", quantity=Decimal("0.2"))
    assert close == case["close"]
    tpsl = signer.tpsl(
        instrument="SOLUSD", kind="stop-loss", side="BUY", quantity=Decimal("0.2"), price=Decimal("81.35")
    )
    assert tpsl == case["tpsl"]


def test_protection_rides_in_body_without_changing_signature():
    signer = OrderSigner(TEST_KEY, 42161)
    case = FIXTURE["cases"][0]["market"]
    protection = [{"type": "stop-loss", "side": "BUY", "quantity": "0", "price": "81.35"}]
    body = signer.market_order(
        order_id=case["id"], instrument="SOLUSD", side="BUY", cash=Decimal("16.46"), protection=protection
    )
    assert body["signature"] == case["signature"] and body["tpsl"] == protection


def test_number_and_id_formats():
    assert (
        plain(Decimal("16.4600")) == "16.46" and plain(Decimal("100")) == "100" and plain(Decimal(0)) == "0"
    )
    assert eth_number(Decimal("16.46")) == 1_646_000_000
    assert re.fullmatch(r"[0-9]{5}:[0-9a-f]{26}", new_order_id())
    assert not hasattr(OrderSigner, "withdraw")


# ---- configuration ceilings ---------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("capital_basis_usd", 150_001),
        # 1% and 5% of the 100 USD default capital basis
        ("risk_per_trade_usd", 1.01),
        ("session_loss_stop_usd", 5.01),
        ("max_trades_per_day", 41),
        ("max_open_positions", 11),
        # the equity guard lies within [capital, 2 x capital]
        ("max_account_equity_usd", 99),
        ("max_account_equity_usd", 201),
        ("position_notional_of_capital", 3.5),
        ("position_notional_of_capital", 0.01),
        ("right_tail_interval_hours", 5),
        ("strategies", []),
        ("strategies", ["unknown_strategy_v1"]),
        ("management_policies", {"regime_aligned_right_tail_v1": {"trail_start_r": 1.0}}),
    ],
)
def test_hard_ceilings_cannot_be_configured_away(field, value):
    config = Config()
    setattr(config, field, value)
    with pytest.raises(ConfigError):
        config.validate()


def test_a_100k_account_is_configured_with_the_same_fractions():
    config = Config()
    config.capital_basis_usd = 100_000
    config.risk_per_trade_usd = 250
    config.session_loss_stop_usd = 3_000
    config.max_account_equity_usd = 125_000
    config.strategies = ["regime_aligned_right_tail_v1"]
    config.position_notional_of_capital = 0.25
    config.management_policies = {
        "regime_aligned_right_tail_v1": {"trail_start_r": 1.0, "trail_distance_r": 2.0}
    }
    config.validate()
    config.risk_per_trade_usd = 1_001
    with pytest.raises(ConfigError):
        config.validate()


# ---- exits ---------------------------------------------------------


def test_software_exit_rules():
    trade = {"side": "LONG", "stop": "81.5", "take": "83.6", "deadline_ms": now_ms() + 60_000}
    assert software_exit(trade, Book(0, Decimal("82.0"), Decimal("82.1")), now_ms()) is None
    assert software_exit(trade, Book(0, Decimal("81.5"), Decimal("81.6")), now_ms()) == "STOP"
    assert software_exit(trade, Book(0, Decimal("83.7"), Decimal("83.8")), now_ms()) == "TAKE"
    assert software_exit(trade, Book(0, Decimal("82.0"), Decimal("82.1")), now_ms() + 120_000) == "TIMEOUT"
    short = {"side": "SHORT", "stop": "83", "take": "81", "deadline_ms": now_ms() + 60_000}
    assert software_exit(short, Book(0, Decimal("82.9"), Decimal("83.0")), now_ms()) == "STOP"
    assert software_exit(short, Book(0, Decimal("80.9"), Decimal("81.0")), now_ms()) == "TAKE"


# ---- candidates ----------------------------------------------------------------


def candidate(**over) -> signals.Routed:
    """A routed SOLUSD long: a real ``CandidateRouteV1`` around a real ``StrategyIntentV1``."""
    return routed(**over)


def test_a_routed_candidate_reads_everything_from_the_contracts():
    c = candidate(signal_time=900_000, score=0.41237)
    intent = c.intent
    assert (c.instrument, intent.symbol, intent.venue) == ("SOLUSD", "SOLUSDT", "BINANCE_UM")
    assert c.id == intent.intent_id and len(c.id) == 64
    assert (c.side, c.setup, c.signal_time) == ("LONG", "PULLBACK_RECLAIM", 900_000)
    assert (c.reference_price, c.stop, c.take, c.timeout_minutes) == (82.35, 81.5, 83.62, 720)
    assert c.atr == 0.8 and c.strength == 0.41237
    record = c.record()
    assert record["id"] == c.id and record["symbol"] == "SOLUSD" and record["score"] == 0.4124
    assert (record["signal_symbol"], record["signal_venue"]) == ("SOLUSDT", "BINANCE_UM")
    assert record["route_id"] == c.route.route_id and record["review_tier"] == "NORMAL"
    assert record["review_deadline_ms"] == intent.decision_ts_ms + 60_000
    assert record["input_bars"] == 1 and record["strategy_id"] == system_engine.HOURLY_TREND_ID
    assert len(json.dumps(record)) < 1_500, "records stay small whatever the intent declares"

    with pytest.raises(ValueError, match="not the signal symbol"):
        signals.Routed("LINKUSD", c.route)
    with pytest.raises(ValueError, match="no Binance signal symbol"):
        signals.signal_symbol("SOLEUR")


def test_outcome_counts_stop_first_and_charges_costs():
    c = candidate(signal_time=900_000).record()
    both = [Candle(900_000, 82.3, 83.7, 81.4, 82.0, 1.0)]
    assert signals.outcome(c, both, cost_fraction=0.0013)["reason"] == "STOP"
    win = [Candle(900_000, 82.3, 83.7, 82.0, 83.6, 1.0)]
    result = signals.outcome(c, win, cost_fraction=0.0013)
    assert result["reason"] == "TAKE" and 1.3 < result["r"] < 1.5
    assert signals.outcome(c, [Candle(0, 82.3, 99, 1, 82.0, 1.0)], cost_fraction=0.0)["r"] is None


# ---- brain --------------------------------------------------------------------


def verdict(cid: str | None = None, kind: str = "ALLOW", confidence: float = 0.7) -> dict:
    return {"candidate_id": cid or candidate().id, "verdict": kind, "confidence": confidence, "reason": "ok"}


def reply(**over):
    base = {"action": "TRADE", "market_view": "v", "reason": "r", "verdicts": [verdict()]}
    base.update(over)
    return base


def test_valid_reply_becomes_trade():
    decision = brain.validate_reply(reply(), [candidate()], min_confidence=0.55)
    assert decision.action == "TRADE" and decision.candidate_id == candidate().id and not decision.problems


@pytest.mark.parametrize(
    "bad",
    [
        {"verdicts": [verdict(confidence=0.3)]},
        {"verdicts": [verdict(confidence=4.0)]},
        {"verdicts": [verdict(kind="VETO")]},
        {"verdicts": [verdict(), verdict(kind="VETO")]},
        {"verdicts": [verdict("GHOST")]},
        {"verdicts": [verdict(), verdict("GHOST")]},
        {"verdicts": []},
        {"action": "WAIT"},
    ],
)
def test_inconsistent_reply_becomes_wait(bad):
    decision = brain.validate_reply(reply(**bad), [candidate()], min_confidence=0.55)
    assert decision.action == "WAIT" and decision.candidate_id is None


def test_several_allowed_candidates_become_ordered_picks_one_per_symbol():
    sol, link = candidate(), candidate(symbol="LINKUSD")
    sol_again = candidate(score=0.45)
    answer = reply(
        verdicts=[
            verdict(sol.id, confidence=0.6),
            verdict(link.id, confidence=0.8),
            verdict(sol_again.id, confidence=0.9),
        ]
    )
    both = brain.validate_reply(answer, [sol, link, sol_again], 0.55, max_picks=3)
    assert both.picks == [sol_again.id, link.id] and both.candidate_id == sol_again.id
    assert both.confidence == 0.9
    one = brain.validate_reply(answer, [sol, link, sol_again], 0.55, max_picks=1)
    assert one.picks == [sol_again.id]
    picky = brain.validate_reply(answer, [sol, link, sol_again], 0.85, max_picks=3)
    assert picky.picks == [sol_again.id]


def test_non_object_reply_is_wait():
    assert brain.validate_reply("TRADE everything", [candidate()], 0.5).action == "WAIT"


def test_output_text_extraction():
    response = {
        "status": "completed",
        "output": [
            {"type": "reasoning"},
            {"type": "message", "content": [{"type": "output_text", "text": '{"a":1}'}]},
        ],
    }
    assert brain.extract_output_text(response) == '{"a":1}'
    with pytest.raises(ValueError):
        brain.extract_output_text({"status": "incomplete", "output": []})


def test_control_decision_takes_best_score():
    a, b = candidate(score=0.3), candidate(score=0.6)
    assert brain.control_decision([a, b]).candidate_id == b.id
    assert brain.control_decision([]).action == "WAIT"
    c = candidate(symbol="LINKUSD", score=0.5)
    assert brain.control_decision([a, b, c], 3).picks == [b.id, c.id], "one pick per instrument"


# ---- engine against a fake venue ----------------------------------------------


class FakeVenue:
    def __init__(self) -> None:
        self.funding = Decimal("109.42")
        self.bid, self.ask = Decimal("82.33"), Decimal("82.35")
        self.position_list: list[Position] = []
        self.orders_open: list[dict] = []
        self.tpsl: list[dict] = []
        self.market_bodies: list[dict] = []
        self.close_bodies: list[dict] = []
        self.cancelled: list[str] = []
        self.close_pnl = Decimal("-0.20")
        self.books: dict[str, tuple[Decimal, Decimal]] = {}

    async def book(self, symbol):
        bid, ask = self.books.get(symbol, (self.bid, self.ask))
        return Book(now_ms(), bid, ask)

    async def metrics(self):
        return {}

    async def candle_rows(self, symbol, group, start_ms, end_ms):
        return []

    async def balance(self):
        return self.funding, self.funding

    async def positions(self):
        return list(self.position_list)

    async def open_orders(self):
        return list(self.orders_open)

    async def active_tpsl(self):
        return list(self.tpsl)

    def venue_exit(self, pnl: str, symbol: str = "SOLUSD") -> None:
        self.position_list = [p for p in self.position_list if p.instrument != symbol]
        self.funding += Decimal(pnl)


class FakePortfolio(LiveBroker):
    """Observe the runner-to-Execution contract without signing or submitting a request."""

    account_id = "evedex-prod-test"
    entry_deadline_utc = datetime.max.replace(tzinfo=UTC)

    async def refresh_performance(self, *, force=False):
        pass

    def __init__(self, venue):
        self.venue = venue
        self.receipt = {"pnl_basis": "unknown", "net_pnl": None, "exit_origin": "unknown"}
        self.baseline = venue.funding
        self.flow = Decimal(0)

    def pilot_status(self):
        return {
            "started_at": datetime.now(UTC).date().isoformat(),
            "funding_balance": str(self.venue.funding),
            "baseline": str(self.baseline),
            "external_flow": str(self.flow),
            "stopped": self.baseline - (self.venue.funding - self.flow) >= 5,
        }

    async def reconcile_start(self, trades, pending):
        own = {trade["symbol"] for trade in trades}
        if pending or await self.foreign_exposure(own):
            raise NeedsOperator("unowned position or pending entry")

    async def prepare_admission(self, own):
        if await self.foreign_exposure(own):
            raise NeedsOperator("unexpected exposure")
        return self.venue.funding

    async def equity(self):
        return self.venue.funding

    async def available(self):
        return self.venue.funding

    async def account_reads(self):
        return *await self.venue.balance(), await self.venue.positions(), await self.venue.open_orders()

    async def foreign_exposure(self, own):
        return (
            [p.instrument for p in await self.venue.positions() if p.instrument not in own]
            + self.venue.orders_open
            + self.venue.tpsl
        )

    async def position_open(self, trade):
        return any(p.instrument == trade["symbol"] for p in await self.venue.positions())

    async def enter(self, candidate, quantity, book, instrument, stop, take, *, leverage, decision):
        self.venue.market_bodies.append(
            {
                "candidate": candidate.id,
                "decision_id": decision.decision_id,
                "trade_id": decision.trade_id,
                "quantity": quantity,
                "leverage": leverage,
                "stop": stop,
                "take": take,
            }
        )
        side = "BUY" if candidate.side == "LONG" else "SELL"
        price = book.ask if side == "BUY" else book.bid
        self.venue.position_list.append(Position(instrument.name, side, quantity, price, True, "stop-test"))
        return {
            "quantity": quantity,
            "price": price,
            "trade_id": decision.trade_id,
            "order_id": f"test-{instrument.name}",
            "protected": True,
        }

    async def close(self, trade, book, reason):
        self.venue.close_bodies.append({"quantity": trade["quantity"], "leverage": trade["leverage"]})
        self.venue.venue_exit(str(self.venue.close_pnl), trade["symbol"])

    async def settle(self, trade):
        return self.receipt


def live_trader(tmp_path, venue: FakeVenue, **config) -> Trader:
    cfg = Config(**{"use_llm": False, "cooldown_minutes": 0, "exit_confirm_polls": 1, **config})
    cfg.validate()
    store = Store(tmp_path / "rt")
    broker = FakePortfolio(venue)
    return Trader(
        api=venue,
        session=None,
        store=store,
        config=cfg,
        broker=broker,
        fee=FEE,
        instruments={"SOLUSD": SOL, "LINKUSD": LINK},
        openai_key=None,
    )


def only(owner):
    """The single open trade of a trader or store, or None."""
    trades = getattr(owner, "store", owner).state.get("open_trades", [])
    assert len(trades) <= 1
    return trades[0] if trades else None


def link_candidate() -> signals.Routed:
    return candidate(
        symbol="LINKUSD",
        ref_price=10.30,
        stop=10.18,
        take=10.48,
        atr=0.1,
    )


def events(trader: Trader) -> list[str]:
    path = trader.store.directory / "journal.jsonl"
    return [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]


def enter(trader: Trader) -> None:
    asyncio.run(trader.try_enter(candidate(), brain.control_decision([candidate()])))


def test_live_handoff_preserves_the_risk_identity_quantity_and_exit_plan(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    (handoff,) = venue.market_bodies
    trade = only(trader)
    assert trade["execution_provider"] == "execution_portfolio"
    assert trade["risk_trade_id"] == trade["trade_id"] == handoff["trade_id"]
    assert trade["risk_decision_id"] == handoff["decision_id"]
    assert trade["quantity"] == str(handoff["quantity"]) == "0.2"
    assert trade["order_id"] == trade["id"]
    assert (trade["stop"], trade["take"]) == (str(handoff["stop"]), str(handoff["take"]))
    assert trade["intent_id"] == handoff["candidate"]


def test_venue_exit_is_accounted_from_balance_and_leftovers_cancelled(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    venue.venue_exit("0.31")
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert only(trader) is None
    assert not venue.close_bodies
    closed = [
        json.loads(line)
        for line in (trader.store.directory / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if '"trade_closed"' in line
    ][0]
    assert closed["reason"] == "VENUE_EXIT" and Decimal(closed["pnl_usd"]) == Decimal("0.31")


def test_timeout_closes_position_with_reduce_only_order(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    asyncio.run(trader.manage(trade, trade["deadline_ms"] + 1))
    assert len(venue.close_bodies) == 1 and venue.close_bodies[0]["quantity"] == "0.2"
    assert only(trader) is None and "trade_closed" in events(trader)


def test_software_stop_waits_one_poll_for_the_venue_stop(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    venue.bid, venue.ask = Decimal("81.40"), Decimal("81.42")
    trade = only(trader)
    asyncio.run(trader.manage(trade, now_ms()))
    assert not venue.close_bodies
    asyncio.run(trader.manage(trade, now_ms()))
    assert len(venue.close_bodies) == 1


def test_foreign_position_blocks_trading(tmp_path):
    venue = FakeVenue()
    venue.position_list = [Position("ETHUSD", "BUY", Decimal("0.01"), Decimal("2000"))]
    trader = live_trader(tmp_path, venue)
    with pytest.raises(NeedsOperator):
        asyncio.run(trader.start())
    assert not trader.trades() and not venue.market_bodies and not venue.close_bodies


def test_an_old_pending_entry_cannot_adopt_or_close_a_position(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    trader.store.state["pending_entry"] = {
        "order_id": "00474:" + "a" * 26,
        "symbol": "SOLUSD",
        "side": "LONG",
        "stop": "81.5",
        "take": "83.62",
        "candidate": candidate().record(),
    }
    trader.store.save()
    venue.position_list = [Position("SOLUSD", "BUY", Decimal("0.2"), Decimal("82.35"), False)]
    restarted = live_trader(tmp_path, venue)
    with pytest.raises(NeedsOperator, match="pending"):
        asyncio.run(restarted.start())
    assert restarted.store.state["pending_entry"] and not restarted.trades()
    assert not venue.close_bodies and not venue.market_bodies


def test_two_flat_snapshots_archive_a_legacy_trade_without_guessing_its_pnl(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue, poll_seconds=2)
    trader.store.state["open_trade"] = {"id": "old", "symbol": "SOLUSD", "candidate": {"id": "c"}}
    asyncio.run(trader.start())
    assert not trader.trades()
    (archived,) = [r for r in journal_of(trader) if r["event"] == "legacy_position_flat_unattributed"]
    assert archived["pnl_usd"] is None and archived["exit_origin"] == "unknown"
    assert not venue.market_bodies and not venue.close_bodies


def test_new_live_session_preserves_the_budget_halt_and_open_records(tmp_path, capsys):
    trader = live_trader(tmp_path, FakeVenue())
    trader.broker.baseline = Decimal("104.20")
    asyncio.run(trader.start())
    before = dict(trader.store.state["session"])
    assert before["start_equity"] != "104.20"
    assert "Начальный баланс пилота 104.20 USDT" in capsys.readouterr().out
    trader.store.state["halted"] = "earlier loss"
    trader.store.state["round_trips"] = 2
    asyncio.run(trader.start())
    assert "Начальный баланс пилота 104.20 USDT" in capsys.readouterr().out
    asyncio.run(trader.start(new_session=True))
    assert "Начальный баланс пилота 104.20 USDT" in capsys.readouterr().out
    starts = [row for row in journal_of(trader) if row["event"] == "session_start"]
    assert len(starts) == 2
    assert all(
        row["risk_start_equity"] == "104.20" and row["equity"] == before["start_equity"] for row in starts
    )
    assert trader.store.state["session"]["start_equity"] == before["start_equity"]
    assert trader.store.state["session"]["started_at"] == before["started_at"]
    assert trader.store.state["round_trips"] == 2 and trader.store.state["halted"] == "earlier loss"
    assert trader.store.state["session"]["reporting_started_at"]


def test_entry_deadline_stops_new_risk_and_keeps_managing_open_positions(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    trader.broker.entry_deadline_utc = datetime.now(UTC) - timedelta(seconds=1)
    asyncio.run(trader.try_enter(link_candidate(), brain.control_decision([link_candidate()])))
    assert len(venue.market_bodies) == 1 and trader.entries_blocked() == "signed entry window ended"
    trade["deadline_ms"] = 0
    asyncio.run(trader.step())
    assert len(venue.close_bodies) == 1 and not trader.trades()


@pytest.mark.parametrize("evidence", ["verified", "offline", "open", "unknown"])
def test_expired_pilot_completes_only_with_a_fresh_flat_observation(tmp_path, monkeypatch, evidence):
    trader = live_trader(tmp_path, FakeVenue())
    asyncio.run(trader.start())
    cutoff = datetime.now(UTC) - timedelta(minutes=2)
    trader.broker.entry_deadline_utc = cutoff
    trader._pilot_journal_path = tmp_path / "pilot.db"
    refreshed = []

    async def refresh(*, force=False):
        refreshed.append(force)

    def observation(path):
        assert path == trader._pilot_journal_path
        return {
            "observed_at": (cutoff if evidence == "offline" else datetime.now(UTC)).isoformat(),
            "final_flat_observation": {"observed_at": cutoff.isoformat()} if evidence != "open" else None,
            "open_positions": int(evidence == "open"),
            "unresolved_effects": int(evidence == "unknown"),
        }

    monkeypatch.setattr(trader.broker, "refresh_performance", refresh)
    monkeypatch.setattr("kairos_autotrader.engine.read_pilot_status", observation)
    assert asyncio.run(trader._can_stop_flat()) == (evidence == "verified")
    assert refreshed == [True]
    assert not trader.broker.venue.market_bodies and not trader.broker.venue.close_bodies


def test_session_loss_limit_halts_entries(tmp_path):
    venue = FakeVenue()
    venue.close_pnl = Decimal("-5.10")
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    asyncio.run(trader.manage(trade, trade["deadline_ms"] + 1))
    assert trader.entries_blocked() == "portfolio loss stop or unavailable performance baseline"


def test_live_review_only_sees_verified_system_outcomes(tmp_path):
    trader = live_trader(tmp_path, FakeVenue())
    owned = {
        "mode": "live",
        "pnl_basis": "execution_fills",
        "exit_origin": "system",
        "risk_trade_id": "id",
        "risk_decision_id": "decision",
        "result_r": 0.3,
    }
    trader.store.state["history"] = [
        {"result_r": 20},
        {**owned, "mode": "paper"},
        {**owned, "pnl_basis": "account_change"},
        {**owned, "exit_origin": "operator"},
        {**owned, "risk_trade_id": None},
        owned,
    ]
    assert trader._recent_outcomes(20) == [owned]


@pytest.mark.parametrize("unknown_effect", [False, True])
def test_failure_to_admit_a_new_trade_keeps_existing_trades_managed(tmp_path, unknown_effect):
    from kairos_execution.one_canary_account import CanaryAccountError

    venue = FakeVenue()
    two_books(venue)
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    if unknown_effect:

        async def fail(*args, **kwargs):
            raise Unknown("order acknowledgement was lost")

        trader.broker.enter = fail
    else:

        async def unavailable(own):
            raise CanaryAccountError("cash-flow observation unavailable")

        trader.broker.prepare_admission = unavailable
    link = link_candidate()
    asyncio.run(trader.try_enter(link, brain.control_decision([link])))
    assert len(trader.trades()) == 1 and bool(trader.store.state.get("halted")) == unknown_effect
    only(trader)["deadline_ms"] = 0
    trader.store.state["last_bar"] = trader.scan_slot(now_ms())
    asyncio.run(trader.step())
    assert len(venue.close_bodies) == 1 and not trader.trades()


def test_old_estimates_and_cash_withdrawal_do_not_trigger_a_portfolio_loss(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    trader.store.state["session"] = {"start_equity": "170", "realized_usd": "-7"}
    trader.store.state["day"] = {
        "date": datetime.now(UTC).date().isoformat(),
        "start_equity": "170",
        "peak_equity": "170",
        "realized_usd": "-7",
        "trades": 0,
    }
    asyncio.run(trader.start())
    enter(trader)
    trader.broker.flow = Decimal("-6")
    venue.venue_exit("-6.1")
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert trader.entries_blocked() is None
    evidence = asyncio.run(trader._account_evidence())
    assert evidence["daily_realized_pnl_usd"] == pytest.approx(-0.10)
    assert evidence["day_start_equity_usd"] == pytest.approx(103.42)


def test_daily_trade_limit_and_stop_file(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue, max_trades_per_day=1)
    asyncio.run(trader.start())
    assert trader.entries_blocked() is None
    enter(trader)
    venue.venue_exit("0.1")
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert trader.entries_blocked() == "daily trade limit reached"
    other = live_trader(tmp_path / "b", FakeVenue())
    asyncio.run(other.start())
    (other.store.directory / "STOP").write_text("")
    assert other.entries_blocked() == "stop file present"


def test_entry_skipped_when_price_ran_away(tmp_path):
    venue = FakeVenue()
    venue.bid, venue.ask = Decimal("83.00"), Decimal("83.02")
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert not venue.market_bodies and "entry_skipped" in events(trader)


def test_entry_skipped_when_the_barriers_are_off_the_venue_tick(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    off_tick = candidate(take=83.625)
    asyncio.run(trader.try_enter(off_tick, brain.control_decision([off_tick])))
    skipped = journal_of(trader)[-1]
    assert not venue.market_bodies and skipped["event"] == "entry_skipped"
    assert skipped["reason"] == "the exit barriers are not on the instrument's price tick"


def test_entry_skipped_after_the_intent_entry_window(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    old = candidate(signal_time=now_ms() // QUARTER * QUARTER - 2 * QUARTER)
    asyncio.run(trader.try_enter(old, brain.control_decision([old])))
    skipped = journal_of(trader)[-1]
    assert not venue.market_bodies and skipped["event"] == "entry_skipped"
    assert skipped["reason"] == "the candidate's entry window has expired" and skipped["candidate"] == old.id


def paper_trader(tmp_path, venue: FakeVenue, cfg: Config) -> Trader:
    cfg.validate()
    store = Store(tmp_path / "paper")
    return Trader(
        api=venue,
        session=None,
        store=store,
        config=cfg,
        broker=PaperBroker(store, cfg, FEE),
        fee=FEE,
        instruments={"SOLUSD": SOL, "LINKUSD": LINK},
        openai_key=None,
    )


def test_paper_cycle_updates_equity_with_fees(tmp_path):
    venue = FakeVenue()
    cfg = Config(use_llm=False, cooldown_minutes=0)
    trader = paper_trader(tmp_path, venue, cfg)
    store = trader.store
    asyncio.run(trader.start())
    enter(trader)
    assert not venue.market_bodies, "paper mode must not send orders"
    venue.bid, venue.ask = Decimal("83.70"), Decimal("83.72")
    asyncio.run(trader.manage(only(store), now_ms()))
    equity = Decimal(store.state["paper_equity"])
    assert only(store) is None and Decimal("100.2") < equity < Decimal("100.26")


def test_runtime_directory_is_bound_to_one_mode(tmp_path):
    venue = FakeVenue()
    asyncio.run(live_trader(tmp_path, venue).start())
    cfg = Config(use_llm=False)
    store = Store(tmp_path / "rt")
    paper = Trader(
        api=venue,
        session=None,
        store=store,
        config=cfg,
        broker=PaperBroker(store, cfg, FEE),
        fee=FEE,
        instruments={"SOLUSD": SOL},
        openai_key=None,
    )
    with pytest.raises(ConfigError):
        asyncio.run(paper.start())


# ---- several positions at once ------------------------------------------------


def two_books(venue: FakeVenue) -> None:
    venue.books = {"LINKUSD": (Decimal("10.299"), Decimal("10.300"))}


def test_one_scan_can_open_two_positions_but_never_two_in_one_instrument(tmp_path):
    venue = FakeVenue()
    two_books(venue)
    trader = paper_trader(tmp_path, venue, Config(use_llm=False, max_open_positions=2))
    asyncio.run(trader.start())
    offered = [candidate(), candidate(score=0.9), link_candidate()]

    async def scan(now):
        return [], offered

    trader.scan = scan
    asyncio.run(trader.step())
    trades = trader.store.state["open_trades"]
    assert sorted(t["symbol"] for t in trades) == ["LINKUSD", "SOLUSD"]
    assert all(t["solo"] is False for t in trades)
    for trade in trades:
        notional = Decimal(trade["quantity"]) * Decimal(trade["entry_price"])
        assert notional <= Decimal(50), "each position gets an equal share of the capital at most"
    asyncio.run(trader.try_enter(candidate(score=0.2), brain.control_decision(offered)))
    assert len(trader.store.state["open_trades"]) == 2 and "entry_skipped" in events(trader)


def test_signed_portfolio_can_accept_two_distinct_positions_on_a_fresh_state(tmp_path):
    venue = FakeVenue()
    two_books(venue)
    trader = live_trader(tmp_path, venue, max_open_positions=3)
    asyncio.run(trader.start())
    assert trader.slot_limit() == 3
    enter(trader)
    asyncio.run(trader.try_enter(link_candidate(), brain.control_decision([link_candidate()])))
    assert len(venue.market_bodies) == len(trader.trades()) == 2


def test_overlapping_live_trades_are_accounted_from_prices(tmp_path):
    venue = FakeVenue()
    two_books(venue)
    trader = live_trader(tmp_path, venue, max_open_positions=2)
    asyncio.run(trader.start())
    trader.store.state["round_trips"] = 1
    enter(trader)
    asyncio.run(trader.try_enter(link_candidate(), brain.control_decision([link_candidate()])))
    assert len(venue.market_bodies) == 2 and len(trader.store.state["open_trades"]) == 2
    # SOL reaches its take on the venue while LINK stays open; the balance moved by both.
    venue.bid, venue.ask = Decimal("83.60"), Decimal("83.62")
    venue.venue_exit("0.90")
    sol = next(t for t in trader.store.state["open_trades"] if t["symbol"] == "SOLUSD")
    asyncio.run(trader.manage(sol, now_ms()))
    closed = [
        json.loads(line)
        for line in (trader.store.directory / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if '"trade_closed"' in line
    ][0]
    assert closed["pnl_exact"] is False and Decimal(closed["exit_price"]) == venue.bid
    assert closed["pnl_basis"] == "price_estimate" and closed["exit_origin"] == "unknown"
    assert Decimal("0.20") < Decimal(closed["pnl_usd"]) < Decimal("0.26")
    assert [t["symbol"] for t in trader.store.state["open_trades"]] == ["LINKUSD"]
    assert "last_flat_equity" not in trader.store.state["session"]


def test_wide_spread_and_instrument_cooldown_block_an_entry(tmp_path):
    venue = FakeVenue()
    venue.bid, venue.ask = Decimal("82.25"), Decimal("82.35")
    trader = paper_trader(tmp_path, venue, Config(use_llm=False))
    asyncio.run(trader.start())
    enter(trader)
    assert only(trader) is None and "entry_skipped" in events(trader)

    venue.bid = Decimal("82.33")
    trader.store.state["cooldowns"] = {"SOLUSD": now_ms() + 600_000}

    async def scan(now):
        return [], [candidate()]

    trader.scan = scan
    asyncio.run(trader.step())
    assert only(trader) is None, "an instrument in cooldown is not offered"
    trader.store.state["cooldowns"] = {}
    trader.store.state["last_bar"] = None
    asyncio.run(trader.step())
    assert only(trader)["symbol"] == "SOLUSD"


def test_state_from_the_single_position_version_is_migrated(tmp_path):
    venue = FakeVenue()
    trader = paper_trader(tmp_path, venue, Config(use_llm=False))
    trader.store.state["open_trade"] = {"id": "x", "symbol": "SOLUSD"}
    assert trader.trades() == [{"id": "x", "symbol": "SOLUSD"}] and "open_trade" not in trader.store.state


def test_one_failing_position_does_not_stop_the_others_being_managed(tmp_path):
    venue = FakeVenue()
    two_books(venue)
    trader = paper_trader(tmp_path, venue, Config(use_llm=False, max_open_positions=2))
    asyncio.run(trader.start())
    enter(trader)
    asyncio.run(trader.try_enter(link_candidate(), brain.control_decision([link_candidate()])))
    venue.books["LINKUSD"] = (Decimal("10.49"), Decimal("10.491"))
    real_book = venue.book

    async def book(symbol):
        if symbol == "SOLUSD":
            raise Unknown("timeout")
        return await real_book(symbol)

    venue.book = book
    with pytest.raises(Unknown):
        asyncio.run(trader.step())
    assert [t["symbol"] for t in trader.store.state["open_trades"]] == ["SOLUSD"], (
        "LINK still took its profit"
    )


def test_runner_passes_adaptive_leverage_to_the_single_portfolio_owner(tmp_path, monkeypatch):
    from support import approve_entries

    approve_entries(monkeypatch, leverage=5.0)
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    assert venue.market_bodies[0]["leverage"] == trade["leverage"] == 5
    asyncio.run(trader.manage(trade, trade["deadline_ms"] + 1))
    assert venue.close_bodies[-1]["leverage"] == 5


def test_one_empty_position_answer_does_not_end_a_trade_or_cancel_its_protection(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue, exit_confirm_polls=3)
    asyncio.run(trader.start())
    enter(trader)
    held = list(venue.position_list)
    venue.position_list = []  # the venue answers with nothing for a moment
    asyncio.run(trader.manage(only(trader), now_ms()))
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert only(trader) is not None and not venue.cancelled, "two empty answers are not enough"
    venue.position_list = held
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert only(trader)["missing_polls"] == 0
    venue.venue_exit("0.31")
    for _ in range(3):
        asyncio.run(trader.manage(only(trader), now_ms())) if only(trader) else None
    assert only(trader) is None and "trade_closed" in events(trader)


# ---- model call glue and secrets ---------------------------------------------


class FakeResponse:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def text(self):
        return json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, status, payload):
        self.status, self.payload, self.calls = status, payload, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.status, self.payload)


def model_response(obj):
    return {
        "status": "completed",
        "usage": {"input_tokens": 900, "output_tokens": 120},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(obj)}]}],
    }


def ask(session):
    payload = brain.build_payload([], [candidate()], trades_today=0)
    return asyncio.run(
        brain.ask_llm(
            session,
            api_key="sk-test",
            model="m",
            effort="low",
            payload=payload,
            candidates=[candidate()],
            min_confidence=0.55,
            timeout_s=5,
        )
    )


def test_model_call_builds_structured_request_and_parses_reply():
    session = FakeSession(200, model_response(reply()))
    decision = ask(session)
    url, kwargs = session.calls[0]
    assert url == brain.OPENAI_URL and kwargs["json"]["store"] is False
    assert (
        kwargs["json"]["text"]["format"]["type"] == "json_schema"
        and kwargs["json"]["text"]["format"]["strict"]
    )
    assert decision.action == "TRADE" and decision.usage == {"input_tokens": 900, "output_tokens": 120}


def test_model_failure_or_garbage_never_trades():
    assert ask(FakeSession(500, {"error": "boom"})).action == "WAIT"
    broken = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "not json"}]}],
    }
    assert ask(FakeSession(200, broken)).action == "WAIT"


def test_secrets_are_read_by_line_and_validated(tmp_path):
    from kairos_autotrader.__main__ import SecretError, find_openai_line, secret_from

    keys = tmp_path / "k.txt"
    keys.write_text(
        "\n".join(
            ["Wallet", "private key: 0x" + "ab" * 32, "OpenAI key = sk-test-123", "bad line: two words"]
        )
    )
    assert secret_from(keys, 2, None, "wallet") == "0x" + "ab" * 32
    assert find_openai_line(keys) == 3 and secret_from(keys, 3, None, "openai") == "sk-test-123"
    with pytest.raises(SecretError):
        secret_from(keys, 4, None, "x")
    with pytest.raises(SecretError):
        secret_from(keys, 9, None, "x")
    assert secret_from(None, None, None, "x") is None


# ---- scanning four times an hour ------------------------------------------------


HISTORY = system_engine.HOURLY_TREND_HISTORY_MS
SLOT = 1_794_400_000_000 // HOUR * HOUR + QUARTER  # a boundary at :15


def scanning_trader(tmp_path, feed: BarFeed, **config) -> Trader:
    """A paper trader whose three sibling processes are the stand-ins of ``support``."""
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, symbols=["SOLUSD"], **config))
    trader.engine = system_engine.EngineConfig()
    trader._fetch_bars = feed.fetch
    return trader


def journal_of(trader) -> list[dict]:
    return [
        json.loads(line)
        for line in (trader.store.directory / "journal.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_scan_runs_binance_bars_through_the_engine_and_the_router(tmp_path, monkeypatch):
    fake_parser(monkeypatch)
    engine_requests = fake_engine(monkeypatch)
    router_requests = fake_router(monkeypatch)
    feed = BarFeed(last_open_ms=SLOT + 2 * MINUTE)  # two minutes of the next quarter already exist
    trader = scanning_trader(tmp_path, feed, reward_risk=2.0, timeout_minutes=240, review_timeout_s=45.0)
    features, found = asyncio.run(trader.scan(SLOT + 130_000))

    assert {symbol for symbol, _, _ in feed.asked} == {"SOLUSDT"}, "the signal symbol is the Binance one"
    assert [f.symbol for f in features] == ["SOLUSD"], "features are labelled with the EVEDEX instrument"
    request = engine_requests[0]
    assert request["op"] == "hourly_trend_bars"
    assert request["config"] == {"target_reward_to_risk": 2.0, "max_hold_minutes": 240}
    assert request["price_increments"] == {"SOLUSDT": str(trader.instruments["SOLUSD"].price_step)}, (
        "the rules place stop and target on the EVEDEX tick"
    )
    sent = request["series"]["SOLUSDT"]
    assert sent[-1][0] == SLOT - MINUTE, "minutes after the boundary are not part of the decision"
    assert sent[0][0] == SLOT - HISTORY and len(sent) == HISTORY // MINUTE, "exactly the rules' window"
    assert len(sent[0]) == 10 and len(sent[0][9]) == 64, "every bar carries its recorded identity"

    (c,) = found
    assert (c.instrument, c.intent.symbol, c.intent.venue) == ("SOLUSD", "SOLUSDT", "BINANCE_UM")
    assert c.intent.decision_ts_ms == SLOT - 1 and c.signal_time == SLOT
    assert (c.setup, c.reference_price, c.stop, c.take, c.timeout_minutes) == (
        "PULLBACK_RECLAIM",
        80.0,
        79.2,
        81.6,
        240,
    )
    assert c.atr == pytest.approx(0.8) and c.strength == 0.41237
    assert c.intent.provenance.input_bar_sha256s == tuple(row[9] for row in sent)
    assert c.route.review_tier.value == "NORMAL" and c.route.source == routing.ROUTE_SOURCE
    assert c.route.review_deadline_ms == SLOT - 1 + 45_000, "the configured review timeout is in the route"
    assert features[0].trend_ema20 == 79.0 and features[0].dist_ema20_atr == 0.3

    routed_request = router_requests[0]
    assert routed_request["symbols"] == ["SOLUSDT"] and routed_request["review_timeout_ms"] == 45_000
    assert routed_request["intents"][0]["intent_id"] == c.id and routed_request["signals"] == []

    scan = journal_of(trader)[-1]
    assert scan["event"] == "scan" and scan["bar"] == SLOT - HOUR, "the bar under review runs from :15 to :15"
    assert (scan["source"], scan["signal_venue"]) == ("strategy_engine", "BINANCE_UM")
    assert scan["engine"]["strategy_code_sha256"] == system_engine.HOURLY_TREND_CODE_SHA256
    assert scan["features"][0]["change_1h_pct"] == 0.123 and scan["features"][0]["change_7d_pct"] == 3.14
    assert scan["features"][0]["position_in_7d_range"] is None
    (recorded,) = scan["candidates"]
    assert recorded["id"] == c.id and recorded["route_id"] == c.route.route_id
    assert recorded["input_bars"] == HISTORY // MINUTE and scan["rejected"] == [] and scan["skipped"] == []
    assert (trader.store.directory / "closed-bars" / "SOLUSDT.1m.json").exists()

    # The next scan downloads only the newest minutes and five cached ones again.
    feed.asked.clear()
    feed.last_open_ms = SLOT + QUARTER
    asyncio.run(trader.scan(SLOT + QUARTER + 12_000))
    assert feed.asked == [("SOLUSDT", SLOT - 3 * MINUTE, SLOT + QUARTER)]


def test_scan_skips_an_instrument_without_complete_fresh_bars(tmp_path, monkeypatch):
    fake_parser(monkeypatch)
    fake_engine(monkeypatch)
    fake_router(monkeypatch)
    late = scanning_trader(tmp_path / "late", BarFeed(last_open_ms=SLOT - 2 * MINUTE))
    assert asyncio.run(late.scan(SLOT + 12_000)) == ([], []), "the last minute of the bar is not there yet"
    assert "page is incomplete" in journal_of(late)[-1]["skipped"][0]

    holed = scanning_trader(tmp_path / "hole", BarFeed(last_open_ms=SLOT - MINUTE, drop=SLOT - 5 * MINUTE))
    assert asyncio.run(holed.scan(SLOT + 12_000)) == ([], []), "a gap is never bridged with invented bars"
    assert journal_of(holed)[-1]["skipped"][0].startswith("SOLUSD: ")

    hourly = scanning_trader(tmp_path / "hourly", BarFeed(last_open_ms=SLOT + 5 * MINUTE), scan_minutes=60)
    asyncio.run(hourly.scan(SLOT + 12_000))
    assert journal_of(hourly)[-1]["bar"] == SLOT - QUARTER - HOUR, (
        "with hourly scans the bar closes on the hour"
    )


def test_a_revised_newest_bar_is_journaled_and_the_scan_goes_on(tmp_path, monkeypatch):
    fake_parser(monkeypatch)
    fake_engine(monkeypatch)
    fake_router(monkeypatch)
    feed = BarFeed(last_open_ms=SLOT - MINUTE)
    trader = scanning_trader(tmp_path, feed)
    assert len(asyncio.run(trader.scan(SLOT + 12_000))[1]) == 1
    feed.last_open_ms = SLOT + QUARTER - MINUTE
    feed.revised = {SLOT - 2 * MINUTE}  # the venue settled a minute after it was cached
    feed.asked.clear()
    features, found = asyncio.run(trader.scan(SLOT + QUARTER + 12_000))
    assert len(features) == 1 and len(found) == 1, "the instrument is not lost for this scan"
    revised = [e for e in journal_of(trader) if e["event"] == "bars_revised"]
    assert [e["minutes"] for e in revised] == [{"SOLUSDT": [SLOT - 2 * MINUTE]}]
    assert all(start >= SLOT - 5 * MINUTE for _, start, _ in feed.asked), "the older bars are kept"


def test_scan_trades_nothing_when_the_engine_is_missing_fails_or_is_not_the_pinned_source(
    tmp_path, monkeypatch
):
    fake_parser(monkeypatch)
    fake_router(monkeypatch)
    feed = BarFeed(last_open_ms=SLOT - MINUTE)
    fake_engine(monkeypatch, code_sha256="0" * 64)
    other = scanning_trader(tmp_path / "other", feed)
    assert asyncio.run(other.scan(SLOT + 12_000)) == ([], [])
    error = journal_of(other)[-1]
    assert error["event"] == "scan_error" and "differs from the pinned one" in error["message"]

    def broken(config, request):
        raise system_engine.EngineError("Strategy Engine: worker failed")

    monkeypatch.setattr(system_engine, "_run_hourly_trend", broken)
    failing = scanning_trader(tmp_path / "failing", feed)
    assert asyncio.run(failing.scan(SLOT + 12_000)) == ([], [])
    assert journal_of(failing)[-1]["event"] == "scan_error"

    absent = scanning_trader(tmp_path / "absent", feed)
    absent.engine = None
    assert asyncio.run(absent.scan(SLOT + 12_000)) == ([], [])
    assert journal_of(absent)[-1]["message"] == "the signal chain is not available"


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda f: {**f, "bar_hashes": f["bar_hashes"][1:]}, "exactly the cached closed bars"),
        (lambda f: {**f, "decision_ts_ms": f["decision_ts_ms"] - QUARTER}, "another decision"),
        (lambda f: {**f, "symbol": "ETHUSDT"}, "another strategy or symbol"),
        (lambda f: {**f, "strategy_id": "other_rules_v1"}, "another strategy or symbol"),
        (lambda f: {**f, "config_sha256": "9" * 64}, "pinned engine identity"),
    ],
)
def test_an_intent_that_does_not_match_the_cache_or_the_pin_is_not_a_candidate(
    tmp_path, monkeypatch, change, reason
):
    fake_parser(monkeypatch)
    fake_router(monkeypatch)
    fake_engine(monkeypatch, tamper=change)
    trader = scanning_trader(tmp_path, BarFeed(last_open_ms=SLOT - MINUTE))
    assert asyncio.run(trader.scan(SLOT + 12_000)) == ([], [])
    scan = journal_of(trader)[-1]
    assert scan["candidates"] == [] and reason in scan["skipped"][0]


def test_a_candidate_the_router_rejects_or_cannot_route_is_never_offered(tmp_path, monkeypatch):
    fake_parser(monkeypatch)
    fake_engine(monkeypatch)
    feed = BarFeed(last_open_ms=SLOT - MINUTE)

    fake_router(monkeypatch, reject="stale candidate")
    stale = scanning_trader(tmp_path / "stale", feed)
    features, found = asyncio.run(stale.scan(SLOT + 12_000))
    assert len(features) == 1 and found == []
    scan = journal_of(stale)[-1]
    assert scan["candidates"] == [] and scan["rejected"][0]["reason"] == "stale candidate"
    assert scan["rejected"][0]["symbol"] == "SOLUSD"

    def other_intent(intent):
        return strict_intent(
            symbol="SOLUSDT", decision_ts_ms=intent.decision_ts_ms, side="SHORT", stop=83.0, take=81.0
        )

    fake_router(monkeypatch, swap=other_intent)
    swapped = scanning_trader(tmp_path / "swapped", feed)
    features, found = asyncio.run(swapped.scan(SLOT + 12_000))
    assert len(features) == 1 and found == [], "a route around another intent is refused"
    assert "another intent" in journal_of(swapped)[-1]["message"]

    def down(config, request):
        raise routing.RouterError("Router: worker failed")

    monkeypatch.setattr(routing, "_run_worker", down)
    silent = scanning_trader(tmp_path / "silent", feed)
    assert asyncio.run(silent.scan(SLOT + 12_000))[1] == []
    assert journal_of(silent)[-1]["event"] == "scan_error"


def test_conflicting_text_evidence_reaches_the_route(tmp_path, monkeypatch):
    fake_parser(monkeypatch)
    fake_engine(monkeypatch)
    requests = fake_router(monkeypatch, tier="CONFLICT")
    trader = scanning_trader(tmp_path, BarFeed(last_open_ms=SLOT - MINUTE))

    class Desk:
        def raw_signals(self):
            return [{"id": "news-1", "topic": "Probe", "event_time": "2026-11-12T00:00:00+00:00"}]

    trader.news = Desk()
    _, (c,) = asyncio.run(trader.scan(SLOT + 12_000))
    assert requests[0]["signals"][0]["id"] == "news-1"
    assert c.route.review_tier.value == "CONFLICT" and c.route.evidence_ids == ("news-1",)
    assert c.record()["text_evidence_ids"] == ["news-1"]


def test_one_scan_per_quarter_hour_slot(tmp_path):
    venue = FakeVenue()
    trader = paper_trader(tmp_path, venue, Config(use_llm=False))
    asyncio.run(trader.start())
    scans = []

    async def scan(now):
        scans.append(now)
        return [], []

    trader.scan = scan
    now = now_ms()
    assert trader.scan_slot(now) % (15 * 60_000) == 0 and now - trader.scan_slot(now) < 15 * 60_000
    trader.store.state["last_bar"] = trader.scan_slot(now) - 15 * 60_000  # the previous slot was scanned
    asyncio.run(trader.step())
    asyncio.run(trader.step())
    waiting = now - trader.scan_slot(now) < SCAN_AFTER_CLOSE_MS  # the first seconds are left for the candle
    assert len(scans) == (0 if waiting else 1), "a slot is scanned once"
    with pytest.raises(ConfigError):
        Config(scan_minutes=5).validate()


def test_every_loop_replaces_the_heartbeat_for_the_alert_monitor(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.run(max_steps=2))
    beat = json.loads((trader.store.directory / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["pid"] == os.getpid() and set(beat) == {"pid", "started_at", "at"}
    assert datetime.fromisoformat(beat["at"]) >= datetime.fromisoformat(beat["started_at"])


def test_scan_trades_only_the_selected_strategies_and_runs_the_daily_rule_at_one_utc(tmp_path, monkeypatch):
    from support import strict_intent

    from kairos_autotrader import engine as engine_module

    fake_parser(monkeypatch)
    fake_engine(monkeypatch)
    router_requests = fake_router(monkeypatch)
    day = 24 * HOUR
    slot = (SLOT // day + 1) * day + HOUR  # 01:00 UTC
    daily_calls = []

    async def daily(config, series, boundary_ms, **kwargs):
        daily_calls.append(boundary_ms)
        rows = series["SOLUSD"]
        intent = strict_intent(
            decision_ts_ms=boundary_ms - 1,
            side="SHORT",
            setup=system_engine.RIGHT_TAIL_SETUP,
            ref_price=80.0,
            stop=83.2,
            take=67.2,
            timeout_minutes=72 * 60,
            bar_hashes=tuple(r[9] for r in rows if r[0] < boundary_ms)[-3:],
            code_sha256=system_engine.RIGHT_TAIL_CODE_SHA256,
            strategy_id=system_engine.RIGHT_TAIL_ID,
        )
        return {"SOLUSD": intent}, [], {"strategy_id": system_engine.RIGHT_TAIL_ID}

    monkeypatch.setattr(engine_module, "right_tail_scan", daily)
    feed = BarFeed(last_open_ms=slot + 2 * MINUTE)
    trader = scanning_trader(tmp_path, feed, strategies=[system_engine.RIGHT_TAIL_ID])
    _, found = asyncio.run(trader.scan(slot + 30_000))

    assert daily_calls == [slot]
    (c,) = found
    assert c.intent.strategy_id == system_engine.RIGHT_TAIL_ID and c.timeout_minutes == 72 * 60
    assert [i["strategy_id"] for i in router_requests[0]["intents"]] == [system_engine.RIGHT_TAIL_ID]
    scan = journal_of(trader)[-1]
    assert scan["not_traded"] == [
        {"symbol": "SOLUSD", "side": "LONG", "strategy_id": system_engine.HOURLY_TREND_ID}
    ], "the hourly candidate is measured, not traded"
    assert trader.history_ms() == system_engine.RIGHT_TAIL_HISTORY_MS
    assert trader.longest_hold_minutes() == 72 * 60

    # Any other quarter hour: the daily rule is not asked, and nothing is traded.
    router_requests.clear()
    feed.last_open_ms = slot + QUARTER + 2 * MINUTE
    assert asyncio.run(trader.scan(slot + QUARTER + 30_000))[1] == []
    assert daily_calls == [slot] and router_requests == []


def test_a_strategy_with_a_fixed_management_policy_gets_it(tmp_path):
    trader = paper_trader(
        tmp_path,
        FakeVenue(),
        Config(
            use_llm=False,
            symbols=["SOLUSD"],
            strategies=[system_engine.RIGHT_TAIL_ID],
            management_policies={
                system_engine.RIGHT_TAIL_ID: {"trail_start_r": 1.0, "trail_distance_r": 2.0}
            },
        ),
    )
    policy = trader.management_policy(SLOT, system_engine.RIGHT_TAIL_ID)
    assert policy is not None and (policy.trail_start_r, policy.trail_distance_r) == (1.0, 2.0)
    assert trader.management_policy(SLOT, system_engine.HOURLY_TREND_ID) is None, "no measurement yet"


def test_a_due_scan_runs_before_positions_are_managed(tmp_path, monkeypatch):
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, symbols=["SOLUSD"]))
    trader.store.state["open_trades"] = [{"id": "t1", "symbol": "XRPUSD"}]
    calls: list[str] = []

    async def manage(trade, now_ms):
        calls.append("manage")

    async def scan(now_ms):
        calls.append("scan")
        return [], []

    monkeypatch.setattr(trader, "manage", manage)
    monkeypatch.setattr(trader, "scan", scan)
    now = SLOT + 30_000
    monkeypatch.setattr(engine_module_time(), "time", lambda: now / 1000)
    asyncio.run(trader.step())
    assert calls == ["scan"], "the review window is short; exchange stops protect the position meanwhile"
    asyncio.run(trader.step())
    assert calls == ["scan", "manage"], "the next step manages the position"

    # With every slot taken there is no scan, so the position is managed at once.
    calls.clear()
    trader.store.state["last_bar"] = None
    trader.config.max_open_positions = 1
    asyncio.run(trader.step())
    assert calls == ["manage"]


def engine_module_time():
    from kairos_autotrader import engine as engine_module

    return engine_module.time


def test_run_ends_the_background_engine_work_and_stops_the_persistent_workers(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    stopped = []

    class Pool:
        def __init__(self, name):
            self.name = name

        async def stop(self):
            stopped.append(self.name)

    async def warming():
        await asyncio.sleep(3600)

    async def run():
        trader.right_tail_pool, trader.hourly_pool = Pool("right_tail"), Pool("hourly")
        trader._warm_task = asyncio.create_task(warming())
        await trader.run(max_steps=1)
        return trader._warm_task

    warm = asyncio.run(run())
    assert warm.cancelled() and sorted(stopped) == ["hourly", "right_tail"]


def test_the_loop_wakes_up_when_the_next_scan_is_due(tmp_path):
    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False, poll_seconds=5.0))
    slot_s = 1_795_008_600  # a quarter-hour close
    due_s = slot_s + SCAN_AFTER_CLOSE_MS / 1000
    assert trader.pause_s(slot_s - 60) == 5.0, "far from a close: the poll interval"
    assert trader.pause_s(due_s - 2) == pytest.approx(2.05), "near it: wake right when the scan is due"
    assert trader.pause_s(due_s + 0.01) == 5.0, "the next scan is a quarter-hour away"
    assert trader.pause_s(due_s - 0.01) == pytest.approx(0.06)


def test_engine_warm_up_reads_windows_the_cache_covers_and_journals_what_it_skipped(tmp_path, monkeypatch):
    from kairos_autotrader import engine as engine_module
    from kairos_autotrader.system_engine import EngineConfig, TrendScan

    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False))
    trader.engine, trader.right_tail_pool, trader.hourly_pool = EngineConfig(), object(), object()
    asked = {}

    async def right_tail(engine, series, boundary, **kwargs):
        asked["right_tail"] = boundary
        return {}, ["XRPUSD: closed bars do not reach back the daily rule's history"], {}

    async def hourly(engine, series, boundary, **kwargs):
        asked["hourly"] = boundary
        return TrendScan({}, {}, {}, {}, [])

    monkeypatch.setattr(engine_module, "right_tail_scan", right_tail)
    monkeypatch.setattr(engine_module, "hourly_trend_scan", hourly)
    before = now_ms()
    asyncio.run(trader._warm_pools({"SOLUSD": []}))
    hour = 3_600_000
    # The latest whole hour (the cache reaches back the rule's history from about now), not
    # the rule's last 4-hourly decision, whose window can start before the cached bars.
    assert asked["right_tail"] in {before // hour * hour, now_ms() // hour * hour}
    assert asked["hourly"] == trader.scan_slot(before) or asked["hourly"] == trader.scan_slot(now_ms())
    warm = {e["rule"]: e for e in journal_of(trader) if e["event"] == "engine_warm"}
    assert warm["right_tail"]["skipped"] == 1 and warm["hourly"]["skipped"] == 0
