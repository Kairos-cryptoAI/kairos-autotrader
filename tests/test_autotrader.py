"""Offline tests. The venue here is a fake: nothing in this file touches the network."""

from __future__ import annotations

import asyncio
import json
import re
import time
from decimal import Decimal
from pathlib import Path

import pytest

from kairos_autotrader import brain, candidates, system_engine
from kairos_autotrader.engine import (
    Config,
    ConfigError,
    LiveBroker,
    PaperBroker,
    Store,
    Trader,
    size_position,
    software_exit,
)
from kairos_autotrader.evedex import (
    Book,
    Candle,
    Instrument,
    OrderSigner,
    Position,
    Rejected,
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
        ("capital_basis_usd", 101),
        ("risk_per_trade_usd", 1.5),
        ("session_loss_stop_usd", 11),
        ("max_trades_per_day", 13),
        ("max_open_positions", 4),
    ],
)
def test_hard_ceilings_cannot_be_configured_away(field, value):
    config = Config()
    setattr(config, field, value)
    with pytest.raises(ConfigError):
        config.validate()


# ---- sizing and exits ---------------------------------------------------------


def test_sizing_keeps_stop_loss_within_risk():
    book = Book(now_ms(), Decimal("82.33"), Decimal("82.35"))
    quantity, why = size_position(
        side="LONG",
        book=book,
        stop=Decimal("81.50"),
        instrument=SOL,
        risk_usd=Decimal("0.25"),
        max_notional=Decimal(100),
        available=Decimal(109),
        fee=FEE,
        slippage=Decimal("0.0002"),
    )
    assert why == "ok" and quantity == Decimal("0.2")
    worst = quantity * (Decimal("82.35") - Decimal("81.50")) + quantity * Decimal("82.35") * 2 * (
        FEE + Decimal("0.0002")
    )
    assert worst <= Decimal("0.25")


def test_sizing_refuses_when_minimum_lot_is_too_risky():
    book = Book(now_ms(), Decimal("73340"), Decimal("73344"))
    quantity, why = size_position(
        side="LONG",
        book=book,
        stop=Decimal("72600"),
        instrument=BTC,
        risk_usd=Decimal("0.25"),
        max_notional=Decimal(100),
        available=Decimal(109),
        fee=FEE,
        slippage=Decimal("0.0002"),
    )
    assert quantity is None and "minimum lot" in why


def test_sizing_refuses_stop_on_wrong_side():
    book = Book(now_ms(), Decimal("82.33"), Decimal("82.35"))
    quantity, _ = size_position(
        side="SHORT",
        book=book,
        stop=Decimal("81.50"),
        instrument=SOL,
        risk_usd=Decimal("0.25"),
        max_notional=Decimal(100),
        available=Decimal(109),
        fee=FEE,
        slippage=Decimal("0.0002"),
    )
    assert quantity is None


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


def candidate(**over) -> candidates.Candidate:
    base = dict(
        id="SOLUSD-LONG-PULLBACK_RECLAIM-1",
        symbol="SOLUSD",
        side="LONG",
        setup="PULLBACK_RECLAIM",
        bar_time=0,
        signal_time=900_000,
        ref_price=82.35,
        stop=81.5,
        take=83.625,
        timeout_minutes=720,
        atr=0.8,
        score=0.4,
        rationale="test",
    )
    base.update(over)
    return candidates.Candidate(**base)


def test_outcome_counts_stop_first_and_charges_costs():
    c = candidate()
    both = [Candle(900_000, 82.3, 83.7, 81.4, 82.0, 1.0)]
    assert candidates.outcome(c, both, cost_fraction=0.0013)["reason"] == "STOP"
    win = [Candle(900_000, 82.3, 83.7, 82.0, 83.6, 1.0)]
    result = candidates.outcome(c, win, cost_fraction=0.0013)
    assert result["reason"] == "TAKE" and 1.3 < result["r"] < 1.5
    assert candidates.outcome(c, [Candle(0, 82.3, 99, 1, 82.0, 1.0)], cost_fraction=0.0)["r"] is None


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
    sol, link = candidate(), candidate(id="LINK-1", symbol="LINKUSD")
    sol_again = candidate(id="SOL-2")
    answer = reply(
        verdicts=[
            verdict(sol.id, confidence=0.6),
            verdict(link.id, confidence=0.8),
            verdict(sol_again.id, confidence=0.9),
        ]
    )
    both = brain.validate_reply(answer, [sol, link, sol_again], 0.55, max_picks=3)
    assert both.picks == ["SOL-2", "LINK-1"] and both.candidate_id == "SOL-2" and both.confidence == 0.9
    one = brain.validate_reply(answer, [sol, link, sol_again], 0.55, max_picks=1)
    assert one.picks == ["SOL-2"]
    picky = brain.validate_reply(answer, [sol, link, sol_again], 0.85, max_picks=3)
    assert picky.picks == ["SOL-2"]


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
    a, b = candidate(id="A", score=0.3), candidate(id="B", score=0.6)
    assert brain.control_decision([a, b]).candidate_id == "B"
    assert brain.control_decision([]).action == "WAIT"
    c = candidate(id="C", symbol="LINKUSD", score=0.5)
    assert brain.control_decision([a, b, c], 3).picks == ["B", "C"], "one pick per instrument"


# ---- engine against a fake venue ----------------------------------------------


def signature_error() -> Rejected:
    return Rejected(
        "refused", status=400, body=json.dumps({"label": "invalid_signature_error", "message": "x"})
    )


class FakeVenue:
    def __init__(self) -> None:
        self.funding = Decimal("109.42")
        self.bid, self.ask = Decimal("82.33"), Decimal("82.35")
        self.position_list: list[Position] = []
        self.orders_open: list[dict] = []
        self.tpsl: list[dict] = []
        self.market_bodies: list[dict] = []
        self.close_bodies: list[dict] = []
        self.tpsl_bodies: list[dict] = []
        self.cancelled: list[str] = []
        self.accept_chains = {"42161", "161803"}
        self.attach_stop = True
        self.unknown_entry = False
        self.reject_tpsl = False
        self.close_pnl = Decimal("-0.20")
        self.position_reads = 0
        self.account_leverage: dict[str, int] = {}
        self.leverage_calls: list[tuple[str, int]] = []
        self.leverage_locked = False
        self.close_chains = {"42161", "161803"}
        self.tpsl_reject_label = "forbidden_error"
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
        self.position_reads += 1
        return list(self.position_list)

    async def open_orders(self):
        return list(self.orders_open)

    async def order(self, order_id):
        return None

    async def active_tpsl(self, instrument=None):
        return [row for row in self.tpsl if instrument in (None, row["instrument"])]

    async def post_market_order(self, body):
        self.market_bodies.append(body)
        if body["chainId"] not in self.accept_chains:
            raise signature_error()
        symbol = body["instrument"]
        if self.account_leverage.get(symbol, 1) != body["leverage"]:
            raise Rejected(
                "refused",
                status=400,
                body=json.dumps(
                    {"label": "invalid_order_leverage_error", "error": "Invalid order leverage - 10 !== 1"}
                ),
            )
        ask = self.books.get(symbol, (self.bid, self.ask))[1]
        quantity = (Decimal(body["cashQuantity"]) / ask).quantize(Decimal("0.1"))
        self.position_list = [p for p in self.position_list if p.instrument != symbol]
        self.position_list.append(Position(symbol, body["side"], quantity, ask, self.attach_stop))
        if self.attach_stop:
            self.tpsl += [
                {"id": f"{symbol}-t{i}", "instrument": symbol, "type": item["type"], "status": "active"}
                for i, item in enumerate(body.get("tpsl", []))
            ]
        if self.unknown_entry:
            raise Unknown("timeout")
        return {"id": body["id"], "status": "FILLED"}

    async def post_close(self, body):
        self.close_bodies.append(body)
        if body["chainId"] not in self.close_chains:
            raise signature_error()
        self.position_list = [p for p in self.position_list if p.instrument != body["instrument"]]
        self.funding += self.close_pnl
        return {"id": body["id"]}

    async def post_tpsl(self, body):
        self.tpsl_bodies.append(body)
        if self.reject_tpsl:
            if self.tpsl_reject_label != "side" or body["side"] == "BUY":
                label = (
                    "stop_loss_price_above_last_price_error"
                    if self.tpsl_reject_label == "side"
                    else self.tpsl_reject_label
                )
                raise Rejected("refused", status=400, body=json.dumps({"label": label, "error": ""}))
        self.tpsl.append(
            {
                "id": f"s{len(self.tpsl)}",
                "instrument": body["instrument"],
                "type": body["type"],
                "status": "active",
            }
        )
        return {"id": "s", "status": "active"}

    async def set_leverage(self, instrument, leverage):
        self.leverage_calls.append((instrument, leverage))
        if self.leverage_locked:
            raise Rejected(
                "refused",
                status=400,
                body=json.dumps({"label": "insufficient_funds_to_reduce_leverage_error"}),
            )
        self.account_leverage[instrument] = leverage

    async def cancel_tpsl(self, instrument, tpsl_id):
        self.cancelled.append(tpsl_id)
        self.tpsl = [row for row in self.tpsl if row["id"] != tpsl_id]

    def venue_exit(self, pnl: str, symbol: str = "SOLUSD") -> None:
        self.position_list = [p for p in self.position_list if p.instrument != symbol]
        self.funding += Decimal(pnl)


def live_trader(tmp_path, venue: FakeVenue, **config) -> Trader:
    cfg = Config(**{"use_llm": False, "cooldown_minutes": 0, "exit_confirm_polls": 1, **config})
    cfg.validate()
    store = Store(tmp_path / "rt")
    broker = LiveBroker(venue, store, cfg, TEST_KEY, settle_seconds=0, cache_seconds=0)
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


def link_candidate() -> candidates.Candidate:
    return candidate(
        id="LINKUSD-LONG-PULLBACK_RECLAIM-1",
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


def test_live_entry_sends_protected_order_and_records_trade(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    body = venue.market_bodies[0]
    assert body["side"] == "BUY" and body["timeInForce"] == "IOC" and body["leverage"] == 1
    assert Decimal(body["cashQuantity"]) <= Decimal(100)
    assert [p["type"] for p in body["tpsl"]] == ["stop-loss", "take-profit"]
    assert body["tpsl"][0]["price"] == "81.5" and body["tpsl"][0]["quantity"] == "0"
    trade = only(trader)
    assert trade["symbol"] == "SOLUSD" and trade["quantity"] == "0.2"
    assert trader.store.state["chain_id"] == 42161 and "pending_entry" not in trader.store.state
    assert not venue.tpsl_bodies and not trader.store.state.get("halted")


def test_venue_exit_is_accounted_from_balance_and_leftovers_cancelled(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    venue.venue_exit("0.31")
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert only(trader) is None
    assert sorted(venue.cancelled) == ["SOLUSD-t0", "SOLUSD-t1"] and not venue.close_bodies
    closed = [
        json.loads(line)
        for line in (trader.store.directory / "journal.jsonl").read_text().splitlines()
        if '"trade_closed"' in line
    ][0]
    assert closed["reason"] == "VENUE_EXIT" and Decimal(closed["pnl_usd"]) == Decimal("0.31")


def test_second_signing_chain_is_tried_after_invalid_signature(tmp_path):
    venue = FakeVenue()
    venue.accept_chains = {"161803"}
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert [b["chainId"] for b in venue.market_bodies] == ["42161", "161803"]
    assert venue.market_bodies[0]["id"] != venue.market_bodies[1]["id"]
    assert trader.store.state["chain_id"] == 161803 and only(trader)


def test_rejection_on_every_chain_leaves_no_trade(tmp_path):
    venue = FakeVenue()
    venue.accept_chains = set()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert len(venue.market_bodies) == 2 and not only(trader)
    assert "pending_entry" not in trader.store.state and not venue.position_list


def test_unprotected_fill_is_closed_and_entries_halt(tmp_path):
    venue = FakeVenue()
    venue.attach_stop, venue.reject_tpsl = False, True
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert len(venue.close_bodies) == 1 and not venue.position_list
    assert only(trader) is None and trader.store.state["halted"]
    assert trader.entries_blocked()


def test_missing_attachment_is_repaired_with_separate_stop(tmp_path):
    venue = FakeVenue()
    venue.attach_stop = False
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert [b["type"] for b in venue.tpsl_bodies] == ["stop-loss", "take-profit"]
    assert venue.tpsl_bodies[0]["quantity"] == "0" and not venue.close_bodies
    assert only(trader) and not trader.store.state.get("halted")


def test_unknown_entry_answer_is_reconciled_from_positions(tmp_path):
    venue = FakeVenue()
    venue.unknown_entry = True
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert len(venue.market_bodies) == 1, "an unknown outcome must never be retried"
    assert only(trader)["quantity"] == "0.2"


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
    asyncio.run(trader.start())
    assert trader.store.state["halted"] and trader.entries_blocked()
    assert not venue.market_bodies and not venue.close_bodies, "foreign exposure must not be touched"


def test_crash_between_order_and_bookkeeping_is_recovered(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    trader.store.state["pending_entry"] = {
        "order_id": "00474:" + "a" * 26,
        "symbol": "SOLUSD",
        "side": "LONG",
        "stop": "81.5",
        "take": "83.62",
        "candidate": candidate().as_dict(),
    }
    trader.store.save()
    venue.position_list = [Position("SOLUSD", "BUY", Decimal("0.2"), Decimal("82.35"), False)]
    restarted = live_trader(tmp_path, venue)
    asyncio.run(restarted.start())
    assert only(restarted)["symbol"] == "SOLUSD"
    assert [b["type"] for b in venue.tpsl_bodies] == ["stop-loss"], "a recovered position must get a stop"


def test_session_loss_limit_halts_entries(tmp_path):
    venue = FakeVenue()
    venue.close_pnl = Decimal("-5.10")
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    trade = only(trader)
    asyncio.run(trader.manage(trade, trade["deadline_ms"] + 1))
    assert "лимита" in trader.store.state["halted"]


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
    offered = [candidate(), candidate(id="SOL-2", score=0.9), link_candidate()]

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
    asyncio.run(trader.try_enter(candidate(id="SOL-3"), brain.control_decision(offered)))
    assert len(trader.store.state["open_trades"]) == 2 and "entry_skipped" in events(trader)


def test_live_allows_one_position_until_the_first_full_trade(tmp_path):
    venue = FakeVenue()
    two_books(venue)
    trader = live_trader(tmp_path, venue, max_open_positions=3)
    asyncio.run(trader.start())
    assert trader.slot_limit() == 1
    enter(trader)
    asyncio.run(trader.try_enter(link_candidate(), brain.control_decision([link_candidate()])))
    assert len(venue.market_bodies) == 1, "no second position before one trade has gone the full circle"
    venue.venue_exit("0.31")
    asyncio.run(trader.manage(only(trader), now_ms()))
    assert trader.slot_limit() == 3


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
        for line in (trader.store.directory / "journal.jsonl").read_text().splitlines()
        if '"trade_closed"' in line
    ][0]
    assert closed["pnl_exact"] is False and closed["exit_price"] == sol["take"]
    assert Decimal("0.20") < Decimal(closed["pnl_usd"]) < Decimal("0.26")
    assert [t["symbol"] for t in trader.store.state["open_trades"]] == ["LINKUSD"]
    assert venue.cancelled == ["SOLUSD-t0", "SOLUSD-t1"], "only the closed instrument's orders are cancelled"
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


def test_routine_position_checks_share_one_read_and_writes_force_a_fresh_one(tmp_path):
    venue = FakeVenue()
    cfg = Config(use_llm=False)
    store = Store(tmp_path / "rt")
    broker = LiveBroker(venue, store, cfg, TEST_KEY, settle_seconds=0, cache_seconds=60)
    trade = {"symbol": "SOLUSD"}
    venue.position_list = [Position("SOLUSD", "BUY", Decimal("0.2"), Decimal("82.35"), True)]
    assert asyncio.run(broker.position_open(trade)) and asyncio.run(broker.position_open(trade))
    assert venue.position_reads == 1
    venue.position_list = []
    assert asyncio.run(broker.position("SOLUSD")) is None, "a direct read is never served from the cache"
    assert venue.position_reads == 2


def test_stop_refused_for_one_side_reading_is_placed_with_the_other(tmp_path):
    venue = FakeVenue()
    venue.attach_stop, venue.reject_tpsl, venue.tpsl_reject_label = False, True, "side"
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert [b["side"] for b in venue.tpsl_bodies] == ["BUY", "SELL", "SELL"], (
        "stop: BUY refused, SELL accepted; take: SELL"
    )
    assert trader.store.state["tpsl_side_mode"] == "closing" and not trader.store.state.get("halted")
    assert only(trader) and not venue.close_bodies


def test_entry_cash_carries_a_small_margin_against_rounding_down(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    cash = Decimal(venue.market_bodies[0]["cashQuantity"])
    assert Decimal("0.2") * venue.ask < cash <= Decimal("0.2") * venue.ask * Decimal("1.002")


def test_close_falls_back_to_the_other_signing_chain(tmp_path):
    venue = FakeVenue()
    venue.unknown_entry = True  # the entry answer is lost, so no chain has been confirmed yet
    venue.close_chains = {"161803"}
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert "chain_id" not in trader.store.state
    trade = only(trader)
    asyncio.run(trader.manage(trade, trade["deadline_ms"] + 1))
    assert [b["chainId"] for b in venue.close_bodies] == ["42161", "161803"]
    assert only(trader) is None and trader.store.state["chain_id"] == 161803


def test_account_leverage_other_than_1x_is_set_to_1x_and_the_order_resent_once(tmp_path):
    venue = FakeVenue()
    venue.account_leverage["SOLUSD"] = 10
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert venue.leverage_calls == [("SOLUSD", 1)] and len(venue.market_bodies) == 2, (
        "refused once, then accepted"
    )
    assert only(trader)["symbol"] == "SOLUSD" and "leverage_set" in events(trader)
    assert not trader.store.state.get("halted")


def test_leverage_that_cannot_be_changed_leaves_no_trade_and_no_loop(tmp_path):
    venue = FakeVenue()
    venue.account_leverage["SOLUSD"], venue.leverage_locked = 10, True
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    assert venue.leverage_calls == [("SOLUSD", 1)] and len(venue.market_bodies) == 1, "no second attempt"
    assert only(trader) is None and "leverage_change_failed" in events(trader)
    assert "pending_entry" not in trader.store.state


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
    assert only(trader) is None and sorted(venue.cancelled) == ["SOLUSD-t0", "SOLUSD-t1"]


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


MINUTE, QUARTER, HOUR = 60_000, 15 * 60_000, 60 * 60_000


class MinuteApi(FakeVenue):
    """A venue that serves closed one-minute bars up to a chosen moment."""

    def __init__(self, last_open_ms: int, drop: int | None = None) -> None:
        super().__init__()
        self.last_open_ms, self.drop, self.asked = last_open_ms, drop, []

    async def candle_rows(self, symbol, group, start_ms, end_ms):
        self.asked.append(group)
        first = max(start_ms // MINUTE * MINUTE, self.last_open_ms - 3 * 1440 * MINUTE)
        return [
            [t, 80.0, 80.2, 79.8, 80.0 + (t // MINUTE % 7) * 0.01, 5.0, 400.0]
            for t in range(first, min(end_ms, self.last_open_ms + MINUTE), MINUTE)
            if t != self.drop
        ]


def engine_state(symbol: str, rows: list) -> dict | None:
    """What the engine reports for the newest quarter hour that is complete in ``rows``."""
    opens = {int(r[0]) for r in rows}
    quarter = max(opens) // QUARTER * QUARTER
    while quarter >= min(opens) and any(quarter + i * MINUTE not in opens for i in range(15)):
        quarter -= QUARTER
    if quarter < min(opens):
        return None
    return {
        "symbol": symbol,
        "decision_ts_ms": quarter + QUARTER - 1,
        "price": 80.0,
        "trend": "UP",
        "trend_ema_fast": 79.0,
        "trend_ema_slow": 78.0,
        "trend_efficiency": 0.41237,
        "rsi": 55.0,
        "atr": 0.8,
        "atr_pct": 1.0,
        "change_1h_pct": 0.12345,
        "change_4h_pct": 0.5,
        "change_24h_pct": 2.0,
        "change_7d_pct": 3.14159,
        "dist_ema_atr": 0.3,
        "position_in_7d_range": None,
        "last_bar_volume_vs_20bar_avg": 1.2345,
        "consecutive_up_bars": 2,
        "consecutive_down_bars": 0,
    }


def engine_intent(state: dict, *, reward_risk: float, hold_minutes: int) -> dict:
    return {
        "intent_id": "f" * 64,
        "sleeve_id": system_engine.HOURLY_TREND_ID,
        "symbol": state["symbol"],
        "side": "LONG",
        "decision_ts_ms": state["decision_ts_ms"],
        "entry_eligible_ts_ms": state["decision_ts_ms"] + 1,
        "entry_expires_ts_ms": state["decision_ts_ms"] + QUARTER,
        "reference_price": 80.0,
        "signal_strength": state["trend_efficiency"],
        "gross_reward_bps": 100.0,
        "stop_price": 79.2,
        "target_price": 80.0 + reward_risk * 0.8,
        "max_holding_ms": hold_minutes * MINUTE,
        "trailing_activation_price": None,
        "trailing_distance": None,
        "metadata": {"setup": "PULLBACK_RECLAIM", "atr": "0.80000000000000004", "trend": "UP"},
    }


def fake_engine(monkeypatch, *, code_sha256: str = system_engine.HOURLY_TREND_CODE_SHA256, signal=True):
    """Replace the engine process with a function; returns the list of requests it received."""
    requests: list[dict] = []

    def runner(config, request):
        requests.append(request)
        results = {}
        for symbol, rows in request["series"].items():
            state = engine_state(symbol, rows)
            intent = None
            if state is not None and signal:
                intent = engine_intent(
                    state,
                    reward_risk=request["config"]["target_reward_to_risk"],
                    hold_minutes=request["config"]["max_hold_minutes"],
                )
            results[symbol] = {"state": state, "intent": intent}
        identity = {
            "strategy_id": system_engine.HOURLY_TREND_ID,
            "revision": "1",
            "strategy_code_sha256": code_sha256,
            "config_sha256": "c" * 64,
        }
        return {"identity": identity, "results": results}

    monkeypatch.setattr(system_engine, "_run_hourly_trend", runner)
    return requests


def scanning_trader(tmp_path, venue, **config) -> Trader:
    trader = paper_trader(tmp_path, venue, Config(use_llm=False, symbols=["SOLUSD"], **config))
    trader.engine = system_engine.EngineConfig()  # the engine process itself is replaced by fake_engine
    return trader


def journal_of(trader) -> list[dict]:
    return [json.loads(line) for line in (trader.store.directory / "journal.jsonl").read_text().splitlines()]


def test_scan_asks_the_engine_about_the_hourly_bar_that_closed_at_the_last_quarter_hour(
    tmp_path, monkeypatch
):
    slot = 1_794_400_000_000 // HOUR * HOUR + QUARTER  # a boundary at :15
    requests = fake_engine(monkeypatch)
    venue = MinuteApi(last_open_ms=slot + 2 * MINUTE)  # two minutes of the next quarter already exist
    trader = scanning_trader(tmp_path, venue, reward_risk=2.0, timeout_minutes=240)
    features, found = asyncio.run(trader.scan(slot + 130_000))

    assert set(venue.asked) == {"1m"} and [f.symbol for f in features] == ["SOLUSD"]
    request = requests[0]
    assert request["op"] == "hourly_trend" and request["all"] is False
    assert request["config"] == {"target_reward_to_risk": 2.0, "max_hold_minutes": 240}
    sent = request["series"]["SOLUSD"]
    assert sent[-1][0] == slot - MINUTE, "minutes after the boundary are not part of the decision"
    assert sent[0][0] >= slot - system_engine.HOURLY_TREND_HISTORY_MS

    (c,) = found
    assert c.id == f"SOLUSD-LONG-PULLBACK_RECLAIM-{slot - HOUR}"
    assert (c.bar_time, c.signal_time, c.setup) == (slot - HOUR, slot, "PULLBACK_RECLAIM")
    assert (c.ref_price, c.stop, c.take, c.timeout_minutes) == (80.0, 79.2, 81.6, 240)
    assert c.atr == pytest.approx(0.8) and c.score == 0.4124 and "uptrend" in c.rationale
    assert features[0].trend_ema20 == 79.0 and features[0].dist_ema20_atr == 0.3

    scan = journal_of(trader)[-1]
    assert scan["event"] == "scan" and scan["bar"] == slot - HOUR, "the bar under review runs from :15 to :15"
    assert scan["source"] == "strategy_engine"
    assert scan["engine"]["strategy_code_sha256"] == system_engine.HOURLY_TREND_CODE_SHA256
    assert scan["features"][0]["change_1h_pct"] == 0.123 and scan["features"][0]["change_7d_pct"] == 3.14
    assert scan["features"][0]["position_in_7d_range"] is None
    assert scan["candidates"][0]["id"] == c.id
    assert (trader.store.directory / "minute-cache" / "SOLUSD.1m.json").exists()


def test_scan_skips_an_instrument_without_a_fresh_bar_and_follows_the_scan_interval(tmp_path, monkeypatch):
    slot = 1_794_400_000_000 // HOUR * HOUR + QUARTER
    fake_engine(monkeypatch)
    stale = scanning_trader(tmp_path / "stale", MinuteApi(last_open_ms=slot - 2 * MINUTE))
    assert asyncio.run(stale.scan(slot + 12_000)) == ([], []), "the last minute of the bar is not there yet"
    assert journal_of(stale)[-1]["skipped"] == ["SOLUSD: no fresh closed bar"]

    holed = scanning_trader(tmp_path / "hole", MinuteApi(last_open_ms=slot - MINUTE, drop=slot - 5 * MINUTE))
    features, _ = asyncio.run(holed.scan(slot + 12_000))
    assert [f.symbol for f in features] == ["SOLUSD"], "one missing minute inside a traded quarter is bridged"
    assert journal_of(holed)[-1]["filled_minutes"] == 1

    hourly = scanning_trader(tmp_path / "hourly", MinuteApi(last_open_ms=slot + 5 * MINUTE), scan_minutes=60)
    asyncio.run(hourly.scan(slot + 12_000))
    assert journal_of(hourly)[-1]["bar"] == slot - QUARTER - HOUR, (
        "with hourly scans the bar closes on the hour"
    )


def test_scan_trades_nothing_when_the_engine_is_missing_fails_or_is_not_the_pinned_source(
    tmp_path, monkeypatch
):
    slot = 1_794_400_000_000 // HOUR * HOUR
    fake_engine(monkeypatch, code_sha256="0" * 64)
    other = scanning_trader(tmp_path / "other", MinuteApi(last_open_ms=slot - MINUTE))
    assert asyncio.run(other.scan(slot + 12_000)) == ([], [])
    error = journal_of(other)[-1]
    assert error["event"] == "scan_error" and "differs from the pinned one" in error["message"]

    def broken(config, request):
        raise system_engine.EngineError("Strategy Engine: worker failed")

    monkeypatch.setattr(system_engine, "_run_hourly_trend", broken)
    failing = scanning_trader(tmp_path / "failing", MinuteApi(last_open_ms=slot - MINUTE))
    assert asyncio.run(failing.scan(slot + 12_000)) == ([], [])
    assert journal_of(failing)[-1]["event"] == "scan_error"

    absent = scanning_trader(tmp_path / "absent", MinuteApi(last_open_ms=slot - MINUTE))
    absent.engine = None
    assert asyncio.run(absent.scan(slot + 12_000)) == ([], [])
    assert journal_of(absent)[-1]["message"] == "Strategy Engine is not available"


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
    waiting = now - trader.scan_slot(now) < 10_000  # the first ten seconds of a slot are left for the candle
    assert len(scans) == (0 if waiting else 1), "a slot is scanned once"
    with pytest.raises(ConfigError):
        Config(scan_minutes=5).validate()
