"""Offline tests of the bridge to the real Kairos Macro Strategist and Risk Manager.

``test_real_*`` start the real repository code (no network, no model); they are skipped
where the checkouts or their environments are absent.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from test_autotrader import FakeVenue, candidate, enter, events, live_trader, only

from kairos_autotrader import engine, system_rules
from kairos_autotrader.engine import Store
from kairos_autotrader.evedex import Evedex
from kairos_autotrader.system_rules import MacroDesk, RulesConfig, RulesError

EQUITY = 109.33


def rules(kind: str) -> RulesConfig:
    config = RulesConfig()
    try:
        config.check_macro() if kind == "macro" else config.check_risk()
    except RulesError:
        pytest.skip(f"{kind} checkout or environment is not present")
    return config


def allocation(regime="BEAR", *, weights=None, reserve=0.5, leverage=1.0, hours_ago=1.0) -> dict:
    return {
        "source": "macro-strategist",
        "regime": regime,
        "stable_reserve_pct": reserve,
        "strategy_weights": {"PULLBACK_RECLAIM": 0.5} if weights is None else weights,
        "max_gross_leverage": leverage,
        "triggered_by": "schedule",
        "rationale": "test",
        "origin": "model",
        "produced_at": (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(),
    }


def item(cid="short", side="SHORT", *, strategy="PULLBACK_RECLAIM", depth=20_000.0, symbol="SOLUSD") -> dict:
    stop, take = (83.9, 81.65) if side == "SHORT" else (82.1, 84.35)
    return {
        "id": cid,
        "symbol": symbol,
        "side": side,
        "strategy_id": strategy,
        "stop": stop,
        "take": take,
        "bid": 83.00,
        "ask": 83.01,
        "depth_usd": depth,
        "slippage_bps": 2.0,
        "taker_fee_bps": 4.5,
    }


def account(**over) -> dict:
    base = {
        "equity_usd": EQUITY,
        "available_usd": EQUITY,
        "day_start_equity_usd": EQUITY,
        "peak_equity_usd": EQUITY,
        "daily_realized_pnl_usd": 0.0,
        "positions": [],
    }
    base.update(over)
    return base


def risk(config, candidates, alloc, acct=None) -> dict[str, dict]:
    result = system_rules._run_worker(
        config,
        "risk",
        {
            "op": "evaluate",
            "now_ms": int(time.time() * 1000),
            "candidates": candidates,
            "account": acct or account(),
            "allocation": alloc,
        },
    )
    assert result["limits"]["per_trade_risk_fraction"] == 0.0025
    return {d["id"]: d for d in result["decisions"]}


def test_real_risk_manager_sizes_by_loss_at_stop_within_a_quarter_percent_of_equity():
    config = rules("risk")
    decision = risk(config, [item()], allocation())["short"]
    assert decision["approved"] and decision["rejection_reasons"] == []
    budget = EQUITY * 0.0025
    assert decision["loss_budget_usd"] == pytest.approx(budget)
    assert decision["worst_case_loss_usd"] <= budget + 1e-6
    per_unit = (83.9 - 83.0) + (83.0 + 83.9) * 4.5 / 10_000 + 83.005 * 2.0 / 10_000
    assert decision["quantity"] == pytest.approx(budget / per_unit, rel=1e-6)
    assert decision["leverage"] == 1.0 and decision["notional_usd"] < EQUITY * 0.5


def test_real_risk_manager_applies_the_macro_regime_and_weight_vetoes():
    config = rules("risk")
    bear = risk(
        config, [item(), item("long", "LONG"), item("orphan", strategy="BREAKOUT")], allocation("BEAR")
    )
    assert bear["short"]["approved"]
    assert bear["long"]["rejection_reasons"] == ["macro_regime_forbids_long"]
    assert bear["orphan"]["rejection_reasons"] == ["strategy_has_no_macro_allocation"]
    assert risk(config, [item("long", "LONG")], allocation("BULL"))["long"]["approved"]
    assert risk(config, [item()], allocation("BULL"))["short"]["rejection_reasons"] == [
        "macro_regime_forbids_short"
    ]
    assert risk(config, [item()], allocation("CHOP"))["short"]["rejection_reasons"] == ["macro_regime_chop"]
    assert risk(config, [item()], None)["short"]["rejection_reasons"] == ["strategic_allocation_missing"]
    stale = risk(config, [item()], allocation(hours_ago=30))["short"]
    assert stale["rejection_reasons"] == ["strategic_allocation_stale"] and stale["quantity"] == 0.0


def test_real_risk_manager_account_limits():
    config = rules("risk")
    held = {
        "symbol": "SOLUSD",
        "side": "SHORT",
        "quantity": 0.2,
        "entry_price": 83.0,
        "stop": 83.9,
        "strategy_id": "PULLBACK_RECLAIM",
    }
    same_symbol = risk(config, [item()], allocation(), account(positions=[held]))["short"]
    assert "symbol_has_active_position" in same_symbol["rejection_reasons"]
    heavy = [
        {**held, "symbol": f"X{i}USD", "quantity": 0.4} for i in range(4)
    ]  # 4 x 0.36 = 1.44 > 1% of equity
    assert (
        "portfolio_open_risk_limit_exhausted"
        in risk(config, [item()], allocation(), account(positions=heavy))["short"]["rejection_reasons"]
    )
    drawdown = risk(config, [item()], allocation(), account(day_start_equity_usd=113.0))["short"]
    assert drawdown["rejection_reasons"] == ["daily_drawdown_limit_reached"]
    realized = risk(config, [item()], allocation(), account(daily_realized_pnl_usd=-3.5))["short"]
    assert realized["rejection_reasons"] == ["daily_drawdown_limit_reached"]
    thin = risk(config, [item(depth=0.0)], allocation())["short"]
    assert thin["rejection_reasons"] == ["position_capacity_exhausted"]
    capped = risk(config, [item()], allocation(weights={"PULLBACK_RECLAIM": 0.05}, reserve=0.95))["short"]
    assert capped["approved"] and capped["notional_usd"] == pytest.approx(EQUITY * 0.05, rel=1e-6)


def macro_request(**over) -> dict:
    base = {
        "op": "prepare",
        "allowed_strategy_ids": ["PULLBACK_RECLAIM", "BREAKOUT"],
        "portfolio": {"equity_usd": EQUITY},
        "performance": {"trades_today": 1},
        "symbols": ["BTCUSD", "ETHUSD", "SOLUSD"],
        "units": {"rsi": "index_0_100"},
        "biases": {"BTCUSD": "SHORT", "ETHUSD": "SHORT", "SOLUSD": "FLAT"},
        "market_values": {"BTCUSD": {"rsi": 41.0}},
        "changes_1h_pct": {"BTCUSD": -0.4, "SOLUSD": -12.5},
        "trigger": {"kind": "schedule"},
    }
    base.update(over)
    return base


def test_real_macro_strategist_prompt_context_regime_hint_and_shock_detector():
    config = rules("macro")
    prepared = system_rules._run_worker(config, "macro", macro_request())
    assert (
        "Macro-Strategist" in prepared["system"] and '["BREAKOUT", "PULLBACK_RECLAIM"]' in prepared["system"]
    )
    assert prepared["regime_hint"] == "BEAR" and prepared["regime_evidence"]["required_majority"] == 2
    assert [(s["symbol"], s["kind"]) for s in prepared["shocks"]] == [("SOLUSD", "price_crash")]
    context = json.loads(prepared["user"])
    assert (
        context["macro_factors"]["status"] == "unavailable"
        and context["onchain_factors"]["status"] == "unavailable"
    )
    assert (
        context["market_factors"]["values"] == {"BTCUSD": {"rsi": 41.0}} and context["regime_hint"] == "BEAR"
    )
    split = system_rules._run_worker(config, "macro", macro_request(biases={"A": "LONG", "B": "SHORT"}))
    assert split["regime_hint"] == "CHOP", "no strict majority"


def test_real_macro_strategist_validation_and_defensive_fallback():
    config = rules("macro")

    def validate(output):
        return system_rules._run_worker(
            config,
            "macro",
            {
                "op": "validate",
                "model_output": output,
                "trigger": "schedule",
                "allowed_strategy_ids": ["BREAKOUT", "PULLBACK_RECLAIM"],
            },
        )["allocation"]

    good = validate(
        {
            "regime": "BEAR",
            "stable_reserve_pct": 0.6,
            "max_gross_leverage": 1.0,
            "rationale": "r",
            "strategy_weights": [{"strategy_name": "PULLBACK_RECLAIM", "weight": 0.4}],
        }
    )
    assert (good["regime"], good["stable_reserve_pct"], good["origin"]) == ("BEAR", 0.6, "model")
    assert good["strategy_weights"] == {"PULLBACK_RECLAIM": 0.4} and good["triggered_by"] == "schedule"
    for broken in (
        None,
        {
            "regime": "BULL",
            "stable_reserve_pct": 0.5,
            "max_gross_leverage": 1.0,
            "rationale": "r",
            "strategy_weights": [{"strategy_name": "PULLBACK_RECLAIM", "weight": 0.2}],
        },  # does not sum to one
        {
            "regime": "BULL",
            "stable_reserve_pct": 0.5,
            "max_gross_leverage": 1.0,
            "rationale": "r",
            "strategy_weights": [{"strategy_name": "INVENTED", "weight": 0.5}],
        },
    ):
        fallback = validate(broken)
        assert (fallback["regime"], fallback["stable_reserve_pct"], fallback["strategy_weights"]) == (
            "CHOP",
            1.0,
            {},
        )
        assert fallback["origin"] == "defensive fallback"


class MacroWorker:
    def __init__(self, shocks=()):
        self.shocks, self.requests = list(shocks), []

    def __call__(self, config, kind, request):
        assert kind == "macro"
        self.requests.append(request)
        if request["op"] == "prepare":
            return {
                "system": "SYS",
                "user": "CTX",
                "regime_hint": "BEAR",
                "regime_evidence": {},
                "shocks": self.shocks,
                "shock_cooldown_s": 3600.0,
                "allowed_strategy_ids": request["allowed_strategy_ids"],
            }
        return {"allocation": {**allocation(), "triggered_by": request["trigger"]}}


def macro_desk(tmp_path, worker):
    calls = []

    async def extract(session, *, api_key, model, system, user, effort, timeout_s):
        calls.append((model, system, user, effort))
        return {"regime": "BEAR"}, {"input_tokens": 2000}, ""

    desk = MacroDesk(
        Store(tmp_path / "rt"),
        RulesConfig(),
        ("PULLBACK_RECLAIM", "BREAKOUT"),
        runner=worker,
        extract=extract,
    )
    return desk, calls


MARKET = {"BTCUSD": {"quant_bias": "SHORT", "change_1h_pct": -0.2}}


def refresh(desk, now_ms, key="sk-test"):
    return asyncio.run(
        desk.refresh(
            object(),
            openai_key=key,
            model="m",
            effort="high",
            market=MARKET,
            portfolio={},
            performance={},
            now_ms=now_ms,
        )
    )


def test_macro_runs_once_a_day_and_again_after_a_shock_with_a_cooldown(tmp_path):
    worker = MacroWorker()
    desk, calls = macro_desk(tmp_path, worker)
    now = 1_794_400_000_000
    assert refresh(desk, now, key=None) is None and not calls, "no model key, no allocation"
    first = refresh(desk, now)
    assert first["regime"] == "BEAR" and calls == [("m", "SYS", "CTX", "high")]
    assert desk.allocation() == first and desk.summary()["regime"] == "BEAR"
    assert refresh(desk, now + 3_600_000) is None and len(calls) == 1, "not due within a day"
    worker.shocks = [{"symbol": "BTCUSD", "kind": "price_crash"}]
    shocked = refresh(desk, now + 2 * 3_600_000)
    assert shocked["triggered_by"] == "shock_event" and len(calls) == 2
    assert refresh(desk, now + 2 * 3_600_000 + 60_000) is None and len(calls) == 2, "shock cooldown"
    assert refresh(desk, now + 25 * 3_600_000 + 2 * 3_600_000) is not None and len(calls) == 3, "next day"
    journal = [
        json.loads(line)["event"]
        for line in (desk.store.directory / "journal.jsonl").read_text().splitlines()
    ]
    assert journal.count("macro_allocation") == 3


def kairos_trader(tmp_path, monkeypatch, verdict):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue, risk_mode="kairos")
    trader.risk_rules = RulesConfig()
    asked = []

    async def fake_review(config, *, candidates, account, allocation, now_ms):
        asked.append((candidates, account))
        if isinstance(verdict, Exception):
            raise verdict
        return {"decisions": [{"id": candidates[0]["id"], **verdict}], "limits": {}}

    monkeypatch.setattr(engine, "risk_review", fake_review)
    asyncio.run(trader.start())
    return trader, venue, asked


def test_kairos_risk_mode_can_only_shrink_or_refuse_an_entry(tmp_path, monkeypatch):
    smaller, venue, asked = kairos_trader(
        tmp_path / "a", monkeypatch, {"approved": True, "rejection_reasons": [], "quantity": 0.1234}
    )
    enter(smaller)
    assert only(smaller)["quantity"] == "0.1", "the Risk Manager's size, floored to the lot step"
    sent, acct = asked[0]
    assert sent[0]["strategy_id"] == "PULLBACK_RECLAIM" and sent[0]["stop"] == 81.5
    assert acct["equity_usd"] == pytest.approx(109.42) and acct["positions"] == []
    assert "risk_review" in events(smaller)

    larger, venue, _ = kairos_trader(
        tmp_path / "b", monkeypatch, {"approved": True, "rejection_reasons": [], "quantity": 5.0}
    )
    enter(larger)
    assert only(larger)["quantity"] == "0.2", "never more than this program's own size"

    refused, venue, _ = kairos_trader(
        tmp_path / "c",
        monkeypatch,
        {"approved": False, "rejection_reasons": ["macro_regime_chop"], "quantity": 0.0},
    )
    enter(refused)
    assert only(refused) is None and not venue.market_bodies and "entry_skipped" in events(refused)

    broken, venue, _ = kairos_trader(tmp_path / "d", monkeypatch, RulesError("worker failed"))
    enter(broken)
    assert only(broken) is None and not venue.market_bodies, "no Risk Manager answer means no entry"

    tiny, venue, _ = kairos_trader(
        tmp_path / "e", monkeypatch, {"approved": True, "rejection_reasons": [], "quantity": 0.04}
    )
    enter(tiny)
    assert only(tiny) is None and not venue.market_bodies, "below the instrument's minimum lot"


def test_shadow_risk_review_is_recorded_and_never_blocks(tmp_path, monkeypatch):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    trader.risk_rules = RulesConfig()

    async def refuse(config, *, candidates, account, allocation, now_ms):
        return {
            "decisions": [
                {
                    "id": c["id"],
                    "approved": False,
                    "rejection_reasons": ["strategic_allocation_missing"],
                    "quantity": 0.0,
                }
                for c in candidates
            ],
            "limits": {},
        }

    monkeypatch.setattr(engine, "risk_review", refuse)
    asyncio.run(trader.start())
    asyncio.run(trader.risk_shadow([candidate()], int(time.time() * 1000)))
    enter(trader)
    assert only(trader) is not None, "in shadow mode the Risk Manager does not decide"
    assert events(trader).count("risk_review") == 1

    async def crash(config, **kwargs):
        raise RulesError("worker failed")

    monkeypatch.setattr(engine, "risk_review", crash)
    asyncio.run(trader.risk_shadow([candidate()], int(time.time() * 1000)))
    assert "risk_error" in events(trader)


def test_day_realized_result_is_kept_for_the_drawdown_rule(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    venue.venue_exit("-0.21")
    asyncio.run(trader.manage(only(trader), int(time.time() * 1000)))
    account_view = asyncio.run(trader._risk_account())
    assert account_view["daily_realized_pnl_usd"] == pytest.approx(-0.21)
    assert account_view["day_start_equity_usd"] == pytest.approx(109.21) and account_view["positions"] == []


def test_order_book_sizes_are_read_when_the_venue_sends_them():
    api = Evedex(None)

    async def request(method, url, **kwargs):
        return {"t": 1, "bids": [{"price": "83.00", "quantity": "120.5"}], "asks": [{"price": "83.01"}]}

    api._request = request
    book = asyncio.run(api.book("SOLUSD"))
    assert (book.bid, book.ask, book.bid_qty, book.ask_qty) == (Decimal("83.00"), Decimal("83.01"), 0, 0)

    async def full(method, url, **kwargs):
        return {
            "t": 1,
            "bids": [{"price": "83.00", "quantity": "120.5"}],
            "asks": [{"price": "83.01", "quantity": "7"}],
        }

    api._request = full
    book = asyncio.run(api.book("SOLUSD"))
    assert (book.bid_qty, book.ask_qty) == (Decimal("120.5"), Decimal("7"))
