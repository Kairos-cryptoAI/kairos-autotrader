"""Offline tests of the bridge to the real Kairos Macro Strategist.

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
from test_autotrader import FakeVenue, enter, live_trader, only

from kairos_autotrader import system_rules
from kairos_autotrader.engine import Store
from kairos_autotrader.evedex import Evedex
from kairos_autotrader.system_rules import MacroDesk, RulesConfig, RulesError

# Entries are decided by the Risk Manager; here a stand-in approves with a loss-at-stop size.
pytestmark = pytest.mark.usefixtures("risk_approves")
EQUITY = 109.33


def allocation(regime="BEAR", *, weights=None, reserve=0.5, leverage=1.0, hours_ago=1.0) -> dict:
    return {
        "source": "macro-strategist",
        "regime": regime,
        "stable_reserve_pct": reserve,
        "strategy_weights": weights if weights is not None else {"hourly_trend_candidates_v1": 0.5},
        "max_gross_leverage": leverage,
        "triggered_by": "schedule",
        "rationale": "test",
        "produced_at": (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(),
    }


def rules(kind: str) -> RulesConfig:
    config = RulesConfig()
    try:
        config.check_macro()
    except RulesError:
        pytest.skip(f"{kind} checkout or environment is not present")
    return config


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


def test_day_realized_result_is_kept_for_the_drawdown_rule(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    asyncio.run(trader.start())
    enter(trader)
    venue.venue_exit("-0.21")
    asyncio.run(trader.manage(only(trader), int(time.time() * 1000)))
    evidence = asyncio.run(trader._account_evidence())
    assert evidence["daily_realized_pnl_usd"] == pytest.approx(-0.21)
    assert evidence["day_start_equity_usd"] == pytest.approx(109.42), "the first observation of the day"
    assert evidence["positions"] == [] and evidence["balance_usd"] == pytest.approx(109.21)


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
