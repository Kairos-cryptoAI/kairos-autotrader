"""The Kairos Risk Manager deciding entries.

``test_real_*`` run ``risk_worker.decide`` with the real Quant Scouts and Risk Manager code
in Quant Scouts' environment, on synthetic order books: no network. They are skipped where
those checkouts or the environment are absent. The other tests check how the trading loop
uses the decision, with a stand-in for the Risk Manager.
"""

from __future__ import annotations

import asyncio
import json
import subprocess  # nosec B404 - the test starts the real worker code with a fixed argument list
import time
from decimal import Decimal
from pathlib import Path

import pytest
from kairos_core.contracts import CandidateReviewV1, RiskTradeDecisionV1
from support import MINUTE, approve_entries, route_of, routed, strict_intent
from test_autotrader import FakeVenue, enter, events, live_trader, only

from kairos_autotrader import engine
from kairos_autotrader.engine import Config
from kairos_autotrader.risk import (
    AGGREGATE_OPEN_RISK_FRACTION,
    LIVE_ACCOUNT_SOURCE,
    MIN_NOTIONAL_USD,
    PER_TRADE_RISK_FRACTION,
    RiskConfig,
    RiskError,
    evaluate_entry,
)

RUNNER_PACKAGE = Path(engine.__file__).parent
LIMITS = {
    "per_trade_risk_fraction": PER_TRADE_RISK_FRACTION,
    "aggregate_open_risk_fraction": AGGREGATE_OPEN_RISK_FRACTION,
    "max_leverage": 1.0,
    "per_trade_dollar_cap_usd": 0.25,
    "aggregate_dollar_cap_usd": 100.0,
    "daily_loss_cap_usd": 5.0,
    "max_account_equity_usd": 200.0,
    "max_position_notional_usd": 33.33,
    "min_notional_usd": MIN_NOTIONAL_USD,
}
SCOPE = {
    "instruments": ["BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ZECUSD"],
    "account_id": "evedex-prod-test",
    "context": {"composer": "test"},
    "sources": {"test": "1"},
    "policy": {"test": "1"},
}


def review(
    *, symbol="SOLUSDT", decision="ALLOW", side="SHORT", ref=80.0, stop=81.0, take=78.5
) -> CandidateReviewV1:
    decided = int(time.time() * 1000) // MINUTE * MINUTE - 1
    intent = strict_intent(
        symbol=symbol,
        decision_ts_ms=decided,
        side=side,
        ref_price=ref,
        stop=stop,
        take=take,
        valid_ms=15 * MINUTE,
    )
    return CandidateReviewV1(
        source="aggregator",
        route=route_of(intent, review_timeout_ms=180_000),
        intent=intent,
        decision=decision,
        priority=60 if decision == "ALLOW" else 0,
        reviewed_at_ms=decided + 1,
        reviewer="DETERMINISTIC",
        reason_codes=("TEST",),
    )


def allocation(regime="BEAR", weights=None) -> dict:
    from datetime import UTC, datetime

    return {
        "source": "macro-strategist",
        "regime": regime,
        "stable_reserve_pct": 0.4,
        "strategy_weights": {"hourly_trend_candidates_v1": 0.6} if weights is None else weights,
        "max_gross_leverage": 1.0,
        "triggered_by": "schedule",
        "rationale": "test",
        "produced_at": datetime.now(UTC).isoformat(),
    }


def account(**over) -> dict:
    base = {
        "source_id": LIVE_ACCOUNT_SOURCE,
        "account_id": "evedex-prod-test",
        "captured_at_ms": int(time.time() * 1000),
        "balance_usd": 109.4,
        "available_usd": 109.4,
        "day_start_equity_usd": 109.4,
        "peak_equity_usd": 109.4,
        "daily_realized_pnl_usd": 0.0,
        "reconciliation_seq": 7,
        "pending_entry": False,
        "open_orders": [],
        "positions": [],
    }
    base.update(over)
    return base


def books(*, evedex="SOLUSD", binance="SOLUSDT", bid=79.99, ask=80.01, age_ms=100) -> dict:
    levels = (
        [(round(bid - i * 0.01, 2), 50.0) for i in range(10)],
        [(round(ask + i * 0.01, 2), 50.0) for i in range(10)],
    )
    return {
        "binance": {"symbol": binance, "bids": levels[0], "asks": levels[1], "age_ms": 100},
        "evedex": {"symbol": evedex, "bids": levels[0], "asks": levels[1], "age_ms": age_ms},
    }


def lineage(symbol="XRPUSD") -> dict:
    return {
        "venue_symbol": symbol,
        "side": "SHORT",
        "quantity": 20.0,
        "entry_price": 1.03,
        "mark_price": 1.02,
        "stop_order_id": "stop-00000001",
        "trade": {
            "stop": 1.045,
            "take": 1.0075,
            "max_holding_ms": 720 * MINUTE,
            "strategy_id": "hourly_trend_candidates_v1",
            "strategy_revision": "1",
            "intent_id": "a" * 64,
            "risk_decision_id": "b" * 64,
            "risk_trade_id": "c" * 64,
            "first_fill_ms": int(time.time() * 1000) - 3_600_000,
            "entry_order_id": "00476:" + "f" * 26,
        },
    }


def decide(*, review_=None, allocation_=..., account_=None, books_=None, refusal_only=False) -> dict:
    config = RiskConfig()
    try:
        config.check()
    except RiskError:
        pytest.skip("Quant Scouts or Risk Manager checkout or environment is not present")
    request = {
        "review": (review_ or review()).model_dump(mode="json"),
        "allocation": allocation() if allocation_ is ... else allocation_,
        "account": (account_() if callable(account_) else account_) or account(),
        "scope": SCOPE,
        "limits": LIMITS,
        "venue": {"taker_fee_bps": 4.5, "assessed_notional_usd": 33.33},
    }
    done = subprocess.run(  # nosec B603
        [
            str(config.python),
            "-s",
            "-B",
            str(Path(__file__).with_name("risk_offline.py")),
            str(config.quant_source),
            str(config.risk_source),
            str(RUNNER_PACKAGE),
        ],
        input=json.dumps(
            {"request": request, "books": books_ or books(), "refusal_only": refusal_only}
        ).encode(),
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-600:]
    return json.loads(done.stdout)


def reasons(answer: dict) -> list[str]:
    return answer["evaluation"]["decision"]["rejection_reasons"]


def test_real_risk_manager_sizes_an_allowed_short_by_loss_at_stop():
    answer = decide()
    decision = RiskTradeDecisionV1.model_validate(answer["evaluation"]["decision"])
    assert decision.approved and decision.rejection_reasons == ()
    assert (decision.trading_mode.value, decision.evedex_profile.value) == ("LIVE", "PROD")
    assert decision.venue_symbol == "SOLUSD" and decision.leverage == 1.0
    assert decision.worst_entry_price == 79.99, "a short sells at the EVEDEX best bid"
    assert decision.loss_budget_usd == 0.25 and decision.worst_case_loss_usd <= 0.25
    assert 0 < decision.quantity <= 33.33 / 79.99 and decision.notional_usd <= 33.33
    assert decision.exit_plan == decision.intent.exit_plan, "the Risk Manager never moves the exits"
    assert answer["evaluation"]["source_id"] == LIVE_ACCOUNT_SOURCE
    assert answer["evaluation"]["account_equity_usd"] == pytest.approx(109.4)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"allocation_": allocation("CHOP")}, "macro_regime_chop"),
        ({"allocation_": allocation("BULL")}, "macro_regime_forbids_short"),
        ({"allocation_": None}, "strategic_allocation_missing"),
        ({"allocation_": allocation(weights={"BREAKOUT": 0.6})}, "strategy_has_no_macro_allocation"),
        ({"review_": review(decision="DEFER")}, "review_defer"),
        (
            {"account_": lambda: account(balance_usd=260.0, available_usd=260.0, peak_equity_usd=260.0)},
            "account_equity_above_release_ceiling",
        ),
        ({"account_": lambda: account(daily_realized_pnl_usd=-5.0)}, "daily_loss_limit_exhausted"),
        ({"account_": lambda: account(positions=[lineage("SOLUSD")])}, "symbol_has_active_position"),
        ({"books_": books(age_ms=3_000)}, "venue_book_stale"),
        ({"books_": books(bid=81.5, ask=81.52)}, "short_entry_exit_geometry_invalid"),
    ],
)
def test_real_risk_manager_refuses(change, reason):
    answer = decide(**change)
    assert answer["evaluation"]["decision"]["approved"] is False
    assert reason in reasons(answer), reasons(answer)
    assert answer["evaluation"]["decision"]["quantity"] == 0.0


@pytest.mark.parametrize(
    "evidence",
    [
        lambda: account(captured_at_ms=int(time.time() * 1000) - 60_000),
        lambda: account(positions=[{**lineage(), "trade": None}]),
        lambda: account(open_orders=[{"instrument": "SOLUSD", "id": "x"}]),
        lambda: account(pending_entry=True),
        lambda: account(account_id="evedex-prod-other"),
    ],
    ids=["stale", "position-without-lineage", "resting-order", "pending-entry", "other-account"],
)
def test_real_account_evidence_that_cannot_be_verified_refuses_the_entry(evidence):
    answer = decide(account_=evidence)
    assert "account_source_verification_failed" in reasons(answer)


def test_real_open_positions_use_up_the_risk_and_capital_room():
    alone = decide()["evaluation"]["decision"]
    held = decide(account_=lambda: account(positions=[lineage("XRPUSD")]))
    evaluation = held["evaluation"]
    assert evaluation["aggregate_open_risk_usd"] == pytest.approx(20 * 0.015)
    assert evaluation["decision"]["approved"]
    assert evaluation["decision"]["notional_usd"] <= alone["notional_usd"] + 1e-9


def test_real_signal_outside_the_approved_prod_map_is_refused_before_any_measurement():
    answer = decide(review_=review(symbol="ZECUSDT", ref=950.0, stop=962.0, take=932.0), refusal_only=True)
    assert answer == {"refused": "ZECUSDT is outside Quant Scouts' approved PROD symbol map"}


# ---- how the trading loop uses the decision ---------------------------------------------


def started(tmp_path, venue=None, **config):
    venue = venue or FakeVenue()
    trader = live_trader(tmp_path, venue, **config)
    asyncio.run(trader.start())
    return trader, venue


def test_an_entry_needs_a_review_and_the_risk_manager(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("the Risk Manager is not asked without a review")

    monkeypatch.setattr(engine, "evaluate_entry", unexpected)
    trader, venue = started(tmp_path / "a")
    trader.risk = RiskConfig()
    enter(trader)
    assert only(trader) is None and not venue.market_bodies
    assert (
        "the Aggregator has not reviewed"
        in json.loads((trader.store.directory / "journal.jsonl").read_text().splitlines()[-1])["reason"]
    )

    approve_entries(monkeypatch)
    absent, venue = started(tmp_path / "b")
    absent.risk = None
    enter(absent)
    assert only(absent) is None and not venue.market_bodies, "no Risk Manager, no entry"


def test_refusal_and_failure_of_the_risk_manager_open_nothing(tmp_path, monkeypatch):
    approve_entries(monkeypatch, approve=False, reasons=("macro_regime_chop",))
    refused, venue = started(tmp_path / "a")
    enter(refused)
    assert only(refused) is None and not venue.market_bodies
    assert events(refused)[-2:] == ["risk_decision", "entry_skipped"]

    approve_entries(monkeypatch, error="worker failed")
    broken, venue = started(tmp_path / "b")
    enter(broken)
    assert only(broken) is None and not venue.market_bodies


def test_the_risk_managers_quantity_is_floored_to_the_lot_and_never_raised(tmp_path, monkeypatch):
    approve_entries(monkeypatch, quantity=0.2789)
    trader, venue = started(tmp_path / "a")
    enter(trader)
    trade = only(trader)
    assert trade["quantity"] == "0.2", "floored to the 0.1 lot step"
    assert (trade["risk_decision_id"], trade["risk_trade_id"]) == ("d" * 64, "e" * 64)
    assert (
        trade["intent_id"] == trade["candidate"]["id"]
        and trade["strategy_id"] == "hourly_trend_candidates_v1"
    )
    assert trade["max_holding_ms"] == 720 * MINUTE and trade["first_fill_ms"] == trade["opened_ms"]

    approve_entries(monkeypatch, quantity=0.04)
    tiny, venue = started(tmp_path / "b")
    enter(tiny)
    assert only(tiny) is None and not venue.market_bodies, "below the instrument's minimum lot"


def test_account_evidence_carries_each_positions_lineage(tmp_path, monkeypatch):
    calls = approve_entries(monkeypatch)
    trader, venue = started(tmp_path, max_open_positions=2)
    trader.store.state["round_trips"] = 1
    enter(trader)
    trade = only(trader)
    venue.books = {"LINKUSD": (Decimal("10.299"), Decimal("10.300"))}
    asyncio.run(
        trader.try_enter(
            routed(symbol="LINKUSD", ref_price=10.30, stop=10.18, take=10.48, atr=0.1),
            engine.brain.Decision("TRADE", None, 0, "", "TEST"),
        )
    )
    evidence = calls[-1]["account"]
    assert evidence["source_id"] == LIVE_ACCOUNT_SOURCE and evidence["account_id"].startswith("evedex-prod-")
    (position,) = evidence["positions"]
    assert position["venue_symbol"] == "SOLUSD" and position["side"] == "LONG"
    assert position["trade"]["risk_decision_id"] == trade["risk_decision_id"]
    assert position["trade"]["entry_order_id"] == trade["id"]
    assert evidence["reconciliation_seq"] == 2 and calls[-1]["limits"]["max_position_notional_usd"] == 50.0

    del trade["risk_decision_id"]
    assert asyncio.run(trader._account_evidence())["positions"][0]["trade"] is None, (
        "a trade opened before the Risk Manager decided has no lineage, so new entries are refused"
    )


def test_retired_risk_keys_in_a_config_file_are_ignored_with_a_note(tmp_path, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"risk_mode": "autotrader", "risk_shadow": True, "review_budget_usd": 5}))
    config = Config.load(path)
    assert config.review_budget_usd == 5 and config.review_mode == "aggregator"
    assert "risk_mode больше не действует" in capsys.readouterr().out
    with pytest.raises(engine.ConfigError):
        Config(max_account_equity_usd=300.0).validate()


def test_bridge_refuses_a_decision_about_another_candidate():
    own, other = review(), review(symbol="XRPUSDT", ref=1.0, stop=1.02, take=0.97)

    def answer(decision_review):
        def runner(config, request):
            return {"evaluation": {"decision": _rejected(decision_review)}}

        return runner

    good = asyncio.run(
        evaluate_entry(
            RiskConfig(),
            review=own,
            allocation=None,
            account={},
            scope={},
            limits={},
            taker_fee_bps=4.5,
            assessed_notional_usd=33.0,
            runner=answer(own),
        )
    )
    assert not good.approved and good.quantity == 0 and good.record["decision_id"]
    with pytest.raises(RiskError):
        asyncio.run(
            evaluate_entry(
                RiskConfig(),
                review=own,
                allocation=None,
                account={},
                scope={},
                limits={},
                taker_fee_bps=4.5,
                assessed_notional_usd=33.0,
                runner=answer(other),
            )
        )
    refused = asyncio.run(
        evaluate_entry(
            RiskConfig(),
            review=own,
            allocation=None,
            account={},
            scope={},
            limits={},
            taker_fee_bps=4.5,
            assessed_notional_usd=33.0,
            runner=lambda c, r: {"refused": "outside the map"},
        )
    )
    assert refused.reasons == ("outside the map",) and refused.decision is None


def _rejected(review_: CandidateReviewV1) -> dict:
    from kairos_core.contracts import VenueQualityV1
    from kairos_core.enums import EvedexProfile, TradingMode

    now = review_.reviewed_at_ms + 1_000
    venue = VenueQualityV1(
        source="test",
        profile=EvedexProfile.PROD,
        symbol="SOLUSD",
        observed_at_ms=now,
        expires_at_ms=now + 5_000,
        reference_timestamp_ms=now,
        book_timestamp_ms=now,
        reference_mid_price=80.0,
        best_bid=79.99,
        best_ask=80.01,
        venue_mid_price=80.0,
        basis_bps=0.0,
        spread_bps=(80.01 - 79.99) / 80.0 * 10_000,
        assessed_notional_usd=33.0,
        depth_usd=1_000.0,
        buy_slippage_bps=1.0,
        sell_slippage_bps=1.0,
        taker_fee_bps=4.5,
        reference_age_ms=0,
        book_age_ms=0,
        latency_ms=10,
        timestamp_skew_ms=0,
        entry_allowed=True,
    )
    return RiskTradeDecisionV1(
        source="test",
        intent=review_.intent,
        review=review_,
        venue_quality=venue,
        approved=False,
        rejection_reasons=("macro_regime_chop",),
        decided_at_ms=now,
        trading_mode=TradingMode.LIVE,
        evedex_profile=EvedexProfile.PROD,
        account_id="evedex-prod-test",
        venue_symbol="SOLUSD",
        quantity=0.0,
        leverage=1.0,
        notional_usd=0.0,
        loss_budget_usd=0.0,
        worst_case_loss_usd=0.0,
        worst_entry_price=79.99 if review_.intent.side.value == "SHORT" else 80.01,
        estimated_fees_usd=0.0,
        estimated_slippage_usd=0.0,
        exit_plan=review_.intent.exit_plan,
    ).model_dump(mode="json")
