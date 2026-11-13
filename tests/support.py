"""Builders of real Kairos contracts and offline stand-ins for the sibling processes."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from kairos_core.contracts import (
    CandidateRouteV1,
    ExitPlanV1,
    StrategyIntentV1,
    StrategyProvenanceV1,
)
from kairos_core.enums import CandidateReviewTier, ReasoningEffort, Side

from kairos_autotrader import binance_bars, routing, system_engine
from kairos_autotrader.signals import Routed, signal_symbol

MINUTE, QUARTER, HOUR = 60_000, 15 * 60_000, 60 * 60_000
DAY = 24 * HOUR
CONFIG_SHA256 = "c" * 64
# A decision of today, so that entry windows of test candidates are open.
TODAY_SLOT = 1_794_400_000_000 // HOUR * HOUR


def strict_intent(
    *,
    symbol: str = "SOLUSDT",
    decision_ts_ms: int,
    side: str = "LONG",
    setup: str = "PULLBACK_RECLAIM",
    ref_price: float = 82.35,
    stop: float = 81.5,
    take: float = 83.625,
    timeout_minutes: int = 720,
    atr: float = 0.8,
    score: float = 0.4,
    valid_ms: int = QUARTER,
    bar_hashes: tuple[str, ...] = ("e" * 64,),
    code_sha256: str = system_engine.HOURLY_TREND_CODE_SHA256,
    config_sha256: str = CONFIG_SHA256,
    strategy_id: str = system_engine.HOURLY_TREND_ID,
) -> StrategyIntentV1:
    """A valid ``StrategyIntentV1`` shaped like the engine's hourly_trend output."""
    return StrategyIntentV1(
        source="strategy-engine",
        strategy_id=strategy_id,
        strategy_revision="1",
        symbol=symbol,
        side=Side(side),
        decision_ts_ms=decision_ts_ms,
        entry_eligible_ts_ms=decision_ts_ms + 1,
        entry_expires_ts_ms=decision_ts_ms + valid_ms,
        reference_price=ref_price,
        signal_strength=score,
        gross_reward_bps=abs(take - ref_price) / ref_price * 10_000,
        exit_plan=ExitPlanV1(stop_price=stop, target_price=take, max_holding_ms=timeout_minutes * MINUTE),
        provenance=StrategyProvenanceV1(
            strategy_code_sha256=code_sha256,
            config_sha256=config_sha256,
            input_window_sha256="a" * 64,
            features_sha256="b" * 64,
            input_bar_sha256s=bar_hashes,
        ),
        metadata=(("atr", repr(atr)), ("setup", setup), ("trend", "UP" if side == "LONG" else "DOWN")),
    )


def route_of(
    intent: StrategyIntentV1,
    *,
    tier: str = "NORMAL",
    evidence_ids: tuple[str, ...] = (),
    review_timeout_ms: int = 60_000,
) -> CandidateRouteV1:
    conflict = tier == "CONFLICT"
    return CandidateRouteV1(
        source=routing.ROUTE_SOURCE,
        intent=intent,
        review_tier=CandidateReviewTier(tier),
        requested_reasoning_effort=ReasoningEffort.HIGH if conflict else ReasoningEffort.MEDIUM,
        routed_at_ms=intent.decision_ts_ms,
        review_deadline_ms=min(intent.entry_expires_ts_ms, intent.decision_ts_ms + review_timeout_ms),
        evidence_ids=evidence_ids,
        conflict_rationale="strategy_side=LONG text_side=SHORT" if conflict else None,
    )


def routed(
    *,
    symbol: str = "SOLUSD",
    signal_time: int | None = None,
    tier: str = "NORMAL",
    evidence_ids: tuple[str, ...] = (),
    valid_ms: int | None = None,
    **intent_fields: Any,
) -> Routed:
    """A routed candidate for an EVEDEX instrument.

    Without ``signal_time`` the decision is recent and the entry window stays open for the
    whole test session; an explicit ``signal_time`` gets the engine's 15-minute window.
    """
    if signal_time is None:
        import time

        signal_time = int(time.time() * 1000) // QUARTER * QUARTER
        valid_ms = valid_ms or DAY
    intent = strict_intent(
        symbol=signal_symbol(symbol),
        decision_ts_ms=signal_time - 1,
        valid_ms=valid_ms or QUARTER,
        **intent_fields,
    )
    return Routed(symbol, route_of(intent, tier=tier, evidence_ids=evidence_ids))


# ---- offline stand-ins for the sibling processes ---------------------------------------


def kline(open_ms: int) -> list[Any]:
    """One closed minute as Binance returns it."""
    close = 80.0 + (open_ms // MINUTE % 7) * 0.01
    return [
        open_ms,
        "80.00",
        "80.20",
        "79.80",
        f"{close:.2f}",
        "5.0",
        open_ms + MINUTE - 1,
        "400.0",
        12,
        "2.5",
        "200.0",
        "0",
    ]


class BarFeed:
    """Serves closed one-minute klines up to a chosen moment, like the public endpoint."""

    def __init__(self, last_open_ms: int, drop: int | None = None) -> None:
        self.last_open_ms, self.drop = last_open_ms, drop
        self.asked: list[tuple[str, int, int]] = []
        self.revised: set[int] = set()

    async def fetch(self, symbol: str, start_ms: int, end_ms: int) -> object:
        self.asked.append((symbol, start_ms, end_ms))
        rows = []
        for open_ms in range(start_ms, min(end_ms, self.last_open_ms + MINUTE), MINUTE):
            if open_ms == self.drop:
                continue
            row = kline(open_ms)
            if open_ms in self.revised:
                row[4] = "81.11"
            rows.append(row)
        return rows


def parse_pages(config, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What ``quant_scouts_worker.py`` answers: complete pages become bars, others an error."""
    answers = []
    for page in pages:
        head = {"symbol": page["symbol"], "start": page["start"], "end": page["end"]}
        rows = page["rows"]
        complete = len(rows) == (page["end"] - page["start"]) // MINUTE and all(
            row[0] == page["start"] + index * MINUTE for index, row in enumerate(rows)
        )
        if not complete:
            answers.append({**head, "error": "RecoveryError: recovery REST page is incomplete"})
            continue
        bars = []
        for row in rows:
            values = [row[0], *(float(row[i]) for i in (1, 2, 3, 4, 5, 7, 9, 10))]
            identity = hashlib.sha256(json.dumps([page["symbol"], *values]).encode()).hexdigest()
            bars.append([*values, identity])
        answers.append({**head, "bars": bars})
    return answers


def engine_state(symbol: str, decision_ts_ms: int) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "decision_ts_ms": decision_ts_ms,
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


def fake_engine(
    monkeypatch,
    *,
    code_sha256: str = system_engine.HOURLY_TREND_CODE_SHA256,
    signal: bool = True,
    tamper=None,
) -> list[dict]:
    """Replace the engine process with a function; returns the list of requests it received.

    The stand-in answers like ``system_engine_worker.py``: a state per symbol and, with
    ``signal``, a strict intent that declares exactly the bars it was given. ``tamper`` may
    change the intent fields before the contract is built.
    """
    requests: list[dict] = []

    def runner(config, request):
        requests.append(request)
        results = {}
        for symbol, rows in request["series"].items():
            decision = int(rows[-1][0]) + MINUTE - 1
            intent = None
            if signal:
                fields = dict(
                    symbol=symbol,
                    decision_ts_ms=decision,
                    ref_price=80.0,
                    stop=79.2,
                    take=80.0 + request["config"]["target_reward_to_risk"] * 0.8,
                    timeout_minutes=request["config"]["max_hold_minutes"],
                    atr=0.8,
                    score=0.41237,
                    bar_hashes=tuple(row[9] for row in rows),
                    code_sha256=code_sha256,
                )
                if tamper is not None:
                    fields = tamper(fields)
                intent = strict_intent(**fields).model_dump(mode="json")
            results[symbol] = {
                "decision_ts_ms": decision,
                "input_first_open_time_ms": int(rows[0][0]),
                "input_bar_count": len(rows),
                "state": engine_state(symbol, decision),
                "intent": intent,
            }
        identity = {
            "strategy_id": system_engine.HOURLY_TREND_ID,
            "revision": "1",
            "strategy_code_sha256": code_sha256,
            "config_sha256": CONFIG_SHA256,
        }
        return {"identity": identity, "results": results}

    monkeypatch.setattr(system_engine, "_run_hourly_trend", runner)
    return requests


def fake_router(monkeypatch, *, tier: str = "NORMAL", reject: str | None = None, swap=None) -> list[dict]:
    """Replace the Router process; returns the list of requests it received."""
    requests: list[dict] = []

    def runner(config, request):
        requests.append(request)
        rows = []
        for raw in request["intents"]:
            intent = StrategyIntentV1.model_validate(raw)
            if reject is not None:
                rows.append({"intent_id": intent.intent_id, "rejected": reject})
                continue
            evidence = tuple(str(s["id"]) for s in request["signals"]) if tier == "CONFLICT" else ()
            carried = swap(intent) if swap is not None else intent
            built = route_of(
                carried, tier=tier, evidence_ids=evidence, review_timeout_ms=request["review_timeout_ms"]
            )
            rows.append(
                {
                    "intent_id": intent.intent_id,
                    "route": built.model_dump(mode="json"),
                    "text": {"bias": "FLAT", "score": 0.0, "confidence": 0.0},
                }
            )
        return {"routes": rows, "settings": {"review_timeout_ms": request["review_timeout_ms"]}}

    monkeypatch.setattr(routing, "_run_worker", runner)
    return requests


def fake_parser(monkeypatch) -> None:
    monkeypatch.setattr(binance_bars, "_run_parser", parse_pages)


# ---- the Risk Manager in tests of the trading loop ---------------------------------------


def approve_entries(monkeypatch, *, quantity=None, approve=True, reasons=(), error=None) -> list[dict]:
    """Every candidate counts as reviewed, and the Risk Manager's answer is a stand-in.

    Without ``quantity`` the stand-in sizes like a loss-at-stop rule: the per-trade risk
    over the stop distance plus round-trip costs, capped by the position notional. Returns
    the list of requests the stand-in received.
    """
    from decimal import Decimal
    from types import SimpleNamespace

    from kairos_autotrader import engine
    from kairos_autotrader.risk import RiskError, RiskOutcome

    calls: list[dict] = []

    async def evaluate(config, *, review, allocation, account, scope, limits, taker_fee_bps, **_):
        calls.append({"review": review, "allocation": allocation, "account": account, "limits": limits})
        if error is not None:
            raise RiskError(error)
        if not approve:
            record = {"approved": False, "rejection_reasons": list(reasons)}
            return RiskOutcome(False, Decimal(0), tuple(reasons), None, record)
        size = quantity
        if size is None:
            ref, stop = review.reference_price, review.stop
            per_unit = abs(ref - stop) + ref * 2 * (taker_fee_bps / 1e4 + 0.0002)
            size = min(
                limits["per_trade_dollar_cap_usd"] / per_unit, limits["max_position_notional_usd"] / ref
            )
        decision = SimpleNamespace(decision_id="d" * 64, trade_id="e" * 64)
        record = {"approved": True, "quantity": float(size), "decision_id": "d" * 64}
        return RiskOutcome(True, Decimal(repr(float(size))), (), decision, record)

    monkeypatch.setattr(engine, "evaluate_entry", evaluate)
    monkeypatch.setattr(engine.Trader, "_review_of", lambda self, candidate: candidate)
    monkeypatch.setattr(engine.Trader, "__init__", _with_risk(engine.Trader.__init__))
    return calls


def _with_risk(init):
    """Give every trader a Risk Manager configuration, as a complete workstation would."""

    def wrapped(self, *args, **kwargs):
        from kairos_autotrader.risk import RiskConfig

        init(self, *args, **kwargs)
        self.risk = RiskConfig()

    return wrapped
