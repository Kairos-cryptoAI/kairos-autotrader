"""Runs the real Kairos Macro Strategist and Risk Manager rules and prints JSON.

Executed as a script by the respective repository's own Python environment.

Macro (``kairos-macro-strategist``), unchanged code used: the allocation system prompt,
the context assembler, the ``AllocationOutput`` schema and its conversion into the
``StrategicAllocation`` contract, the deterministic defensive fallback and the price
shock detector. The regime hint is the service's strict-majority rule, written out here
because the original is a method bound to the running service.

Risk (``kairos-risk-manager``), unchanged code used: ``RiskSettings`` limits, the
allocation gate (regime vetoes, strategy weight, allocation age), loss-at-stop sizing
(``_size_quantity``), the fee and slippage economics and the open-risk measure of a
position. The account gate is written out here with the same formulas and the same
settings, because the original needs an ``AccountSnapshotV2`` that only a PAPER account
on the EVEDEX test venue can produce. Gates that exist only to keep the test profile
closed (strategy promotion allowlist, DEV-only venue, PAPER-only account) are not applied
and are reported as such.

argv: macro <macro source> <kairos-llm source> <kairos-core source>
      risk  <risk source>
stdin: {"op": ...}; stdout: a marker line followed by JSON.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

MARKER = "@@KAIROS-WORKER@@"
NOT_APPLIED = [
    "strategy_not_paper_approved (promotion allowlist of the test profile)",
    "venue must be EVEDEX DEV",
    "account must be the PAPER test account",
]


# ---- Macro ---------------------------------------------------------------------


def macro_prepare(request: dict) -> dict:
    from kairos_macro.config import MacroSettings, validate_strategy_ids
    from kairos_macro.context import build_macro_context
    from kairos_macro.prompts import allocation_system_prompt
    from kairos_macro.triggers import ShockDetector

    defaults = {name: field.default for name, field in MacroSettings.model_fields.items()}
    allowed = validate_strategy_ids(tuple(request["allowed_strategy_ids"]))
    biases = request.get("biases", {})
    values = list(biases.values())
    long_count, short_count = values.count("LONG"), values.count("SHORT")
    majority = len(values) // 2 + 1
    hint = "BULL" if long_count >= majority else "BEAR" if short_count >= majority else "CHOP"
    evidence = {
        "method": "strict_majority_of_fresh_quant_biases",
        "counts": {"LONG": long_count, "SHORT": short_count, "FLAT": values.count("FLAT")},
        "fresh_market_count": len(values),
        "required_majority": majority if values else 1,
    }
    market = request.get("market_values", {})
    detector = ShockDetector(defaults["crash_pct_1h"])
    shocks = []
    for symbol, change in sorted((request.get("changes_1h_pct") or {}).items()):
        event = detector.check_price(float(change))
        if event is not None:
            shocks.append(
                {"symbol": symbol, "kind": event.kind, "detail": event.detail, "severity": event.severity}
            )
    context = build_macro_context(
        portfolio=request.get("portfolio", {}),
        performance=request.get("performance", {}),
        regime_hint=hint,
        regime_evidence=evidence,
        market_factors={
            "status": "available" if market else "unavailable",
            "source": "EVEDEX public candles and instrument metrics, via the autotrader scan",
            "coverage": {
                "fresh_symbols": sorted(market),
                "configured_symbols": sorted(request.get("symbols", [])),
            },
            "units": request.get("units", {}),
            "values": market,
        },
        macro_factors={"status": "unavailable", "reason": "no structured macro-release source is connected"},
        onchain_factors={"status": "unavailable", "reason": "no structured on-chain source is connected"},
        trigger=request.get("trigger", {}),
    )
    return {
        "system": allocation_system_prompt(allowed),
        "user": context,
        "regime_hint": hint,
        "regime_evidence": evidence,
        "shocks": shocks,
        "shock_cooldown_s": defaults["shock_cooldown_s"],
        "allowed_strategy_ids": list(allowed),
    }


def macro_validate(request: dict) -> dict:
    from kairos_core.contracts import StrategicAllocation
    from kairos_core.enums import StrategicTrigger
    from kairos_macro.strategist import AllocationOutput, MacroStrategist

    allowed = tuple(request["allowed_strategy_ids"])
    trigger = StrategicTrigger(request.get("trigger", "schedule"))
    strategist = MacroStrategist(None, allowed_strategy_ids=allowed)
    origin = "model"
    try:
        output = AllocationOutput.model_validate(request["model_output"])
        allocation = StrategicAllocation(
            source=strategist.source,
            regime=output.regime,
            stable_reserve_pct=output.stable_reserve_pct,
            strategy_weights=output.strategy_weight_map(strategist.allowed_strategy_ids),
            max_gross_leverage=output.max_gross_leverage,
            triggered_by=trigger,
            rationale=output.rationale,
        )
    except Exception as exc:  # noqa: BLE001 - the strategist's own capital-preservation fallback
        origin = "defensive fallback"
        allocation = strategist.defensive(
            trigger, detail=f"invalid or failed model output ({type(exc).__name__})"
        )
    dumped = allocation.model_dump(mode="json")
    dumped["origin"] = origin
    return {"allocation": dumped}


# ---- Risk ----------------------------------------------------------------------


def _risk_one(pipeline, settings, item: dict, shared: dict, allocation, now_ms: int) -> dict:
    from kairos_core.enums import Side
    from kairos_risk.paper import PaperReservations, _AccountMetrics

    side = Side(item["side"])
    stop, target = float(item["stop"]), float(item["take"])
    bid, ask = float(item["bid"]), float(item["ask"])
    worst_entry = ask if side is Side.LONG else bid
    equity = float(shared["equity_usd"])
    reasons: list[str] = []

    positions = shared.get("positions", [])
    open_risk = sum(
        pipeline._position_stop_risk(
            SimpleNamespace(
                side=Side(p["side"]),
                signed_quantity=float(p["quantity"]),
                entry_price=float(p["entry_price"]),
                exit_plan=SimpleNamespace(stop_price=float(p["stop"])),
            )
        )
        for p in positions
    )
    gross = sum(abs(float(p["quantity"])) * float(p.get("mark_price", p["entry_price"])) for p in positions)
    strategy_notional = sum(
        abs(float(p["quantity"])) * float(p.get("mark_price", p["entry_price"]))
        for p in positions
        if p.get("strategy_id") == item["strategy_id"]
    )

    # Account gate, same formulas and settings as PaperRiskPipeline._account_gate.
    if any(p["symbol"] == item["symbol"] for p in positions):
        reasons.append("symbol_has_active_position")
    if open_risk >= equity * settings.paper_max_total_open_risk_fraction - 1e-9:
        reasons.append("portfolio_open_risk_limit_exhausted")
    day_start = float(shared.get("day_start_equity_usd") or equity)
    peak = float(shared.get("peak_equity_usd") or equity)
    drawdown = max(
        0.0,
        (day_start - equity) / day_start,
        -float(shared.get("daily_realized_pnl_usd", 0.0)) / day_start,
        (peak - equity) / peak,
    )
    if drawdown * 100 >= settings.max_daily_drawdown_pct:
        reasons.append("daily_drawdown_limit_reached")
    account = _AccountMetrics(
        equity_usd=equity,
        available_balance_usd=float(shared["available_usd"]),
        total_open_risk_usd=open_risk,
        gross_notional_usd=gross,
        strategy_notional_usd=strategy_notional,
    )

    allocation_reasons, allocation_metrics = pipeline._allocation_gate(
        allocation, strategy_id=item["strategy_id"], side=side, decided_at_ms=now_ms
    )
    reasons.extend(allocation_reasons)

    if side is Side.LONG and not stop < worst_entry < target:
        reasons.append("long_entry_exit_geometry_invalid")
    elif side is Side.SHORT and not target < worst_entry < stop:
        reasons.append("short_entry_exit_geometry_invalid")

    loss_budget, leverage, quantity = 0.0, 1.0, 0.0
    if allocation_metrics is not None:
        remaining = max(0.0, equity * settings.paper_max_total_open_risk_fraction - open_risk)
        loss_budget = min(equity * settings.paper_per_trade_risk_fraction, remaining)
        if loss_budget <= 0 and "portfolio_open_risk_limit_exhausted" not in reasons:
            reasons.append("portfolio_open_risk_limit_exhausted")
        leverage = min(settings.paper_max_leverage, allocation_metrics.max_gross_leverage)
        if leverage < 1:
            reasons.append("macro_leverage_below_one")
            leverage = 1.0

    depth = float(item["depth_usd"])
    venue = SimpleNamespace(
        buy_slippage_bps=float(item["slippage_bps"]),
        sell_slippage_bps=float(item["slippage_bps"]),
        taker_fee_bps=float(item["taker_fee_bps"]),
        venue_mid_price=(bid + ask) / 2,
        assessed_notional_usd=depth,
        depth_usd=depth,
    )
    if not reasons and allocation_metrics is not None:
        quantity = pipeline._size_quantity(
            intent_side=side,
            stop_price=stop,
            worst_entry=worst_entry,
            venue=venue,
            loss_budget=loss_budget,
            leverage=leverage,
            account=account,
            allocation=allocation_metrics,
            reservations=PaperReservations(),
        )
        if quantity <= 0:
            reasons.append("position_capacity_exhausted")
        elif quantity * worst_entry + 1e-9 < settings.paper_min_notional_usd:
            reasons.append("position_below_minimum_notional")
            quantity = 0.0
    fees, slippage, worst_loss = pipeline._economics(
        quantity=quantity, stop_price=stop, worst_entry=worst_entry, intent_side=side, venue=venue
    )
    if worst_loss > loss_budget + 1e-6:
        reasons.append("computed_loss_exceeds_budget")
        quantity = 0.0
        fees = slippage = worst_loss = 0.0
    approved = not reasons
    if not approved:
        quantity = 0.0
        fees = slippage = worst_loss = 0.0
    return {
        "id": item["id"],
        "approved": approved,
        "rejection_reasons": reasons,
        "quantity": quantity,
        "leverage": leverage,
        "notional_usd": quantity * worst_entry,
        "loss_budget_usd": loss_budget,
        "worst_case_loss_usd": worst_loss,
        "worst_entry_price": worst_entry,
        "estimated_fees_usd": fees,
        "estimated_slippage_usd": slippage,
        "open_risk_usd": open_risk,
    }


def risk_evaluate(request: dict) -> dict:
    from kairos_core.contracts import StrategicAllocation
    from kairos_risk.config import RiskSettings
    from kairos_risk.paper import PaperRiskPipeline

    settings = RiskSettings()
    pipeline = PaperRiskPipeline(settings)
    now_ms = int(request["now_ms"])
    allocation = None
    raw = request.get("allocation")
    if raw:
        try:
            allocation = StrategicAllocation.model_validate({k: v for k, v in raw.items() if k != "origin"})
        except Exception:  # noqa: BLE001 - an unreadable allocation is a missing one
            allocation = None
    decisions = []
    for item in request["candidates"]:
        try:
            decision = _risk_one(pipeline, settings, item, request["account"], allocation, now_ms)
            if not all(math.isfinite(float(decision[k])) for k in ("quantity", "worst_case_loss_usd")):
                raise ValueError("non-finite risk result")
        except Exception as exc:  # noqa: BLE001 - one candidate must not hide the others; fail closed
            decision = {
                "id": item.get("id"),
                "approved": False,
                "quantity": 0.0,
                "rejection_reasons": [f"risk_evaluation_error_{type(exc).__name__}"],
            }
        decisions.append(decision)
    return {
        "decisions": decisions,
        "not_applied": NOT_APPLIED,
        "limits": {
            "per_trade_risk_fraction": settings.paper_per_trade_risk_fraction,
            "max_total_open_risk_fraction": settings.paper_max_total_open_risk_fraction,
            "max_leverage": settings.paper_max_leverage,
            "min_notional_usd": settings.paper_min_notional_usd,
            "max_daily_drawdown_pct": settings.max_daily_drawdown_pct,
            "allocation_max_age_s": settings.strategic_allocation_max_age_s,
        },
        "evaluated_at": datetime.fromtimestamp(now_ms / 1000, UTC).isoformat(),
    }


def main() -> int:
    kind = sys.argv[1]
    sys.path[0] = sys.argv[2]
    for extra in sys.argv[3:]:
        sys.path.insert(1, extra)
    request = json.load(sys.stdin)
    try:
        if kind == "macro":
            result = macro_prepare(request) if request.get("op") == "prepare" else macro_validate(request)
        else:
            result = risk_evaluate(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
