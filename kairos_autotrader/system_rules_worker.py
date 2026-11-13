"""Runs the real Kairos Macro Strategist and prints JSON.

Executed as a script by the Macro Strategist repository's own Python environment.

Macro (``kairos-macro-strategist``), unchanged code used: the allocation system prompt,
the context assembler, the ``AllocationOutput`` schema and its conversion into the
``StrategicAllocation`` contract, the deterministic defensive fallback and the price
shock detector. The regime hint is the service's strict-majority rule, written out here
because the original is a method bound to the running service.

argv: macro <macro source> <kairos-llm source> <kairos-core source>
stdin: {"op": ...}; stdout: a marker line followed by JSON.
"""

from __future__ import annotations

import json
import sys

MARKER = "@@KAIROS-WORKER@@"


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


def main() -> int:
    kind = sys.argv[1]
    sys.path[0] = sys.argv[2]
    for extra in sys.argv[3:]:
        sys.path.insert(1, extra)
    request = json.load(sys.stdin)
    try:
        if kind != "macro":
            raise ValueError(f"unknown rules worker {kind!r}")
        result = macro_prepare(request) if request.get("op") == "prepare" else macro_validate(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
