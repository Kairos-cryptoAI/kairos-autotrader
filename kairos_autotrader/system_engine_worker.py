"""Runs the real Kairos Strategy Engine on closed 1m bars and prints its output as JSON.

This file is executed as a script by the Strategy Engine's own Python environment, with
the engine's source directory on ``PYTHONPATH``. It imports nothing from the autotrader,
does no network or file access and changes no engine code: it only calls the engine's
pure generators and serialises what they return.

Registered sleeves (shadow and replay):

stdin:  {"symbol": "SOLUSD", "strategies": ["trend_pullback_reclaim_v1", ...],
         "rows": [[open_time_ms, open, high, low, close, base_volume, quote_volume], ...]}
stdout: {"results": {"<strategy>": {"revision": "1", "status": "...", "seconds": 0.4,
                                    "intents": [...]} | {"error": "..."}}}

The live rules, ``kairos_strategy.hourly_trend``:

stdin:  {"op": "hourly_trend", "series": {"SOLUSD": [[...], ...], ...}, "all": false,
         "config": {"target_reward_to_risk": 1.5, "max_hold_minutes": 720}}
stdout: {"identity": {"strategy_id": ..., "revision": ..., "strategy_code_sha256": ...,
                      "config_sha256": ...},
         "results": {"SOLUSD": {"state": {...} | null, "intent": {...} | null}
                               | {"intents": [...]}      (with "all": true)
                               | {"error": "..."}}}
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from typing import Any

MINUTE_MS = 60_000
MARKER = "@@KAIROS-WORKER@@"


def _candles(candle_type: Any, symbol: str, rows: list) -> list:
    return [
        candle_type(
            symbol=symbol,
            timeframe="1m",
            open_time_ms=int(row[0]),
            close_time_ms=int(row[0]) + MINUTE_MS - 1,  # the engine uses inclusive close times
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            quote_volume=float(row[6]),
        )
        for row in rows
    ]


def _intent(intent: Any) -> dict:
    return {
        "intent_id": intent.intent_id,
        "sleeve_id": intent.sleeve_id,
        "symbol": intent.symbol,
        "side": str(getattr(intent.side, "value", intent.side)).upper(),
        "decision_ts_ms": intent.decision_ts_ms,
        "entry_eligible_ts_ms": intent.entry_eligible_ts_ms,
        "entry_expires_ts_ms": intent.entry_expires_ts_ms,
        "reference_price": intent.reference_price,
        "signal_strength": intent.signal_strength,
        "gross_reward_bps": intent.gross_reward_bps,
        "stop_price": intent.exit_plan.stop_price,
        "target_price": intent.exit_plan.target_price,
        "max_holding_ms": intent.exit_plan.max_holding_ms,
        "trailing_activation_price": intent.exit_plan.trailing_activation_price,
        "trailing_distance": intent.exit_plan.trailing_distance,
        "metadata": dict(intent.metadata),
    }


def _registered(request: dict, candle_type: Any) -> dict:
    from kairos_strategy import registry

    candles = _candles(candle_type, str(request["symbol"]), request["rows"])
    results: dict[str, dict] = {}
    for strategy_id in request["strategies"]:
        started = time.perf_counter()
        try:
            definition = registry.get_strategy(strategy_id)
            intents = registry.generate_sleeve_intents(strategy_id, candles)
        except Exception as exc:  # noqa: BLE001 - one broken sleeve must not hide the others
            results[strategy_id] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            continue
        results[strategy_id] = {
            "revision": str(definition.revision),
            "status": str(getattr(definition.status, "value", definition.status)),
            "seconds": round(time.perf_counter() - started, 3),
            "intents": [_intent(intent) for intent in intents],
        }
    return {"results": results}


def _hourly_trend(request: dict, candle_type: Any) -> dict:
    from kairos_strategy.hourly_trend import (
        HourlyTrendCandidatesConfig,
        evaluate_hourly_trend_state,
        generate_hourly_trend_candidates_intents,
        hourly_trend_source_identity,
    )

    # Only fields of the engine's own validated config are accepted; the identity reports it.
    config = HourlyTrendCandidatesConfig(**dict(request.get("config") or {}))
    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        try:
            candles = _candles(candle_type, str(symbol), rows)
            if request.get("all"):
                found = generate_hourly_trend_candidates_intents(candles, config)
                results[symbol] = {"intents": [_intent(intent) for intent in found]}
            else:
                state, intent = evaluate_hourly_trend_state(candles, config)
                results[symbol] = {
                    "state": None if state is None else asdict(state),
                    "intent": None if intent is None else _intent(intent),
                }
        except Exception as exc:  # noqa: BLE001 - one broken instrument must not hide the others
            results[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"identity": asdict(hourly_trend_source_identity(config)), "results": results}


def main() -> int:
    # argv[1] is the engine's source directory. It replaces this script's own directory
    # on the import path, so nothing from the autotrader package can shadow engine imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        from kairos_strategy.candles import Candle

        answer = (
            _hourly_trend(request, Candle)
            if request.get("op") == "hourly_trend"
            else _registered(request, Candle)
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
