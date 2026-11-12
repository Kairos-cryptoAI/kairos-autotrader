"""Runs the real Kairos Strategy Engine on closed 1m bars and prints its output as JSON.

This file is executed as a script by the Strategy Engine's own Python environment, with
the engine's source directory on ``PYTHONPATH``. It imports nothing from the runner,
does no network or file access and changes no engine code: it only calls the engine's
own functions and serialises what they return.

Registered sleeves (shadow and replay), research ``SleeveIntent`` output:

stdin:  {"symbol": "SOLUSD", "strategies": ["trend_pullback_reclaim_v1", ...],
         "rows": [[open_time_ms, open, high, low, close, base_volume, quote_volume], ...]}
stdout: {"results": {"<strategy>": {"revision": "1", "status": "...", "seconds": 0.4,
                                    "intents": [...]} | {"error": "..."}}}

The live rules, ``kairos_strategy.hourly_trend``, through the engine's strict adapter. Every
row is a cached Binance USD-M closed bar with the ``bar_sha256`` Quant Scouts gave it; the
bar is rebuilt with the engine's own ``candle_to_closed_bar`` and has to hash to the same
identity before the rules read it:

stdin:  {"op": "hourly_trend_bars",
         "series": {"SOLUSDT": [[open_time_ms, open, high, low, close, base_volume, quote_volume,
                                 taker_buy_base_volume, taker_buy_quote_volume, bar_sha256], ...]},
         "config": {"target_reward_to_risk": 1.5, "max_hold_minutes": 720}}
stdout: {"identity": {"strategy_id", "revision", "strategy_code_sha256", "config_sha256"},
         "results": {"SOLUSDT": {"decision_ts_ms", "input_first_open_time_ms", "input_bar_count",
                                 "state": {...} | null,
                                 "intent": <StrategyIntentV1 as JSON> | null}
                                | {"error": "..."}}}

The same rules over a whole history, research ``SleeveIntent`` output, for the replay:

stdin:  {"op": "hourly_trend_history", "series": {"SOLUSDT": [[...], ...]}, "config": {...}}
stdout: {"identity": {...}, "results": {"SOLUSDT": {"intents": [...]} | {"error": "..."}}}
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


def _hourly_trend_bars(request: dict, candle_type: Any) -> dict:
    from kairos_strategy.hourly_trend import (
        HourlyTrendCandidatesConfig,
        evaluate_hourly_trend_closed_bars,
        hourly_trend_source_identity,
    )
    from kairos_strategy.runtime import candle_to_closed_bar

    # Only fields of the engine's own validated config are accepted; the identity reports it.
    config = HourlyTrendCandidatesConfig(**dict(request.get("config") or {}))
    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        try:
            bars = []
            for row in rows:
                bar = candle_to_closed_bar(
                    candle_type(
                        symbol=str(symbol),
                        timeframe="1m",
                        open_time_ms=int(row[0]),
                        close_time_ms=int(row[0]) + MINUTE_MS - 1,
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                        quote_volume=float(row[6]),
                        taker_buy_volume=float(row[7]),
                        taker_buy_quote_volume=float(row[8]),
                    )
                )
                if bar.bar_sha256 != row[9]:
                    raise ValueError(f"cached bar {int(row[0])} does not hash to its recorded identity")
                bars.append(bar)
            evaluation = evaluate_hourly_trend_closed_bars(bars, config)
            results[symbol] = {
                "decision_ts_ms": evaluation.decision_ts_ms,
                "input_first_open_time_ms": evaluation.input_first_open_time_ms,
                "input_bar_count": evaluation.input_bar_count,
                "state": None if evaluation.state is None else asdict(evaluation.state),
                "intent": None if evaluation.intent is None else evaluation.intent.model_dump(mode="json"),
            }
        except Exception as exc:  # noqa: BLE001 - one broken symbol must not hide the others
            results[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"identity": asdict(hourly_trend_source_identity(config)), "results": results}


def _hourly_trend_history(request: dict, candle_type: Any) -> dict:
    from kairos_strategy.hourly_trend import (
        HourlyTrendCandidatesConfig,
        generate_hourly_trend_candidates_intents,
        hourly_trend_source_identity,
    )

    config = HourlyTrendCandidatesConfig(**dict(request.get("config") or {}))
    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        try:
            found = generate_hourly_trend_candidates_intents(_candles(candle_type, str(symbol), rows), config)
            results[symbol] = {"intents": [_intent(intent) for intent in found]}
        except Exception as exc:  # noqa: BLE001 - one broken symbol must not hide the others
            results[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"identity": asdict(hourly_trend_source_identity(config)), "results": results}


def main() -> int:
    # argv[1] is the engine's source directory. It replaces this script's own directory
    # on the import path, so nothing from the runner package can shadow engine imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        from kairos_strategy.candles import Candle

        operations = {
            "hourly_trend_bars": _hourly_trend_bars,
            "hourly_trend_history": _hourly_trend_history,
            None: _registered,
        }
        answer = operations[request.get("op")](request, Candle)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
