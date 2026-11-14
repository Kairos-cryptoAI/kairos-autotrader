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
from decimal import Decimal
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


def _closed_bars(symbol: str, rows: list, candle_type: Any) -> list:
    """The strict closed bars of cached rows, each checked against its recorded identity."""
    from kairos_strategy.runtime import candle_to_closed_bar

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
    return bars


def _adaptive_bars(request: dict, candle_type: Any) -> dict:
    """The research ``adaptive`` strategy in its own DRY_RUN observation mode, per symbol."""
    from kairos_strategy.adaptive.adapter import evaluate_adaptive_closed_bars
    from kairos_strategy.adaptive.config import STRATEGY_ID, STRATEGY_REVISION, UNIVERSE

    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        if symbol not in UNIVERSE:
            continue
        try:
            evaluation = evaluate_adaptive_closed_bars(
                _closed_bars(symbol, rows, candle_type),
                observed_at_ms=int(request["observed_at_ms"]),
                source_set_sha256=str(request["source_set_sha256"]),
            )
            results[symbol] = {
                "status": evaluation.decision.status,
                "reason": evaluation.decision.reason,
                "regime": str(evaluation.decision.regime.value),
                "intent": None if evaluation.intent is None else evaluation.intent.model_dump(mode="json"),
            }
        except Exception as exc:  # noqa: BLE001 - one broken symbol must not hide the others
            results[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return {"strategy_id": STRATEGY_ID, "strategy_revision": STRATEGY_REVISION, "results": results}


def _hourly_trend_bars(request: dict, candle_type: Any) -> dict:
    from kairos_strategy.hourly_trend import (
        HourlyTrendCandidatesConfig,
        evaluate_hourly_trend_closed_bars,
        hourly_trend_source_identity,
    )

    # Only fields of the engine's own validated config are accepted; the identity reports it.
    config = HourlyTrendCandidatesConfig(**dict(request.get("config") or {}))
    increments = dict(request.get("price_increments") or {})
    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        try:
            bars = _closed_bars(str(symbol), rows, candle_type)
            tick = increments.get(symbol)
            evaluation = evaluate_hourly_trend_closed_bars(
                bars, config, price_increment=None if tick is None else Decimal(str(tick))
            )
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


def _hourly_trend_history_step(request: dict, candle_type: Any) -> dict:
    """Research only: the live rules decided every ``step_minutes`` instead of every quarter.

    The engine's rules read hourly bars that close at the decision. Live, the hour is
    rebuilt to close at each quarter; here it is rebuilt to close every 1, 5 or 15
    minutes and the engine's own decision function is applied unchanged. With a 15-minute
    step this reproduces ``generate_hourly_trend_candidates_intents`` exactly.
    """
    from kairos_strategy.hourly_trend import HourlyTrendCandidatesConfig, rules
    from kairos_strategy.timeframes import aggregate

    step = int(request["step_minutes"])
    if step not in (1, 5, 15):
        raise ValueError("step_minutes must be 1, 5 or 15")
    step_ms, hour_ms = step * MINUTE_MS, 60 * MINUTE_MS
    four_hours_ms, per_hour = 4 * hour_ms, 60 // step
    settings = rules._settings(HourlyTrendCandidatesConfig(**dict(request.get("config") or {})))
    results: dict[str, dict] = {}
    for symbol, rows in request["series"].items():
        try:
            candles = _candles(candle_type, str(symbol), rows)
            base = candles if step == 1 else aggregate(candles, f"{step}m")
            bars = [rules._Bar(c.open_time_ms, c.open, c.high, c.low, c.close, c.quote_volume) for c in base]
            phases = {}
            for offset in range(0, hour_ms, step_ms):
                buckets: dict[int, list] = {}
                for bar in bars:
                    buckets.setdefault((bar.open_time_ms - offset) // hour_ms * hour_ms + offset, []).append(
                        bar
                    )
                hours = [
                    rules._merge(items, opened)
                    for opened, items in sorted(buckets.items())
                    if len(items) == per_hour
                ]
                trend: dict[int, list] = {}
                for hour in hours:
                    trend.setdefault(hour.open_time_ms // four_hours_ms * four_hours_ms, []).append(hour)
                complete = [(opened, items) for opened, items in sorted(trend.items()) if len(items) == 4]
                phases[offset] = rules._Phase(
                    hours=hours,
                    hour_opens=[hour.open_time_ms for hour in hours],
                    trend_bars=[rules._merge(items, opened) for opened, items in complete],
                    trend_first_hour_opens=[items[0].open_time_ms for _, items in complete],
                    trend_last_hour_opens=[items[-1].open_time_ms for _, items in complete],
                )
            found = []
            for candle in base:
                offset = (candle.close_time_ms + 1) % hour_ms
                _, intent = rules._decide(str(symbol), candle.close_time_ms, phases[offset], settings)
                if intent is not None:
                    found.append(_intent(intent))
            results[symbol] = {"intents": found}
        except Exception as exc:  # noqa: BLE001 - one broken symbol must not hide the others
            results[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    from kairos_strategy.hourly_trend import hourly_trend_source_identity

    return {"identity": asdict(hourly_trend_source_identity(settings)), "results": results}


def main() -> int:
    # argv[1] is the engine's source directory. It replaces this script's own directory
    # on the import path, so nothing from the runner package can shadow engine imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        from kairos_strategy.candles import Candle

        operations = {
            "hourly_trend_bars": _hourly_trend_bars,
            "adaptive_bars": _adaptive_bars,
            "hourly_trend_history": _hourly_trend_history,
            "hourly_trend_history_step": _hourly_trend_history_step,
            None: _registered,
        }
        answer = operations[request.get("op")](request, Candle)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
