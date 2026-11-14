"""Bridge to the real Kairos Strategy Engine (repository ``kairos-strategy-engine``).

The engine's code is run unchanged, in the engine's own Python environment, by
``system_engine_worker.py``.

Two things come from the engine:

- the live rules, ``kairos_strategy.hourly_trend``, through the engine's strict adapter:
  closed Binance USD-M bars in, a ``StrategyIntentV1`` out (``hourly_trend_scan``). The
  package is pinned by the engine's own source identity, and every intent is checked to
  declare exactly the bars of the runner's cache;
- the registered research sleeves on EVEDEX minute bars, for the shadow recorder and the
  replay (``collect``). Their output stays the engine's research ``SleeveIntent``.

Nothing here places orders or needs a key.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairos_core.contracts import StrategyIntentV1
from pydantic import ValidationError

from .evedex import Candle, Evedex, EvedexError
from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .signals import Features, signal_symbol

MINUTE_MS = 60_000
QUARTER_MS = 15 * MINUTE_MS
HOUR_MS = 60 * MINUTE_MS
DAY_MS = 86_400_000
WORKER = Path(__file__).with_name("system_engine_worker.py")

# Sleeves of the engine that need only prices and volumes, which EVEDEX provides.
USABLE_SLEEVES = (
    "trend_pullback_reclaim_v1",
    "range_mean_reversion_v1",
    "right_tail_trend_v1",
    "regime_aligned_right_tail_v1",
)
# Registered in the engine but not run here, with the reason.
EXCLUDED_SLEEVES = {
    "trend_breakout_v1": "нужен объём покупок по рынку, которого в истории EVEDEX нет",
    "regime_veto_retest_reclaim_v1": "нужен объём покупок по рынку, которого в истории EVEDEX нет",
    "quarter_hour_flow_v1": "нужен объём покупок по рынку, которого в истории EVEDEX нет",
    "orderflow_volatility_expansion_v1": "нужен объём покупок по рынку, которого в истории EVEDEX нет",
    "crowded_trend_continuation_v1": "нужна история ставки финансирования и открытого интереса",
    "donchian_ensemble_long_v1": "выдаёт доли портфеля, а не сделки",
    "four_hour_sma200_long_v1": "выдаёт доли портфеля, а не сделки",
}


# The live rules. The pin is the engine's own fingerprint of the package's source closure
# (kairos_strategy/hourly_trend/provenance.py): a different engine source is refused, not traded.
HOURLY_TREND_ID = "hourly_trend_candidates_v1"
HOURLY_TREND_CODE_SHA256 = "1390b94bcf38440efa2f3b8b466c10bc1044e60252ed7c9979b25453db9b59ad"
HOURLY_TREND_HISTORY_MS = 504 * HOUR_MS  # how far back from a decision the rules read


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EngineConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-strategy-engine"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-strategy-engine"))
    sleeves: tuple[str, ...] = USABLE_SLEEVES
    window_days: int = 36  # regime_aligned_right_tail_v1 needs 200 complete 4h bars
    timeout_s: float = 180.0

    def check(self) -> None:
        if not self.python.exists():
            raise EngineError(f"Strategy Engine Python not found: {self.python}")
        if not (self.source / "kairos_strategy" / "registry.py").exists():
            raise EngineError(f"Strategy Engine source not found: {self.source}")
        unknown = [s for s in self.sleeves if s not in USABLE_SLEEVES]
        if unknown:
            raise EngineError("sleeves not usable on EVEDEX data: " + ", ".join(unknown))

    def check_hourly_trend(self) -> None:
        if not self.python.exists():
            raise EngineError(f"Strategy Engine Python not found: {self.python}")
        if not (self.source / "kairos_strategy" / "hourly_trend" / "adapter.py").exists():
            raise EngineError(
                f"Strategy Engine checkout {self.source} has no strict kairos_strategy.hourly_trend adapter"
            )


def engine_revision(source: Path) -> str:
    """Git revision of the engine source, with a mark if it has local changes."""
    return git_revision(source, "kairos_strategy")


@dataclass(slots=True)
class MinuteStore:
    """Closed 1m bars per instrument, cached on disk so restarts do not refetch weeks of history."""

    directory: Path
    _rows: dict[str, list[list[float]]] = field(default_factory=dict)

    def _path(self, symbol: str) -> Path:
        return self.directory / f"{symbol}.1m.json"

    def _load(self, symbol: str) -> list[list[float]]:
        if symbol not in self._rows:
            path = self._path(symbol)
            try:
                self._rows[symbol] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            except ValueError:
                self._rows[symbol] = []
        return self._rows[symbol]

    async def update(self, api: Evedex, symbol: str, now_ms: int, days: int) -> list[list[float]]:
        """Bring the cache up to the last closed minute and return the window, oldest first."""
        self.directory.mkdir(parents=True, exist_ok=True)
        first = (now_ms - days * DAY_MS) // MINUTE_MS * MINUTE_MS
        rows = [r for r in self._load(symbol) if r[0] >= first]
        if rows and rows[0][0] - first > DAY_MS:
            # A longer window than the cache was built for: fetch the missing older part once.
            older = await api.candle_rows(symbol, "1m", first, int(rows[0][0]))
            rows = [r for r in older if first <= r[0] < rows[0][0]] + rows
        start = int(rows[-1][0]) + MINUTE_MS if rows else first
        if start < now_ms:
            fresh = await api.candle_rows(symbol, "1m", start, now_ms)
            known = {int(r[0]) for r in rows[-5:]}
            rows.extend(r for r in fresh if r[0] >= start and int(r[0]) not in known)
        rows = [r for r in rows if r[0] + MINUTE_MS <= now_ms]
        rows.sort(key=lambda r: r[0])
        self._rows[symbol] = rows
        temp = self._path(symbol).with_suffix(".tmp")
        temp.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, self._path(symbol))
        return rows


def gaps(rows: list[list[float]]) -> int:
    """Number of places where consecutive bars are not one minute apart."""
    return sum(1 for a, b in zip(rows, rows[1:], strict=False) if int(b[0]) - int(a[0]) != MINUTE_MS)


def to_candles(rows: list[list[float]]) -> list[Candle]:
    return [Candle(int(r[0]), r[1], r[2], r[3], r[4], r[6]) for r in rows]


def _run_worker(config: EngineConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        answer = run_worker(config.python, WORKER, [str(config.source)], request, timeout_s=config.timeout_s)
        return answer["results"]
    except SiblingError as exc:
        raise EngineError(f"Strategy Engine: {exc}") from exc
    except (KeyError, TypeError) as exc:
        raise EngineError("Strategy Engine worker returned malformed output") from exc


async def run_engine(
    config: EngineConfig, symbol: str, rows: list[list[float]], runner=_run_worker
) -> dict[str, Any]:
    """Intents of every configured sleeve for one instrument's closed 1m bars."""
    request = {"symbol": symbol, "strategies": list(config.sleeves), "rows": rows}
    return await asyncio.to_thread(runner, config, request)


def sleeve_record(intent: dict[str, Any], *, instrument: str | None = None, note: str = "") -> dict[str, Any]:
    """A research ``SleeveIntent`` of the engine as a signal record for reports and the replay.

    Side, prices and hold time are kept as they are. The engine closes its bars at an
    inclusive last millisecond, so the signal time is one millisecond later. A trailing
    stop is not applied by the replay; that difference is written into the record.
    """
    side = str(intent["side"]).upper()
    if side not in ("LONG", "SHORT"):
        raise EngineError(f"unexpected intent side {side!r}")
    ref, stop, take = (
        float(intent["reference_price"]),
        float(intent["stop_price"]),
        float(intent["target_price"]),
    )
    if not (stop < ref < take if side == "LONG" else take < ref < stop):
        raise EngineError("intent prices are not ordered stop / reference / target")
    decision = int(intent["decision_ts_ms"])
    sleeve = str(intent["sleeve_id"])
    metadata = intent.get("metadata") or {}
    setup = str(metadata.get("setup") or sleeve)
    symbol = instrument or str(intent["symbol"])
    if intent.get("trailing_distance"):
        note = (note + "; " if note else "") + "engine trailing stop not applied"
    return {
        "id": f"{symbol}-{side}-{setup}-{decision}",
        "symbol": symbol,
        "side": side,
        "setup": setup,
        "signal_time": decision + 1,
        "ref_price": ref,
        "stop": stop,
        "take": take,
        "timeout_minutes": max(1, int(intent["max_holding_ms"]) // MINUTE_MS),
        # Without an ATR in the metadata the stop distance stands in for it.
        "atr": float(metadata["atr"]) if "atr" in metadata else abs(ref - stop),
        "score": round(float(intent["signal_strength"]), 4),
        "strategy_id": sleeve,
        "note": note,
    }


def records_from(results: dict[str, Any], *, since_ms: int = 0) -> tuple[list[dict[str, Any]], list[str]]:
    """Signal records decided after ``since_ms`` and a list of sleeve problems, if any."""
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for sleeve, result in results.items():
        if "error" in result:
            problems.append(f"{sleeve}: {result['error'][:200]}")
            continue
        note = f"Kairos Strategy Engine rev {result.get('revision')}, engine status {result.get('status')}"
        for intent in result.get("intents", []):
            if int(intent["decision_ts_ms"]) < since_ms:
                continue
            try:
                records.append(sleeve_record(intent, note=note))
            except (EngineError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{sleeve}: bad intent ({exc})")
    records.sort(key=lambda r: (r["signal_time"], r["id"]))
    return records, problems


async def collect(
    api: Evedex,
    store: MinuteStore,
    config: EngineConfig,
    symbols: list[str],
    now_ms: int,
    *,
    since_ms: int = 0,
    runner=_run_worker,
) -> tuple[list[dict[str, Any]], list[str], dict[str, list]]:
    """Update EVEDEX bars and run the registered sleeves for every instrument.

    Returns signal records, problems and the bars.
    """
    gate = asyncio.Semaphore(3)
    bars: dict[str, list] = {}

    async def one(symbol: str):
        async with gate:
            rows = await store.update(api, symbol, now_ms, config.window_days)
        bars[symbol] = rows
        if len(rows) < 2:
            return [], [f"{symbol}: no minute history"]
        results = await run_engine(config, symbol, rows, runner)
        found, problems = records_from(results, since_ms=since_ms)
        missing = gaps(rows)
        if missing:
            problems.append(f"{symbol}: {missing} gaps in minute history")
        return found, [f"{symbol} {p}" if not p.startswith(symbol) else p for p in problems]

    outcomes = await asyncio.gather(*(one(s) for s in symbols), return_exceptions=True)
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for symbol, outcome in zip(symbols, outcomes, strict=False):
        if isinstance(outcome, (EvedexError, EngineError, OSError)):
            problems.append(f"{symbol}: {type(outcome).__name__}: {str(outcome)[:200]}")
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            records.extend(outcome[0])
            problems.extend(outcome[1])
    records.sort(key=lambda r: (r["signal_time"], r["id"]))
    return records, problems, bars


# ---- the live rules: kairos_strategy.hourly_trend -----------------------------------


def _run_hourly_trend(config: EngineConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(config.python, WORKER, [str(config.source)], request, timeout_s=config.timeout_s)
    except SiblingError as exc:
        raise EngineError(f"Strategy Engine: {exc}") from exc


ADAPTIVE_HISTORY_MS = 3_240 * MINUTE_MS


async def adaptive_shadow_scan(
    config: EngineConfig,
    series: dict[str, list[list[Any]]],
    boundary_ms: int,
    *,
    observed_at_ms: int,
    source_set_sha256: str,
    runner=None,
) -> dict[str, Any]:
    """The research ``adaptive`` strategy on the same closed bars, in its DRY_RUN observation mode.

    Its intents are recorded and measured, never traded: the Strategy Engine allows it
    PAPER or LIVE only after the owner lifts that gate, and the learning ledger has to show
    a proven edge first.
    """
    first_ms = boundary_ms - ADAPTIVE_HISTORY_MS
    windows = {
        signal_symbol(instrument): [row for row in bars if first_ms <= row[0] < boundary_ms]
        for instrument, bars in series.items()
    }
    request = {
        "op": "adaptive_bars",
        "series": {symbol: rows for symbol, rows in windows.items() if rows},
        "observed_at_ms": int(observed_at_ms),
        "source_set_sha256": source_set_sha256,
    }
    answer = await asyncio.to_thread(runner or _run_hourly_trend, config, request)
    if "error" in answer or not isinstance(answer.get("results"), dict):
        raise EngineError(f"adaptive observation failed: {str(answer.get('error', 'malformed'))[:200]}")
    return answer


def hourly_trend_config(*, reward_risk: float, timeout_minutes: int) -> dict[str, Any]:
    """The two controls the runner sets; the engine validates and fingerprints them."""
    return {"target_reward_to_risk": float(reward_risk), "max_hold_minutes": int(timeout_minutes)}


def checked_hourly_answer(answer: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Identity and per-symbol results, refused unless the engine source is the pinned one."""
    identity, results = answer.get("identity"), answer.get("results")
    if not isinstance(identity, dict) or not isinstance(results, dict):
        raise EngineError("Strategy Engine worker returned malformed output")
    if identity.get("strategy_id") != HOURLY_TREND_ID:
        raise EngineError(f"unexpected strategy {identity.get('strategy_id')!r}")
    if identity.get("strategy_code_sha256") != HOURLY_TREND_CODE_SHA256:
        raise EngineError(
            "the Strategy Engine source of hourly_trend differs from the pinned one "
            f"({str(identity.get('strategy_code_sha256'))[:12]} instead of {HOURLY_TREND_CODE_SHA256[:12]})"
        )
    return {key: str(value) for key, value in identity.items()}, results


def state_to_features(state: dict[str, Any], instrument: str) -> tuple[Features, dict[str, Any]]:
    """The engine's state as the reviewer's features, plus its descriptive context."""

    def rounded(name: str, digits: int) -> float | None:
        return None if state[name] is None else round(float(state[name]), digits)

    features = Features(
        symbol=instrument,
        price=float(state["price"]),
        trend=str(state["trend"]),
        trend_ema20=float(state["trend_ema_fast"]),
        trend_ema50=float(state["trend_ema_slow"]),
        trend_efficiency=float(state["trend_efficiency"]),
        rsi=float(state["rsi"]),
        atr=float(state["atr"]),
        atr_pct=float(state["atr_pct"]),
        change_4h_pct=float(state["change_4h_pct"]),
        change_24h_pct=float(state["change_24h_pct"]),
        dist_ema20_atr=float(state["dist_ema_atr"]),
    )
    context = {
        "change_1h_pct": rounded("change_1h_pct", 3),
        "change_7d_pct": rounded("change_7d_pct", 2),
        "position_in_7d_range": rounded("position_in_7d_range", 3),
        "last_bar_volume_vs_20bar_avg": rounded("last_bar_volume_vs_20bar_avg", 2),
        "consecutive_up_bars": int(state["consecutive_up_bars"]),
        "consecutive_down_bars": int(state["consecutive_down_bars"]),
    }
    return features, context


def checked_intent(
    raw: Any, *, symbol: str, boundary_ms: int, window: list[list[Any]], identity: dict[str, str]
) -> StrategyIntentV1:
    """The engine's strict intent, validated by the contract and tied to the cached bars."""
    intent = StrategyIntentV1.model_validate(raw)
    if intent.strategy_id != HOURLY_TREND_ID or intent.symbol != symbol:
        raise EngineError("intent belongs to another strategy or symbol")
    if intent.decision_ts_ms != boundary_ms - 1:
        raise EngineError("intent belongs to another decision")
    provenance = intent.provenance
    if (
        provenance.strategy_code_sha256 != HOURLY_TREND_CODE_SHA256
        or provenance.config_sha256 != identity.get("config_sha256")
    ):
        raise EngineError("intent provenance differs from the pinned engine identity")
    if provenance.input_bar_sha256s != tuple(row[9] for row in window):
        raise EngineError("intent does not declare exactly the cached closed bars")
    return intent


@dataclass(frozen=True, slots=True)
class TrendScan:
    """What the live rules said at one decision, per EVEDEX instrument that had fresh bars."""

    identity: dict[str, str]
    features: dict[str, Features]
    context: dict[str, dict[str, Any]]
    intents: dict[str, StrategyIntentV1]
    skipped: list[str]


async def hourly_trend_scan(
    config: EngineConfig,
    series: dict[str, list[list[Any]]],
    boundary_ms: int,
    *,
    reward_risk: float,
    timeout_minutes: int,
    runner=None,
    processes: int = 4,
    price_increments: dict[str, Any] | None = None,
) -> TrendScan:
    """Run the live rules for the decision at ``boundary_ms`` on closed Binance bars.

    ``series`` maps an EVEDEX instrument to the cached bars of its Binance signal symbol.
    An instrument whose newest closed bar does not end at the boundary is skipped: its
    history is late. Building some 30 000 strict bars per symbol takes about a second,
    so the symbols are spread over a few engine processes. ``price_increments`` (EVEDEX
    instrument to its price tick) makes the rules place stop and target on that tick.
    """
    first_ms = boundary_ms - HOURLY_TREND_HISTORY_MS
    windows: dict[str, list[list[Any]]] = {}
    skipped: list[str] = []
    for instrument, bars in series.items():
        window = [row for row in bars if first_ms <= row[0] < boundary_ms]
        if not window or int(window[-1][0]) != boundary_ms - MINUTE_MS:
            skipped.append(f"{instrument}: no fresh closed bar")
        elif int(window[0][0]) != first_ms:
            # A shorter window would still be declared exactly, but it is not the rules' window.
            skipped.append(f"{instrument}: closed bars do not reach back the whole history window")
        else:
            windows[instrument] = window
    run = runner or _run_hourly_trend
    names = list(windows)
    groups = [names[index::processes] for index in range(max(1, min(processes, len(names))))]
    tuning = hourly_trend_config(reward_risk=reward_risk, timeout_minutes=timeout_minutes)
    ticks = dict(price_increments or {})

    def request(group: list[str]) -> dict[str, Any]:
        return {
            "op": "hourly_trend_bars",
            "series": {signal_symbol(instrument): windows[instrument] for instrument in group},
            "config": tuning,
            "price_increments": {
                signal_symbol(instrument): str(ticks[instrument])
                for instrument in group
                if instrument in ticks
            },
        }

    answers = await asyncio.gather(*(asyncio.to_thread(run, config, request(group)) for group in groups))
    identity: dict[str, str] = {}
    results: dict[str, Any] = {}
    for answer in answers:
        found_identity, found = checked_hourly_answer(answer)
        if identity and found_identity != identity:
            raise EngineError("engine processes reported different identities")
        identity = found_identity
        results.update(found)
    features: dict[str, Features] = {}
    context: dict[str, dict[str, Any]] = {}
    intents: dict[str, StrategyIntentV1] = {}
    for instrument in names:
        symbol = signal_symbol(instrument)
        result = results.get(symbol)
        if not isinstance(result, dict) or "error" in result:
            reason = str(result.get("error"))[:200] if isinstance(result, dict) else "no answer"
            skipped.append(f"{instrument}: {reason}")
            continue
        state = result.get("state")
        if result.get("decision_ts_ms") != boundary_ms - 1:
            skipped.append(f"{instrument}: the engine decided on another clock")
            continue
        if not isinstance(state, dict):
            skipped.append(f"{instrument}: history is too short for the rules")
            continue
        try:
            found_features, found_context = state_to_features(state, instrument)
            intent = None
            if result.get("intent") is not None:
                intent = checked_intent(
                    result["intent"],
                    symbol=symbol,
                    boundary_ms=boundary_ms,
                    window=windows[instrument],
                    identity=identity,
                )
        except (EngineError, ValidationError, KeyError, TypeError, ValueError) as exc:
            skipped.append(f"{instrument}: bad engine output ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        features[instrument], context[instrument] = found_features, found_context
        if intent is not None:
            intents[instrument] = intent
    return TrendScan(identity, features, context, intents, skipped)


def hourly_trend_history(
    config: EngineConfig,
    series: dict[str, list[list[Any]]],
    *,
    reward_risk: float,
    timeout_minutes: int,
    runner=None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Every candidate the live rules produce over a whole bar history, for the replay.

    ``series`` maps an EVEDEX instrument to the bars of its Binance signal symbol. The
    replay reads the engine's research ``SleeveIntent``: a strict intent per candidate
    would hash the whole window thousands of times and add nothing to a report.
    """
    request = {
        "op": "hourly_trend_history",
        "series": {signal_symbol(instrument): bars for instrument, bars in series.items()},
        "config": hourly_trend_config(reward_risk=reward_risk, timeout_minutes=timeout_minutes),
    }
    _, results = checked_hourly_answer((runner or _run_hourly_trend)(config, request))
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for instrument in series:
        result = results.get(signal_symbol(instrument))
        if not isinstance(result, dict) or "error" in result:
            reason = result.get("error") if isinstance(result, dict) else "no answer"
            problems.append(f"{instrument}: {reason}")
            continue
        for intent in result.get("intents", []):
            try:
                records.append(sleeve_record(intent, instrument=instrument))
            except (EngineError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{instrument}: bad intent ({exc})")
    records.sort(key=lambda r: (r["signal_time"], r["id"]))
    return records, problems
