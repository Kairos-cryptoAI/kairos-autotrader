"""Bridge to the real Kairos Strategy Engine (repository ``kairos-strategy-engine``).

The engine's generators are run unchanged, in the engine's own Python environment, by
``system_engine_worker.py``. This module feeds it closed 1m bars from EVEDEX, keeps a
local cache of those bars and turns the engine's output into the feature and candidate
types the rest of the runner understands.

Two things come from the engine:

- the live rules, ``kairos_strategy.hourly_trend`` (``hourly_trend_scan``), pinned by the
  engine's own source identity of that package;
- the registered research sleeves, for the shadow recorder and the replay (``collect``).

Nothing here places orders or needs a key.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .candidates import Candidate, Features
from .evedex import Candle, Evedex, EvedexError
from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python

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
HOURLY_TREND_CODE_SHA256 = "5600f453953b2404b64e9a20445d9736240b02bee6fce6d0a7a6b161f52c8045"
HOURLY_TREND_HISTORY_MS = 504 * HOUR_MS  # how far back from a decision the rules read
HOURLY_TREND_CACHE_DAYS = 22
_RATIONALE = {
    ("LONG", "PULLBACK_RECLAIM"): (
        "higher-timeframe uptrend; price dipped to the EMA20 and closed back above it"
    ),
    ("LONG", "BREAKOUT"): "higher-timeframe uptrend; close above the 20-bar high",
    ("SHORT", "PULLBACK_RECLAIM"): (
        "higher-timeframe downtrend; price rallied to the EMA20 and closed back below it"
    ),
    ("SHORT", "BREAKOUT"): "higher-timeframe downtrend; close below the 20-bar low",
}


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
        if not (self.source / "kairos_strategy" / "hourly_trend" / "rules.py").exists():
            raise EngineError(
                f"Strategy Engine checkout {self.source} has no kairos_strategy.hourly_trend package"
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


def intent_to_candidate(intent: dict[str, Any], *, revision: str, status: str) -> Candidate:
    """An engine intent as an autotrader candidate. Side, stop, target and hold time are kept as is.

    The engine closes its bars at an inclusive last millisecond, so the signal time is one
    millisecond later. A trailing stop in the intent is not carried: the autotrader has
    fixed exits only, and that difference is written into the rationale.
    """
    side = str(intent["side"]).upper()
    if side not in ("LONG", "SHORT"):
        raise EngineError(f"unexpected intent side {side!r}")
    ref, stop, take = (
        float(intent["reference_price"]),
        float(intent["stop_price"]),
        float(intent["target_price"]),
    )
    ordered = stop < ref < take if side == "LONG" else take < ref < stop
    if not ordered:
        raise EngineError("intent prices are not ordered stop / reference / target")
    decision = int(intent["decision_ts_ms"])
    sleeve = str(intent["sleeve_id"])
    note = "; engine trailing stop not applied" if intent.get("trailing_distance") else ""
    return Candidate(
        id=f"{intent['symbol']}-{side}-{sleeve}-{decision}",
        symbol=str(intent["symbol"]),
        side=side,
        setup=sleeve,
        bar_time=decision + 1 - MINUTE_MS,
        signal_time=decision + 1,
        ref_price=ref,
        stop=stop,
        take=take,
        timeout_minutes=max(1, int(intent["max_holding_ms"]) // MINUTE_MS),
        atr=abs(ref - stop),  # the stop distance stands in for ATR in the entry drift check
        score=round(float(intent["signal_strength"]), 4),
        rationale=f"Kairos Strategy Engine {sleeve} rev {revision}, engine status {status}{note}",
    )


def candidates_from(results: dict[str, Any], *, since_ms: int = 0) -> tuple[list[Candidate], list[str]]:
    """Candidates decided after ``since_ms`` and a list of sleeve problems, if any."""
    candidates: list[Candidate] = []
    problems: list[str] = []
    for sleeve, result in results.items():
        if "error" in result:
            problems.append(f"{sleeve}: {result['error'][:200]}")
            continue
        for intent in result.get("intents", []):
            if int(intent["decision_ts_ms"]) < since_ms:
                continue
            try:
                candidates.append(
                    intent_to_candidate(
                        intent, revision=str(result.get("revision")), status=str(result.get("status"))
                    )
                )
            except (EngineError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{sleeve}: bad intent ({exc})")
    candidates.sort(key=lambda c: (c.signal_time, c.id))
    return candidates, problems


async def collect(
    api: Evedex,
    store: MinuteStore,
    config: EngineConfig,
    symbols: list[str],
    now_ms: int,
    *,
    since_ms: int = 0,
    runner=_run_worker,
) -> tuple[list[Candidate], list[str], dict[str, list]]:
    """Update bars and run the engine for every instrument. Returns candidates, problems and the bars."""
    gate = asyncio.Semaphore(3)
    bars: dict[str, list] = {}

    async def one(symbol: str):
        async with gate:
            rows = await store.update(api, symbol, now_ms, config.window_days)
        bars[symbol] = rows
        if len(rows) < 2:
            return [], [f"{symbol}: no minute history"]
        results = await run_engine(config, symbol, rows, runner)
        found, problems = candidates_from(results, since_ms=since_ms)
        missing = gaps(rows)
        if missing:
            problems.append(f"{symbol}: {missing} gaps in minute history")
        return found, [f"{symbol} {p}" if not p.startswith(symbol) else p for p in problems]

    outcomes = await asyncio.gather(*(one(s) for s in symbols), return_exceptions=True)
    candidates: list[Candidate] = []
    problems: list[str] = []
    for symbol, outcome in zip(symbols, outcomes, strict=False):
        if isinstance(outcome, (EvedexError, EngineError, OSError)):
            problems.append(f"{symbol}: {type(outcome).__name__}: {str(outcome)[:200]}")
        elif isinstance(outcome, BaseException):
            raise outcome
        else:
            candidates.extend(outcome[0])
            problems.extend(outcome[1])
    candidates.sort(key=lambda c: (c.signal_time, c.id))
    return candidates, problems, bars


# ---- the live rules: kairos_strategy.hourly_trend -----------------------------------


def fill_quiet_minutes(rows: list[list[float]]) -> tuple[list[list[float]], int]:
    """Give the engine the complete minute series its bar builder requires.

    EVEDEX history has no row for a minute without trades, while a minute polled live comes
    back as a zero-volume row with transient prices. Both are treated the same way, so the
    series does not depend on how it was collected: a minute without trades inside a
    quarter hour that traded becomes a flat zero-volume bar at the previous close, the way
    Binance publishes it. A quarter hour without a single trade stays missing (the venue
    has no 15-minute candle for it either), and the rules skip the decisions it touches.
    """
    rows = [row for row in rows if row[5] > 0]
    present = {int(row[0]) // QUARTER_MS for row in rows}
    out: list[list[float]] = []
    added = 0
    for row in rows:
        if out:
            minute = int(out[-1][0]) + MINUTE_MS
            while minute < int(row[0]):
                if minute // QUARTER_MS in present:
                    close = out[-1][4]
                    out.append([minute, close, close, close, close, 0.0, 0.0])
                    added += 1
                minute += MINUTE_MS
        out.append(row)
    return out, added


def decision_window(rows: list[list[float]], boundary_ms: int) -> tuple[list[list[float]], int]:
    """The closed minutes the rules read for the decision at ``boundary_ms``, gaps filled."""
    first = boundary_ms - HOURLY_TREND_HISTORY_MS
    return fill_quiet_minutes([row for row in rows if first <= row[0] < boundary_ms])


def _run_hourly_trend(config: EngineConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(config.python, WORKER, [str(config.source)], request, timeout_s=config.timeout_s)
    except SiblingError as exc:
        raise EngineError(f"Strategy Engine: {exc}") from exc


def hourly_trend_request(
    series: dict[str, list[list[float]]], *, reward_risk: float, timeout_minutes: int, every: bool = False
) -> dict[str, Any]:
    return {
        "op": "hourly_trend",
        "series": series,
        "all": every,
        "config": {"target_reward_to_risk": float(reward_risk), "max_hold_minutes": int(timeout_minutes)},
    }


def checked_hourly_answer(answer: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    """Identity and per-instrument results, refused unless the engine source is the pinned one."""
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


def state_to_features(state: dict[str, Any]) -> tuple[Features, dict[str, Any]]:
    """The engine's state as the runner's features, plus the descriptive context for the reviewer."""

    def rounded(name: str, digits: int) -> float | None:
        return None if state[name] is None else round(float(state[name]), digits)

    features = Features(
        symbol=str(state["symbol"]),
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


def hourly_intent_to_candidate(intent: dict[str, Any]) -> Candidate:
    """An intent of the live rules as a runner candidate; prices and hold time are kept as is.

    The candidate keeps the setup name (``PULLBACK_RECLAIM`` or ``BREAKOUT``) that Macro,
    Risk and the reports know. The engine's inclusive close becomes the next millisecond.
    """
    side, setup = str(intent["side"]).upper(), str(intent["metadata"]["setup"])
    if (side, setup) not in _RATIONALE:
        raise EngineError(f"unexpected intent {side} {setup}")
    ref, stop, take = (
        float(intent["reference_price"]),
        float(intent["stop_price"]),
        float(intent["target_price"]),
    )
    if not (stop < ref < take if side == "LONG" else take < ref < stop):
        raise EngineError("intent prices are not ordered stop / reference / target")
    boundary = int(intent["decision_ts_ms"]) + 1
    bar_time = boundary - HOUR_MS
    return Candidate(
        id=f"{intent['symbol']}-{side}-{setup}-{bar_time}",
        symbol=str(intent["symbol"]),
        side=side,
        setup=setup,
        bar_time=bar_time,
        signal_time=boundary,
        ref_price=ref,
        stop=stop,
        take=take,
        timeout_minutes=max(1, int(intent["max_holding_ms"]) // MINUTE_MS),
        atr=float(intent["metadata"]["atr"]),
        score=round(float(intent["signal_strength"]), 4),
        rationale=_RATIONALE[(side, setup)],
    )


@dataclass(frozen=True, slots=True)
class TrendScan:
    """What the live rules said at one decision, for every instrument that had fresh bars."""

    identity: dict[str, str]
    features: dict[str, Features]
    context: dict[str, dict[str, Any]]
    candidates: list[Candidate]
    skipped: list[str]


async def hourly_trend_scan(
    config: EngineConfig,
    series: dict[str, list[list[float]]],
    boundary_ms: int,
    *,
    reward_risk: float,
    timeout_minutes: int,
    runner=None,
) -> TrendScan:
    """Run the live rules for the decision at ``boundary_ms``.

    An instrument whose newest closed quarter hour is not the one ending at the boundary is
    skipped: its history is late or has a hole next to the decision.
    """
    request = hourly_trend_request(series, reward_risk=reward_risk, timeout_minutes=timeout_minutes)
    answer = await asyncio.to_thread(runner or _run_hourly_trend, config, request)
    identity, results = checked_hourly_answer(answer)
    features: dict[str, Features] = {}
    context: dict[str, dict[str, Any]] = {}
    candidates: list[Candidate] = []
    skipped: list[str] = []
    for symbol in series:
        result = results.get(symbol)
        if not isinstance(result, dict) or "error" in result:
            reason = str(result.get("error"))[:200] if isinstance(result, dict) else "no answer"
            skipped.append(f"{symbol}: {reason}")
            continue
        state = result.get("state")
        if not isinstance(state, dict) or state.get("decision_ts_ms") != boundary_ms - 1:
            skipped.append(f"{symbol}: no fresh closed bar")
            continue
        try:
            found_features, found_context = state_to_features(state)
            intent = result.get("intent")
            candidate = None
            if intent is not None:
                if int(intent["decision_ts_ms"]) != boundary_ms - 1:
                    raise EngineError("intent belongs to another decision")
                candidate = hourly_intent_to_candidate(intent)
        except (EngineError, KeyError, TypeError, ValueError) as exc:
            skipped.append(f"{symbol}: bad engine output ({type(exc).__name__}: {str(exc)[:120]})")
            continue
        features[symbol], context[symbol] = found_features, found_context
        if candidate is not None:
            candidates.append(candidate)
    return TrendScan(identity, features, context, candidates, skipped)


def hourly_trend_history(
    config: EngineConfig,
    series: dict[str, list[list[float]]],
    *,
    reward_risk: float,
    timeout_minutes: int,
    runner=None,
) -> tuple[list[Candidate], list[str]]:
    """Every candidate the live rules produce over a whole minute history, for the replay."""
    filled = {symbol: fill_quiet_minutes(rows)[0] for symbol, rows in series.items()}
    request = hourly_trend_request(
        filled, reward_risk=reward_risk, timeout_minutes=timeout_minutes, every=True
    )
    _, results = checked_hourly_answer((runner or _run_hourly_trend)(config, request))
    candidates: list[Candidate] = []
    problems: list[str] = []
    for symbol in series:
        result = results.get(symbol)
        if not isinstance(result, dict) or "error" in result:
            problems.append(f"{symbol}: {result.get('error') if isinstance(result, dict) else 'no answer'}")
            continue
        for intent in result.get("intents", []):
            try:
                candidates.append(hourly_intent_to_candidate(intent))
            except (EngineError, KeyError, TypeError, ValueError) as exc:
                problems.append(f"{symbol}: bad intent ({exc})")
    candidates.sort(key=lambda c: (c.signal_time, c.id))
    return candidates, problems
