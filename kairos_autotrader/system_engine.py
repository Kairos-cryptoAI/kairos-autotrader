"""Bridge to the real Kairos Strategy Engine (repository ``kairos-strategy-engine``).

The engine's registered generators are run unchanged, in the engine's own Python
environment, by ``system_engine_worker.py``. This module feeds it closed 1m bars from
EVEDEX, keeps a local cache of those bars and turns the engine's intents into the
candidate type the rest of the autotrader understands.

Nothing here places orders or needs a key.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evedex import Candle, Evedex, EvedexError
from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .strategy import Candidate

MINUTE_MS = 60_000
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
