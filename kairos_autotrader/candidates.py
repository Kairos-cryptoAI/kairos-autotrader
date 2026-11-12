"""The runner's local shapes for a market state and a trade candidate.

The rules that produce them live in the Strategy Engine (``kairos_strategy.hourly_trend``)
and are called through ``system_engine``. Every candidate carries a fixed stop, take and
timeout that nothing downstream (including the LLM) is allowed to change.

These dataclasses are a known deviation from ``StrategyIntentV1``; see docs/ALIGNMENT.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .evedex import Candle


@dataclass(frozen=True, slots=True)
class Features:
    symbol: str
    price: float
    trend: str  # UP, DOWN or FLAT, from the higher timeframe
    trend_ema20: float
    trend_ema50: float
    trend_efficiency: float
    rsi: float
    atr: float
    atr_pct: float
    change_4h_pct: float
    change_24h_pct: float
    dist_ema20_atr: float

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    symbol: str
    side: str  # LONG or SHORT
    setup: str
    bar_time: int
    signal_time: int  # close time of the signal bar, epoch ms
    ref_price: float
    stop: float
    take: float
    timeout_minutes: int
    atr: float
    score: float
    rationale: str

    def as_dict(self) -> dict:
        return {k: (round(v, 8) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def outcome(candidate: Candidate, later_bars: list[Candle], *, cost_fraction: float) -> dict:
    """Hypothetical result of a candidate on later bars, in R multiples net of costs.

    A bar that touches both levels counts as the stop. ``cost_fraction`` is the round
    trip fee plus slippage as a fraction of notional. Used only for reports.
    """
    entry, risk = candidate.ref_price, abs(candidate.ref_price - candidate.stop)
    sign = 1.0 if candidate.side == "LONG" else -1.0
    deadline = candidate.signal_time + candidate.timeout_minutes * 60_000
    cost_r = cost_fraction * entry / risk
    exit_price, reason = None, "OPEN"
    for bar in later_bars:
        if bar.t < candidate.signal_time:
            continue
        hit_stop = bar.l <= candidate.stop if sign > 0 else bar.h >= candidate.stop
        hit_take = bar.h >= candidate.take if sign > 0 else bar.l <= candidate.take
        if hit_stop:
            exit_price, reason = candidate.stop, "STOP"
            break
        if hit_take:
            exit_price, reason = candidate.take, "TAKE"
            break
        if bar.t >= deadline:
            exit_price, reason = bar.c, "TIMEOUT"
            break
    if exit_price is None:
        return {"reason": reason, "r": None}
    return {"reason": reason, "r": round(sign * (exit_price - entry) / risk - cost_r, 4)}
