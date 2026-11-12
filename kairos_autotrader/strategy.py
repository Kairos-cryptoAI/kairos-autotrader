"""Chart analysis: indicators, market features and trade candidates.

The algorithms only *propose*. Every candidate carries a fixed stop, take and timeout
that nothing downstream (including the LLM) is allowed to change. The setups here are
plain trend-following rules; they are a starting point for measurement, not a proven edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .evedex import Candle

MIN_SIGNAL_BARS = 60
MIN_TREND_BARS = 60


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for value in values[1:]:
        out.append(out[-1] + alpha * (value - out[-1]))
    return out


def rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains = losses = 0.0
    for prev, cur in zip(closes[:period], closes[1 : period + 1], strict=False):
        change = cur - prev
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for prev, cur in zip(closes[period:-1], closes[period + 1 :], strict=False):
        change = cur - prev
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def atr(candles: list[Candle], period: int = 14) -> float:
    if len(candles) < 2:
        return 0.0
    ranges = []
    for prev, cur in zip(candles[:-1], candles[1:], strict=False):
        ranges.append(max(cur.h - cur.l, abs(cur.h - prev.c), abs(cur.l - prev.c)))
    window = ranges[-period:]
    return sum(window) / len(window)


def efficiency_ratio(closes: list[float], period: int = 20) -> float:
    """Net move divided by the path travelled: near 1 is a clean trend, near 0 is chop."""
    if len(closes) <= period:
        return 0.0
    window = closes[-period - 1 :]
    path = sum(abs(b - a) for a, b in zip(window[:-1], window[1:], strict=False))
    return abs(window[-1] - window[0]) / path if path else 0.0


def closed_only(candles: list[Candle], step_ms: int, now_ms: int) -> list[Candle]:
    """Drop the bar that is still forming."""
    return [c for c in candles if c.t + step_ms <= now_ms]


def aggregate(candles: list[Candle], step_ms: int, target_ms: int, offset_ms: int = 0) -> list[Candle]:
    """Merge bars into a larger timeframe, keeping only complete groups.

    ``offset_ms`` shifts the group boundaries: with 15-minute bars, a one-hour target and an
    offset of 15 minutes the hourly bars run from :15 to :15.
    """
    groups: dict[int, list[Candle]] = {}
    for candle in candles:
        groups.setdefault((candle.t - offset_ms) // target_ms * target_ms + offset_ms, []).append(candle)
    need = target_ms // step_ms
    return [
        Candle(start, g[0].o, max(c.h for c in g), min(c.l for c in g), g[-1].c, sum(c.v for c in g))
        for start, g in sorted(groups.items())
        if len(g) == need
    ]


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


def bar_context(bars: list[Candle]) -> dict:
    """Extra descriptive context from closed 1h bars. Shown to the reviewer; not used by the rules."""
    if len(bars) < 30:
        return {}
    last = bars[-1]
    week = bars[-168:]
    high, low = max(b.h for b in week), min(b.l for b in week)
    volumes = [b.v for b in bars[-21:-1]]
    average = sum(volumes) / len(volumes) if volumes else 0.0
    ups = 0
    for bar in reversed(bars[-12:]):
        if bar.c <= bar.o:
            break
        ups += 1
    downs = 0
    for bar in reversed(bars[-12:]):
        if bar.c >= bar.o:
            break
        downs += 1
    return {
        "change_1h_pct": round((last.c / bars[-2].c - 1) * 100, 3),
        "change_7d_pct": round((last.c / week[0].o - 1) * 100, 2),
        "position_in_7d_range": round((last.c - low) / (high - low), 3) if high > low else None,
        "last_bar_volume_vs_20bar_avg": round(last.v / average, 2) if average > 0 else None,
        "consecutive_up_bars": ups,
        "consecutive_down_bars": downs,
    }


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


def analyse(
    symbol: str,
    bars: list[Candle],
    trend_bars: list[Candle],
    *,
    reward_risk: float = 1.5,
    timeout_minutes: int = 720,
) -> tuple[Features | None, list[Candidate]]:
    """Features plus zero or more candidates from closed bars only.

    ``bars`` is the signal timeframe, ``trend_bars`` a higher timeframe for direction.
    """
    if len(bars) < MIN_SIGNAL_BARS or len(trend_bars) < MIN_TREND_BARS:
        return None, []
    step = bars[-1].t - bars[-2].t
    back_4h = min(len(bars) - 1, max(1, 4 * 3_600_000 // step))
    back_24h = min(len(bars) - 1, max(1, 24 * 3_600_000 // step))
    closes_h = [c.c for c in trend_bars]
    e20h, e50h = ema(closes_h, 20)[-1], ema(closes_h, 50)[-1]
    er = efficiency_ratio(closes_h, 20)
    if e20h > e50h and closes_h[-1] > e50h and er >= 0.2:
        trend = "UP"
    elif e20h < e50h and closes_h[-1] < e50h and er >= 0.2:
        trend = "DOWN"
    else:
        trend = "FLAT"

    closes = [c.c for c in bars]
    e20 = ema(closes, 20)
    last = bars[-1]
    a = atr(bars, 14)
    if a <= 0 or last.c <= 0:
        return None, []
    strength = rsi(closes, 14)
    features = Features(
        symbol=symbol,
        price=last.c,
        trend=trend,
        trend_ema20=e20h,
        trend_ema50=e50h,
        trend_efficiency=er,
        rsi=strength,
        atr=a,
        atr_pct=100.0 * a / last.c,
        change_4h_pct=100.0 * (last.c / bars[-1 - back_4h].c - 1.0),
        change_24h_pct=100.0 * (last.c / bars[-1 - back_24h].c - 1.0),
        dist_ema20_atr=(last.c - e20[-1]) / a,
    )

    candidates: list[Candidate] = []
    recent = bars[-3:]
    recent_ema = e20[-3:]
    prior_high = max(c.h for c in bars[-21:-1])
    prior_low = min(c.l for c in bars[-21:-1])

    def add(side: str, setup: str, raw_stop: float, why: str) -> None:
        distance = abs(last.c - raw_stop)
        distance = max(distance, 0.8 * a)
        if distance > 2.5 * a:
            return
        sign = 1.0 if side == "LONG" else -1.0
        candidates.append(
            Candidate(
                id=f"{symbol}-{side}-{setup}-{last.t}",
                symbol=symbol,
                side=side,
                setup=setup,
                bar_time=last.t,
                signal_time=last.t + (last.t - bars[-2].t),
                ref_price=last.c,
                stop=last.c - sign * distance,
                take=last.c + sign * reward_risk * distance,
                timeout_minutes=timeout_minutes,
                atr=a,
                score=round(er, 4),
                rationale=why,
            )
        )

    if trend == "UP":
        touched = any(bar.l <= level for bar, level in zip(recent, recent_ema, strict=False))
        if touched and last.c > e20[-1] and last.c > last.o and 35.0 <= strength <= 68.0:
            add(
                "LONG",
                "PULLBACK_RECLAIM",
                min(bar.l for bar in recent) - 0.25 * a,
                "higher-timeframe uptrend; price dipped to the EMA20 and closed back above it",
            )
        elif last.c > prior_high and strength < 75.0:
            add("LONG", "BREAKOUT", last.c - 1.2 * a, "higher-timeframe uptrend; close above the 20-bar high")
    elif trend == "DOWN":
        touched = any(bar.h >= level for bar, level in zip(recent, recent_ema, strict=False))
        if touched and last.c < e20[-1] and last.c < last.o and 32.0 <= strength <= 65.0:
            add(
                "SHORT",
                "PULLBACK_RECLAIM",
                max(bar.h for bar in recent) + 0.25 * a,
                "higher-timeframe downtrend; price rallied to the EMA20 and closed back below it",
            )
        elif last.c < prior_low and strength > 25.0:
            add(
                "SHORT",
                "BREAKOUT",
                last.c + 1.2 * a,
                "higher-timeframe downtrend; close below the 20-bar low",
            )
    return features, candidates


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
