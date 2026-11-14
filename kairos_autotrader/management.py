"""Position management a professional would use, applied only where it measurably helps.

A ``Management`` policy moves the stop toward less risk (to break-even, or trailing the
best price) and may take part of the position at a profit. ``managed_outcome`` replays
one signal on later minute bars under a policy; with no management it gives exactly the
plan's result (``signals.outcome``). ``choose_policy`` compares every policy with the
plan on the same signals, paired, with errors clustered by instrument and day (the
signals of one instrument on one day overlap), and returns a policy only if its gain
over the plan is proven at the selection confidence. Otherwise positions keep the plan.

Iron rules, enforced here and again by the brokers: a stop only ever moves toward less
risk; a new stop is in place before the old one is removed; a partial exit never leaves
less than the instrument's minimum lot.
"""

from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MINUTE_MS = 60_000
DAY_MS = 86_400_000
POLICY_FILE = "management-policy.json"
# Several policies are compared with the plan, so the selection asks for 99% one-sided
# confidence, not 95%.
SELECTION_Z = 2.326
MIN_SIGNALS = 200
MAX_POLICY_AGE_MS = 35 * DAY_MS


@dataclass(frozen=True, slots=True)
class Management:
    name: str
    breakeven_at_r: float | None = None
    trail_start_r: float | None = None
    trail_distance_r: float | None = None
    partial_at_r: float | None = None
    partial_fraction: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Management:
        policy = cls(**{key: data[key] for key in cls.__dataclass_fields__ if key in data})
        policy.check()
        return policy

    def check(self) -> None:
        if self.trail_start_r is not None and not (self.trail_distance_r and self.trail_distance_r > 0):
            raise ValueError("a trailing stop needs a positive distance")
        if self.partial_at_r is not None and not 0 < self.partial_fraction < 1:
            raise ValueError("a partial exit takes a fraction strictly between 0 and 1")
        for value in (self.breakeven_at_r, self.trail_start_r, self.partial_at_r):
            if value is not None and value <= 0:
                raise ValueError("management levels are positive R multiples")


PLAN = Management("plan")


def policies_for(reward_risk: float) -> tuple[Management, ...]:
    """The compared policies, placed at fractions of the distance to the take.

    Levels at or beyond the take would never act, so the grid follows the plan's reward.
    """

    def r(fraction: float) -> float:
        return round(reward_risk * fraction, 4)

    third, half, two_thirds = r(1 / 3), r(1 / 2), r(2 / 3)
    return (
        Management(f"breakeven_at_{third}r", breakeven_at_r=third),
        Management(f"breakeven_at_{two_thirds}r", breakeven_at_r=two_thirds),
        Management(f"trail_from_{third}r_by_{third}r", trail_start_r=third, trail_distance_r=third),
        Management(f"trail_from_{third}r_by_{two_thirds}r", trail_start_r=third, trail_distance_r=two_thirds),
        Management(f"trail_from_{two_thirds}r_by_{third}r", trail_start_r=two_thirds, trail_distance_r=third),
        Management(
            f"half_at_{half}r_then_breakeven", partial_at_r=half, partial_fraction=0.5, breakeven_at_r=half
        ),
        Management(
            f"half_at_{two_thirds}r_then_breakeven",
            partial_at_r=two_thirds,
            partial_fraction=0.5,
            breakeven_at_r=two_thirds,
        ),
    )


def tighter(side: str, current: float, proposed: float) -> float:
    """The stop with less risk: the higher one for a long, the lower one for a short."""
    return max(current, proposed) if side == "LONG" else min(current, proposed)


def managed_stop(policy: Management, side: str, entry: float, initial_stop: float, best: float) -> float:
    """The stop the policy wants given the best price reached since entry (never looser than initial)."""
    risk = abs(entry - initial_stop)
    if risk <= 0:
        return initial_stop
    sign = 1.0 if side == "LONG" else -1.0
    gain_r = sign * (best - entry) / risk
    stop = initial_stop
    if policy.breakeven_at_r is not None and gain_r >= policy.breakeven_at_r:
        stop = tighter(side, stop, entry)
    if policy.trail_start_r is not None and policy.trail_distance_r and gain_r >= policy.trail_start_r:
        stop = tighter(side, stop, best - sign * policy.trail_distance_r * risk)
    return stop


def partial_level(policy: Management, side: str, entry: float, initial_stop: float) -> float | None:
    if policy.partial_at_r is None:
        return None
    sign = 1.0 if side == "LONG" else -1.0
    return entry + sign * policy.partial_at_r * abs(entry - initial_stop)


def managed_outcome(
    record: Mapping[str, Any], later_bars: Iterable[Any], *, cost_fraction: float, policy: Management = PLAN
) -> dict[str, Any]:
    """One signal's result in R under a management policy, net of the round-trip cost.

    Conservative inside a bar: the stop in force is checked first, then a partial exit and
    the take; the stop only moves after the bar closes, from that bar's best price.
    """
    entry, stop0, take = float(record["ref_price"]), float(record["stop"]), float(record["take"])
    risk = abs(entry - stop0)
    if risk <= 0:
        return {"reason": "INVALID", "r": None}
    side = record["side"]
    sign = 1.0 if side == "LONG" else -1.0
    start = int(record["signal_time"])
    deadline = start + int(record["timeout_minutes"]) * MINUTE_MS
    level = partial_level(policy, side, entry, stop0)
    stop, best, remaining, realized, reason = stop0, entry, 1.0, 0.0, "OPEN"
    exit_ms = None
    for bar in later_bars:
        exit_ms = bar.t
        if bar.t < start:
            continue
        if bar.l <= stop if sign > 0 else bar.h >= stop:
            realized += remaining * sign * (stop - entry) / risk
            reason = "STOP" if stop == stop0 else "MANAGED_STOP"
            break
        if level is not None and remaining == 1.0 and (bar.h >= level if sign > 0 else bar.l <= level):
            if (take - level) * sign > 0:
                realized += policy.partial_fraction * float(policy.partial_at_r or 0)
                remaining -= policy.partial_fraction
        if bar.h >= take if sign > 0 else bar.l <= take:
            realized += remaining * sign * (take - entry) / risk
            reason = "TAKE"
            break
        if bar.t >= deadline:
            realized += remaining * sign * (bar.c - entry) / risk
            reason = "TIMEOUT"
            break
        best = max(best, bar.h) if sign > 0 else min(best, bar.l)
        stop = tighter(side, stop, managed_stop(policy, side, entry, stop0, best))
        if remaining < 1.0 and policy.breakeven_at_r is not None:
            stop = tighter(side, stop, entry)
    else:
        return {"reason": reason, "r": None}
    return {"reason": reason, "r": round(realized - cost_fraction * entry / risk, 4), "exit_ms": exit_ms}


def scale_in_evidence(
    records: list[dict[str, Any]],
    candles_by_symbol: Mapping[str, list[Any]],
    cost_fraction: float,
    *,
    min_open_gain_r: float,
) -> dict[str, Any]:
    """Do new signals that agree with an open winner earn on their own? (Adding to winners.)

    Per instrument, a flat book takes the first signal as a base position held to its plan.
    While it is open, the first later signal on the same side, at a moment the base shows
    at least ``min_open_gain_r``, is an add-on; its own plan result is measured. One add-on
    per base, as the runner would do. Errors are clustered by instrument and day.
    """
    addons: list[float] = []
    groups: list[str] = []
    for symbol, candles in candles_by_symbol.items():
        times = [c.t for c in candles]
        base: dict[str, Any] | None = None
        open_until, added = -1, False
        for rec in sorted((r for r in records if r["symbol"] == symbol), key=lambda r: r["signal_time"]):
            begin = bisect_left(times, rec["signal_time"])
            window = candles[begin : begin + int(rec["timeout_minutes"]) + 2]
            if base is None or rec["signal_time"] >= open_until:
                result = managed_outcome(rec, window, cost_fraction=cost_fraction)
                if result["r"] is None:
                    continue
                base, open_until, added = rec, int(result["exit_ms"]), False
                continue
            if added or rec["side"] != base["side"] or begin == 0:
                continue
            sign = 1.0 if base["side"] == "LONG" else -1.0
            price = candles[begin - 1].c
            gain = (
                sign
                * (price - float(base["ref_price"]))
                / abs(float(base["ref_price"]) - float(base["stop"]))
            )
            if gain < min_open_gain_r:
                continue
            result = managed_outcome(rec, window, cost_fraction=cost_fraction)
            if result["r"] is None:
                continue
            added = True
            addons.append(float(result["r"]))
            groups.append(f"{symbol}:{int(rec['signal_time']) // DAY_MS}")
    if len(addons) < 2:
        return {"min_open_gain_r": min_open_gain_r, "signals": len(addons), "proven": False}
    mean, error = _clustered(addons, groups)
    lower = mean - SELECTION_Z * error
    return {
        "min_open_gain_r": min_open_gain_r,
        "signals": len(addons),
        "clusters": len(set(groups)),
        "mean_r": round(mean, 4),
        "error_r": round(error, 4),
        "lower_r": round(lower, 4),
        "proven": len(addons) >= MIN_SIGNALS // 2 and lower > 0,
    }


def _clustered(values: list[float], clusters: list[str]) -> tuple[float, float]:
    """Mean and its standard error with clusters of correlated values (CR1)."""
    n = len(values)
    mean = sum(values) / n
    sums: dict[str, float] = defaultdict(float)
    for value, cluster in zip(values, clusters, strict=True):
        sums[cluster] += value - mean
    count = len(sums)
    if count < 2:
        return mean, math.inf
    variance = sum(total**2 for total in sums.values()) / n**2 * count / (count - 1)
    return mean, math.sqrt(variance)


def scan_cadence_summary(
    records: list[dict[str, Any]],
    candles_by_symbol: Mapping[str, list[Any]],
    cost_fraction: float,
    *,
    cooldown_minutes: int,
    days: float,
) -> dict[str, Any]:
    """How the live rules do when decided at a given cadence: every signal, and the trades taken.

    Taken trades follow the runner: one position per instrument, none while one is open
    and none during the cooldown after its exit. Errors are clustered by instrument and day.
    """
    times = {symbol: [c.t for c in candles] for symbol, candles in candles_by_symbol.items()}
    every: list[tuple[str, float]] = []
    taken: list[tuple[str, float]] = []
    busy_until: dict[str, int] = {}
    for record in sorted(records, key=lambda r: (r["signal_time"], r["id"])):
        candles = candles_by_symbol.get(record["symbol"])
        if not candles:
            continue
        begin = bisect_left(times[record["symbol"]], record["signal_time"])
        window = candles[begin : begin + int(record["timeout_minutes"]) + 2]
        result = managed_outcome(record, window, cost_fraction=cost_fraction)
        if result["r"] is None:
            continue
        cluster = f"{record['symbol']}:{int(record['signal_time']) // DAY_MS}"
        every.append((cluster, float(result["r"])))
        if record["signal_time"] >= busy_until.get(record["symbol"], -1):
            taken.append((cluster, float(result["r"])))
            busy_until[record["symbol"]] = int(result["exit_ms"]) + cooldown_minutes * MINUTE_MS

    def stats(rows: list[tuple[str, float]]) -> dict[str, Any]:
        if len(rows) < 2:
            return {"signals": len(rows)}
        mean, error = _clustered([r for _, r in rows], [c for c, _ in rows])
        return {
            "signals": len(rows),
            "per_day": round(len(rows) / days, 1),
            "mean_r": round(mean, 4),
            "error_r": round(error, 4),
            "total_r": round(sum(r for _, r in rows), 2),
            "wins": sum(r > 0 for _, r in rows),
        }

    return {"every_signal": stats(every), "taken": stats(taken)}


def compare_policies(
    records: list[dict[str, Any]],
    candles_by_symbol: Mapping[str, list[Any]],
    cost_fraction: float,
    policies: tuple[Management, ...],
) -> list[dict[str, Any]]:
    """Each policy's paired gain over the plan on the same finished signals."""
    times = {symbol: [c.t for c in candles] for symbol, candles in candles_by_symbol.items()}
    base: dict[str, float] = {}
    windows: dict[str, list[Any]] = {}
    clusters: dict[str, str] = {}
    for record in records:
        candles = candles_by_symbol.get(record["symbol"])
        if not candles:
            continue
        begin = bisect_left(times[record["symbol"]], record["signal_time"])
        window = candles[begin : begin + int(record["timeout_minutes"]) + 2]
        plan = managed_outcome(record, window, cost_fraction=cost_fraction)
        if plan["r"] is None:
            continue
        base[record["id"]] = plan["r"]
        windows[record["id"]] = window
        clusters[record["id"]] = f"{record['symbol']}:{int(record['signal_time']) // DAY_MS}"
    by_id = {record["id"]: record for record in records}
    results = []
    for policy in policies:
        deltas, groups, values = [], [], []
        for key, plan_r in base.items():
            managed = managed_outcome(by_id[key], windows[key], cost_fraction=cost_fraction, policy=policy)
            if managed["r"] is None:
                continue
            deltas.append(managed["r"] - plan_r)
            values.append(managed["r"])
            groups.append(clusters[key])
        if len(deltas) < 2:
            continue
        gain, error = _clustered(deltas, groups)
        results.append(
            {
                "policy": policy.as_dict(),
                "signals": len(deltas),
                "clusters": len(set(groups)),
                "plan_mean_r": round(sum(base[k] for k in base) / len(base), 4),
                "mean_r": round(sum(values) / len(values), 4),
                "gain_r": round(gain, 4),
                "gain_error_r": round(error, 4),
                "gain_lower_r": round(gain - SELECTION_Z * error, 4),
            }
        )
    return results


def choose_policy(results: list[dict[str, Any]], *, min_signals: int = MIN_SIGNALS) -> dict[str, Any] | None:
    """The policy with the best proven gain over the plan, or None (keep the plan)."""
    proven = [r for r in results if r["signals"] >= min_signals and r["gain_lower_r"] > 0]
    return max(proven, key=lambda r: r["gain_lower_r"]) if proven else None


def write_policy(
    directory: Path,
    results: list[dict[str, Any]],
    *,
    now_ms: int,
    reward_risk: float,
    scale_in: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    chosen = choose_policy(results)
    proven_scale_in = [item for item in scale_in or [] if item.get("proven")]
    document = {
        "version": "kairos.management-policy.v1",
        "reward_risk": reward_risk,
        # Adding to an open winner: the lowest proven open-gain threshold, or None.
        "scale_in": min(proven_scale_in, key=lambda item: item["min_open_gain_r"])
        if proven_scale_in
        else None,
        "scale_in_evidence": scale_in or [],
        "measured_at_ms": now_ms,
        "measured_utc": datetime.fromtimestamp(now_ms / 1000, tz=UTC).isoformat(timespec="seconds"),
        "selection_z": SELECTION_Z,
        "min_signals": MIN_SIGNALS,
        "policy": chosen["policy"] if chosen else None,
        "evidence": results,
    }
    path = directory / POLICY_FILE
    partial = path.with_suffix(".partial")
    partial.write_text(json.dumps(document, indent=1), encoding="utf-8")
    partial.replace(path)
    return document


def scale_in_status(directory: Path, *, now_ms: int, reward_risk: float) -> str:
    """The measured verdict on adding to an open winner, in words for the journal.

    The runner keeps one position per instrument either way: EVEDEX nets an instrument
    into one position with full-position protection, so an add-on would need per-leg
    stops and takes, which are built only once adding is proven.
    """
    path = directory / POLICY_FILE
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "adding to it is not measured yet"
    if (
        float(document.get("reward_risk", -1)) != float(reward_risk)
        or now_ms - int(document.get("measured_at_ms", 0)) > MAX_POLICY_AGE_MS
    ):
        return "adding to it is not measured for the current plan"
    proven = document.get("scale_in")
    if not proven:
        return "adding to it is not proven profitable"
    return (
        f"adding from +{proven['min_open_gain_r']} R is proven, but per-leg protection "
        "for add-ons is not built yet"
    )


def load_policy(directory: Path, *, now_ms: int, reward_risk: float) -> tuple[Management | None, str]:
    """The measured policy for open positions, or None with the reason the plan is kept."""
    path = directory / POLICY_FILE
    if not path.exists():
        return None, "no management measurement yet"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != "kairos.management-policy.v1":
            return None, "unknown management measurement version"
        if float(document["reward_risk"]) != float(reward_risk):
            return None, "the management was measured for another take distance"
        if now_ms - int(document["measured_at_ms"]) > MAX_POLICY_AGE_MS:
            return None, "the management measurement is older than 35 days"
        if not document.get("policy"):
            return None, "no policy beat the plan with proof"
        return Management.from_dict(document["policy"]), "measured"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"management measurement unreadable: {type(exc).__name__}"
