"""What the runner passes between its layers: the Kairos contracts themselves.

A trade candidate is the Router's ``CandidateRouteV1``, which carries the Strategy
Engine's immutable ``StrategyIntentV1``, together with the EVEDEX instrument the runner
trades it on. As in the Kairos design, the signal is computed on Binance USD-M bars and
the order goes to EVEDEX; the exit plan of the intent is used unchanged.

Nothing downstream, a model included, can change a side, a stop, a target or a timeout:
they are read from the contract every time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from kairos_core.contracts import CandidateRouteV1, StrategyIntentV1

from .evedex import Candle

MINUTE_MS = 60_000
SIGNAL_VENUE = "BINANCE_UM"


def signal_symbol(instrument: str) -> str:
    """The Binance USD-M symbol whose closed bars give the signal for an EVEDEX instrument."""
    if not instrument.isalnum() or not instrument.isupper() or not instrument.endswith("USD"):
        raise ValueError(f"no Binance signal symbol is known for instrument {instrument!r}")
    return instrument + "T"


@dataclass(frozen=True, slots=True)
class Features:
    """The market state the rules saw at a decision, labelled with the EVEDEX instrument.

    Prices are those of the signal venue (Binance USD-M). Evidence for a reviewer only.
    """

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
class Routed:
    """A routed strategy candidate and the EVEDEX instrument it is traded on."""

    instrument: str
    route: CandidateRouteV1

    def __post_init__(self) -> None:
        intent = self.route.intent
        if intent.intent_id is None or self.route.route_id is None:
            raise ValueError("a routed candidate needs its canonical identities")
        if signal_symbol(self.instrument) != intent.symbol:
            raise ValueError(f"{intent.symbol} is not the signal symbol of {self.instrument}")
        metadata = dict(intent.metadata)
        if not metadata.get("setup") or float(metadata.get("atr", "0")) <= 0:
            raise ValueError("the intent carries no setup name or no positive ATR")

    @property
    def intent(self) -> StrategyIntentV1:
        return self.route.intent

    @property
    def id(self) -> str:
        return str(self.intent.intent_id)

    @property
    def side(self) -> str:
        return self.intent.side.value

    @property
    def setup(self) -> str:
        return dict(self.intent.metadata)["setup"]

    @property
    def atr(self) -> float:
        return float(dict(self.intent.metadata)["atr"])

    @property
    def reference_price(self) -> float:
        return self.intent.reference_price

    @property
    def stop(self) -> float:
        return self.intent.exit_plan.stop_price

    @property
    def take(self) -> float:
        return self.intent.exit_plan.target_price

    @property
    def timeout_minutes(self) -> int:
        return max(1, self.intent.exit_plan.max_holding_ms // MINUTE_MS)

    @property
    def signal_time(self) -> int:
        """The first millisecond after the decision bar closed."""
        return self.intent.decision_ts_ms + 1

    @property
    def strength(self) -> float:
        return self.intent.signal_strength

    def record(self) -> dict[str, Any]:
        """The compact form written to the journal, the state file and reviewer prompts.

        The full contracts are megabytes (an intent declares every input bar), so records
        keep their identities and the fields a person or a report reads.
        """
        intent, route = self.intent, self.route
        return {
            "id": self.id,
            "symbol": self.instrument,
            "side": self.side,
            "setup": self.setup,
            "signal_time": self.signal_time,
            "ref_price": round(self.reference_price, 8),
            "stop": round(self.stop, 8),
            "take": round(self.take, 8),
            "timeout_minutes": self.timeout_minutes,
            "atr": round(self.atr, 8),
            "score": round(self.strength, 4),
            "signal_venue": intent.venue,
            "signal_symbol": intent.symbol,
            "strategy_id": intent.strategy_id,
            "strategy_revision": intent.strategy_revision,
            "entry_expires_ts_ms": intent.entry_expires_ts_ms,
            "strategy_code_sha256": intent.provenance.strategy_code_sha256,
            "config_sha256": intent.provenance.config_sha256,
            "input_window_sha256": intent.provenance.input_window_sha256,
            "input_bars": len(intent.provenance.input_bar_sha256s),
            "route_id": route.route_id,
            "review_tier": route.review_tier.value,
            "review_deadline_ms": route.review_deadline_ms,
            "text_evidence_ids": list(route.evidence_ids),
        }


def held_outcome(plan: Mapping[str, Any], later_bars: list[Candle], *, cost_fraction: float) -> float | None:
    """The R a position would have made had it been held to its plan, or None if not yet known.

    ``plan`` has the entry, the stop and take in force, the initial risk per unit
    (``risk_price``), the start and the deadline. A bar touching both levels counts as
    the stop. R is measured on the initial risk, so a stop moved to break-even counts 0 R.
    """
    entry, stop, take = float(plan["entry"]), float(plan["stop"]), float(plan["take"])
    risk = float(plan["risk_price"])
    if risk <= 0:
        return None
    sign = 1.0 if plan["side"] == "LONG" else -1.0
    start, deadline = int(plan["start_ms"]), int(plan["deadline_ms"])
    for bar in later_bars:
        if bar.t < start:
            continue
        if bar.l <= stop if sign > 0 else bar.h >= stop:
            price = stop
        elif bar.h >= take if sign > 0 else bar.l <= take:
            price = take
        elif bar.t >= deadline:
            price = bar.c
        else:
            continue
        return round(sign * (price - entry) / risk - cost_fraction * entry / risk, 4)
    return None


def outcome(record: Mapping[str, Any], later_bars: list[Candle], *, cost_fraction: float) -> dict:
    """Hypothetical result of a recorded signal on later bars, in R multiples net of costs.

    ``record`` is a signal as the journal holds it (``Routed.record`` or an older entry).
    A bar that touches both levels counts as the stop. ``cost_fraction`` is the round trip
    fee plus slippage as a fraction of notional. Used only for reports.
    """
    entry, stop, take = float(record["ref_price"]), float(record["stop"]), float(record["take"])
    risk = abs(entry - stop)
    sign = 1.0 if record["side"] == "LONG" else -1.0
    signal_time = int(record["signal_time"])
    deadline = signal_time + int(record["timeout_minutes"]) * MINUTE_MS
    cost_r = cost_fraction * entry / risk
    exit_price, reason = None, "OPEN"
    for bar in later_bars:
        if bar.t < signal_time:
            continue
        hit_stop = bar.l <= stop if sign > 0 else bar.h >= stop
        hit_take = bar.h >= take if sign > 0 else bar.l <= take
        if hit_stop:
            exit_price, reason = stop, "STOP"
            break
        if hit_take:
            exit_price, reason = take, "TAKE"
            break
        if bar.t >= deadline:
            exit_price, reason = bar.c, "TIMEOUT"
            break
    if exit_price is None:
        return {"reason": reason, "r": None}
    return {"reason": reason, "r": round(sign * (exit_price - entry) / risk - cost_r, 4)}
