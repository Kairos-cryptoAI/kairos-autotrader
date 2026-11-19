"""The trading loop: scan, decide, size, enter, protect, exit, account.

A few positions at a time, never two in one instrument. Hard ceilings below are code
constants, not configuration: raising them is a deliberate code change, not a typo in a
JSON file.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from pathlib import Path
from typing import Any

import aiohttp
from kairos_core.contracts import StrategyIntentV1
from kairos_execution.one_canary_account import CanaryAccountError
from kairos_execution.portfolio_broker import NeedsOperator
from kairos_execution.portfolio_broker import PortfolioBroker as LiveBroker
from kairos_execution.portfolio_journal import PortfolioJournalError
from kairos_execution.portfolio_performance import read_pilot_status

from . import brain
from .aggregator_review import (
    OPENAI_MONTHLY_CEILING_USD,
    AggregatorConfig,
    AggregatorError,
    AggregatorSession,
    Reviews,
    allowed_in_order,
    review_candidates,
    revisions,
)
from .binance_bars import (
    BarsError,
    ClosedBarStore,
    FetchPage,
    QuantConfig,
    http_fetcher,
    market_snapshots,
    quant_revision,
)
from .evedex import (
    Book,
    Evedex,
    EvedexError,
    Instrument,
    Position,
    Unknown,
    ceil_step,
    dec,
    floor_step,
)
from .execution_session import legacy_session_result
from .management import Management, load_policy, managed_stop, scale_in_status, tighter
from .outcomes import recorded_closes
from .risk import (
    AGGREGATE_OPEN_RISK_FRACTION,
    MIN_NOTIONAL_USD,
    PAPER_ACCOUNT_SOURCE,
    PER_TRADE_RISK_FRACTION,
    RiskConfig,
    RiskError,
    RiskSession,
    evaluate_entry,
    risk_revisions,
)
from .routing import (
    MAX_REVIEW_TIMEOUT_MS,
    RouterConfig,
    RouterError,
    route_intents,
    router_revision,
    router_session,
)
from .sibling import SiblingError
from .signals import SIGNAL_VENUE, Features, Routed, held_outcome, outcome, signal_symbol
from .system_engine import (
    HOURLY_TREND_HISTORY_MS,
    HOURLY_TREND_ID,
    RIGHT_TAIL_HISTORY_MS,
    RIGHT_TAIL_ID,
    RIGHT_TAIL_INTERVALS,
    EngineConfig,
    EngineError,
    EnginePool,
    TrendScan,
    adaptive_shadow_scan,
    engine_revision,
    hourly_trend_scan,
    right_tail_due,
    right_tail_scan,
    to_candles,
)
from .system_rules import MacroDesk, RulesConfig, RulesError, bias_of
from .system_rules import revisions as rules_revisions
from .text_scouts import NewsDesk, TextScoutsError
from .worker_session import WorkerSession

# Hard bounds of the configuration. Dollar limits are fractions of the capital basis, so the
# same rules hold for a 100 USD canary and a 100 000 USD account.
HARD_MAX_CAPITAL_USD = Decimal(150_000)
# One trade's stop-loss dollars: at most this share of the capital basis (the Risk Manager's
# constitutional fraction is lower; this only bounds the configured dollar cap).
HARD_MAX_RISK_PER_TRADE_OF_CAPITAL = Decimal("0.01")
# The session's realised loss stop: at most this share of the capital basis.
HARD_MAX_SESSION_LOSS_OF_CAPITAL = Decimal("0.05")
HARD_MAX_TRADES_PER_DAY = 40
HARD_MAX_OPEN_POSITIONS = 10
# The owner's leverage constitution (2026-11-16). The Risk Manager picks each trade's
# leverage from 1x up to this by need, evidence, the stop and the venue's maximum.
MAX_LEVERAGE = 30.0
# One position's notional is at most this multiple of the capital basis; its stop-loss risk
# is set by the Risk Manager, and the number of positions by its aggregate open risk (1%),
# the Macro Strategist's gross cap and free margin at the chosen leverage.
POSITION_NOTIONAL_OF_CAPITAL = 3
# The configurable multiple may only be lower: a large account trades thin books.
MIN_POSITION_NOTIONAL_OF_CAPITAL = 0.05
# The account equity above which the Risk Manager refuses every entry (LiveRiskLimitsV1):
# at most this multiple of the capital basis, so a bigger account is never traded unnoticed.
HARD_MAX_ACCOUNT_EQUITY_OF_CAPITAL = Decimal(2)
HISTORY_LENGTH = 20
HOUR_MS = 60 * 60_000
BAR_MS = HOUR_MS  # signals come from closed 1h bars
# The Aggregator's review context carries this many of the newest closed bars the rules read
# (AggregatorSettings.candidate_context_max_bars).
REVIEW_CONTEXT_BARS = 60
SNAPSHOT_BARS = 200  # the indicator window of Quant Scouts' snapshot builder (price_window)
# The market snapshot for a decision is taken in this window before its boundary: it must be
# produced before the decision and stay within the snapshot's 120-second life until the
# review has finished.
# Started 75-10 s before the boundary: 23 instruments on a slow line take up to ~30 s, and a
# snapshot lives 120 s, past the review 15-40 s after the boundary.
SNAPSHOT_LEAD_MS = (10_000, 75_000)
# The Aggregator must answer within 60 s of the bar's close (the Router's longest review
# window), so the scan starts soon after the close and runs its parts wide: the last
# minute bar is on Binance within a second or two of the close.
SCAN_AFTER_CLOSE_MS = 5_000
POSITION_MISMATCH = "position quantity, side or price differs from the owned entry"
MANUAL_CHECK_PREFIX = "нужна ручная проверка счёта: "
REPEATED_ERRORS_HALT = "слишком много ошибок подряд"
BAR_CONCURRENCY = 8
ENGINE_PROCESSES = 8
# The right-tail rule reads 35 days of minute bars per instrument; it runs beside the hourly
# rules, so it gets its own, larger share of the machine's cores.
RIGHT_TAIL_PROCESSES = 12
# The strategies the runner can trade. ``Config.strategies`` selects among them; the Macro
# Strategist allocates capital to the selected ones, and the Risk Manager gives an entry
# capital only through the weight of its intent's strategy id.
KNOWN_STRATEGIES = (HOURLY_TREND_ID, RIGHT_TAIL_ID)
# Config keys of earlier versions that no longer select anything.
RETIRED_CONFIG_KEYS = {
    "risk_mode": "every entry is decided by the Kairos Risk Manager",
    "risk_shadow": "every entry is decided by the Kairos Risk Manager",
}


class ConfigError(ValueError):
    pass


@dataclass(slots=True)
class Config:
    # Liquid EVEDEX perpetuals with a tight spread on 2026-11-11. BTC is kept mostly as
    # market context: its minimum lot is too large for this account's risk per trade.
    symbols: list[str] = field(
        default_factory=lambda: [
            "BTCUSD",
            "ETHUSD",
            "HYPEUSD",
            "SOLUSD",
            "XRPUSD",
            "DOGEUSD",
            "LINKUSD",
            "SUIUSD",
            "AAVEUSD",
            "ZECUSD",
        ]
    )
    # The strategies whose intents are traded. hourly_trend always runs, because its state is
    # the reviewers' market features; when it is not listed its intents are only journaled.
    strategies: list[str] = field(default_factory=lambda: [HOURLY_TREND_ID])
    # A fixed position-management policy per strategy (management.Management fields), chosen
    # by an offline study; a strategy without one uses the measured policy below.
    management_policies: dict[str, dict[str, Any]] = field(default_factory=dict)
    # How often the right-tail rule decides (hours): 24 is the frozen Trial 15 clock.
    right_tail_interval_hours: int = 24
    capital_basis_usd: float = 100.0
    # One position's notional at most this multiple of the capital basis. The venue gate
    # measures the book's slippage for this notional, so a large account sets it to what
    # the thinner books can absorb (0.25 of 100 000 USD is 25 000 USD).
    position_notional_of_capital: float = float(POSITION_NOTIONAL_OF_CAPITAL)
    risk_per_trade_usd: float = 0.25
    session_loss_stop_usd: float = 5.0
    max_trades_per_day: int = 12
    max_open_positions: int = 10
    cooldown_minutes: int = 30  # per instrument, after its trade closes
    reward_risk: float = 1.5
    timeout_minutes: int = 720
    max_entry_drift_atr: float = 0.35
    max_spread_bps: float = 4.0
    slippage_fraction: float = 0.0002
    use_llm: bool = True
    llm_model: str = "gpt-6-luna"
    llm_effort: str = "medium"
    llm_timeout_s: float = 120.0
    max_llm_calls_per_day: int = 600
    min_confidence: float = 0.5
    # Who decides: "probability" is the single review of all candidates with a probability
    # threshold; "aggregator" is the Kairos Router + Aggregator review, one call per
    # candidate, ALLOW / VETO / DEFER. With review_shadow the other one runs too and is
    # only recorded, so the two can be compared in the report.
    review_mode: str = "aggregator"
    review_shadow: bool = True
    # The review deadline the Router writes into every route, counted from the decision.
    # RouterSettings allows at most 60 seconds.
    review_timeout_s: float = 60.0
    # The monthly ceiling of the Aggregator review's model calls, kept in llm-budget.json
    # in the runtime directory. kairos-llm registers 12 USD for OpenAI as the maximum.
    review_budget_usd: float = 12.0
    # The Kairos Macro Strategist: one strategic allocation a day, and after a price crash.
    use_macro: bool = True
    macro_effort: str = "high"
    # The Kairos Risk Manager refuses every entry while the account equity is above this.
    max_account_equity_usd: float = 200.0
    use_text_scouts: bool = True  # the Kairos Text Scouts news layer
    news_poll_minutes: float = 5.0
    max_news_llm_calls_per_day: int = 150
    chain_ids: list[int] = field(default_factory=lambda: [42161, 161803])
    poll_seconds: float = 5.0
    scan_minutes: int = 15  # how often the market is scanned: 15, 30 or 60
    exit_confirm_polls: int = 3  # polls in a row without the position before it counts as closed
    # Older runner profiles use "runner" for the same Execution Engine portfolio client.
    executor: str = "execution_portfolio"
    execution_profile: str = ""
    # The Risk Manager's adaptive per-trade risk (kairos_risk.adaptive_risk) from the learning
    # ledger of closed trades; always at most the fixed ceiling.
    adaptive_risk: bool = True
    # The research adaptive strategy runs alongside in observation only; its measured
    # outcomes feed the learning ledger (as replay-weight evidence).
    adaptive_shadow: bool = True
    # The Macro Strategist re-judges the regime this often (hours); shocks trigger it sooner.
    macro_refresh_hours: float = 6.0
    # Close an open position early when the regime turns against it (measured; see
    # REGIME_EXIT_MIN_SAMPLES).
    regime_exit: bool = True
    # Break-even, trailing stops and partial exits: "measured" applies the policy a replay
    # proved better than the plan (management-policy.json in the runtime directory);
    # "off" always keeps the plan.
    management: str = "measured"
    execution_python: str = ""
    execution_session_config: str = ""

    @classmethod
    def load(cls, path: Path | None) -> Config:
        config = cls()
        if path is not None:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ConfigError("config file must contain a JSON object")
            for key, value in raw.items():
                if key.startswith("_"):
                    continue
                if key in RETIRED_CONFIG_KEYS:
                    say(f"Настройка {key} больше не действует: {RETIRED_CONFIG_KEYS[key]}.")
                    continue
                if key not in cls.__dataclass_fields__:
                    raise ConfigError(f"unknown config key: {key}")
                setattr(config, key, value)
            if config.execution_profile:
                config.execution_profile = str((path.parent / config.execution_profile).resolve())
        config.validate()
        return config

    def validate(self) -> None:
        if not self.symbols or any(not isinstance(s, str) or not s.isalnum() for s in self.symbols):
            raise ConfigError("symbols must be plain instrument names like SOLUSD")
        try:
            signals = [signal_symbol(s) for s in self.symbols]
        except ValueError as exc:
            raise ConfigError(f"symbols must be EVEDEX instruments quoted in USD: {exc}") from exc
        if len(set(signals)) != len(signals):
            raise ConfigError("symbols must not repeat")
        if not 1 <= self.review_timeout_s * 1000 <= MAX_REVIEW_TIMEOUT_MS:
            raise ConfigError("review_timeout_s must be within 0.001..60")
        if not 0 < self.review_budget_usd <= OPENAI_MONTHLY_CEILING_USD:
            raise ConfigError(f"review_budget_usd must be within (0, {OPENAI_MONTHLY_CEILING_USD}]")
        capital = Decimal(str(self.capital_basis_usd))
        if not 0 < capital <= HARD_MAX_CAPITAL_USD:
            raise ConfigError(f"capital_basis_usd must be within (0, {HARD_MAX_CAPITAL_USD}]")
        if not 0 < Decimal(str(self.risk_per_trade_usd)) <= capital * HARD_MAX_RISK_PER_TRADE_OF_CAPITAL:
            raise ConfigError(
                "risk_per_trade_usd must be within (0, "
                f"{HARD_MAX_RISK_PER_TRADE_OF_CAPITAL:%} of capital_basis_usd]"
            )
        if not 0 < Decimal(str(self.session_loss_stop_usd)) <= capital * HARD_MAX_SESSION_LOSS_OF_CAPITAL:
            raise ConfigError(
                "session_loss_stop_usd must be within (0, "
                f"{HARD_MAX_SESSION_LOSS_OF_CAPITAL:%} of capital_basis_usd]"
            )
        if (
            not MIN_POSITION_NOTIONAL_OF_CAPITAL
            <= float(self.position_notional_of_capital)
            <= float(POSITION_NOTIONAL_OF_CAPITAL)
        ):
            raise ConfigError(
                f"position_notional_of_capital must be within {MIN_POSITION_NOTIONAL_OF_CAPITAL}.."
                f"{POSITION_NOTIONAL_OF_CAPITAL}"
            )
        if not self.strategies or any(s not in KNOWN_STRATEGIES for s in self.strategies):
            raise ConfigError(f"strategies must be a non-empty subset of {', '.join(KNOWN_STRATEGIES)}")
        if len(set(self.strategies)) != len(self.strategies):
            raise ConfigError("strategies must not repeat")
        if self.right_tail_interval_hours not in RIGHT_TAIL_INTERVALS:
            raise ConfigError(f"right_tail_interval_hours must be one of {RIGHT_TAIL_INTERVALS}")
        if not isinstance(self.management_policies, dict):
            raise ConfigError("management_policies must map a strategy id to a policy")
        for strategy, policy in self.management_policies.items():
            if strategy not in KNOWN_STRATEGIES or not isinstance(policy, dict):
                raise ConfigError(f"management_policies: unknown strategy or malformed policy for {strategy}")
            try:
                Management.from_dict({"name": f"{strategy}-fixed", **policy})
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"management_policies[{strategy}]: {exc}") from exc
        if not 1 <= int(self.max_trades_per_day) <= HARD_MAX_TRADES_PER_DAY:
            raise ConfigError(f"max_trades_per_day must be within 1..{HARD_MAX_TRADES_PER_DAY}")
        if not 1 <= int(self.max_open_positions) <= HARD_MAX_OPEN_POSITIONS:
            raise ConfigError(f"max_open_positions must be within 1..{HARD_MAX_OPEN_POSITIONS}")
        if not 0.5 <= self.max_spread_bps <= 50:
            raise ConfigError("max_spread_bps must be within 0.5..50")
        if self.scan_minutes not in (15, 30, 60):
            raise ConfigError("scan_minutes must be 15, 30 or 60")
        if not 1 <= int(self.exit_confirm_polls) <= 20:
            raise ConfigError("exit_confirm_polls must be within 1..20")
        if self.macro_effort not in ("low", "medium", "high"):
            raise ConfigError("macro_effort must be low, medium or high")
        if not 1 <= float(self.macro_refresh_hours) <= 24:
            raise ConfigError("macro_refresh_hours must be within 1..24")
        if self.management not in ("measured", "off"):
            raise ConfigError("management must be measured or off")
        ceiling = capital * HARD_MAX_ACCOUNT_EQUITY_OF_CAPITAL
        if not capital <= Decimal(str(self.max_account_equity_usd)) <= ceiling:
            raise ConfigError(
                "max_account_equity_usd must be within [capital_basis_usd, "
                f"{HARD_MAX_ACCOUNT_EQUITY_OF_CAPITAL} x capital_basis_usd]"
            )
        if self.review_mode not in ("probability", "aggregator"):
            raise ConfigError("review_mode must be probability or aggregator")
        if self.executor not in ("execution_portfolio", "runner", "execution_engine"):
            raise ConfigError("executor must be execution_portfolio, runner, or historical execution_engine")
        if self.executor == "execution_engine" and not (
            self.execution_python and self.execution_session_config
        ):
            raise ConfigError("executor execution_engine needs execution_python and execution_session_config")
        if not 1 <= self.news_poll_minutes <= 60 or not 0 <= int(self.max_news_llm_calls_per_day) <= 400:
            raise ConfigError("news_poll_minutes must be 1..60 and max_news_llm_calls_per_day 0..400")
        if not 1.0 <= self.reward_risk <= 5.0 or not 15 <= self.timeout_minutes <= 1440:
            raise ConfigError("reward_risk must be 1..5 and timeout_minutes 15..1440")
        if not 0.0 <= self.min_confidence <= 1.0 or not 0 <= self.slippage_fraction <= 0.01:
            raise ConfigError("min_confidence must be 0..1 and slippage_fraction 0..0.01")
        if not self.chain_ids or any(not isinstance(c, int) or c <= 0 for c in self.chain_ids):
            raise ConfigError("chain_ids must be positive integers")
        if self.poll_seconds < 2 or self.llm_effort not in ("low", "medium", "high"):
            raise ConfigError("poll_seconds must be at least 2 and llm_effort low/medium/high")


def software_exit(trade: dict[str, Any], book: Book, now_ms: int) -> str | None:
    """Exit reason from the engine's own watch of the book, or None."""
    stop, take = dec(trade["stop"]), dec(trade["take"])
    if trade["side"] == "LONG":
        if book.bid <= stop:
            return "STOP"
        if book.bid >= take:
            return "TAKE"
    else:
        if book.ask >= stop:
            return "STOP"
        if book.ask <= take:
            return "TAKE"
    return "TIMEOUT" if now_ms >= trade["deadline_ms"] else None


OPPOSING_REGIME = {"LONG": "BEAR", "SHORT": "BULL"}
# Early exits on a regime change keep running only while they measurably help: after this
# many measured exits, a negative total of (exit R - held-to-plan R) switches them off.
REGIME_EXIT_MIN_SAMPLES = 20


def _iso_ms(text: Any) -> int | None:
    try:
        return int(datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def regime_exit_reason(
    side: str | None,
    entry_regime: str | None,
    allocation: dict[str, Any] | None,
    decided_at_ms: int | None,
) -> str | None:
    """Why to close an open position early: the Macro regime has turned against it since entry.

    Only an allocation produced after the entry decision counts, and only a regime that
    opposes the side (LONG in BEAR, SHORT in BULL) and differs from the entry's regime.
    """
    if not allocation or side not in OPPOSING_REGIME:
        return None
    regime = allocation.get("regime")
    if regime != OPPOSING_REGIME[side] or regime == entry_regime:
        return None
    produced = _iso_ms(allocation.get("produced_at"))
    if produced is None or (decided_at_ms is not None and produced <= int(decided_at_ms)):
        return None
    return f"Macro regime {regime} opposes the open {side} (entered in {entry_regime or 'UNKNOWN'})"


def management_step(
    trade: dict[str, Any], book: Book, instrument: Instrument | None
) -> tuple[Decimal | None, Decimal | None]:
    """What the trade's management policy asks for now: a tighter stop and/or a partial exit.

    The best price since entry is updated from the book. A new stop is on the price tick,
    strictly less risky than the stop in force and away from the market by one tick; a
    partial exit is a multiple of the quantity step and leaves at least the minimum lot.
    """
    if not trade.get("management"):
        return None, None
    policy = Management.from_dict(trade["management"])
    side, entry = trade["side"], float(trade["entry_price"])
    initial = float(trade.get("initial_stop", trade["stop"]))
    touch = book.bid if side == "LONG" else book.ask
    best = float(trade.get("best_price", trade["entry_price"]))
    best = max(best, float(touch)) if side == "LONG" else min(best, float(touch))
    trade["best_price"] = repr(best)
    part: Decimal | None = None
    targets = policy.targets()
    if (
        instrument is not None
        and targets
        and not trade.get("partial_done")
        and not trade.get("native_partial_targets")
    ):
        initial_quantity = dec(trade.setdefault("initial_quantity", trade["quantity"]))
        quantity = dec(trade["quantity"])
        if quantity > initial_quantity:
            raise NeedsOperator("managed position exceeds its original fill")
        sign = 1 if side == "LONG" else -1
        gain = sign * (touch - dec(trade["entry_price"]))
        risk = abs(dec(trade["entry_price"]) - dec(trade.get("initial_stop", trade["stop"])))
        fraction = sum(
            (Decimal(str(t.fraction)) for t in targets if risk > 0 and gain >= Decimal(str(t.at_r)) * risk),
            Decimal(0),
        )
        if fraction > 0:
            # The reconciled remainder also covers a restart just after a confirmed fill.
            step = instrument.quantity_step
            cumulative = (initial_quantity * fraction / step).to_integral_value(rounding=ROUND_DOWN) * step
            wanted = min(cumulative - (initial_quantity - quantity), quantity - instrument.min_quantity)
            rounded = (wanted / step).to_integral_value(rounding=ROUND_DOWN) * step
            if rounded > 0 and rounded >= instrument.min_quantity:
                part = rounded
            elif len(targets) == 1 and initial_quantity == quantity:
                trade["partial_done"] = "skipped: the position is not divisible into minimum lots"
    desired = managed_stop(policy, side, entry, initial, best)
    if (
        policy.partial_at_r is not None
        and trade.get("partial_done") == "done"
        and policy.breakeven_at_r is not None
    ):
        desired = tighter(side, desired, entry)
    if instrument is None:
        return None, part
    tick = instrument.price_step
    rounding = ROUND_DOWN if side == "LONG" else ROUND_UP
    stop = (Decimal(repr(desired)) / tick).to_integral_value(rounding=rounding) * tick
    current = dec(trade["stop"])
    safe = stop <= touch - tick if side == "LONG" else stop >= touch + tick
    better = stop > current if side == "LONG" else stop < current
    return (stop if better and safe else None), part


def estimate_pnl(trade: dict[str, Any], exit_price: Decimal, fee: Decimal) -> Decimal:
    """Result of one trade from its prices and the taker fee (used when positions overlap)."""
    quantity, entry = dec(trade["quantity"]), dec(trade["entry_price"])
    sign = 1 if trade["side"] == "LONG" else -1
    return sign * (exit_price - entry) * quantity - (entry + exit_price) * quantity * fee


class Store:
    """State file (atomic replace) and append-only journal in the runtime directory."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._state_path = directory / "state.json"
        self._journal_path = directory / "journal.jsonl"
        self.state: dict[str, Any] = {}
        if self._state_path.exists():
            self.state = json.loads(self._state_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        temp = self._state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str), encoding="utf-8")
        os.replace(temp, self._state_path)

    def journal(self, event: str, **data: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": event, **data}
        with self._journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def publish_close(self, *, recovering: bool = False) -> None:
        pending = self.state.get("pending_close_report")
        if pending is None:
            return
        published = False
        if recovering and self._journal_path.exists():
            with self._journal_path.open(encoding="utf-8") as records:
                for line in records:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get("close_report_id") == pending["close_report_id"]:
                        published = True
                        break
        if not published:
            self.journal("trade_closed", **pending)
        del self.state["pending_close_report"]
        self.save()

    def flag(self, name: str) -> bool:
        return (self.directory / name).exists()


def say(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')} UTC] {message}", flush=True)


class PaperBroker:
    """Simulated fills at the top of the real book, with taker fee and slippage."""

    mode = "paper"
    account_source = PAPER_ACCOUNT_SOURCE
    account_id = "evedex-prod-paper-ledger"

    def __init__(self, store: Store, config: Config, fee: Decimal) -> None:
        self._store, self._fee = store, fee
        self._slip = Decimal(str(config.slippage_fraction))
        store.state.setdefault("paper_equity", str(config.capital_basis_usd))

    async def account_reads(self) -> tuple[Decimal, Decimal, list[Position], list[dict[str, Any]]]:
        """Balance, available balance, positions and resting orders of the simulated ledger."""
        equity = await self.equity()
        positions = [
            Position(
                t["symbol"],
                "BUY" if t["side"] == "LONG" else "SELL",
                dec(t["quantity"]),
                dec(t["entry_price"]),
                True,
                f"paper-stop-{t['id']}",
            )
            for t in self._store.state.get("open_trades", [])
        ]
        return equity, equity, positions, []

    async def equity(self) -> Decimal:
        return dec(self._store.state["paper_equity"])

    async def available(self) -> Decimal:
        return await self.equity()

    async def foreign_exposure(self, own: set[str]) -> list[str]:
        return []

    async def position_open(self, trade: dict[str, Any]) -> bool:
        return True

    async def enter(
        self,
        candidate: Routed,
        quantity: Decimal,
        book: Book,
        instrument: Instrument,
        stop: Decimal,
        take: Decimal,
        leverage: int = 1,
        decision: Any = None,
    ) -> dict[str, Any] | None:
        price = book.ask * (1 + self._slip) if candidate.side == "LONG" else book.bid * (1 - self._slip)
        return {"quantity": quantity, "price": price, "order_id": "paper", "protected": True}

    async def close(self, trade: dict[str, Any], book: Book, reason: str) -> None:
        quantity, entry = dec(trade["quantity"]), dec(trade["entry_price"])
        sign = 1 if trade["side"] == "LONG" else -1
        if reason == "TAKE":
            price = dec(trade["take"])
        else:
            touch = book.bid if sign > 0 else book.ask
            price = touch * (1 - sign * self._slip)
        fees = (entry + price) * quantity * self._fee
        pnl = sign * (price - entry) * quantity - fees
        trade["exit_price"] = str(price)
        trade["pnl_usd"] = str(pnl + dec(trade.get("partial_pnl_usd", "0")))
        self._store.state["paper_equity"] = str(await self.equity() + pnl)

    async def move_stop(self, trade: dict[str, Any], stop: Decimal) -> bool:
        current = dec(trade["stop"])
        if stop <= current if trade["side"] == "LONG" else stop >= current:
            raise ValueError("a stop may only move toward less risk")
        trade["stop"] = str(stop)
        return True

    async def partial_close(self, trade: dict[str, Any], book: Book, quantity: Decimal) -> Decimal | None:
        """Close part of the position at the book; returns the price."""
        if not 0 < quantity < dec(trade["quantity"]):
            return None
        entry = dec(trade["entry_price"])
        sign = 1 if trade["side"] == "LONG" else -1
        touch = book.bid if sign > 0 else book.ask
        price = touch * (1 - sign * self._slip)
        pnl = sign * (price - entry) * quantity - (entry + price) * quantity * self._fee
        trade["partial_pnl_usd"] = str(dec(trade.get("partial_pnl_usd", "0")) + pnl)
        trade["quantity"] = str(dec(trade["quantity"]) - quantity)
        self._store.state["paper_equity"] = str(await self.equity() + pnl)
        return price

    async def settle(self, trade: dict[str, Any]) -> dict[str, Any] | None:
        return None


def collect_learning_outcomes(directory: Path, config: Config, journal=None) -> list[dict[str, Any]]:
    """Closed trades in R for the learning ledger: this runner's and the Execution Engine's.

    The regime of an Execution Engine trade is the Macro regime journaled before its
    decision (``macro_allocation``); the runner's own trades carry theirs.
    """
    path = directory / "journal.jsonl"
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    # Balance changes and estimated exits include effects from other positions,
    # funding, account transfers or manual actions. They cannot establish strategy edge.
    eligible = {}
    for close in recorded_closes(records):
        row = close.row
        trade_id = row.get("trade_id")
        if (
            close.mode == "live"
            and close.pnl_basis == "execution_fills"
            and close.exit_origin == "system"
            and row.get("pnl_exact") is True
            and isinstance(trade_id, str)
            and trade_id
            and row.get("candidate_id")
        ):
            eligible[trade_id] = row
    outcomes = [
        {
            "strategy_id": r["strategy_id"],
            "symbol": r["signal_symbol"],
            "regime": r.get("regime") or "UNKNOWN",
            "r_multiple": r["r_multiple"],
            "decided_at_ms": r["decided_at_ms"],
            "source": "LIVE",
        }
        for r in eligible.values()
        if r.get("r_multiple") is not None
        and r.get("strategy_id")
        and r.get("signal_symbol")
        and isinstance(r.get("decided_at_ms"), int)
    ]
    outcomes.extend(
        {
            "strategy_id": r["strategy_id"],
            "symbol": r["signal_symbol"],
            "regime": r.get("regime") or "UNKNOWN",
            "r_multiple": r["r_multiple"],
            "decided_at_ms": r["decided_at_ms"],
            "source": "REPLAY",
        }
        for r in records
        if r.get("event") == "shadow_outcome" and r.get("r_multiple") is not None
    )
    replay = directory / "learning-replay.jsonl"
    if replay.exists():
        for line in replay.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("source") == "REPLAY" and row.get("r_multiple") is not None:
                outcomes.append(row)
    if config.execution_python and config.execution_session_config:
        from .execution_session import closed_trades

        timeline = sorted(
            (
                int(datetime.fromisoformat(str(r["produced_at"]).replace("Z", "+00:00")).timestamp() * 1000),
                r["regime"],
            )
            for r in records
            if r.get("event") == "macro_allocation" and r.get("produced_at") and r.get("regime")
        )

        def regime_at(at_ms: int) -> str:
            regime = "UNKNOWN"
            for produced, value in timeline:
                if produced > at_ms:
                    break
                regime = value
            return regime

        try:
            rows = closed_trades(Path(config.execution_python), Path(config.execution_session_config))
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            (journal or (lambda *a, **k: None))("learning_error", message=str(exc)[:200])
            rows = []
        # The accepted accounting report is scoped to the Execution database. Bind
        # it to completed live handoffs in this journal before using it in Risk.
        finished = {
            r["trade_id"]
            for r in records
            if r.get("event") == "execution_session_finished"
            and r.get("outcome") == "CLOSED_AND_ACCOUNTED"
            and r.get("candidate")
            and not r.get("exit_requested")
            and r.get("trade_id")
            and r["trade_id"] not in eligible
        }
        outcomes.extend(
            {
                "strategy_id": row["strategy_id"],
                "symbol": row["symbol"],
                "regime": regime_at(int(row["decided_at_ms"])),
                "r_multiple": row["r_multiple"],
                "decided_at_ms": int(row["decided_at_ms"]),
                "source": "LIVE",
            }
            for row in rows
            if row.get("trade_id") in finished
            and row.get("version") == "kairos.execution-closed-trade.v1"
            and row.get("settled_at")
            and row.get("exit_reason") in {"STOP", "TARGET", "TIMEOUT"}
            and not row.get("early_exit")
            and row.get("exit_origin") == "system"
            and row.get("r_multiple") is not None
        )
    return outcomes


class Trader:
    def __init__(
        self,
        *,
        api: Evedex,
        session: aiohttp.ClientSession,
        store: Store,
        config: Config,
        broker: PaperBroker | LiveBroker,
        fee: Decimal,
        instruments: dict[str, Instrument],
        openai_key: str | None,
        bars: ClosedBarStore | None = None,
        fetch_bars: FetchPage | None = None,
        pilot_journal_path: Path | None = None,
    ) -> None:
        self.api, self.session, self.store, self.config = api, session, store, config
        self.broker, self.fee, self.instruments, self.openai_key = broker, fee, instruments, openai_key
        self._pilot_journal_path = pilot_journal_path
        self._next_flat_probe = 0.0
        self._extras: dict[str, dict[str, Any]] = {}
        self._status_at: dict[str, int] = {}  # when each position was last reported on screen
        # The signal chain of the Kairos design: closed Binance bars checked by Quant Scouts,
        # the Strategy Engine's rules, the Router's route. Without any of the three the
        # runner only manages positions.
        self.bars = bars or ClosedBarStore(store.directory / "closed-bars")
        self._fetch_bars = fetch_bars or http_fetcher(session)
        self.engine: EngineConfig | None = EngineConfig()
        self.quant, self.router = QuantConfig(), RouterConfig()
        self._router_session: WorkerSession | None = None
        self._router_warm_task: asyncio.Task | None = None
        self._signal_revisions: dict[str, str] = {}
        # The Aggregator review's evidence: the market snapshots taken before a boundary and
        # the newest closed bars of the last scan, keyed by the Binance signal symbol.
        self._market: dict[str, Any] = {"boundary": None, "items": {}, "problems": {}}
        self._market_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task[dict[str, dict[str, Any]]] | None = None
        self._metrics_boundary: int | None = None
        self._probability_shadow: asyncio.Task[None] | None = None
        self._review_bars: dict[str, list[Any]] = {}
        try:
            self.engine.check_hourly_trend()
            self.quant.check()
            self.router.check()
            self._signal_revisions = {
                "engine_revision": engine_revision(self.engine.source),
                "quant_scouts_revision": quant_revision(self.quant),
                "router_revision": router_revision(self.router),
            }
        except (EngineError, BarsError, RouterError) as exc:
            self.engine = None
            say(f"Сигнальная цепочка Kairos недоступна, новых сигналов не будет: {exc}")
        # The right-tail rule reads a 35-day window: persistent engine workers keep the bars
        # they have checked, so a decision lands well inside the 60-second review window.
        self.right_tail_pool: EnginePool | None = (
            EnginePool(self.engine, RIGHT_TAIL_PROCESSES)
            if self.engine is not None and RIGHT_TAIL_ID in config.strategies
            else None
        )
        # The hourly rules read 21 days every scan and give the reviewers their features.
        self.hourly_pool: EnginePool | None = (
            EnginePool(self.engine, ENGINE_PROCESSES) if self.engine is not None else None
        )
        self._warm_task: asyncio.Task | None = None
        self._adaptive_task: asyncio.Task | None = None
        self.aggregator: AggregatorConfig | None = None
        self._aggregator_session: AggregatorSession | None = None
        self._aggregator_warm_task: asyncio.Task | None = None
        self._aggregator_task: asyncio.Task[Reviews] | None = None
        self._position_faults: dict[str, tuple[str, str]] = {}
        self._reported_unprotected: dict[str, str] = {}
        if config.use_llm and (config.review_mode == "aggregator" or config.review_shadow):
            candidate_review = AggregatorConfig()
            try:
                candidate_review.check()
                self.aggregator = candidate_review
                self._aggregator_revisions = revisions(candidate_review)
            except AggregatorError as exc:
                say(f"Обзор Aggregator отключён: {exc}")
        rules = RulesConfig()
        self.macro: MacroDesk | None = None
        self._macro_task: asyncio.Task | None = None
        if config.use_llm and config.use_macro:
            try:
                rules.check_macro()
                self.macro = MacroDesk(
                    store,
                    rules,
                    tuple(config.strategies),
                    refresh_ms=int(float(config.macro_refresh_hours) * 3_600_000),
                )
            except RulesError as exc:
                say(f"Macro Strategist отключён: {exc}")
        self._rules_revisions = rules_revisions(rules) if self.macro else {}
        # Every entry is decided by the Kairos Risk Manager; without it nothing is opened.
        self.risk: RiskConfig | None = RiskConfig()
        self.risk_session: RiskSession | None = None
        self._risk_revisions: dict[str, str] = {}
        try:
            self.risk.check()
            self._risk_revisions = risk_revisions(self.risk)
            self.risk_session = RiskSession(self.risk)
        except RiskError as exc:
            self.risk = None
            say(f"Risk Manager недоступен, новых входов не будет: {exc}")
        self._reviews: dict[str, Any] = {}
        self._policy: tuple[int, Management | None, str] = (0, None, "not loaded")
        self.news: NewsDesk | None = None
        if config.use_text_scouts:
            desk = NewsDesk(store)
            try:
                desk.config.check()
                self.news = desk
            except TextScoutsError as exc:
                say(f"Слой новостей Text Scouts отключён: {exc}")

    async def news_tick(self) -> None:
        """One Text Scouts poll. Its failures are logged and never reach the trading loop."""
        if self.news is None:
            return
        try:
            summary = await self.news.poll(
                self.session,
                openai_key=self.openai_key,
                model=self.config.llm_model,
                max_llm_calls=self.config.max_news_llm_calls_per_day,
            )
        except (TimeoutError, TextScoutsError, aiohttp.ClientError, OSError) as exc:
            self.store.journal("news_error", kind=type(exc).__name__, message=str(exc)[:300])
            return
        if summary["signals"]:
            say(
                "Новости (Text Scouts): "
                + "; ".join(
                    f"{s['topic']} — {s['impact']}, уверенность {s['confidence']:.2f}"
                    + (" (грубая оценка без модели)" if s["origin"] == "local" else "")
                    for s in summary["signals"]
                )
            )

    async def _news_loop(self) -> None:
        while True:
            try:
                await self.news_tick()
            except Exception as exc:  # noqa: BLE001 - the news layer must never stop, or disturb, trading
                self.store.journal("news_error", kind=type(exc).__name__, message=str(exc)[:300])
            await asyncio.sleep(max(60.0, self.config.news_poll_minutes * 60))

    # ---- bookkeeping --------------------------------------------------------

    def trades(self) -> list[dict[str, Any]]:
        """Open trades. A state file written by the single-position version is migrated."""
        state = self.store.state
        trades = state.setdefault("open_trades", [])
        old = state.pop("open_trade", None)
        if old:
            trades.append(old)
        return trades

    def _day(self) -> dict[str, Any]:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        day = self.store.state.get("day")
        if not isinstance(day, dict) or day.get("date") != today:
            day = {"date": today, "trades": 0, "llm_calls": 0}
            self.store.state["day"] = day
        return day

    # ---- Kairos Macro Strategist and Risk Manager -------------------------------

    @staticmethod
    def _market_view(
        features: list[Features], scan_extras: dict[str, dict[str, Any]], metrics: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        view: dict[str, dict[str, Any]] = {}
        for f in features:
            extras = {**scan_extras.get(f.symbol, {}), **metrics.get(f.symbol, {})}
            view[f.symbol] = {
                "price": round(f.price, 8),
                "quant_bias": bias_of(f.trend),
                "trend_4h": f.trend,
                "rsi": round(f.rsi, 1),
                "atr_pct": round(f.atr_pct, 3),
                "change_24h_pct": round(f.change_24h_pct, 2),
                **{
                    key: extras[key]
                    for key in (
                        "change_1h_pct",
                        "change_7d_pct",
                        "funding_rate_pct",
                        "open_interest_usd",
                        "volume_24h_usd",
                    )
                    if key in extras
                },
            }
        return view

    def _day_equity(self, equity: Decimal) -> dict[str, Any]:
        """Today's start and peak equity, kept in the state file."""
        if isinstance(self.broker, LiveBroker):
            status = self.broker.pilot_status()
            if status is None:
                raise NeedsOperator("portfolio performance has no verified baseline")
            flow = dec(status["external_flow"])
            normalized = dec(status["funding_balance"]) - flow
            today = datetime.now(UTC).date().isoformat()
            key = (status["started_at"], today)
            day = self.store.state.get("portfolio_day", {})
            if (day.get("pilot"), day.get("date")) != key:
                start = dec(status["baseline"]) if status["started_at"][:10] == today else normalized
                day = {"pilot": key[0], "date": today, "start": str(start), "peak": str(start)}
                self.store.state["portfolio_day"] = day
            day["peak"] = str(max(normalized, dec(day["peak"])))
            return {
                "start_equity": str(dec(day["start"]) + flow),
                "peak_equity": str(dec(day["peak"]) + flow),
                "realized_usd": str(normalized - dec(day["start"])),
            }
        day = self._day()
        if not day.get("start_equity"):
            day["start_equity"] = str(equity)
        day["peak_equity"] = str(max(equity, dec(day.get("peak_equity") or equity)))
        return day

    async def _account_view(self) -> dict[str, Any]:
        """The account as the Macro Strategist's prompt sees it."""
        equity, available = await self.broker.equity(), await self.broker.available()
        day = self._day_equity(equity)
        return {
            "equity_usd": float(equity),
            "available_usd": float(available),
            "daily_realized_pnl_usd": float(dec(day.get("realized_usd", "0"))),
            "positions": [
                {
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "quantity": float(dec(t["quantity"])),
                    "entry_price": float(dec(t["entry_price"])),
                    "stop": float(dec(t["stop"])),
                    "strategy_id": t.get("strategy_id"),
                }
                for t in self.trades()
            ],
        }

    @staticmethod
    def _lineage(trade: dict[str, Any] | None) -> dict[str, Any] | None:
        """What the Risk Manager needs to know about the trade behind a position, if it has it."""
        if not trade or not trade.get("risk_decision_id") or not trade.get("leverage"):
            return None  # opened before entries were decided by the Risk Manager
        return {
            key: trade[key]
            for key in (
                "stop",
                "take",
                "max_holding_ms",
                "strategy_id",
                "strategy_revision",
                "intent_id",
                "risk_decision_id",
                "risk_trade_id",
                "first_fill_ms",
                "leverage",
            )
        } | {"entry_order_id": trade["id"]}

    async def _account_evidence(self) -> dict[str, Any]:
        """The account as the venue and the runner's ledger show it now, for the Risk Manager."""
        captured = int(time.time() * 1000)
        balance, available, positions, orders = await self.broker.account_reads()
        instruments = list(dict.fromkeys(p.instrument for p in positions))
        quotes = await asyncio.gather(*(self.api.book(symbol) for symbol in instruments))
        books = dict(zip(instruments, quotes, strict=True))
        trades = {t["symbol"]: t for t in self.trades()}
        day = self._day_equity(balance)
        state = self.store.state
        state["reconciliation_seq"] = int(state.get("reconciliation_seq", 0)) + 1
        return {
            "source_id": self.broker.account_source,
            "account_id": self.broker.account_id,
            "captured_at_ms": captured,
            "balance_usd": float(balance),
            "available_usd": float(available),
            "day_start_equity_usd": float(dec(day["start_equity"])),
            "peak_equity_usd": float(dec(day["peak_equity"])),
            "daily_realized_pnl_usd": float(dec(day.get("realized_usd", "0"))),
            "reconciliation_seq": state["reconciliation_seq"],
            "pending_entry": bool(state.get("pending_entry")),
            "open_orders": [{"instrument": o.get("instrument"), "id": o.get("id")} for o in orders],
            "positions": [
                {
                    "venue_symbol": p.instrument,
                    "side": "LONG" if p.side == "BUY" else "SHORT",
                    "quantity": float(p.quantity),
                    "entry_price": float(p.avg_price),
                    "mark_price": float(
                        books[p.instrument].bid if p.side == "BUY" else books[p.instrument].ask
                    ),
                    "stop_order_id": p.stop_id,
                    "trade": self._lineage(trades.get(p.instrument)),
                }
                for p in positions
            ],
        }

    def _risk_scope(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """The frozen scope inputs and limits of LiveRiskEvaluator, from this configuration."""
        config = self.config
        capital = float(config.capital_basis_usd)
        position_cap = capital * float(config.position_notional_of_capital)
        limits = {
            "per_trade_risk_fraction": PER_TRADE_RISK_FRACTION,
            "aggregate_open_risk_fraction": AGGREGATE_OPEN_RISK_FRACTION,
            "max_leverage": MAX_LEVERAGE,
            "per_trade_dollar_cap_usd": float(config.risk_per_trade_usd),
            "aggregate_dollar_cap_usd": capital * MAX_LEVERAGE,
            "daily_loss_cap_usd": float(config.session_loss_stop_usd),
            "max_account_equity_usd": float(config.max_account_equity_usd),
            "max_position_notional_usd": round(position_cap, 2),
            "min_notional_usd": MIN_NOTIONAL_USD,
        }
        scope = {
            "instruments": list(config.symbols),
            "account_id": self.broker.account_id,
            "context": {
                "composer": "kairos-autotrader",
                "venue": "EVEDEX",
                "profile": "PROD",
                "mode": self.broker.mode,
            },
            "sources": {
                **self._signal_revisions,
                **self._risk_revisions,
                **getattr(self, "_aggregator_revisions", {}),
            },
            "policy": {"limits": limits, "symbols": list(config.symbols), "reward_risk": config.reward_risk},
        }
        return scope, limits

    def _recent_outcomes(self, limit: int) -> list[dict]:
        history = self.store.state.get("history", [])
        if self.broker.mode == "live":
            history = [
                row
                for row in history
                if row.get("mode") == "live"
                and row.get("pnl_basis") == "execution_fills"
                and row.get("exit_origin") == "system"
                and row.get("risk_trade_id")
                and row.get("risk_decision_id")
                and row.get("result_r") is not None
            ]
        return history[-limit:]

    async def macro_tick(self, features: list[Features], now_ms: int) -> None:
        """Ask the Macro Strategist if an allocation is due. Its failures never reach the trading loop."""
        if self.macro is None or not features:
            return
        try:
            # A slow account read may cross a scan boundary. Capture this scan's features and
            # optional metrics together before another scan can replace their source.
            scan_extras = {symbol: fields.copy() for symbol, fields in self._extras.items()}
            account, metrics = await asyncio.gather(
                self._account_view(), self._metrics_at(self.scan_slot(now_ms))
            )
            if isinstance(self.broker, LiveBroker):
                pilot = self.broker.pilot_status() or {}
                performance = {"pilot_account_result_usd": pilot.get("account_result")}
            else:
                performance = {
                    "session_realized_usd": self.store.state.get("session", {}).get("realized_usd")
                }
            allocation = await self.macro.refresh(
                self.session,
                openai_key=self.openai_key,
                model=self.config.llm_model,
                effort=self.config.macro_effort,
                market=self._market_view(features, scan_extras, metrics),
                portfolio={k: account[k] for k in ("equity_usd", "available_usd", "positions")},
                performance={
                    **performance,
                    "daily_realized_pnl_usd": account["daily_realized_pnl_usd"],
                    "trades_today": self._day()["trades"],
                    "recent_closed_trades": self._recent_outcomes(20),
                },
                now_ms=now_ms,
            )
        except Exception as exc:  # noqa: BLE001 - the strategic layer must never stop, or disturb, trading
            self.store.journal("macro_error", kind=type(exc).__name__, message=str(exc)[:300])
            return
        if allocation:
            self.store.journal(
                "macro_allocation",
                regime=allocation.get("regime"),
                produced_at=allocation.get("produced_at"),
                origin=allocation.get("origin"),
                allocation=allocation,
            )
            weights = ", ".join(
                f"{name} {weight:.0%}" for name, weight in allocation["strategy_weights"].items()
            )
            say(
                f"Macro Strategist: режим {allocation['regime']}, "
                f"резерв {allocation['stable_reserve_pct']:.0%}, "
                f"доли: {weights or 'нет'}"
                + (" (защитное решение: модель не ответила)" if allocation.get("origin") != "model" else "")
            )

    def halt(self, reason: str, *, reconcile_entry: bool = False) -> None:
        if self.store.state.get("halted") != reason:
            self.store.state["halted"] = reason
            self.store.state["entry_reconciliation_halt"] = reconcile_entry
            self.store.save()
            self.store.journal("halt", reason=reason)
            say(f"НОВЫЕ ВХОДЫ ОСТАНОВЛЕНЫ: {reason}")

    async def _observe_adaptive(self, series: dict[str, list[list[Any]]], slot: int) -> None:
        """Record the adaptive strategy's intents at this boundary; nothing is traded."""
        if self.engine is None:
            return
        try:
            answer = await adaptive_shadow_scan(
                self.engine,
                series,
                slot,
                observed_at_ms=int(time.time() * 1000),
                source_set_sha256=hashlib.sha256(
                    json.dumps(self._signal_revisions, sort_keys=True).encode()
                ).hexdigest(),
            )
        except EngineError as exc:
            self.store.journal("shadow_error", message=str(exc)[:300])
            return
        allocation = self.macro.allocation() if self.macro else None
        pending = self.store.state.setdefault("shadow_pending", [])
        for symbol, result in sorted(answer["results"].items()):
            intent = result.get("intent")
            if not intent:
                continue
            record = {
                "id": intent["intent_id"],
                "strategy_id": answer["strategy_id"],
                "signal_symbol": symbol,
                "side": intent["side"],
                "ref_price": intent["reference_price"],
                "stop": intent["exit_plan"]["stop_price"],
                "take": intent["exit_plan"]["target_price"],
                "timeout_minutes": int(intent["exit_plan"]["max_holding_ms"]) // 60_000,
                "signal_time": int(intent["decision_ts_ms"]),
                "regime": (allocation or {}).get("regime") or "UNKNOWN",
                "detector_regime": result.get("regime"),
            }
            if all(item["id"] != record["id"] for item in pending):
                pending.append(record)
                self.store.journal("shadow_signal", **record)
        self.store.save()
        self._measure_shadow(series)

    def _measure_shadow(self, series: dict[str, list[list[Any]]]) -> None:
        """Measure every observed signal whose holding time has passed, on the cached closed bars."""
        pending = self.store.state.get("shadow_pending", [])
        if not pending:
            return
        by_signal = {signal_symbol(instrument): rows for instrument, rows in series.items()}
        cost_fraction = 2 * (float(self.fee) + self.config.slippage_fraction)
        remaining = []
        for record in pending:
            rows = by_signal.get(record["signal_symbol"], [])
            deadline = record["signal_time"] + record["timeout_minutes"] * 60_000
            if not rows or rows[-1][0] < deadline:
                remaining.append(record)
                continue
            result = outcome(record, to_candles(rows), cost_fraction=cost_fraction)
            if result["r"] is None:
                remaining.append(record)
                continue
            self.store.journal(
                "shadow_outcome",
                strategy_id=record["strategy_id"],
                signal_symbol=record["signal_symbol"],
                regime=record["regime"],
                decided_at_ms=record["signal_time"],
                r_multiple=str(result["r"]),
                reason=result["reason"],
            )
        self.store.state["shadow_pending"] = remaining[-500:]
        self.store.save()

    def learning_outcomes(self) -> list[dict[str, Any]]:
        """Closed trades in R for the learning ledger (``collect_learning_outcomes``)."""
        return collect_learning_outcomes(self.store.directory, self.config, self.store.journal)

    def history_ms(self) -> int:
        """How far back the closed-bar cache must reach for the selected strategies."""
        need = HOURLY_TREND_HISTORY_MS
        if RIGHT_TAIL_ID in self.config.strategies:
            need = max(need, RIGHT_TAIL_HISTORY_MS)
        return need

    def longest_hold_minutes(self) -> int:
        """The longest holding time any selected strategy plans (72 h for the daily rule)."""
        longest = int(self.config.timeout_minutes)
        if RIGHT_TAIL_ID in self.config.strategies:
            longest = max(longest, 72 * 60)
        return longest

    def management_policy(self, now_ms: int, strategy_id: str | None = None) -> Management | None:
        """The management policy for a new trade of ``strategy_id``.

        A strategy with a fixed policy in the configuration gets it; otherwise the measured
        policy (re-read every ten minutes), which the replay of the hourly rules selects.
        """
        if self.config.management == "off":
            return None
        fixed = self.config.management_policies.get(strategy_id or "")
        if fixed is not None:
            return Management.from_dict({"name": f"{strategy_id}-fixed", **fixed})
        loaded_at, policy, reason = self._policy
        if now_ms - loaded_at >= 600_000:
            policy, reason = load_policy(
                self.store.directory, now_ms=now_ms, reward_risk=self.config.reward_risk
            )
            if reason != self._policy[2]:
                self.store.journal(
                    "management_policy", policy=policy.as_dict() if policy else None, reason=reason
                )
            self._policy = (now_ms, policy, reason)
        return policy

    def _regime_exit(self, side: Any, entry_regime: Any, decided_at_ms: Any) -> str | None:
        """The early-exit reason for an open position, while early exits measurably help."""
        if not self.config.regime_exit or self.macro is None:
            return None
        stats = self.store.state.get("early_exit_stats") or {}
        if int(stats.get("n", 0)) >= REGIME_EXIT_MIN_SAMPLES and float(stats.get("saved_r", 0)) < 0:
            if not stats.get("disabled"):
                stats["disabled"] = True
                self.store.journal("regime_exit_disabled", measured=stats["n"], saved_r=stats["saved_r"])
                say(
                    f"Досрочные выходы по смене режима отключены: за {stats['n']} измеренных выходов "
                    f"они дали {stats['saved_r']:+.2f} R против удержания по плану."
                )
                self.store.save()
            return None
        return regime_exit_reason(
            side,
            entry_regime,
            self.macro.allocation(),
            int(decided_at_ms) if decided_at_ms is not None else None,
        )

    def _queue_early_exit(self, plan: dict[str, Any]) -> None:
        """Keep an early exit's plan until the bars show what holding it would have made."""
        pending = self.store.state.setdefault("early_exit_pending", [])
        pending.append(plan)
        del pending[:-200]

    def _measure_early_exits(self, series: dict[str, list[list[Any]]]) -> None:
        """Compare each early exit with holding the position to its plan, on the closed bars."""
        pending = self.store.state.get("early_exit_pending", [])
        if not pending:
            return
        by_signal = {signal_symbol(instrument): rows for instrument, rows in series.items()}
        cost_fraction = 2 * (float(self.fee) + self.config.slippage_fraction)
        stats = self.store.state.setdefault("early_exit_stats", {"n": 0, "saved_r": 0.0})
        remaining = []
        for plan in pending:
            rows = by_signal.get(plan.get("signal_symbol"), [])
            held = (
                held_outcome(plan, to_candles(rows), cost_fraction=cost_fraction)
                if rows and rows[-1][0] >= int(plan["deadline_ms"])
                else None
            )
            if held is None:
                remaining.append(plan)
                continue
            saved = round(float(plan["exit_r"]) - held, 4)
            stats["n"] = int(stats.get("n", 0)) + 1
            stats["saved_r"] = round(float(stats.get("saved_r", 0)) + saved, 4)
            self.store.journal(
                "early_exit_measured",
                id=plan.get("id"),
                signal_symbol=plan.get("signal_symbol"),
                exit_r=str(plan["exit_r"]),
                held_r=str(held),
                saved_r=str(saved),
            )
        self.store.state["early_exit_pending"] = remaining
        self.store.save()

    def _protection_block(self, instrument: str | None = None) -> str | None:
        if not isinstance(self.broker, LiveBroker):
            return None
        if instrument is not None:
            for_instrument = getattr(self.broker, "protection_unavailable_for", None)
            if for_instrument is not None:
                return for_instrument(instrument)
        # An older installed execution package cannot certify position-reducing protection.
        return getattr(
            self.broker,
            "native_protection_unavailable",
            "Execution Engine does not report qualified position-reducing native protection",
        )

    def _account_entry_block(self) -> str | None:
        state = self.store.state
        if state.get("halted"):
            return str(state["halted"])
        if isinstance(self.broker, LiveBroker):
            if datetime.now(UTC) >= self.broker.entry_deadline_utc:
                return "signed entry window ended"
            status = self.broker.pilot_status()
            if status is None or status["stopped"]:
                return "portfolio loss stop or unavailable performance baseline"
        if self.store.flag("STOP") or self.store.flag("CLOSE"):
            return "stop file present"
        if self._day()["trades"] >= self.config.max_trades_per_day:
            return "daily trade limit reached"
        return None

    def entries_blocked(self, instrument: str | None = None) -> str | None:
        return self._account_entry_block() or self._protection_block(instrument)

    def _ranking_limit(self, candidates: list[Routed], free_slots: int) -> int:
        # An instrument with unqualified exits may be reviewed but cannot take a live slot.
        blocked = {c.instrument for c in candidates if self._protection_block(c.instrument)}
        return free_slots + len(blocked)

    def slot_limit(self) -> int:
        return min(HARD_MAX_OPEN_POSITIONS, int(self.config.max_open_positions))

    async def start(self, *, new_session: bool = False, resume_reconciled: bool = False) -> None:
        state = self.store.state
        if state.get("mode") not in (None, self.broker.mode):
            raise ConfigError("this runtime directory belongs to another mode; use a separate directory")
        if resume_reconciled and not isinstance(self.broker, LiveBroker):
            raise ConfigError("--resume-reconciled requires a live portfolio")
        state["mode"] = self.broker.mode
        self.store.publish_close(recovering=True)
        trades = self.trades()
        pilot_baseline = None
        if isinstance(self.broker, LiveBroker):
            if self.config.executor == "execution_engine":
                raise ConfigError("single-trade sessions are retired; configure execution_portfolio")
            legacy = legacy_session_result(self.store)
            if legacy is not None:
                self.store.journal("execution_session_finished", **legacy)
                state.pop("execution_session")
                self.store.save()
            if resume_reconciled:
                await self._resume_reconciled(self.broker, trades)
            else:
                await self._reconcile_portfolio(trades)
            blocked = {
                symbol: reason for symbol in self.config.symbols if (reason := self._protection_block(symbol))
            }
            if blocked and blocked != self._reported_unprotected:
                self.store.journal(
                    "native_protection_unavailable",
                    reason="; ".join(dict.fromkeys(blocked.values())),
                    symbols=list(blocked),
                )
                say(f"Без подтверждённой защиты EVEDEX: {len(blocked)} из {len(self.config.symbols)}.")
            self._reported_unprotected = blocked
            status = self.broker.pilot_status()
            pilot_baseline = status["baseline"] if status else None
        if new_session and self.broker.mode == "paper":
            state.pop("session", None)
            state.pop("halted", None)
            state.pop("entry_reconciliation_halt", None)
        if new_session or "session" not in state:
            equity = await self.broker.equity()
            if "session" in state:
                state["session"]["reporting_started_at"] = datetime.now(UTC).isoformat()
            else:
                state["session"] = {
                    "start_equity": str(equity),
                    "started_at": datetime.now(UTC).isoformat(),
                    "realized_usd": "0",
                }
            self.store.journal(
                "session_start",
                equity=str(equity),
                mode=self.broker.mode,
                config=asdict(self.config),
                **({"risk_start_equity": pilot_baseline} if isinstance(self.broker, LiveBroker) else {}),
            )
        self.store.save()
        await self.warm_history()
        if isinstance(self.broker, LiveBroker):
            opening = (
                f"Начальный баланс пилота {pilot_baseline} USDT"
                if pilot_baseline is not None
                else ("Начальный баланс пилота пока не подтверждён")
            )
        else:
            opening = f"Стартовый баланс сессии: {state['session']['start_equity']} USDT"
        say(f"Режим: {self.broker.mode}. {opening}. Позиций одновременно: до {self.slot_limit()}.")
        for trade in self.trades():
            say(
                f"Веду открытую позицию: {trade['symbol']} {trade['side']} {trade['quantity']} "
                f"по {trade['entry_price']}, "
                f"стоп {trade['stop']}, тейк {trade['take']}."
            )
        if len(self.trades()) >= self.slot_limit():
            say(
                "Все места заняты: новые входы начнутся после закрытия позиции. "
                "Состояние печатается раз в 10 минут."
            )

    async def reconcile_while_stopped(self) -> None:
        if not isinstance(self.broker, LiveBroker) or self.store.state.get("mode") != "live":
            raise ConfigError("stopped reconciliation requires an existing live portfolio")
        await self._resume_reconciled(self.broker, self.trades(), while_stopped=True)

    async def _resume_reconciled(
        self, broker: LiveBroker, trades: list[dict[str, Any]], *, while_stopped: bool = False
    ) -> None:
        state = self.store.state
        halt = state.get("halted")
        eligible = state.get("entry_reconciliation_halt")
        # Old runners wrote this specific ownership fault with a false flag. The same
        # flat-account and venue/journal checks below are required for an explicit resume.
        if halt == MANUAL_CHECK_PREFIX + POSITION_MISMATCH:
            eligible = True
        if eligible is None and isinstance(halt, str):
            # Older live runners recorded only the reason, including the affected instrument.
            prefixes = tuple(f"вход {symbol} требует сверки: " for symbol in self.config.symbols)
            eligible = halt.startswith(("портфель требует сверки: ", *prefixes))
        if not halt or eligible is not True:
            raise ConfigError("no execution-reconciliation halt to resume")
        if trades or state.get("pending_entry") or state.get("execution_session"):
            raise NeedsOperator("runner still has an open or pending trade")
        if self.store.flag("STOP") != while_stopped or self.store.flag("CLOSE"):
            raise NeedsOperator("operator stop state does not match the requested reconciliation")
        if any(await asyncio.gather(self.api.positions(), self.api.open_orders(), self.api.active_tpsl())):
            raise NeedsOperator("venue still has a position or an open order or protection")

        started = datetime.now(UTC)
        await broker.reconcile_start([], None)
        if await broker.foreign_exposure(set()):
            raise NeedsOperator("portfolio ownership or a venue request remains unresolved")
        status = broker.pilot_status()
        now = datetime.now(UTC)
        if (
            not status
            or status["stopped"]
            or status["open_positions"] != 0
            or not max(started, now - timedelta(seconds=60))
            <= datetime.fromisoformat(status["observed_at"])
            <= now
            or now >= broker.entry_deadline_utc
            or self.store.flag("STOP") != while_stopped
            or self.store.flag("CLOSE")
        ):
            raise NeedsOperator("a fresh flat account with an active pilot mandate is required")
        self.store.journal(
            "entry_halt_reconciled",
            reason=halt,
            observed_at=status["observed_at"],
            while_stopped=while_stopped,
        )
        state.pop("halted")
        state.pop("entry_reconciliation_halt", None)
        self.store.save()

    async def _reconcile_portfolio(self, trades: list[dict[str, Any]]) -> None:
        broker = self.broker
        if not isinstance(broker, LiveBroker):
            return
        pending = self.store.state.get("pending_entry")
        legacy = [trade for trade in trades if trade.get("execution_provider") != "execution_portfolio"]
        if pending and not pending.get("trade_id"):
            raise NeedsOperator("legacy pending entry must be reconciled before portfolio execution")
        if not legacy:
            await broker.reconcile_start(trades, pending)
            return
        if pending or len(legacy) != len(trades):
            raise NeedsOperator("mixed legacy and portfolio ownership requires reconciliation")
        # Old records contain no Execution ownership proof. Only archive them
        # when the entire account is observed flat and has no open obligations.
        for observation in range(2):
            if await broker.foreign_exposure(set()):
                raise NeedsOperator("legacy exposure must be reconciled before portfolio execution")
            if observation == 0:
                await asyncio.sleep(self.config.poll_seconds)
        for trade in legacy:
            self.store.journal(
                "legacy_position_flat_unattributed",
                id=trade.get("id"),
                symbol=trade.get("symbol"),
                candidate_id=(trade.get("candidate") or {}).get("id"),
                mode="live",
                exit_origin="unknown",
                pnl_basis="unknown",
                pnl_usd=None,
            )
        trades[:] = [trade for trade in trades if trade not in legacy]
        self.store.save()
        await broker.reconcile_start(trades, None)

    # ---- scanning and deciding ---------------------------------------------

    def scan_slot(self, now_ms: int) -> int:
        """The latest scan boundary: the moment the hourly bar under review closed."""
        scan_ms = int(self.config.scan_minutes) * 60_000
        return now_ms // scan_ms * scan_ms

    async def _update_bars(self, now_ms: int) -> tuple[dict[str, list[Any]], dict[str, str]]:
        """Closed Binance bars per EVEDEX instrument, and why an instrument has none."""
        by_signal = {signal_symbol(instrument): instrument for instrument in self.config.symbols}
        bars, problems = await self.bars.update(
            self.quant,
            list(by_signal),
            now_ms,
            self.history_ms() + HOUR_MS,
            fetch=self._fetch_bars,
            concurrency=BAR_CONCURRENCY,
        )
        if self.bars.revisions:
            # Binance settled a just-closed minute after it was cached: its later value is kept.
            self.store.journal("bars_revised", minutes=self.bars.revisions)
        return (
            {by_signal[symbol]: rows for symbol, rows in bars.items()},
            {by_signal[symbol]: reason for symbol, reason in problems.items()},
        )

    async def warm_history(self) -> None:
        """Bring the closed-bar cache up to date before the loop starts.

        The very first start downloads three weeks of Binance minute bars per instrument,
        which takes a couple of minutes; later starts fetch only what is missing. A failure
        here is not fatal: the next scan tries again.
        """
        if self.engine is None:
            return
        series, problems = await self._update_bars(int(time.time() * 1000))
        if series and self._warm_task is None:
            self._warm_task = asyncio.create_task(self._warm_pools(series))
        if problems:
            say(
                "Минутная история Binance загружена не полностью ("
                + ", ".join(f"{symbol}: {reason[:60]}" for symbol, reason in problems.items())
                + "); повторю на скане."
            )

    async def _warm_pools(self, series: dict[str, list[Any]]) -> None:
        """Run each pooled rule once at its latest past decision, only to fill the workers.

        The answers are discarded: they decide nothing. The first real scan after a start
        then checks only the bars that closed since, instead of the whole window.
        """
        engine = self.engine
        if engine is None:
            return
        now_ms = int(time.time() * 1000)
        ticks = {name: item.price_step for name, item in self.instruments.items()}
        parts: dict[str, tuple[int, Any]] = {}
        if self.hourly_pool is not None:
            boundary = self.scan_slot(now_ms)
            parts["hourly"] = (
                boundary,
                hourly_trend_scan(
                    engine,
                    series,
                    boundary,
                    reward_risk=self.config.reward_risk,
                    timeout_minutes=self.config.timeout_minutes,
                    price_increments=ticks,
                    pool=self.hourly_pool,
                ),
            )
        if self.right_tail_pool is not None:
            # The latest whole hour, not the rule's last decision: the cached window reaches
            # back the rule's history from about now, and the next decision reuses nearly all of it.
            boundary = now_ms // HOUR_MS * HOUR_MS
            parts["right_tail"] = (
                boundary,
                right_tail_scan(
                    engine,
                    series,
                    boundary,
                    price_increments=ticks,
                    interval_hours=self.config.right_tail_interval_hours,
                    pool=self.right_tail_pool,
                ),
            )

        async def timed(name: str, boundary: int, work: Any) -> None:
            started = time.monotonic()
            try:
                result = await work
            except EngineError as exc:
                say(f"Прогрев движка ({name}) не удался, первый скан будет медленнее: {str(exc)[:200]}")
                return
            # An instrument the rule skipped was not warmed: its first scan checks its whole window.
            skipped = result.skipped if isinstance(result, TrendScan) else result[1]
            self.store.journal(
                "engine_warm",
                rule=name,
                boundary=boundary,
                seconds=round(time.monotonic() - started, 2),
                skipped=len(skipped),
            )

        await asyncio.gather(*(timed(name, boundary, work) for name, (boundary, work) in parts.items()))

    def _market_due(self, now_ms: int) -> int | None:
        """The boundary whose snapshot should be taken now, if any."""
        if self.engine is None or self.aggregator is None or not self.openai_key:
            return None
        boundary = self.scan_slot(now_ms) + int(self.config.scan_minutes) * 60_000
        lead = boundary - now_ms
        if not SNAPSHOT_LEAD_MS[0] <= lead <= SNAPSHOT_LEAD_MS[1] or self._market["boundary"] == boundary:
            return None
        if self._market_task is not None and not self._market_task.done():
            return None
        return boundary

    async def capture_market(self, boundary_ms: int) -> None:
        """Market snapshots for the decision at ``boundary_ms``, built by Quant Scouts' code.

        Runs before the boundary, in the background. Its failures are recorded and only
        leave the review without market evidence, which the Aggregator turns into DEFER.
        """
        self._market = {"boundary": boundary_ms, "items": {}, "problems": {}}
        try:
            series, problems = await self._update_bars(int(time.time() * 1000))
            bars = {signal_symbol(i): rows[-SNAPSHOT_BARS:] for i, rows in series.items() if rows}
            items, failed = await market_snapshots(self.quant, bars) if bars else ({}, {})
        except (TimeoutError, BarsError, aiohttp.ClientError, OSError) as exc:
            items, problems, failed = {}, {}, {"*": f"{type(exc).__name__}: {str(exc)[:160]}"}
        self._market = {
            "boundary": boundary_ms,
            "items": items,
            "problems": {**{signal_symbol(i): r for i, r in problems.items()}, **failed},
        }
        produced = sorted(item["payload"]["produced_at"] for item in items.values())
        self.store.journal(
            "market_snapshots",
            boundary=boundary_ms,
            symbols=sorted(items),
            produced_from=produced[0] if produced else None,
            produced_to=produced[-1] if produced else None,
            problems=self._market["problems"],
        )

    async def _optional_metrics(self) -> dict[str, dict[str, Any]]:
        try:
            return await self.api.metrics()
        except Exception as exc:  # noqa: BLE001 - supplementary public data cannot fail the review
            self.store.journal("venue_metrics_unavailable", kind=type(exc).__name__)
            return {}

    async def _metrics_at(self, boundary: int) -> dict[str, dict[str, Any]]:
        if self._metrics_boundary == boundary and self._metrics_task is not None:
            return await asyncio.shield(self._metrics_task)
        return {}

    async def scan(self, now_ms: int) -> tuple[list[Features], list[Routed]]:
        """Ask the Kairos signal chain about the hourly bar that has just closed.

        Closed Binance USD-M one-minute bars, accepted by the Quant Scouts parser, go to
        ``kairos_strategy.hourly_trend`` in the Strategy Engine, which answers with a
        ``StrategyIntentV1`` per candidate. The Router's policy turns each intent into a
        ``CandidateRouteV1``. The rules read one-hour bars rebuilt so that the newest one
        ends at this scan's boundary; with ``scan_minutes=15`` that is four decisions an
        hour (:00, :15, :30 and :45) on the same timeframe.
        """
        slot = self.scan_slot(now_ms)
        last_closed = slot - BAR_MS
        self._review_bars = {}
        self._extras = {}
        if self.engine is None:
            self.store.journal("scan_error", bar=last_closed, message="the signal chain is not available")
            return [], []
        started = time.monotonic()
        timing: dict[str, float] = {"start_after_close_s": round((time.time() * 1000 - slot) / 1000, 2)}
        # These public metrics enrich the scan report, Macro and the probability review.
        # The Router and Aggregator use their own evidence at this bar's boundary.
        metrics_task = None
        if self._metrics_task is None or self._metrics_task.done():
            self._metrics_boundary = slot
            metrics_task = self._metrics_task = asyncio.create_task(self._optional_metrics())
        series, problems = await self._update_bars(now_ms)
        timing["bars_s"] = round(time.monotonic() - started, 2)
        skipped = [f"{symbol}: {reason}" for symbol, reason in problems.items()]
        self._review_bars = {
            signal_symbol(instrument): [row for row in rows if row[0] < slot][-REVIEW_CONTEXT_BARS:]
            for instrument, rows in series.items()
        }
        found = None
        daily_task = None
        ticks = {name: item.price_step for name, item in self.instruments.items()}
        if (
            series
            and RIGHT_TAIL_ID in self.config.strategies
            and right_tail_due(slot, self.config.right_tail_interval_hours)
        ):
            # Alongside the hourly rules, in their own engine processes: the review window
            # (60 s after the close) is shared by every candidate of this scan.
            daily_task = asyncio.create_task(
                right_tail_scan(
                    self.engine,
                    series,
                    slot,
                    price_increments=ticks,
                    processes=RIGHT_TAIL_PROCESSES,
                    interval_hours=self.config.right_tail_interval_hours,
                    pool=self.right_tail_pool,
                )
            )
        if series:
            try:
                found = await hourly_trend_scan(
                    self.engine,
                    series,
                    slot,
                    reward_risk=self.config.reward_risk,
                    timeout_minutes=self.config.timeout_minutes,
                    price_increments={name: item.price_step for name, item in self.instruments.items()},
                    processes=ENGINE_PROCESSES,
                    pool=self.hourly_pool,
                )
            except EngineError as exc:
                say(f"Strategy Engine не дал ответа, скан пропущен: {str(exc)[:200]}")
                self.store.journal("scan_error", bar=last_closed, message=str(exc)[:300])
                if daily_task is not None:
                    daily_task.cancel()
                return [], []
            skipped.extend(found.skipped)
            timing["engine_s"] = round(time.monotonic() - started - timing["bars_s"], 2)
            if self.config.adaptive_shadow and (self._adaptive_task is None or self._adaptive_task.done()):
                # Only observed, never traded: it must not take time from the review window.
                self._adaptive_task = asyncio.create_task(self._observe_adaptive(series, slot))
            self._measure_early_exits(series)
        features = [found.features[s] for s in self.config.symbols if found and s in found.features]
        self._extras = {f.symbol: dict(found.context[f.symbol]) for f in features} if found else {}
        candidates: list[Routed] = []
        rejected: list[dict[str, Any]] = []
        router: dict[str, Any] = {}
        intents: dict[str, StrategyIntentV1] = {}
        shadow: list[dict[str, Any]] = []
        if found and found.intents:
            if HOURLY_TREND_ID in self.config.strategies:
                intents.update(found.intents)
            else:
                # Not traded: recorded so the hourly rules keep being measured.
                shadow = [
                    {"symbol": i, "side": x.side.value, "strategy_id": x.strategy_id}
                    for i, x in found.intents.items()
                ]
        engines = {**(found.identity if found else {})}
        if daily_task is not None:
            try:
                daily, daily_skipped, daily_identity = await daily_task
            except EngineError as exc:
                say(f"Дневная стратегия не дала ответа: {str(exc)[:200]}")
                self.store.journal("scan_error", bar=last_closed, message=f"right tail: {str(exc)[:300]}")
            else:
                skipped.extend(daily_skipped)
                engines["right_tail"] = json.dumps(daily_identity, sort_keys=True)
                for instrument, intent in daily.items():
                    replaced = intents.get(instrument)
                    if replaced is not None:
                        # One position per instrument: the daily rule's candidate is reviewed.
                        shadow.append(
                            {
                                "symbol": instrument,
                                "side": replaced.side.value,
                                "strategy_id": replaced.strategy_id,
                            }
                        )
                    intents[instrument] = intent
                timing["right_tail_s"] = round(time.monotonic() - started, 2)
        if intents:
            routing_started = time.monotonic()
            try:
                routing = await route_intents(
                    self.router,
                    intents,
                    signals=self.news.raw_signals() if self.news else [],
                    now_ms=int(time.time() * 1000),
                    review_timeout_ms=int(self.config.review_timeout_s * 1000),
                    session=self._router_session,
                )
            except RouterError as exc:
                say(f"Router не дал ответа, сигналы этого скана пропущены: {str(exc)[:200]}")
                self.store.journal("scan_error", bar=last_closed, message=str(exc)[:300])
                return features, []
            candidates, rejected, router = routing.routed, routing.rejected, routing.settings
            timing["router_s"] = round(time.monotonic() - routing_started, 2)
        metrics_ready = metrics_task is not None and metrics_task.done() and not metrics_task.cancelled()
        metrics = metrics_task.result() if metrics_task is not None and metrics_ready else {}
        for symbol, extras in self._extras.items():
            extras.update(metrics.get(symbol, {}))
        timing["done_after_close_s"] = round((time.time() * 1000 - slot) / 1000, 2)
        self.store.journal(
            "scan",
            timing=timing,
            venue_metrics_ready=metrics_ready,
            bar=last_closed,
            source="strategy_engine",
            signal_venue=SIGNAL_VENUE,
            engine={**engines, **self._signal_revisions},
            router=router,
            features=[{**f.as_dict(), **self._extras.get(f.symbol, {})} for f in features],
            candidates=[c.record() for c in candidates],
            rejected=rejected,
            skipped=skipped,
            not_traded=shadow,
        )
        return features, candidates

    async def decide(
        self, features: list[Features], candidates: list[Routed], free_slots: int = 1
    ) -> brain.Decision:
        control = brain.control_decision(candidates, self._ranking_limit(candidates, free_slots))
        day = self._day()
        if not (self.config.use_llm and self.openai_key):
            decision = control
        elif day["llm_calls"] >= self.config.max_llm_calls_per_day:
            decision = brain.Decision("WAIT", None, 0.0, "daily model call limit reached", "FALLBACK")
        else:
            by_aggregator = self.config.review_mode == "aggregator" and self.aggregator is not None
            left = self.config.max_llm_calls_per_day - day["llm_calls"]
            if by_aggregator and len(candidates) > left:
                ranked = sorted(candidates, key=lambda c: (-c.strength, c.id))
                self.store.journal("review_budget_limited", skipped=[c.id for c in ranked[left:]])
                candidates = ranked[:left]
            shadow_slot = self._probability_shadow is None or self._probability_shadow.done()
            ask_probability = not by_aggregator or (
                self.config.review_shadow and left > len(candidates) and shadow_slot
            )
            ask_aggregator = self.aggregator is not None and (
                by_aggregator or (self.config.review_shadow and left > len(candidates))
            )
            day["llm_calls"] += int(ask_probability) + (len(candidates) if ask_aggregator else 0)
            self.store.save()
            # Snapshot portfolio and signal context before an entry can change them. Only the
            # probability reviewer consumes this payload or fetches the external sentiment index.
            payload = (
                self._probability_payload(features, candidates, free_slots, day["trades"])
                if ask_probability
                else {}
            )
            if by_aggregator:
                if ask_probability:
                    self._probability_shadow = asyncio.create_task(
                        self._record_probability_shadow(payload, candidates, free_slots)
                    )
                chosen = await self._aggregator_decision(candidates, free_slots)
                if not ask_probability:
                    self.store.journal(
                        "review_context",
                        candidate_ids=[c.id for c in candidates],
                        fear_greed=None,
                        text_signals=self.news.context(int(time.time() * 1000))
                        if self.news
                        else "news layer is off",
                        macro=(self.macro.summary() if self.macro else None) or "UNAVAILABLE",
                    )
            elif ask_aggregator:
                # The probability choice still needs the Aggregator contract for the Risk Manager.
                chosen, other = await asyncio.gather(
                    self._probability_decision(payload, candidates, free_slots),
                    self._aggregator_decision(candidates, free_slots),
                )
                self.store.journal(
                    "shadow_decision", decision=other.as_dict(), candidate_ids=[c.id for c in candidates]
                )
            else:
                chosen = await self._probability_decision(payload, candidates, free_slots)
            decision = chosen or brain.Decision(
                "WAIT", None, 0.0, "the selected reviewer did not answer", "FALLBACK"
            )
        self.store.journal(
            "decision",
            decision=decision.as_dict(),
            control_pick=control.candidate_id,
            control_picks=control.picks,
            candidate_ids=[c.id for c in candidates],
        )
        return decision

    def _probability_payload(
        self, features: list[Features], candidates: list[Routed], free_slots: int, trades_today: int
    ) -> dict[str, Any]:
        now_ms = int(time.time() * 1000)
        return brain.build_payload(
            features,
            candidates,
            trades_today=trades_today,
            free_slots=free_slots,
            open_positions=[
                {
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "entry_price": t["entry_price"],
                    "stop": t["stop"],
                    "take": t["take"],
                    "age_minutes": (now_ms - t["opened_ms"]) // 60_000,
                }
                for t in self.trades()
            ],
            extras=self._extras,
            recent_trades=self._recent_outcomes(10),
            text_signals=self.news.context(now_ms) if self.news else None,
            macro=self.macro.summary() if self.macro else None,
        )

    async def _probability_decision(
        self, payload: dict[str, Any], candidates: list[Routed], free_slots: int
    ) -> brain.Decision:
        fear_greed, metrics = await asyncio.gather(
            brain.fetch_fear_greed(self.session),
            self._metrics_at(max((c.signal_time for c in candidates), default=0)),
        )
        for feature in payload["market_features"]:
            feature.update(metrics.get(feature["symbol"], {}))
        payload["crypto_fear_greed_index"] = fear_greed or "unavailable"
        self.store.journal(
            "review_context",
            candidate_ids=[c.id for c in candidates],
            fear_greed=fear_greed,
            text_signals=payload["text_scouts_signals"],
            macro=payload["macro_allocation"],
        )
        return await brain.ask_llm(
            self.session,
            api_key=self.openai_key or "",
            model=self.config.llm_model,
            effort=self.config.llm_effort,
            payload=payload,
            candidates=candidates,
            min_confidence=self.config.min_confidence,
            timeout_s=self.config.llm_timeout_s,
            max_picks=free_slots,
        )

    async def _record_probability_shadow(
        self, payload: dict[str, Any], candidates: list[Routed], free_slots: int
    ) -> None:
        candidate_ids = [c.id for c in candidates]
        try:
            decision = await self._probability_decision(payload, candidates, free_slots)
        except Exception as exc:  # noqa: BLE001 - an optional observation must not fail a primary review
            self.store.journal("shadow_review_error", error=type(exc).__name__, candidate_ids=candidate_ids)
        else:
            self.store.journal("shadow_decision", decision=decision.as_dict(), candidate_ids=candidate_ids)

    async def _aggregator_decision(self, candidates: list[Routed], free_slots: int) -> brain.Decision:
        """The Kairos Aggregator review of every routed candidate, as one decision."""
        if self.aggregator is None or self.openai_key is None:
            raise AggregatorError("the Aggregator review needs its checkout and a model key")
        boundary = max(candidate.signal_time for candidate in candidates)
        if self._market_task is not None and not self._market_task.done():
            # A capture still running would deliver its snapshot after the decision anyway.
            self._market_task.cancel()
        market = self._market["items"] if self._market["boundary"] == boundary else {}
        started_after_close_s = round((time.time() * 1000 - boundary) / 1000, 2)
        deadline = min(c.route.review_deadline_ms for c in candidates)
        # Reserve the final half second for publishing the selected decision.
        remaining = (deadline - 500) / 1000 - time.time()
        try:
            if remaining <= 0:
                raise TimeoutError("review window has elapsed")
            if self._aggregator_task is not None and not self._aggregator_task.done():
                raise AggregatorError("previous review is still settling")
            self._aggregator_task = asyncio.create_task(
                review_candidates(
                    self.aggregator,
                    candidates=candidates,
                    market=market,
                    closed_bars=self._review_bars,
                    signals=self.news.raw_signals() if self.news else [],
                    macro=self.macro.allocation_message() if self.macro else None,
                    openai_key=self.openai_key,
                    ledger=self.store.directory / "llm-budget.json",
                    monthly_budget_usd=self.config.review_budget_usd,
                    session=self._aggregator_session,
                )
            )
            # Leave any dispatched usage reservation to settle. A late result cannot authorize an entry.
            self._aggregator_task.add_done_callback(self._observe_review_completion)
            async with asyncio.timeout(remaining):
                result = await asyncio.shield(self._aggregator_task)
            if time.time() * 1000 >= deadline:
                raise TimeoutError("review returned after the route deadline")
        except (TimeoutError, AggregatorError, OSError) as exc:
            self._reviews = {}
            self.store.journal(
                "aggregator_error",
                kind=type(exc).__name__,
                message=str(exc)[:300] or "review window elapsed while waiting for worker",
                done_after_close_s=round((time.time() * 1000 - boundary) / 1000, 3),
            )
            return brain.Decision(
                "WAIT",
                None,
                0.0,
                f"Aggregator review failed: {type(exc).__name__}",
                "AGGREGATOR",
                problems=[str(exc)[:200]],
            )
        reviews = result.records
        self._reviews = dict(result.contracts)
        self.store.journal(
            "aggregator_review",
            revisions=self._aggregator_revisions,
            reviews=reviews,
            ingest_problems=result.ingest_problems[:20],
            market_snapshots=sorted(market),
            timing={
                "start_after_close_s": started_after_close_s,
                "done_after_close_s": round((time.time() * 1000 - boundary) / 1000, 2),
                **result.timing,
            },
        )
        picks = allowed_in_order(reviews, candidates, self._ranking_limit(candidates, free_slots))
        counts = {name: sum(r["decision"] == name for r in reviews) for name in ("ALLOW", "VETO", "DEFER")}
        top = next((r for r in reviews if picks and r["id"] == picks[0]), None)
        return brain.Decision(
            "TRADE" if picks else "WAIT",
            picks[0] if picks else None,
            (top["priority"] / 100.0) if top else 0.0,
            f"Aggregator: ALLOW {counts['ALLOW']}, VETO {counts['VETO']}, DEFER {counts['DEFER']}"
            + (f"; {', '.join(top['reason_codes'])}" if top else ""),
            "AGGREGATOR",
            picks=picks,
            verdicts=[
                {
                    "candidate_id": r["id"],
                    "verdict": r["decision"],
                    "confidence": r["priority"] / 100.0,
                    "reason": f"{r['tier']}: " + ", ".join(r["reason_codes"]),
                }
                for r in reviews
            ],
        )

    def _review_of(self, candidate: Routed) -> Any:
        """The Aggregator's CandidateReviewV1 of this candidate in the current scan, or None."""
        return self._reviews.get(candidate.id)

    async def try_enter(self, candidate: Routed, decision: brain.Decision) -> None:
        blocked = self.entries_blocked(candidate.instrument)
        if blocked:
            self.store.journal("entry_skipped", candidate=candidate.id, reason=blocked)
            return
        trades = self.trades()
        instrument = self.instruments.get(candidate.instrument)
        if instrument is None or not instrument.tradable:
            self.store.journal("entry_skipped", candidate=candidate.id, reason="instrument is not tradable")
            return
        book = await self.api.book(candidate.instrument)
        now_ms = int(time.time() * 1000)
        ref = book.ask if candidate.side == "LONG" else book.bid
        # The reference is the Binance close of the decision bar, so this also bounds the
        # price difference between the two venues.
        drift = abs(float(ref) - candidate.reference_price)
        spread_bps = float((book.ask - book.bid) / book.ask) * 1e4
        # Strategy Engine places the barriers on this tick (``price_increments``).
        stop, take = dec(repr(candidate.stop)), dec(repr(candidate.take))
        reason = None
        if stop % instrument.price_step or take % instrument.price_step:
            reason = "the exit barriers are not on the instrument's price tick"
        elif any(t["symbol"] == candidate.instrument for t in trades):
            reason = "there is already a position in this instrument; " + scale_in_status(
                self.store.directory, now_ms=now_ms, reward_risk=self.config.reward_risk
            )
        elif len(trades) >= self.slot_limit():
            reason = "no free position slot"
        elif now_ms > candidate.intent.entry_expires_ts_ms:
            reason = "the candidate's entry window has expired"
        elif abs(now_ms - book.time_ms) > 30_000:
            reason = "order book is stale"
        elif spread_bps > self.config.max_spread_bps:
            reason = f"spread {spread_bps:.1f} bps is wider than allowed"
        elif drift > self.config.max_entry_drift_atr * candidate.atr:
            reason = f"price moved {drift:.6g} since the signal bar, more than allowed"
        elif not (stop < ref < take if candidate.side == "LONG" else take < ref < stop):
            reason = "market is already beyond the stop or the take"
        quantity: Decimal | None = None
        equity_before: Decimal | None = None
        risk_decision = None
        review = self._review_of(candidate)
        if reason is None and review is None:
            reason = "the Aggregator has not reviewed this candidate"
        elif reason is None and self.risk is None:
            reason = "the Risk Manager is not available"
        if reason is None and self.risk is not None:
            scope, limits = self._risk_scope()
            step = instrument.quantity_step
            minimum = max(
                instrument.min_quantity,
                ceil_step(instrument.min_volume * Decimal("1.03") / ref, step),
            )
            allocation = self.macro.allocation() if self.macro else None
            try:
                if isinstance(self.broker, LiveBroker):
                    equity_before = await self.broker.prepare_admission({t["symbol"] for t in trades})
                outcome = await evaluate_entry(
                    self.risk,
                    review=review,
                    allocation={k: v for k, v in allocation.items() if k != "origin"} if allocation else None,
                    # Capture independent account and book observations concurrently.
                    account=self._account_evidence,
                    session=self.risk_session,
                    scope=scope,
                    limits=limits,
                    taker_fee_bps=float(self.fee) * 1e4,
                    assessed_notional_usd=limits["max_position_notional_usd"],
                    venue_max_leverage=instrument.max_leverage,
                    min_quantity=minimum,
                    adaptive={"outcomes": self.learning_outcomes()} if self.config.adaptive_risk else None,
                )
            except (NeedsOperator, PortfolioJournalError) as exc:
                self.halt(f"портфель требует сверки: {str(exc)[:160]}", reconcile_entry=True)
                return
            except (RiskError, EvedexError, TimeoutError, aiohttp.ClientError, OSError) as exc:
                reason = f"the Risk Manager could not decide ({type(exc).__name__}: {str(exc)[:120]})"
            else:
                self.store.journal("risk_decision", candidate=candidate.id, **outcome.record)
                if not outcome.approved:
                    reason = "Risk Manager: " + ", ".join(outcome.reasons)
                else:
                    risk_decision = outcome.decision
                    # The Risk Manager's quantity, down to the venue's lot: never more.
                    book = await self.api.book(candidate.instrument)
                    ref = book.ask if candidate.side == "LONG" else book.bid
                    quantity = floor_step(outcome.quantity, step)
                    minimum = max(
                        instrument.min_quantity,
                        ceil_step(instrument.min_volume * Decimal("1.03") / ref, step),
                    )
                    if quantity < minimum:
                        reason = (
                            f"the Risk Manager's size {outcome.quantity} is below "
                            f"the instrument's minimum lot {minimum}"
                        )
                        quantity = None
        if quantity is None or risk_decision is None:
            self.store.journal("entry_skipped", candidate=candidate.id, reason=reason)
            say(f"Вход пропущен ({candidate.instrument} {candidate.side}): {reason}")
            return
        if equity_before is None:
            equity_before = await self.broker.equity()
        leverage = max(1, int(risk_decision.leverage))
        policy = self.management_policy(now_ms, candidate.intent.strategy_id)
        target_args: dict[str, Any] = {}
        if policy and policy.targets() and isinstance(self.broker, LiveBroker):
            qualifies = getattr(self.broker, "native_ladder_qualified_for", None)
            if qualifies and qualifies(candidate.instrument):
                target_args["partial_targets"] = [
                    {"at_r": target.at_r, "fraction": target.fraction} for target in policy.targets()
                ]
        try:
            fill = await self.broker.enter(
                candidate,
                quantity,
                book,
                instrument,
                stop,
                take,
                leverage=leverage,
                decision=risk_decision,
                **target_args,
            )
        except CanaryAccountError as exc:
            self.store.journal("entry_skipped", candidate=candidate.id, reason=str(exc)[:200])
            return
        except (Unknown, NeedsOperator, PortfolioJournalError) as exc:
            self.halt(f"вход {candidate.instrument} требует сверки: {str(exc)[:160]}", reconcile_entry=True)
            return
        if fill is None:
            self.store.journal("entry_skipped", candidate=candidate.id, reason="execution refused the entry")
            return
        trade: dict[str, Any] = {
            "id": (
                fill["order_id"] if fill["order_id"] != "paper" else f"paper-{now_ms}-{candidate.instrument}"
            ),
            "symbol": candidate.instrument,
            "side": candidate.side,
            "quantity": str(fill["quantity"]),
            "entry_price": str(fill["price"]),
            "stop": str(stop),
            "take": str(take),
            "opened_ms": now_ms,
            "deadline_ms": now_ms + candidate.timeout_minutes * 60_000,
            "equity_before": str(equity_before),
            "candidate": candidate.record(),
            "decision": decision.as_dict(),
            # A solo trade can use the account movement as an estimate.
            "solo": not trades,
            # Lineage for the Risk Manager's account evidence while the position is open.
            "intent_id": candidate.id,
            "strategy_id": candidate.intent.strategy_id,
            "strategy_revision": candidate.intent.strategy_revision,
            "risk_decision_id": risk_decision.decision_id,
            "risk_trade_id": risk_decision.trade_id,
            "trade_id": fill.get("trade_id"),
            "order_id": fill["order_id"],
            "execution_provider": "execution_portfolio" if self.broker.mode == "live" else "paper",
            "first_fill_ms": now_ms,
            "max_holding_ms": candidate.intent.exit_plan.max_holding_ms,
            # The learning ledger's inputs: the regime in force and the trade's 1 R in USD.
            "signal_symbol": candidate.intent.symbol,
            "regime": (allocation or {}).get("regime") or "UNKNOWN",
            "decided_at_ms": risk_decision.decided_at_ms,
            "one_r_usd": str(fill["quantity"] * abs(dec(str(fill["price"])) - stop)),
            "leverage": leverage,
            # Position management fixed for this trade's life (None: the plan as it is).
            "initial_stop": str(stop),
            "initial_quantity": str(fill["quantity"]),
            "management": policy.as_dict() if policy else None,
            "native_partial_targets": bool(fill.get("native_partial_targets")),
        }
        for other in trades:
            other["solo"] = False
        trades.append(trade)
        self._day()["trades"] += 1
        self.store.save()
        self.store.journal("trade_opened", trade=trade)
        say(
            f"ВХОД {trade['symbol']} {trade['side']} {trade['quantity']} по {fill['price']}, "
            f"плечо {trade['leverage']}x; "
            f"стоп {stop}, тейк {take}. Решение: {decision.source}. Открыто позиций: {len(trades)}."
        )
        if not fill["protected"]:
            say("Биржа не приняла стоп-заявку. Закрываю позицию и останавливаю входы.")
            await self._finish(trade, "PROTECTION_FAILED", await self.api.book(trade["symbol"]), close=True)
            self.halt("биржа не приняла стоп-заявку; нужна проверка формата")

    # ---- managing open trades ----------------------------------------------

    async def manage(self, trade: dict[str, Any], now_ms: int) -> None:
        if now_ms < trade.get("retry_after_ms", 0):
            return
        is_open = await self.broker.position_open(trade)
        if trade.get("native_target_fills"):
            trade["partial_pnl_usd"] = str(
                sum(
                    estimate_pnl({**trade, "quantity": qty}, dec(price), self.fee)
                    for _, qty, price in trade["native_target_fills"]
                )
            )
        if not is_open:
            # One empty answer is not proof that the position is gone: the venue has had short
            # outages. Its protective orders are cancelled only after several polls agree.
            trade["missing_polls"] = trade.get("missing_polls", 0) + 1
            if trade["missing_polls"] < self.config.exit_confirm_polls:
                return
            await self._finish(trade, "VENUE_EXIT", None, close=False)
            return
        trade["missing_polls"] = 0
        book = await self.api.book(trade["symbol"])
        reason = "OPERATOR_CLOSE" if self.store.flag("CLOSE") else software_exit(trade, book, now_ms)
        if reason is None:
            early = self._regime_exit(trade["side"], trade.get("regime"), trade.get("decided_at_ms"))
            if early:
                trade["early_exit"] = early
                say(f"Досрочный выход {trade['symbol']} {trade['side']}: {early}.")
                reason = "REGIME_EXIT"
        if reason is None and trade.get("management"):
            await self._apply_management(trade, book)
        if reason is None:
            trade["breaches"] = 0
            if now_ms - self._status_at.get(trade["id"], 0) >= 600_000:
                self._status_at[trade["id"]] = now_ms
                price = book.ask if trade["side"] == "SHORT" else book.bid
                sign = 1 if trade["side"] == "LONG" else -1
                unrealized = sign * (price - dec(trade["entry_price"])) * dec(trade["quantity"])
                left = max(0, trade["deadline_ms"] - now_ms) // 60_000
                say(
                    f"Позиция {trade['symbol']} {trade['side']}: цена {price}, вход {trade['entry_price']}, "
                    f"стоп {trade['stop']}, тейк {trade['take']}, "
                    f"сейчас {unrealized:+.4f} USDT без комиссий, "
                    f"до выхода по времени {left // 60} ч {left % 60} мин."
                )
            return
        if self.broker.mode == "live" and reason in ("STOP", "TAKE"):
            # The exchange's own TP/SL should fire first. Step in only if the level is
            # still breached on the next poll and the position is still there.
            trade["breaches"] = trade.get("breaches", 0) + 1
            if trade["breaches"] < 2:
                return
        try:
            await self._finish(trade, reason, book, close=True)
        except NeedsOperator:
            trade["retry_after_ms"] = now_ms + 60_000
            self.store.save()
            raise

    async def _apply_management(self, trade: dict[str, Any], book: Book) -> None:
        """Move the stop toward less risk and take a partial profit, as the trade's policy says."""
        stop, part = management_step(trade, book, self.instruments.get(trade["symbol"]))
        if part is not None:
            price = await self.broker.partial_close(trade, book, part)
            if price is not None:
                if len(trade["management"].get("partial_targets", ())) <= 1:
                    trade["partial_done"] = "done"
                if self.broker.mode == "live":
                    # Partial fills remain price estimates until venue accounting is available.
                    partial = estimate_pnl({**trade, "quantity": str(part)}, price, self.fee)
                    trade["partial_pnl_usd"] = str(dec(trade.get("partial_pnl_usd", "0")) + partial)
                self.store.save()
                self.store.journal(
                    "partial_exit",
                    id=trade["id"],
                    quantity=str(part),
                    price=str(price),
                    left=trade["quantity"],
                )
                say(f"Частичный выход {trade['symbol']}: {part} по {price}, осталось {trade['quantity']}.")
                stop, _ = management_step(trade, book, self.instruments.get(trade["symbol"]))
        if stop is not None:
            before = trade["stop"]
            if await self.broker.move_stop(trade, stop):
                self.store.journal("stop_moved", id=trade["id"], old=before, new=str(stop))
                say(f"Стоп {trade['symbol']} перенесён {before} → {stop} (к меньшему риску).")
        self.store.save()

    async def _trade_result(
        self, trade: dict[str, Any], reason: str, book: Book | None, equity: Decimal
    ) -> tuple[Decimal, bool]:
        """This trade's result in USD and whether it is exact or estimated from prices."""
        if self.broker.mode == "paper" and "pnl_usd" in trade:
            return dec(trade["pnl_usd"]), True
        if trade.get("solo", True):
            return equity - dec(trade["equity_before"]), False
        if book is None:
            book = await self.api.book(trade["symbol"])
        exit_price = book.bid if trade["side"] == "LONG" else book.ask
        trade["exit_price"] = str(exit_price)
        return estimate_pnl(trade, exit_price, self.fee) + dec(trade.get("partial_pnl_usd", "0")), False

    async def _finish(self, trade: dict[str, Any], reason: str, book: Book | None, *, close: bool) -> None:
        if close:
            if book is None:
                raise NeedsOperator("a position cannot be closed without a fresh order book")
            await self.broker.close(trade, book, reason)
        receipt = await self.broker.settle(trade)
        equity = await self.broker.equity()
        pnl, exact = await self._trade_result(trade, reason, book, equity)
        origin = "operator" if reason == "OPERATOR_CLOSE" else "unknown"
        if self.broker.mode == "paper":
            basis = "paper_fills"
            if reason != "OPERATOR_CLOSE":
                origin = "system"
        else:
            basis = "account_change" if trade.get("solo", True) else "price_estimate"
            if receipt and receipt.get("exit_origin") in {"operator", "system", "unknown"}:
                origin = receipt["exit_origin"]
            if (
                receipt
                and receipt.get("pnl_basis") == "execution_fills"
                and receipt.get("net_pnl") is not None
            ):
                pnl, exact, basis = dec(receipt["net_pnl"]), True, "execution_fills"
        state = self.store.state
        trades = self.trades()
        trades[:] = [t for t in trades if t["id"] != trade["id"]]
        state.setdefault("cooldowns", {})[trade["symbol"]] = (
            int(time.time() * 1000) + self.config.cooldown_minutes * 60_000
        )
        session = state["session"]
        if self.broker.mode == "paper":
            session["realized_usd"] = str(dec(session.get("realized_usd", "0")) + pnl)
            day = self._day()
            day["realized_usd"] = str(dec(day.get("realized_usd", "0")) + pnl)
        if not trades:
            session["last_flat_equity"] = str(equity)
        if reason != "PROTECTION_FAILED":
            state["round_trips"] = int(state.get("round_trips", 0)) + 1
        r_multiple = (
            (pnl / dec(trade["one_r_usd"])).quantize(Decimal("0.0001"))
            if trade.get("one_r_usd") and dec(trade["one_r_usd"]) > 0
            else None
        )
        history = state.setdefault("history", [])
        history.append(
            {
                "symbol": trade["symbol"],
                "side": trade["side"],
                "setup": trade["candidate"].get("setup"),
                "exit": reason,
                "pnl_basis": basis,
                "exit_origin": origin,
                "mode": self.broker.mode,
                "risk_trade_id": trade.get("risk_trade_id"),
                "risk_decision_id": trade.get("risk_decision_id"),
                "result_r": float(r_multiple) if r_multiple is not None else None,
                "held_minutes": (int(time.time() * 1000) - trade["opened_ms"]) // 60_000,
                "closed_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
            }
        )
        del history[:-HISTORY_LENGTH]
        if reason == "REGIME_EXIT" and r_multiple is not None and trade.get("signal_symbol"):
            entry = dec(trade["entry_price"])
            self._queue_early_exit(
                {
                    "id": trade["id"],
                    "signal_symbol": trade["signal_symbol"],
                    "side": trade["side"],
                    "entry": str(entry),
                    "stop": trade["stop"],
                    "take": trade["take"],
                    "risk_price": str(
                        dec(trade["one_r_usd"]) / dec(trade.get("initial_quantity", trade["quantity"]))
                    ),
                    "start_ms": trade["opened_ms"],
                    "deadline_ms": trade["deadline_ms"],
                    "exit_r": str(r_multiple),
                }
            )
        self.store.state["pending_close_report"] = dict(
            close_report_id=f"{self.broker.mode}:{trade['id']}",
            ts=datetime.now(UTC).isoformat(timespec="milliseconds"),
            id=trade["id"],
            symbol=trade["symbol"],
            side=trade["side"],
            reason=reason,
            pnl_usd=str(pnl),
            pnl_exact=exact,
            mode=self.broker.mode,
            pnl_basis=basis,
            exit_origin=origin,
            trade_id=trade.get("trade_id"),
            decision_id=trade.get("risk_decision_id"),
            equity=str(equity),
            exit_price=trade.get("exit_price"),
            source=trade["decision"].get("source"),
            candidate_id=trade["candidate"].get("id"),
            strategy_id=trade.get("strategy_id"),
            signal_symbol=trade.get("signal_symbol"),
            regime=trade.get("regime"),
            decided_at_ms=trade.get("decided_at_ms"),
            r_multiple=None if r_multiple is None else str(r_multiple),
            early_exit=trade.get("early_exit"),
        )
        self.store.save()
        self.store.publish_close()
        say(
            f"ВЫХОД {trade['symbol']} ({reason}). Результат сделки: {pnl:+.4f} USDT"
            f"{'' if exact else ' (оценка)'}. Баланс: {equity:.4f} USDT. "
            f"Открыто позиций: {len(trades)}."
        )
        if isinstance(self.broker, LiveBroker):
            await self.broker.refresh_performance(force=True)
            return
        limit = Decimal(str(self.config.session_loss_stop_usd))
        loss = max(dec(session["start_equity"]) - equity, -dec(session["realized_usd"]))
        if loss >= limit:
            self.halt(f"убыток сессии {loss:.2f} USDT достиг лимита {self.config.session_loss_stop_usd}")

    # ---- the loop -----------------------------------------------------------

    async def step(self) -> None:
        now_ms = int(time.time() * 1000)
        trades = self.trades()
        failure: Exception | None = None
        slot = self.scan_slot(now_ms)
        # A scan that will run goes first: its review window closes 60 s after the bar, and each
        # open position costs venue round trips. Exchange stops protect them meanwhile, and they
        # are managed on the next step, seconds later.
        scan_first = (
            self.store.state.get("last_bar") != slot
            and now_ms - slot >= SCAN_AFTER_CLOSE_MS
            and self.slot_limit() - len(trades) > 0
            and not self._account_entry_block()
            and not any(self._protection_block(t["symbol"]) for t in trades)
        )
        for trade in [] if scan_first else list(trades):
            symbol = trade["symbol"]
            try:
                await self.manage(trade, now_ms)
            except (
                TimeoutError,
                EvedexError,
                PortfolioJournalError,
                CanaryAccountError,
                aiohttp.ClientError,
            ) as exc:
                fault = type(exc).__name__, str(exc)[:300]
                if self._position_faults.get(symbol) != fault:
                    self.store.journal(
                        "position_management_error", symbol=symbol, kind=fault[0], message=fault[1]
                    )
                    self._position_faults[symbol] = fault
                if failure is None or (
                    isinstance(exc, (NeedsOperator, PortfolioJournalError))
                    and not isinstance(failure, (NeedsOperator, PortfolioJournalError))
                ):
                    failure = exc  # a transient fault on another instrument cannot mask an ownership fault
            else:
                if self._position_faults.pop(symbol, None) is not None:
                    self.store.journal("position_management_recovered", symbol=symbol)
        if isinstance(self.broker, LiveBroker):
            try:
                await self.broker.refresh_performance()
            except (
                TimeoutError,
                EvedexError,
                PortfolioJournalError,
                CanaryAccountError,
                sqlite3.Error,
                aiohttp.ClientError,
            ) as refresh_error:
                if failure is not None:
                    raise failure from refresh_error
                raise
        if failure is not None:
            raise failure
        if self._account_entry_block():
            return
        free = self.slot_limit() - len(trades)
        if free <= 0:
            return
        boundary = self._market_due(now_ms)
        if boundary is not None:
            # In the background, so that positions are still managed while it runs.
            self._market_task = asyncio.create_task(self.capture_market(boundary))
        slot = self.scan_slot(now_ms)
        if self.store.state.get("last_bar") == slot or now_ms - slot < SCAN_AFTER_CLOSE_MS:
            return
        self.store.state["last_bar"] = slot
        self.store.save()
        features, candidates = await self.scan(now_ms)
        if self.macro is not None and (self._macro_task is None or self._macro_task.done()):
            # In the background: a slow strategic call must not delay this scan's decision.
            self._macro_task = asyncio.create_task(self.macro_tick(features, now_ms))
        busy = {t["symbol"] for t in trades}
        cooldowns = self.store.state.get("cooldowns", {})
        eligible = [
            c for c in candidates if c.instrument not in busy and now_ms >= cooldowns.get(c.instrument, 0)
        ]
        if not eligible:
            say(f"Скан: {len(features)} инструментов, новых сигналов нет. Открыто позиций: {len(trades)}.")
            return
        say("Скан: сигналы — " + ", ".join(f"{c.instrument} {c.side} {c.setup}" for c in eligible))
        self._reviews = {}
        decision = await self.decide(features, eligible, free)
        say(
            f"Решение ({decision.source}): {decision.action} {', '.join(decision.picks)} "
            f"— {decision.reason[:200]}"
        )
        for candidate_id in decision.picks:
            if self._account_entry_block() or len(self.trades()) >= self.slot_limit():
                break
            await self.try_enter(next(c for c in eligible if c.id == candidate_id), decision)

    async def run(
        self, *, max_steps: int | None = None, new_session: bool = False, resume_reconciled: bool = False
    ) -> None:
        news_task = None
        try:
            if self.engine is not None:
                self._router_session = router_session(self.router)
                self._router_warm_task = asyncio.create_task(
                    self._warm_review_worker("router", self._router_session)
                )
            if self.aggregator is not None and self.openai_key:
                self._aggregator_session = AggregatorSession(self.aggregator)
                self._aggregator_warm_task = asyncio.create_task(
                    self._warm_review_worker("aggregator", self._aggregator_session)
                )
            await self.start(new_session=new_session, resume_reconciled=resume_reconciled)
            if self.news is not None and max_steps is None:
                news_task = asyncio.create_task(self._news_loop())
            await self._loop(max_steps)
        finally:
            if news_task is not None:
                news_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await news_task
            await self.stop_engine_workers()

    @staticmethod
    def _observe_review_completion(task: asyncio.Task[Reviews]) -> None:
        if not task.cancelled():
            task.exception()

    async def _warm_review_worker(self, name: str, session: WorkerSession) -> None:
        started = time.monotonic()
        try:
            await session.warm()
        except SiblingError as exc:
            self.store.journal(f"{name}_warm_error", message=str(exc)[:300])
            return
        self.store.journal(f"{name}_warm", seconds=round(time.monotonic() - started, 3))

    async def stop_engine_workers(self) -> None:
        """End the background engine work and the persistent workers before the loop closes."""
        for task in (
            self._aggregator_task,
            self._probability_shadow,
            self._metrics_task,
            self._warm_task,
            self._adaptive_task,
            self._market_task,
            self._macro_task,
            self._router_warm_task,
            self._aggregator_warm_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, EngineError):
                    await task
        for pool in (self.right_tail_pool, self.hourly_pool):
            if pool is not None:
                await pool.stop()
        if self.risk_session is not None:
            await self.risk_session.stop()
        for session in (self._router_session, self._aggregator_session):
            if session is not None:
                await session.stop()

    def _heartbeat(self, started_at: str) -> None:
        """Replace ``heartbeat.json`` on every loop for the single-host alert monitor.

        The format is the one ``kairos_execution.production_alerts`` reads: the process
        id, when this process started and when it last completed a loop.
        """
        path = self.store.directory / "heartbeat.json"
        temporary = path.with_suffix(".tmp")
        beat = {"pid": os.getpid(), "started_at": started_at, "at": datetime.now(UTC).isoformat()}
        # On Windows the monitor's read briefly locks the file; the replace is retried.
        for attempt in range(5):
            try:
                temporary.write_text(json.dumps(beat), encoding="utf-8")
                temporary.replace(path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
            except OSError as exc:
                self.store.journal("heartbeat_error", message=str(exc)[:200])
                return
        self.store.journal("heartbeat_error", message="the heartbeat file stayed locked")

    async def _can_stop_flat(self) -> bool:
        if (
            self.trades()
            or self.store.state.get("execution_session")
            or self.store.state.get("pending_entry")
        ):
            return False
        stop = self.store.flag("STOP")
        if not isinstance(self.broker, LiveBroker):
            return stop
        now = datetime.now(UTC)
        expired = now >= self.broker.entry_deadline_utc
        if not (stop or expired) or time.monotonic() < self._next_flat_probe:
            return False
        self._next_flat_probe = time.monotonic() + 60
        if await self.broker.foreign_exposure(set()):
            return False
        if not expired:
            return True
        if self._pilot_journal_path is None:
            return False
        await self.broker.refresh_performance(force=True)
        status = read_pilot_status(self._pilot_journal_path)
        return bool(
            status
            and status["final_flat_observation"]
            and datetime.fromisoformat(status["observed_at"]) >= now
            and status["open_positions"] == status["unresolved_effects"] == 0
        )

    async def _loop(self, max_steps: int | None) -> None:
        errors = steps = 0
        last_fault: tuple[str, str] | None = None
        started_at = datetime.now(UTC).isoformat()
        while max_steps is None or steps < max_steps:
            steps += 1
            try:
                await self.step()
                if await self._can_stop_flat():
                    self._heartbeat(started_at)
                    say("Новые входы завершены. Открытых позиций и ожидающих операций нет.")
                    return
                errors = 0
                last_fault = None
            except (
                TimeoutError,
                EvedexError,
                PortfolioJournalError,
                CanaryAccountError,
                sqlite3.Error,
                aiohttp.ClientError,
            ) as exc:
                errors += 1
                fault = type(exc).__name__, str(exc)[:300]
                if last_fault != fault:
                    self.store.journal(
                        "error", kind=fault[0], message=fault[1], body=getattr(exc, "body", "")[:300]
                    )
                    say(f"Ошибка торгового цикла ({fault[0]}): {fault[1][:160]}")
                    last_fault = fault
                active_halt = self.store.state.get("halted")
                if isinstance(exc, (NeedsOperator, PortfolioJournalError)):
                    if not active_halt or active_halt == REPEATED_ERRORS_HALT:
                        self.halt(
                            MANUAL_CHECK_PREFIX + str(exc)[:200],
                            reconcile_entry=isinstance(exc, NeedsOperator) and str(exc) == POSITION_MISMATCH,
                        )
                elif errors >= 30 and not active_halt:
                    self.halt(REPEATED_ERRORS_HALT)
            self._heartbeat(started_at)
            await asyncio.sleep(self.pause_s(time.time()) if max_steps is None else 0)

    def pause_s(self, now_s: float) -> float:
        """Seconds until the next loop: the poll interval, or less to start a scan on time.

        The review window closes 60 s after the bar, so the scan starts right when its
        bar has had ``SCAN_AFTER_CLOSE_MS`` to reach Binance, not up to a poll later.
        """
        scan_ms = int(self.config.scan_minutes) * 60_000
        now_ms = now_s * 1000
        due_ms = (now_ms - SCAN_AFTER_CLOSE_MS) // scan_ms * scan_ms + scan_ms + SCAN_AFTER_CLOSE_MS
        return max(0.05, min(float(self.config.poll_seconds), (due_ms - now_ms) / 1000 + 0.05))
