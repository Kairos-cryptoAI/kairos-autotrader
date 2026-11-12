"""The trading loop: scan, decide, size, enter, protect, exit, account.

A few positions at a time, never two in one instrument. Hard ceilings below are code
constants, not configuration: raising them is a deliberate code change, not a typo in a
JSON file.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_UP, Decimal
from pathlib import Path
from typing import Any

import aiohttp

from . import brain, strategy
from .aggregator_review import (
    AggregatorConfig,
    AggregatorError,
    allowed_in_order,
    review_candidates,
    revisions,
)
from .evedex import (
    Book,
    Evedex,
    EvedexError,
    Instrument,
    OrderSigner,
    Rejected,
    Unknown,
    ceil_step,
    dec,
    floor_step,
    new_order_id,
    plain,
    round_step,
)
from .system_rules import MacroDesk, RulesConfig, RulesError, bias_of, risk_review
from .system_rules import revisions as rules_revisions
from .text_scouts import NewsDesk, TextScoutsError

HARD_MAX_CAPITAL_USD = Decimal(100)
HARD_MAX_RISK_PER_TRADE_USD = Decimal(1)
HARD_MAX_SESSION_LOSS_USD = Decimal(10)
HARD_MAX_TRADES_PER_DAY = 12
HARD_MAX_OPEN_POSITIONS = 3
HISTORY_LENGTH = 20
HOUR_MS = 60 * 60_000
BAR_MS = HOUR_MS  # signals come from closed 1h bars
TREND_MS = 4 * HOUR_MS
QUARTER_MS = 15 * 60_000  # the candles the hourly bars are rebuilt from
STRATEGY_IDS = ("BREAKOUT", "PULLBACK_RECLAIM")  # the setups of strategy.analyse, as Macro and Risk know them


# The venue's refusals of a TP/SL whose price is on the wrong side of the market for the
# given ``side``, e.g. ``stop_loss_price_above_last_price_error``.
PRICE_SIDE_LABEL = re.compile(r"^(stop_loss|take_profit)_price_(above|below)_(last|order)_price_error$")


def wrong_protection_side(exc: Rejected) -> bool:
    return bool(PRICE_SIDE_LABEL.match(exc.label)) or "price should be" in exc.body.lower()


class ConfigError(ValueError):
    pass


class NeedsOperator(EvedexError):
    """The account is in a state the engine must not guess about: stop new entries."""


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
    capital_basis_usd: float = 100.0
    risk_per_trade_usd: float = 0.25
    session_loss_stop_usd: float = 5.0
    max_trades_per_day: int = 12
    max_open_positions: int = 3
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
    review_mode: str = "probability"
    review_shadow: bool = True
    # The Kairos Macro Strategist: one strategic allocation a day, and after a price crash.
    use_macro: bool = True
    macro_effort: str = "high"
    # Who admits and sizes an entry: "autotrader" is this program's own sizing; "kairos"
    # additionally requires the Kairos Risk Manager's approval and takes the smaller size.
    # With risk_shadow the Risk Manager judges every candidate and is only recorded.
    risk_mode: str = "autotrader"
    risk_shadow: bool = True
    use_text_scouts: bool = True  # the Kairos Text Scouts news layer
    news_poll_minutes: float = 5.0
    max_news_llm_calls_per_day: int = 150
    chain_ids: list[int] = field(default_factory=lambda: [42161, 161803])
    poll_seconds: float = 5.0
    scan_minutes: int = 15  # how often the market is scanned: 15, 30 or 60
    exit_confirm_polls: int = 3  # polls in a row without the position before it counts as closed

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
                if key not in cls.__dataclass_fields__:
                    raise ConfigError(f"unknown config key: {key}")
                setattr(config, key, value)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.symbols or any(not isinstance(s, str) or not s.isalnum() for s in self.symbols):
            raise ConfigError("symbols must be plain instrument names like SOLUSD")
        if not 0 < Decimal(str(self.capital_basis_usd)) <= HARD_MAX_CAPITAL_USD:
            raise ConfigError(f"capital_basis_usd must be within (0, {HARD_MAX_CAPITAL_USD}]")
        if not 0 < Decimal(str(self.risk_per_trade_usd)) <= HARD_MAX_RISK_PER_TRADE_USD:
            raise ConfigError(f"risk_per_trade_usd must be within (0, {HARD_MAX_RISK_PER_TRADE_USD}]")
        if not 0 < Decimal(str(self.session_loss_stop_usd)) <= HARD_MAX_SESSION_LOSS_USD:
            raise ConfigError(f"session_loss_stop_usd must be within (0, {HARD_MAX_SESSION_LOSS_USD}]")
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
        if self.risk_mode not in ("autotrader", "kairos") or self.macro_effort not in (
            "low",
            "medium",
            "high",
        ):
            raise ConfigError("risk_mode must be autotrader or kairos and macro_effort low/medium/high")
        if self.review_mode not in ("probability", "aggregator"):
            raise ConfigError("review_mode must be probability or aggregator")
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


def size_position(
    *,
    side: str,
    book: Book,
    stop: Decimal,
    instrument: Instrument,
    risk_usd: Decimal,
    max_notional: Decimal,
    available: Decimal,
    fee: Decimal,
    slippage: Decimal,
) -> tuple[Decimal | None, str]:
    """Quantity whose stop-out, with round trip costs, loses at most ``risk_usd``."""
    ref = book.ask if side == "LONG" else book.bid
    distance = ref - stop if side == "LONG" else stop - ref
    if distance <= 0:
        return None, "stop is on the wrong side of the market"
    per_unit = distance + ref * 2 * (fee + slippage)
    step = instrument.quantity_step
    quantity = floor_step(risk_usd / per_unit, step)
    cap = min(max_notional, available * Decimal("0.9"))
    quantity = min(quantity, floor_step(cap / ref, step))
    minimum = max(instrument.min_quantity, ceil_step(instrument.min_volume * Decimal("1.03") / ref, step))
    if quantity < minimum:
        return None, f"minimum lot {minimum} would risk more than {risk_usd} USD or exceed the capital"
    return quantity, "ok"


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


def estimate_pnl(trade: dict[str, Any], exit_price: Decimal, fee: Decimal) -> Decimal:
    """Result of one trade from its prices and the taker fee (used when positions overlap)."""
    quantity, entry = dec(trade["quantity"]), dec(trade["entry_price"])
    sign = 1 if trade["side"] == "LONG" else -1
    return sign * (exit_price - entry) * quantity - (entry + exit_price) * quantity * fee


def nearest_level(trade: dict[str, Any], price: Decimal) -> Decimal:
    """The stop or the take, whichever the market is closer to: where the venue exit happened."""
    stop, take = dec(trade["stop"]), dec(trade["take"])
    return stop if abs(price - stop) <= abs(price - take) else take


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

    def flag(self, name: str) -> bool:
        return (self.directory / name).exists()


def say(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')} UTC] {message}", flush=True)


class PaperBroker:
    """Simulated fills at the top of the real book, with taker fee and slippage."""

    mode = "paper"

    def __init__(self, store: Store, config: Config, fee: Decimal) -> None:
        self._store, self._fee = store, fee
        self._slip = Decimal(str(config.slippage_fraction))
        store.state.setdefault("paper_equity", str(config.capital_basis_usd))

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
        candidate: strategy.Candidate,
        quantity: Decimal,
        book: Book,
        instrument: Instrument,
        stop: Decimal,
        take: Decimal,
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
        trade["pnl_usd"] = str(pnl)
        self._store.state["paper_equity"] = str(await self.equity() + pnl)

    async def settle(self, trade: dict[str, Any]) -> None:
        return None


class LiveBroker:
    """Real orders on EVEDEX. Entry by market IOC, then stop-loss and take-profit."""

    mode = "live"

    def __init__(
        self,
        api: Evedex,
        store: Store,
        config: Config,
        wallet_key: str,
        settle_seconds: float = 1.0,
        cache_seconds: float = 3.0,
    ) -> None:
        self._api, self._store, self._config, self._key = api, store, config, wallet_key
        self._settle_seconds = settle_seconds  # pause between reconciliation reads
        self._cache_seconds = cache_seconds  # routine "is it still open" checks share one read
        self._positions: list[Any] = []
        self._positions_at: float | None = None

    def _signer(self, chain_id: int) -> OrderSigner:
        return OrderSigner(self._key, chain_id)

    def _chains(self) -> list[int]:
        confirmed = self._store.state.get("chain_id")
        return [confirmed] if confirmed else list(self._config.chain_ids)

    async def equity(self) -> Decimal:
        return (await self._api.balance())[0]

    async def available(self) -> Decimal:
        return (await self._api.balance())[1]

    async def _read_positions(self, max_age: float = 0.0) -> list[Any]:
        fresh = self._positions_at is not None and time.monotonic() - self._positions_at <= max_age
        if not (max_age > 0 and fresh):
            self._positions = await self._api.positions()
            self._positions_at = time.monotonic()
        return self._positions

    async def foreign_exposure(self, own: set[str]) -> list[str]:
        """Positions and resting orders in instruments this engine has no trade in."""
        found = [
            f"position {p.instrument} {p.side} {p.quantity}"
            for p in await self._read_positions()
            if p.instrument not in own
        ]
        found += [
            f"open order {o.get('instrument')} {o.get('id')}"
            for o in await self._api.open_orders()
            if o.get("instrument") not in own
        ]
        return found

    async def position(self, instrument: str, max_age: float = 0.0):
        for position in await self._read_positions(max_age):
            if position.instrument == instrument:
                return position
        return None

    async def position_open(self, trade: dict[str, Any]) -> bool:
        return await self.position(trade["symbol"], self._cache_seconds) is not None

    def _protection_sides(self, position_side: str, *, every: bool = False) -> list[str]:
        """Which ``side`` a TP/SL must carry, most likely reading first.

        Project research says it is the protected position's side. If the venue refuses
        that on a price-direction check, the other reading is tried. The venue validates
        the stop price against the market for the given side, so only the reading that
        describes a real protective stop can be accepted. ``every`` returns both readings
        even when one has been confirmed before.
        """
        other = "SELL" if position_side == "BUY" else "BUY"
        mode = self._store.state.get("tpsl_side_mode")
        ordered = [other, position_side] if mode == "closing" else [position_side, other]
        return ordered if every or mode not in ("position", "closing") else ordered[:1]

    def _remember_side(self, position_side: str, used: str) -> None:
        self._store.state["tpsl_side_mode"] = "position" if used == position_side else "closing"

    async def enter(
        self,
        candidate: strategy.Candidate,
        quantity: Decimal,
        book: Book,
        instrument: Instrument,
        stop: Decimal,
        take: Decimal,
    ) -> dict[str, Any] | None:
        store, symbol = self._store, candidate.symbol
        side = "BUY" if candidate.side == "LONG" else "SELL"
        ref = book.ask if side == "BUY" else book.bid
        # The venue turns cash into quantity at the fill price. A 0.1% margin keeps a
        # one-tick move during the request from rounding the quantity down a whole step.
        cash = (quantity * ref * Decimal("1.001")).quantize(Decimal("0.01"), rounding=ROUND_UP)
        attempts = [
            (chain, tpsl_side) for chain in self._chains() for tpsl_side in self._protection_sides(side)
        ]
        order_id, signer, sent = "", None, False
        skip_chain: int | None = None
        leverage_fixed = False
        while attempts:
            chain, tpsl_side = attempts.pop(0)
            if chain == skip_chain:
                continue
            signer = self._signer(chain)
            order_id = new_order_id()
            protection = [
                {"type": "stop-loss", "side": tpsl_side, "quantity": "0", "price": plain(stop)},
                {"type": "take-profit", "side": tpsl_side, "quantity": "0", "price": plain(take)},
            ]
            store.state["pending_entry"] = {
                "order_id": order_id,
                "symbol": symbol,
                "side": candidate.side,
                "stop": str(stop),
                "take": str(take),
                "candidate": candidate.as_dict(),
            }
            store.save()
            store.journal(
                "entry_sent",
                order_id=order_id,
                symbol=symbol,
                side=side,
                cash=str(cash),
                chain=chain,
                tpsl_side=tpsl_side,
            )
            self._positions_at = None
            try:
                answer = await self._api.post_market_order(
                    signer.market_order(
                        order_id=order_id, instrument=symbol, side=side, cash=cash, protection=protection
                    )
                )
                store.journal(
                    "entry_answer",
                    order_id=order_id,
                    **{
                        key: (answer or {}).get(key)
                        for key in (
                            "status",
                            "quantity",
                            "unFilledQuantity",
                            "filledAvgPrice",
                            "fee",
                            "rejectedReason",
                        )
                    },
                )
            except Rejected as exc:
                store.journal(
                    "entry_rejected",
                    order_id=order_id,
                    status=exc.status,
                    label=exc.label,
                    body=exc.body,
                    chain=chain,
                    tpsl_side=tpsl_side,
                )
                store.state.pop("pending_entry", None)
                store.save()
                say(f"Биржа отклонила заявку: HTTP {exc.status} {exc.label or exc.body[:160]}")
                if exc.label == "invalid_signature_error":
                    skip_chain = chain  # the other signing chain, if any, comes next
                    continue
                if wrong_protection_side(exc):
                    continue  # the other reading of the TP/SL side comes next
                if exc.label == "invalid_order_leverage_error" and not leverage_fixed:
                    # The account keeps its own leverage per instrument and refuses an order
                    # that names another one. This engine trades at 1x only, so the setting
                    # is brought to 1x and the same order is sent once more.
                    leverage_fixed = True
                    if await self._set_leverage_1x(symbol):
                        attempts.insert(0, (chain, tpsl_side))
                        continue
                return None
            except Unknown as exc:
                store.journal("entry_unknown", order_id=order_id, error=str(exc))
                say("Ответ биржи на заявку не получен. Проверяю, появилась ли позиция.")
                sent = True
                break
            store.state["chain_id"] = chain  # the venue verified this signature
            store.save()
            sent = True
            break
        if not sent:
            return None

        position = None
        for _ in range(15):
            await asyncio.sleep(self._settle_seconds)
            try:
                position = await self.position(symbol)
                if position is not None:
                    break
                order = await self._api.order(order_id)
            except EvedexError:
                continue
            if order and order.get("status") in ("CANCELLED", "REJECTED", "EXPIRED", "ERROR"):
                store.journal(
                    "entry_not_filled",
                    order_id=order_id,
                    status=order.get("status"),
                    reason=order.get("rejectedReason"),
                )
                store.state.pop("pending_entry", None)
                store.save()
                say(f"Заявка не исполнилась: {order.get('status')} {order.get('rejectedReason') or ''}")
                return None
        if position is None:
            store.journal("entry_outcome_unknown", order_id=order_id)
            raise NeedsOperator("entry outcome is unknown: no position and no terminal order after 15 s")
        if position.side != side:
            raise NeedsOperator("venue position side differs from the order side")

        store.journal(
            "entry_filled",
            order_id=order_id,
            quantity=str(position.quantity),
            price=str(position.avg_price),
            stop_attached=position.has_stop,
        )
        protected = position.has_stop
        try:
            rows = await self._api.active_tpsl(symbol)
            store.journal(
                "protection_state",
                symbol=symbol,
                rows=[
                    {key: row.get(key) for key in ("id", "type", "side", "quantity", "price", "status")}
                    for row in rows
                ],
            )
            protected = protected or any(row.get("type") == "stop-loss" for row in rows)
        except EvedexError as exc:
            store.journal("protection_state_unknown", symbol=symbol, error=str(exc)[:200])
        if protected:
            self._remember_side(side, tpsl_side)  # the attached stop is live, so this reading was accepted
        else:
            protected = await self._protect(symbol, side, stop, "stop-loss")
            if protected:
                await self._protect(symbol, side, take, "take-profit")
        store.state.pop("pending_entry", None)
        store.save()
        return {
            "quantity": position.quantity,
            "price": position.avg_price,
            "order_id": order_id,
            "protected": protected,
        }

    async def _set_leverage_1x(self, symbol: str) -> bool:
        try:
            await self._api.set_leverage(symbol, 1)
        except EvedexError as exc:
            self._store.journal(
                "leverage_change_failed",
                symbol=symbol,
                error=str(exc)[:200],
                body=getattr(exc, "body", "")[:300],
            )
            say(f"Не удалось поставить плечо 1x для {symbol}: {getattr(exc, 'label', '') or str(exc)[:120]}")
            return False
        self._store.journal("leverage_set", symbol=symbol, leverage=1)
        say(f"На счёте для {symbol} стояло другое плечо. Поставил 1x и отправляю заявку повторно.")
        return True

    async def _protect(self, symbol: str, position_side: str, price: Decimal, kind: str) -> bool:
        """Place one full-position TP/SL (quantity zero means the whole position).

        Tries the signing chains and both readings of ``side`` until the venue accepts one.
        """
        for chain in self._chains():
            signer = self._signer(chain)
            for tpsl_side in self._protection_sides(position_side, every=True):
                body = signer.tpsl(
                    instrument=symbol, kind=kind, side=tpsl_side, quantity=Decimal(0), price=price
                )
                try:
                    response = await self._api.post_tpsl(body)
                except Rejected as exc:
                    self._store.journal(
                        "protection_rejected",
                        kind=kind,
                        status=exc.status,
                        label=exc.label,
                        body=exc.body,
                        tpsl_side=tpsl_side,
                        chain=chain,
                    )
                    say(f"Биржа отклонила {kind}: HTTP {exc.status} {exc.label or exc.body[:160]}")
                    if exc.label == "invalid_signature_error":
                        break  # the other signing chain, if any, comes next
                    if wrong_protection_side(exc):
                        continue
                    return False
                except Unknown as exc:
                    self._store.journal("protection_unknown", kind=kind, error=str(exc))
                    try:
                        await asyncio.sleep(self._settle_seconds * 2)
                        return any(row.get("type") == kind for row in await self._api.active_tpsl(symbol))
                    except EvedexError:
                        return False
                self._store.state["chain_id"] = chain
                self._remember_side(position_side, tpsl_side)
                self._store.journal(
                    "protection_placed",
                    kind=kind,
                    price=str(price),
                    tpsl_side=tpsl_side,
                    id=(response or {}).get("id"),
                    status=(response or {}).get("status"),
                )
                return True
        return False

    async def ensure_stop(self, trade: dict[str, Any]) -> bool:
        """After a restart: make sure a live stop-loss exists for the adopted position."""
        symbol = trade["symbol"]
        if any(row.get("type") == "stop-loss" for row in await self._api.active_tpsl(symbol)):
            return True
        position = await self.position(symbol)
        if position is None or position.has_stop:
            return True
        return await self._protect(symbol, position.side, dec(trade["stop"]), "stop-loss")

    async def close(self, trade: dict[str, Any], book: Book, reason: str) -> None:
        symbol = trade["symbol"]
        for attempt in range(2):
            for chain in self._chains():
                position = await self.position(symbol)
                if position is None:
                    return
                order_id = new_order_id()
                self._store.journal(
                    "close_sent",
                    order_id=order_id,
                    symbol=symbol,
                    reason=reason,
                    attempt=attempt,
                    chain=chain,
                )
                self._positions_at = None
                try:
                    answer = await self._api.post_close(
                        self._signer(chain).close_order(
                            order_id=order_id, instrument=symbol, quantity=position.quantity
                        )
                    )
                    self._store.state["chain_id"] = chain
                    self._store.journal(
                        "close_answer",
                        order_id=order_id,
                        **{
                            key: (answer or {}).get(key)
                            for key in ("status", "filledAvgPrice", "realizedPnL", "fee", "rejectedReason")
                        },
                    )
                except Rejected as exc:
                    self._store.journal(
                        "close_rejected", order_id=order_id, status=exc.status, label=exc.label, body=exc.body
                    )
                    if exc.label == "invalid_signature_error":
                        continue  # the other signing chain, if any, comes next
                except Unknown as exc:
                    self._store.journal("close_unknown", order_id=order_id, error=str(exc))
                break
            for _ in range(10):
                await asyncio.sleep(self._settle_seconds)
                try:
                    if await self.position(symbol) is None:
                        return
                except EvedexError:
                    continue
        raise NeedsOperator(f"could not close the {symbol} position; the exchange stop-loss stays in place")

    async def settle(self, trade: dict[str, Any]) -> None:
        """Cancel protective orders left behind after the position is gone."""
        symbol = trade["symbol"]
        try:
            leftovers = await self._api.active_tpsl(symbol)
        except EvedexError:
            return
        for row in leftovers:
            try:
                await self._api.cancel_tpsl(symbol, str(row.get("id")))
                self._store.journal("protection_cancelled", id=row.get("id"), kind=row.get("type"))
            except EvedexError as exc:
                self._store.journal("protection_cancel_failed", id=row.get("id"), error=str(exc))
        await asyncio.sleep(self._settle_seconds * 2)


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
    ) -> None:
        self.api, self.session, self.store, self.config = api, session, store, config
        self.broker, self.fee, self.instruments, self.openai_key = broker, fee, instruments, openai_key
        self._extras: dict[str, dict[str, Any]] = {}
        self._status_at: dict[str, int] = {}  # when each position was last reported on screen
        self.aggregator: AggregatorConfig | None = None
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
                self.macro = MacroDesk(store, rules, STRATEGY_IDS)
            except RulesError as exc:
                say(f"Macro Strategist отключён: {exc}")
        self.risk_rules: RulesConfig | None = None
        if config.risk_mode == "kairos" or config.risk_shadow:
            try:
                rules.check_risk()
                self.risk_rules = rules
            except RulesError as exc:
                say(
                    f"Risk Manager недоступен: {exc}"
                    + (" Входы в режиме kairos будут пропускаться." if config.risk_mode == "kairos" else "")
                )
        self._rules_revisions = rules_revisions(rules) if (self.macro or self.risk_rules) else {}
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

    def _market_view(self, features: list[strategy.Features]) -> dict[str, dict[str, Any]]:
        view: dict[str, dict[str, Any]] = {}
        for f in features:
            extras = self._extras.get(f.symbol, {})
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

    async def _risk_account(self) -> dict[str, Any]:
        equity, available = await self.broker.equity(), await self.broker.available()
        day = self._day()
        if not day.get("start_equity"):
            day["start_equity"] = str(equity)
        day["peak_equity"] = str(max(equity, dec(day.get("peak_equity") or equity)))
        return {
            "equity_usd": float(equity),
            "available_usd": float(available),
            "day_start_equity_usd": float(dec(day["start_equity"])),
            "peak_equity_usd": float(dec(day["peak_equity"])),
            "daily_realized_pnl_usd": float(dec(day.get("realized_usd", "0"))),
            "positions": [
                {
                    "symbol": t["symbol"],
                    "side": t["side"],
                    "quantity": float(dec(t["quantity"])),
                    "entry_price": float(dec(t["entry_price"])),
                    "stop": float(dec(t["stop"])),
                    "strategy_id": t["candidate"].get("setup"),
                }
                for t in self.trades()
            ],
        }

    def _risk_item(
        self, candidate: strategy.Candidate, book: Book, stop: Any = None, take: Any = None
    ) -> dict:
        depth = book.ask_qty * book.ask if candidate.side == "LONG" else book.bid_qty * book.bid
        return {
            "id": candidate.id,
            "symbol": candidate.symbol,
            "side": candidate.side,
            "strategy_id": candidate.setup,
            "stop": float(stop if stop is not None else candidate.stop),
            "take": float(take if take is not None else candidate.take),
            "bid": float(book.bid),
            "ask": float(book.ask),
            "depth_usd": float(depth),
            "slippage_bps": self.config.slippage_fraction * 1e4,
            "taker_fee_bps": float(self.fee) * 1e4,
        }

    async def _risk_decisions(self, items: list[dict[str, Any]], now_ms: int) -> dict[str, dict[str, Any]]:
        """The Risk Manager's verdict per candidate id. Raises ``RulesError`` when it cannot be asked."""
        if self.risk_rules is None:
            raise RulesError("Risk Manager is not available")
        now_ms = int(time.time() * 1000)  # judged at this moment, never before the allocation it uses
        result = await risk_review(
            self.risk_rules,
            candidates=items,
            account=await self._risk_account(),
            allocation=self.macro.allocation() if self.macro else None,
            now_ms=now_ms,
        )
        decisions = {d["id"]: d for d in result["decisions"]}
        self.store.journal(
            "risk_review",
            revisions=self._rules_revisions,
            limits=result.get("limits"),
            regime=(self.macro.allocation() or {}).get("regime") if self.macro else None,
            decisions=[
                {
                    k: d.get(k)
                    for k in (
                        "id",
                        "approved",
                        "rejection_reasons",
                        "quantity",
                        "notional_usd",
                        "loss_budget_usd",
                        "worst_case_loss_usd",
                    )
                }
                for d in decisions.values()
            ],
        )
        return decisions

    async def risk_shadow(self, candidates: list[strategy.Candidate], now_ms: int) -> None:
        """Record what the Risk Manager says about every candidate. Never affects trading."""
        if self.risk_rules is None or not self.config.risk_shadow or not candidates:
            return
        try:
            books = await asyncio.gather(*(self.api.book(c.symbol) for c in candidates))
            await self._risk_decisions(
                [self._risk_item(c, b) for c, b in zip(candidates, books, strict=False)], now_ms
            )
        except (TimeoutError, RulesError, EvedexError, aiohttp.ClientError, OSError) as exc:
            self.store.journal("risk_error", kind=type(exc).__name__, message=str(exc)[:300])

    async def macro_tick(self, features: list[strategy.Features], now_ms: int) -> None:
        """Ask the Macro Strategist if an allocation is due. Its failures never reach the trading loop."""
        if self.macro is None or not features:
            return
        try:
            account = await self._risk_account()
            allocation = await self.macro.refresh(
                self.session,
                openai_key=self.openai_key,
                model=self.config.llm_model,
                effort=self.config.macro_effort,
                market=self._market_view(features),
                portfolio={k: account[k] for k in ("equity_usd", "available_usd", "positions")},
                performance={
                    "daily_realized_pnl_usd": account["daily_realized_pnl_usd"],
                    "session_realized_usd": self.store.state.get("session", {}).get("realized_usd"),
                    "trades_today": self._day()["trades"],
                    "recent_closed_trades": self.store.state.get("history", [])[-20:],
                },
                now_ms=now_ms,
            )
        except Exception as exc:  # noqa: BLE001 - the strategic layer must never stop, or disturb, trading
            self.store.journal("macro_error", kind=type(exc).__name__, message=str(exc)[:300])
            return
        if allocation:
            weights = ", ".join(
                f"{name} {weight:.0%}" for name, weight in allocation["strategy_weights"].items()
            )
            say(
                f"Macro Strategist: режим {allocation['regime']}, "
                f"резерв {allocation['stable_reserve_pct']:.0%}, "
                f"доли: {weights or 'нет'}"
                + (" (защитное решение: модель не ответила)" if allocation.get("origin") != "model" else "")
            )

    def halt(self, reason: str) -> None:
        if self.store.state.get("halted") != reason:
            self.store.state["halted"] = reason
            self.store.save()
            self.store.journal("halt", reason=reason)
            say(f"НОВЫЕ ВХОДЫ ОСТАНОВЛЕНЫ: {reason}")

    def entries_blocked(self) -> str | None:
        state = self.store.state
        if state.get("halted"):
            return str(state["halted"])
        if self.store.flag("STOP") or self.store.flag("CLOSE"):
            return "stop file present"
        if self._day()["trades"] >= self.config.max_trades_per_day:
            return "daily trade limit reached"
        return None

    def slot_limit(self) -> int:
        """How many positions may be open at once right now.

        In live mode only one position is allowed until one full trade (entry, exchange
        protection, exit, accounting) has completed on the real venue.
        """
        if self.broker.mode == "live" and not self.store.state.get("round_trips"):
            return 1
        return int(self.config.max_open_positions)

    async def start(self) -> None:
        state = self.store.state
        if state.get("mode") not in (None, self.broker.mode):
            raise ConfigError("this runtime directory belongs to another mode; use a separate directory")
        state["mode"] = self.broker.mode
        trades = self.trades()
        pending = state.get("pending_entry")
        if isinstance(self.broker, LiveBroker):
            own = {t["symbol"] for t in trades}
            if pending and pending["symbol"] not in own:
                position = await self.broker.position(pending["symbol"])
                if position is not None:
                    trade = self._adopt(
                        pending, position.quantity, position.avg_price, state.get("session", {})
                    )
                    trade["solo"] = not trades
                    for other in trades:
                        other["solo"] = False
                    trades.append(trade)
                    own.add(trade["symbol"])
                    say(f"Найдена позиция после сбоя на входе: {trade['symbol']}. Беру под управление.")
            state.pop("pending_entry", None)
            foreign = await self.broker.foreign_exposure(own)
            if foreign:
                self.halt("на счёте есть чужая позиция или заявка: " + "; ".join(foreign)[:300])
            for trade in list(trades):
                if not await self.broker.ensure_stop(trade):
                    say(f"Не удалось поставить стоп на бирже для позиции {trade['symbol']}. Закрываю её.")
                    await self._finish(
                        trade, "PROTECTION_FAILED", await self.api.book(trade["symbol"]), close=True
                    )
                    self.halt("биржа не приняла стоп-заявку; нужна проверка формата")
        if "session" not in state:
            equity = await self.broker.equity()
            state["session"] = {
                "start_equity": str(equity),
                "started_at": datetime.now(UTC).isoformat(),
                "realized_usd": "0",
            }
            self.store.journal(
                "session_start", equity=str(equity), mode=self.broker.mode, config=asdict(self.config)
            )
        self.store.save()
        say(
            f"Режим: {self.broker.mode}. Стартовый баланс сессии: {state['session']['start_equity']} USDT. "
            f"Позиций одновременно: до {self.slot_limit()}"
            + (
                f" (после первой полной сделки — до {self.config.max_open_positions})."
                if self.slot_limit() < self.config.max_open_positions
                else "."
            )
        )
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

    def _adopt(
        self, pending: dict[str, Any], quantity: Decimal, price: Decimal, session: dict
    ) -> dict[str, Any]:
        candidate = pending["candidate"]
        return {
            "id": pending["order_id"],
            "symbol": pending["symbol"],
            "side": pending["side"],
            "quantity": str(quantity),
            "entry_price": str(price),
            "stop": pending["stop"],
            "take": pending["take"],
            "opened_ms": int(time.time() * 1000),
            "deadline_ms": int(time.time() * 1000) + int(candidate["timeout_minutes"]) * 60_000,
            "equity_before": session.get("last_flat_equity") or session.get("start_equity") or "0",
            "candidate": candidate,
            "decision": {"source": "RECOVERED"},
        }

    # ---- scanning and deciding ---------------------------------------------

    def scan_slot(self, now_ms: int) -> int:
        """The latest scan boundary: the moment the hourly bar under review closed."""
        scan_ms = int(self.config.scan_minutes) * 60_000
        return now_ms // scan_ms * scan_ms

    async def scan(self, now_ms: int) -> tuple[list[strategy.Features], list[strategy.Candidate]]:
        """Look for signals on the hourly bar that has just closed.

        The rules always read one-hour bars. With ``scan_minutes=15`` those bars are rebuilt
        from 15-minute candles four times an hour (closing at :00, :15, :30 and :45), so the
        market is looked at four times as often without moving to a shorter, costlier timeframe.
        """
        slot = self.scan_slot(now_ms)
        last_closed = slot - BAR_MS
        gate = asyncio.Semaphore(4)

        async def one(symbol: str):
            async with gate:
                raw = await self.api.candles(symbol, "15m", now_ms - 21 * 24 * HOUR_MS, now_ms)
            quarters = strategy.closed_only(raw, QUARTER_MS, now_ms)
            if not quarters or quarters[-1].t != slot - QUARTER_MS:
                return symbol, None, [], {}
            bars = strategy.aggregate(quarters, QUARTER_MS, BAR_MS, offset_ms=slot % BAR_MS)
            trend_bars = strategy.aggregate(bars, BAR_MS, TREND_MS)
            # A missing candle would stretch the last step and misplace the signal time.
            if len(bars) < 2 or bars[-1].t != last_closed or bars[-1].t - bars[-2].t != BAR_MS:
                return symbol, None, [], {}
            features, candidates = strategy.analyse(
                symbol,
                bars[-300:],
                trend_bars,
                reward_risk=self.config.reward_risk,
                timeout_minutes=self.config.timeout_minutes,
            )
            return symbol, features, candidates, strategy.bar_context(bars)

        results = await asyncio.gather(*(one(s) for s in self.config.symbols), return_exceptions=True)
        try:
            metrics = await self.api.metrics()
        except EvedexError:
            metrics = {}
        features, candidates, skipped = [], [], []
        self._extras = {}
        for symbol, result in zip(self.config.symbols, results, strict=False):
            if isinstance(result, BaseException):
                skipped.append(f"{symbol}: {type(result).__name__}")
                continue
            if result[1] is None:
                skipped.append(f"{symbol}: no fresh closed bar")
                continue
            features.append(result[1])
            candidates.extend(result[2])
            self._extras[symbol] = {**result[3], **metrics.get(symbol, {})}
        self.store.journal(
            "scan",
            bar=last_closed,
            features=[{**f.as_dict(), **self._extras.get(f.symbol, {})} for f in features],
            candidates=[c.as_dict() for c in candidates],
            skipped=skipped,
        )
        return features, candidates

    async def decide(
        self, features: list[strategy.Features], candidates: list[strategy.Candidate], free_slots: int = 1
    ) -> brain.Decision:
        control = brain.control_decision(candidates, free_slots)
        day = self._day()
        if not (self.config.use_llm and self.openai_key):
            decision = control
        elif day["llm_calls"] >= self.config.max_llm_calls_per_day:
            decision = brain.Decision("WAIT", None, 0.0, "daily model call limit reached", "FALLBACK")
        else:
            fear_greed = await brain.fetch_fear_greed(self.session)
            now_ms = int(time.time() * 1000)
            payload = brain.build_payload(
                features,
                candidates,
                trades_today=day["trades"],
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
                fear_greed=fear_greed,
                recent_trades=self.store.state.get("history", [])[-10:],
                text_signals=self.news.context(now_ms) if self.news else None,
                macro=self.macro.summary() if self.macro else None,
            )
            by_aggregator = self.config.review_mode == "aggregator" and self.aggregator is not None
            api_key = self.openai_key or ""  # this branch runs only with a key
            left = self.config.max_llm_calls_per_day - day["llm_calls"]
            ask_probability = not by_aggregator or (self.config.review_shadow and left > len(candidates))
            ask_aggregator = self.aggregator is not None and (
                by_aggregator or (self.config.review_shadow and left > len(candidates))
            )
            day["llm_calls"] += int(ask_probability) + (len(candidates) if ask_aggregator else 0)
            self.store.save()

            async def probability() -> brain.Decision | None:
                if not ask_probability:
                    return None
                return await brain.ask_llm(
                    self.session,
                    api_key=api_key,
                    model=self.config.llm_model,
                    effort=self.config.llm_effort,
                    payload=payload,
                    candidates=candidates,
                    min_confidence=self.config.min_confidence,
                    timeout_s=self.config.llm_timeout_s,
                    max_picks=free_slots,
                )

            async def aggregator() -> brain.Decision | None:
                if not ask_aggregator:
                    return None
                return await self._aggregator_decision(features, candidates, payload, free_slots, now_ms)

            by_probability, by_review = await asyncio.gather(probability(), aggregator())
            chosen = by_review if by_aggregator else by_probability
            other = by_probability if by_aggregator else by_review
            decision = chosen or brain.Decision(
                "WAIT", None, 0.0, "the selected reviewer did not answer", "FALLBACK"
            )
            if other is not None:
                self.store.journal(
                    "shadow_decision", decision=other.as_dict(), candidate_ids=[c.id for c in candidates]
                )
            self.store.journal(
                "review_context",
                fear_greed=fear_greed,
                text_signals=payload["text_scouts_signals"],
                macro=payload["macro_allocation"],
            )
        self.store.journal(
            "decision",
            decision=decision.as_dict(),
            control_pick=control.candidate_id,
            control_picks=control.picks,
            candidate_ids=[c.id for c in candidates],
        )
        return decision

    async def _aggregator_decision(
        self,
        features: list[strategy.Features],
        candidates: list[strategy.Candidate],
        payload: dict[str, Any],
        free_slots: int,
        now_ms: int,
    ) -> brain.Decision:
        """The Kairos Router + Aggregator review of every candidate, as one decision."""
        if self.aggregator is None or self.openai_key is None:
            raise AggregatorError("the Aggregator review needs its checkout and a model key")
        try:
            reviews = await review_candidates(
                self.session,
                self.aggregator,
                api_key=self.openai_key,
                model=self.config.llm_model,
                candidates=candidates,
                signals=self.news.raw_signals() if self.news else [],
                market={f.symbol: {**f.as_dict(), **self._extras.get(f.symbol, {})} for f in features},
                portfolio=payload["portfolio"],
                context={
                    "crypto_fear_greed_index": payload["crypto_fear_greed_index"],
                    "own_recent_closed_trades": payload["own_recent_closed_trades"],
                    "base_rates": payload["base_rates"],
                },
                macro=payload["macro_allocation"],
                now_ms=now_ms,
                timeout_s=self.config.llm_timeout_s,
            )
        except (TimeoutError, AggregatorError, aiohttp.ClientError, OSError) as exc:
            self.store.journal("aggregator_error", kind=type(exc).__name__, message=str(exc)[:300])
            return brain.Decision(
                "WAIT",
                None,
                0.0,
                f"Aggregator review failed: {type(exc).__name__}",
                "AGGREGATOR",
                problems=[str(exc)[:200]],
            )
        self.store.journal("aggregator_review", revisions=self._aggregator_revisions, reviews=reviews)
        picks = allowed_in_order(reviews, candidates, free_slots)
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
            usage={k: sum(r["usage"].get(k, 0) for r in reviews) for k in ("input_tokens", "output_tokens")},
        )

    async def try_enter(self, candidate: strategy.Candidate, decision: brain.Decision) -> None:
        trades = self.trades()
        instrument = self.instruments.get(candidate.symbol)
        if instrument is None or not instrument.tradable:
            self.store.journal("entry_skipped", candidate=candidate.id, reason="instrument is not tradable")
            return
        book = await self.api.book(candidate.symbol)
        now_ms = int(time.time() * 1000)
        ref = book.ask if candidate.side == "LONG" else book.bid
        drift = abs(float(ref) - candidate.ref_price)
        spread_bps = float((book.ask - book.bid) / book.ask) * 1e4
        stop = round_step(dec(repr(candidate.stop)), instrument.price_step)
        take = round_step(dec(repr(candidate.take)), instrument.price_step)
        reason = None
        if any(t["symbol"] == candidate.symbol for t in trades):
            reason = "there is already a position in this instrument"
        elif len(trades) >= self.slot_limit():
            reason = "no free position slot"
        elif abs(now_ms - book.time_ms) > 30_000:
            reason = "order book is stale"
        elif spread_bps > self.config.max_spread_bps:
            reason = f"spread {spread_bps:.1f} bps is wider than allowed"
        elif drift > self.config.max_entry_drift_atr * candidate.atr:
            reason = f"price moved {drift:.6g} since the signal bar, more than allowed"
        elif not (stop < ref < take if candidate.side == "LONG" else take < ref < stop):
            reason = "market is already beyond the stop or the take"
        quantity = None
        if reason is None:
            capital = Decimal(str(self.config.capital_basis_usd))
            quantity, reason = size_position(
                side=candidate.side,
                book=book,
                stop=stop,
                instrument=instrument,
                risk_usd=Decimal(str(self.config.risk_per_trade_usd)),
                max_notional=capital / int(self.config.max_open_positions),
                available=await self.broker.available(),
                fee=self.fee,
                slippage=Decimal(str(self.config.slippage_fraction)),
            )
        if quantity is not None and self.config.risk_mode == "kairos":
            # The Kairos Risk Manager must approve, and it can only make the position smaller.
            try:
                verdict = (
                    await self._risk_decisions([self._risk_item(candidate, book, stop, take)], now_ms)
                )[candidate.id]
            except (RulesError, OSError) as exc:
                verdict = {
                    "approved": False,
                    "rejection_reasons": [f"risk_manager_unavailable: {str(exc)[:80]}"],
                }
            if not verdict["approved"]:
                quantity, reason = None, "Risk Manager: " + ", ".join(verdict["rejection_reasons"])
            else:
                allowed = floor_step(dec(repr(verdict["quantity"])), instrument.quantity_step)
                quantity = min(quantity, allowed)
                minimum = max(
                    instrument.min_quantity,
                    ceil_step(instrument.min_volume * Decimal("1.03") / ref, instrument.quantity_step),
                )
                if quantity < minimum:
                    quantity, reason = None, "Risk Manager size is below the instrument's minimum lot"
        if quantity is None:
            self.store.journal("entry_skipped", candidate=candidate.id, reason=reason)
            say(f"Вход пропущен ({candidate.symbol} {candidate.side}): {reason}")
            return
        foreign = await self.broker.foreign_exposure({t["symbol"] for t in trades})
        if foreign:
            self.halt("на счёте есть чужая позиция или заявка: " + "; ".join(foreign)[:300])
            return
        equity_before = await self.broker.equity()
        fill = await self.broker.enter(candidate, quantity, book, instrument, stop, take)
        if fill is None:
            return
        trade: dict[str, Any] = {
            "id": fill["order_id"] if fill["order_id"] != "paper" else f"paper-{now_ms}-{candidate.symbol}",
            "symbol": candidate.symbol,
            "side": candidate.side,
            "quantity": str(fill["quantity"]),
            "entry_price": str(fill["price"]),
            "stop": str(stop),
            "take": str(take),
            "opened_ms": now_ms,
            "deadline_ms": now_ms + candidate.timeout_minutes * 60_000,
            "equity_before": str(equity_before),
            "candidate": candidate.as_dict(),
            "decision": decision.as_dict(),
            # Alone for its whole life: the balance change is exactly this trade's result.
            "solo": not trades,
        }
        for other in trades:
            other["solo"] = False
        trades.append(trade)
        self._day()["trades"] += 1
        self.store.save()
        self.store.journal("trade_opened", trade=trade)
        say(
            f"ВХОД {trade['symbol']} {trade['side']} {trade['quantity']} по {fill['price']}; "
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
        if not await self.broker.position_open(trade):
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

    async def _trade_result(
        self, trade: dict[str, Any], reason: str, book: Book | None, equity: Decimal
    ) -> tuple[Decimal, bool]:
        """This trade's result in USD and whether it is exact or estimated from prices."""
        if "pnl_usd" in trade:  # the paper broker computes it itself
            return dec(trade["pnl_usd"]), True
        if trade.get("solo", True):
            return equity - dec(trade["equity_before"]), True
        if book is None:
            book = await self.api.book(trade["symbol"])
        touch = book.bid if trade["side"] == "LONG" else book.ask
        exit_price = nearest_level(trade, touch) if reason == "VENUE_EXIT" else touch
        trade["exit_price"] = str(exit_price)
        return estimate_pnl(trade, exit_price, self.fee), False

    async def _finish(self, trade: dict[str, Any], reason: str, book: Book | None, *, close: bool) -> None:
        if close:
            if book is None:
                raise NeedsOperator("a position cannot be closed without a fresh order book")
            await self.broker.close(trade, book, reason)
        await self.broker.settle(trade)
        equity = await self.broker.equity()
        pnl, exact = await self._trade_result(trade, reason, book, equity)
        state = self.store.state
        trades = self.trades()
        trades[:] = [t for t in trades if t["id"] != trade["id"]]
        state.setdefault("cooldowns", {})[trade["symbol"]] = (
            int(time.time() * 1000) + self.config.cooldown_minutes * 60_000
        )
        session = state["session"]
        session["realized_usd"] = str(dec(session.get("realized_usd", "0")) + pnl)
        day = self._day()
        day["realized_usd"] = str(dec(day.get("realized_usd", "0")) + pnl)
        if not trades:
            session["last_flat_equity"] = str(equity)
        if reason != "PROTECTION_FAILED":
            state["round_trips"] = int(state.get("round_trips", 0)) + 1
        risk = Decimal(str(self.config.risk_per_trade_usd))
        history = state.setdefault("history", [])
        history.append(
            {
                "symbol": trade["symbol"],
                "side": trade["side"],
                "setup": trade["candidate"].get("setup"),
                "exit": reason,
                "result_r": round(float(pnl / risk), 2),
                "held_minutes": (int(time.time() * 1000) - trade["opened_ms"]) // 60_000,
                "closed_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
            }
        )
        del history[:-HISTORY_LENGTH]
        self.store.save()
        self.store.journal(
            "trade_closed",
            id=trade["id"],
            symbol=trade["symbol"],
            side=trade["side"],
            reason=reason,
            pnl_usd=str(pnl),
            pnl_exact=exact,
            equity=str(equity),
            exit_price=trade.get("exit_price"),
            source=trade["decision"].get("source"),
            candidate_id=trade["candidate"].get("id"),
        )
        say(
            f"ВЫХОД {trade['symbol']} ({reason}). Результат сделки: {pnl:+.4f} USDT"
            f"{'' if exact else ' (оценка по ценам)'}. Баланс: {equity:.4f} USDT. "
            f"Открыто позиций: {len(trades)}."
        )
        limit = Decimal(str(self.config.session_loss_stop_usd))
        loss = max(dec(session["start_equity"]) - equity, -dec(session["realized_usd"]))
        if loss >= limit:
            self.halt(f"убыток сессии {loss:.2f} USDT достиг лимита {self.config.session_loss_stop_usd}")

    # ---- the loop -----------------------------------------------------------

    async def step(self) -> None:
        now_ms = int(time.time() * 1000)
        trades = self.trades()
        failure: Exception | None = None
        for trade in list(trades):
            try:
                await self.manage(trade, now_ms)
            except (TimeoutError, EvedexError, aiohttp.ClientError) as exc:
                failure = failure or exc  # keep managing the other positions first
        if failure is not None:
            raise failure
        if self.entries_blocked():
            return
        free = self.slot_limit() - len(trades)
        if free <= 0:
            return
        slot = self.scan_slot(now_ms)
        if self.store.state.get("last_bar") == slot or now_ms - slot < 10_000:
            return
        self.store.state["last_bar"] = slot
        self.store.save()
        features, candidates = await self.scan(now_ms)
        if self.macro is not None and (self._macro_task is None or self._macro_task.done()):
            # In the background: a slow strategic call must not delay this scan's decision.
            self._macro_task = asyncio.create_task(self.macro_tick(features, now_ms))
        busy = {t["symbol"] for t in trades}
        cooldowns = self.store.state.get("cooldowns", {})
        eligible = [c for c in candidates if c.symbol not in busy and now_ms >= cooldowns.get(c.symbol, 0)]
        if not eligible:
            say(f"Скан: {len(features)} инструментов, новых сигналов нет. Открыто позиций: {len(trades)}.")
            return
        say("Скан: сигналы — " + ", ".join(f"{c.symbol} {c.side} {c.setup}" for c in eligible))
        decision, _ = await asyncio.gather(
            self.decide(features, eligible, free), self.risk_shadow(eligible, now_ms)
        )
        say(
            f"Решение ({decision.source}): {decision.action} {', '.join(decision.picks)} "
            f"— {decision.reason[:200]}"
        )
        for candidate_id in decision.picks[:free]:
            if self.entries_blocked() or len(self.trades()) >= self.slot_limit():
                break
            await self.try_enter(next(c for c in eligible if c.id == candidate_id), decision)

    async def run(self, *, max_steps: int | None = None) -> None:
        await self.start()
        # The news layer polls on its own clock so that slow feeds never delay position management.
        news_task = (
            asyncio.create_task(self._news_loop()) if self.news is not None and max_steps is None else None
        )
        try:
            await self._loop(max_steps)
        finally:
            if news_task is not None:
                news_task.cancel()

    async def _loop(self, max_steps: int | None) -> None:
        errors = steps = 0
        while max_steps is None or steps < max_steps:
            steps += 1
            try:
                await self.step()
                errors = 0
            except (TimeoutError, EvedexError, aiohttp.ClientError) as exc:
                errors += 1
                self.store.journal(
                    "error",
                    kind=type(exc).__name__,
                    message=str(exc)[:300],
                    body=getattr(exc, "body", "")[:300],
                )
                say(f"Ошибка связи с биржей ({type(exc).__name__}): {str(exc)[:160]}")
                if isinstance(exc, NeedsOperator):
                    self.halt("нужна ручная проверка счёта: " + str(exc)[:200])
                if errors >= 30:
                    self.halt("слишком много ошибок подряд")
                    errors = 0
            if self.store.flag("STOP") and not self.trades():
                say("Найден файл STOP, позиций нет. Останавливаюсь.")
                return
            await asyncio.sleep(self.config.poll_seconds if max_steps is None else 0)
