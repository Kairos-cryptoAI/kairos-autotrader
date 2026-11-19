"""Bridge to the real Kairos Risk Manager (repository ``kairos-risk-manager``).

Every entry is decided by ``LiveRiskEvaluator`` (``risk_worker.py``): the Aggregator's
``CandidateReviewV1`` must ALLOW the candidate, the Macro allocation must give the strategy
capital and permit the side, the venue measurement must allow an entry, and the account
must have room in every risk and capital ceiling. The answer is a ``RiskTradeDecisionV1``
with the quantity; the runner can only trade that quantity or less, never more.

The runner supplies the account evidence it reads from EVEDEX and its own trade ledger;
the Risk Manager's code decides. Nothing here places orders.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from kairos_core.contracts import RiskTradeDecisionV1
from kairos_execution.portfolio_broker import LIVE_ACCOUNT_SOURCE as LIVE_ACCOUNT_SOURCE
from kairos_execution.portfolio_broker import account_id_for as account_id_for
from pydantic import ValidationError

from .sibling import (
    _PASSED_ENVIRONMENT,
    MARKER,
    SiblingError,
    git_revision,
    run_worker,
    sibling_checkout,
    sibling_python,
)

WORKER = Path(__file__).with_name("risk_worker.py")
# Ceilings of LiveRiskLimitsV1 that the runner uses at their maximum.
PER_TRADE_RISK_FRACTION = 0.0025
AGGREGATE_OPEN_RISK_FRACTION = 0.01
MIN_NOTIONAL_USD = 1.0
PAPER_ACCOUNT_SOURCE = "kairos-autotrader-paper-ledger"


class RiskError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RiskConfig:
    # The worker runs in Quant Scouts' environment, with the Risk Manager's source on the path.
    python: Path = field(default_factory=lambda: sibling_python("kairos-quant-scouts"))
    quant_source: Path = field(default_factory=lambda: sibling_checkout("kairos-quant-scouts"))
    risk_source: Path = field(default_factory=lambda: sibling_checkout("kairos-risk-manager"))
    timeout_s: float = 30.0

    def check(self) -> None:
        if not self.python.exists():
            raise RiskError(f"Quant Scouts Python not found: {self.python}")
        for path, marker in (
            (self.quant_source, "kairos_quant/runtime_venue_gate.py"),
            (self.risk_source, "kairos_risk/live.py"),
        ):
            if not (path / marker).exists():
                raise RiskError(f"source not found: {path / marker}")


def risk_revisions(config: RiskConfig) -> dict[str, str]:
    return {"risk_manager": git_revision(config.risk_source, "kairos_risk")}


def _constant(value: dict[str, Any]) -> Callable[[], Awaitable[dict[str, Any]]]:
    async def read() -> dict[str, Any]:
        return value

    return read


def _run_worker(config: RiskConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(
            config.python,
            WORKER,
            [str(config.quant_source), str(config.risk_source)],
            request,
            timeout_s=config.timeout_s,
        )
    except SiblingError as exc:
        raise RiskError(f"Risk Manager: {exc}") from exc


class RiskSession:
    """One persistent Risk worker (``risk_worker.py --serve``) shared by every entry.

    The worker keeps Quant Scouts and the Risk Manager loaded. For each entry it measures
    the venue and only then asks for the account, which the runner reads at that moment:
    the account evidence reaches the decision well inside its five-second freshness.
    Any failure kills the worker; the next entry starts a fresh one.
    """

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def _read(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RiskError("Risk Manager: worker output is unavailable")
        while True:
            line = await self._process.stdout.readline()
            if not line:
                raise RiskError("Risk Manager: the worker exited")
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith(MARKER):
                value = json.loads(text[len(MARKER) :])
                if not isinstance(value, dict):
                    raise RiskError("Risk Manager: malformed worker message")
                return value

    async def _write(self, value: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RiskError("Risk Manager: worker input is unavailable")
        self._process.stdin.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))
        await self._process.stdin.drain()

    async def _start(self) -> None:
        environment = {key: os.environ[key] for key in _PASSED_ENVIRONMENT if key in os.environ}
        environment["PYTHONIOENCODING"] = "utf-8"
        self._process = await asyncio.create_subprocess_exec(
            str(self.config.python),
            "-s",
            "-B",
            str(WORKER),
            str(self.config.quant_source),
            str(self.config.risk_source),
            "--serve",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=environment,
            limit=32 * 1024 * 1024,
        )
        if not (await asyncio.wait_for(self._read(), timeout=self.config.timeout_s * 2)).get("ready"):
            raise RiskError("Risk Manager: the worker did not get ready")

    async def stop(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()

    async def ask(
        self, request: dict[str, Any], account: Callable[[], Awaitable[dict[str, Any]]]
    ) -> dict[str, Any]:
        async with self._lock:
            try:
                if self._process is None or self._process.returncode is not None:
                    await self._start()
                async with asyncio.timeout(self.config.timeout_s):
                    await self._write(request)
                    answer = await self._read()
                    if answer.get("need_account"):
                        await self._write({"account": await account()})
                        answer = await self._read()
                return answer
            except (TimeoutError, OSError, ValueError, RiskError) as exc:
                await self.stop()
                if isinstance(exc, RiskError):
                    raise
                raise RiskError(f"Risk Manager: {type(exc).__name__}: {str(exc)[:200]}") from exc


@dataclass(frozen=True, slots=True)
class RiskOutcome:
    """The Risk Manager's decision on one entry, and what the journal keeps of it."""

    approved: bool
    quantity: Decimal
    reasons: tuple[str, ...]
    decision: RiskTradeDecisionV1 | None
    record: dict[str, Any]


def _record(decision: RiskTradeDecisionV1, evaluation: dict[str, Any]) -> dict[str, Any]:
    venue = decision.venue_quality
    return {
        "decision_id": decision.decision_id,
        "trade_id": decision.trade_id,
        "approved": decision.approved,
        "rejection_reasons": list(decision.rejection_reasons),
        "quantity": decision.quantity,
        "notional_usd": decision.notional_usd,
        "leverage": decision.leverage,
        "loss_budget_usd": decision.loss_budget_usd,
        "worst_case_loss_usd": decision.worst_case_loss_usd,
        "worst_entry_price": decision.worst_entry_price,
        "decided_at_ms": decision.decided_at_ms,
        "venue": {
            "measurement_id": venue.measurement_id,
            "entry_allowed": venue.entry_allowed,
            "reason_codes": list(venue.reason_codes),
            "basis_bps": round(venue.basis_bps, 3),
            "spread_bps": round(venue.spread_bps, 3),
            "book_age_ms": venue.book_age_ms,
            "reference_age_ms": venue.reference_age_ms,
        },
        "account": {
            key: evaluation.get(key)
            for key in (
                "source_id",
                "account_snapshot_sha256",
                "account_equity_usd",
                "aggregate_open_risk_usd",
                "daily_adverse_loss_usd",
            )
        },
        "scope_sha256": evaluation.get("scope_sha256"),
    }


async def ledger_table(
    config: RiskConfig, outcomes: list[dict[str, Any]], runner=None
) -> list[dict[str, Any]]:
    """The learning ledger's cells (strategy x instrument x regime) as the Risk Manager sees them."""
    answer = await asyncio.to_thread(runner or _run_worker, config, {"op": "ledger", "outcomes": outcomes})
    if "table" not in answer:
        raise RiskError(str(answer.get("error", "the ledger worker returned no table"))[:200])
    return list(answer["table"])


async def evaluate_entry(
    config: RiskConfig,
    *,
    review: Any,
    allocation: dict[str, Any] | None,
    account: dict[str, Any] | Callable[[], Awaitable[dict[str, Any]]],
    scope: dict[str, Any],
    limits: dict[str, Any],
    taker_fee_bps: float,
    assessed_notional_usd: float,
    adaptive: dict[str, Any] | None = None,
    venue_max_leverage: int = 1,
    min_quantity: Decimal | None = None,
    runner=None,
    session: RiskSession | None = None,
) -> RiskOutcome:
    """The Risk Manager's decision on one reviewed candidate. Raises ``RiskError`` if it cannot answer.

    With a ``session`` and an ``account`` reader, the account is read after the venue is
    measured, just before the decision.
    """
    request: dict[str, Any] = {
        "review": review.model_dump(mode="json"),
        "allocation": allocation,
        "scope": scope,
        "limits": limits,
        "venue": {
            "taker_fee_bps": taker_fee_bps,
            "assessed_notional_usd": assessed_notional_usd,
            "max_leverage": int(venue_max_leverage),
        },
    }
    if min_quantity is not None:
        # The venue's minimum lot: a smaller adaptive size is raised to it within the ceiling.
        request["venue"]["min_quantity"] = str(min_quantity)
    if adaptive is not None:
        request["adaptive"] = adaptive
    if session is not None and runner is None:
        reader = account if callable(account) else _constant(account)
        answer = await session.ask(request, reader)
    else:
        request["account"] = await account() if callable(account) else account
        answer = await asyncio.to_thread(runner or _run_worker, config, request)
    if "refused" in answer:
        reason = str(answer["refused"])[:200]
        record = {"approved": False, "refused": reason}
        if "adaptive" in answer:
            record["adaptive"] = answer["adaptive"]
        return RiskOutcome(False, Decimal(0), (reason,), None, record)
    try:
        evaluation = answer["evaluation"]
        decision = RiskTradeDecisionV1.model_validate(evaluation["decision"])
    except (KeyError, TypeError, ValidationError) as exc:
        raise RiskError(f"Risk Manager returned a malformed decision ({type(exc).__name__})") from exc
    if decision.intent.intent_id != review.intent.intent_id or decision.review.review_id != review.review_id:
        raise RiskError("the risk decision belongs to another candidate")
    quantity = Decimal(repr(decision.quantity)) if decision.approved else Decimal(0)
    record = _record(decision, evaluation)
    if "adaptive" in answer:
        record["adaptive"] = answer["adaptive"]
    return RiskOutcome(decision.approved, quantity, decision.rejection_reasons, decision, record)
