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
import hashlib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from kairos_core.contracts import RiskTradeDecisionV1
from pydantic import ValidationError

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python

WORKER = Path(__file__).with_name("risk_worker.py")
# Ceilings of LiveRiskLimitsV1 that the runner uses at their maximum.
PER_TRADE_RISK_FRACTION = 0.0025
AGGREGATE_OPEN_RISK_FRACTION = 0.01
MIN_NOTIONAL_USD = 1.0
LIVE_ACCOUNT_SOURCE = "evedex-prod-runner-session"
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


def account_id_for(wallet_address: str) -> str:
    """A stable, non-secret account identity: a hash of the wallet address."""
    return "evedex-prod-" + hashlib.sha256(wallet_address.lower().encode()).hexdigest()[:20]


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


async def evaluate_entry(
    config: RiskConfig,
    *,
    review: Any,
    allocation: dict[str, Any] | None,
    account: dict[str, Any],
    scope: dict[str, Any],
    limits: dict[str, Any],
    taker_fee_bps: float,
    assessed_notional_usd: float,
    adaptive: dict[str, Any] | None = None,
    runner=None,
) -> RiskOutcome:
    """The Risk Manager's decision on one reviewed candidate. Raises ``RiskError`` if it cannot answer."""
    request = {
        "review": review.model_dump(mode="json"),
        "allocation": allocation,
        "account": account,
        "scope": scope,
        "limits": limits,
        "venue": {"taker_fee_bps": taker_fee_bps, "assessed_notional_usd": assessed_notional_usd},
    }
    if adaptive is not None:
        request["adaptive"] = adaptive
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
