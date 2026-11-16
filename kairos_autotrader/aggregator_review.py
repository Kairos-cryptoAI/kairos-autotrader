"""Bridge to the Kairos Aggregator's candidate review (repository ``kairos-aggregator``).

Every routed candidate is reviewed by the Aggregator's own ``CandidateReviewBrain`` on a
``DecisionContextV1`` frozen by its own ``CandidateContextStore``, with the model called
through ``kairos-llm``'s budgeted gateway (``aggregator_worker.py``). The runner hands over
what it has observed, with the moment it received each source:

- the market snapshot that Quant Scouts' code built before the decision;
- the newest closed Binance bars the strategy read;
- the exact Text Scouts signals the route names, and the Macro allocation.

The result per candidate is a ``CandidateReviewV1``: ALLOW, VETO or DEFER, a priority 0-100
and reason codes. A missing or stale source, a missed deadline, an exhausted budget or a
broken model answer is a DEFER decided by the Aggregator's code, never by the runner. The
reviewer cannot change a side, a price, a stop, a size or propose a trade.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairos_core.contracts import CandidateReviewV1
from pydantic import ValidationError

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .signals import Routed

WORKER = Path(__file__).with_name("aggregator_worker.py")
# The ceiling kairos-llm registers for OpenAI; a runner budget may be lower, never higher.
OPENAI_MONTHLY_CEILING_USD = 40.0


class AggregatorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AggregatorConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-aggregator"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-aggregator"))
    timeout_s: float = 150.0
    # An OpenAI-compatible endpoint instead of OpenAI itself; tests point it at a local stub.
    openai_base_url: str | None = None

    def check(self) -> None:
        if not self.python.exists():
            raise AggregatorError(f"Aggregator Python not found: {self.python}")
        if not (self.source / "kairos_aggregator" / "candidate_review.py").exists():
            raise AggregatorError(f"Aggregator source not found: {self.source}")


def revisions(config: AggregatorConfig) -> dict[str, str]:
    return {"aggregator": git_revision(config.source, "kairos_aggregator")}


def _run_worker(config: AggregatorConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(config.python, WORKER, [str(config.source)], request, timeout_s=config.timeout_s)
    except SiblingError as exc:
        raise AggregatorError(f"Aggregator: {exc}") from exc


def _record(candidate: Routed, review: CandidateReviewV1, context: dict[str, Any] | None) -> dict[str, Any]:
    """The compact journal form of one review; the contract itself is several megabytes."""
    provenance = review.model_provenance
    return {
        "id": candidate.id,
        "symbol": candidate.instrument,
        "review_id": review.review_id,
        "decision": review.decision.value,
        "priority": review.priority,
        "reason_codes": list(review.reason_codes),
        "reviewer": review.reviewer,
        "tier": review.route.review_tier.value,
        "effort": review.route.requested_reasoning_effort.value,
        "reviewed_at_ms": review.reviewed_at_ms,
        "review_deadline_ms": review.route.review_deadline_ms,
        "model": None
        if provenance is None
        else {
            "model": provenance.model,
            "effort": provenance.reasoning_effort,
            "latency_ms": provenance.latency_ms,
            "cost_usd": provenance.cost_usd,
            "request_id": provenance.request_id,
            "budget_reservation_id": provenance.budget_reservation_id,
            "prompt_sha256": provenance.prompt_sha256,
        },
        "context": context,
    }


@dataclass(frozen=True, slots=True)
class Reviews:
    records: list[dict[str, Any]]
    ingest_problems: list[str]
    # The validated CandidateReviewV1 per intent id; the Risk Manager decides on it.
    contracts: dict[str, CandidateReviewV1] = field(default_factory=dict)


async def review_candidates(
    config: AggregatorConfig,
    *,
    candidates: list[Routed],
    market: dict[str, dict[str, Any]],
    closed_bars: dict[str, list[list[Any]]],
    signals: list[dict[str, Any]],
    macro: dict[str, Any] | None,
    openai_key: str,
    ledger: Path,
    monthly_budget_usd: float,
    runner=None,
) -> Reviews:
    """One Aggregator review per candidate. Raises ``AggregatorError`` only if the worker fails.

    ``market`` and ``closed_bars`` are keyed by the Binance signal symbol.
    """
    if not candidates:
        return Reviews([], [])
    if not 0 < monthly_budget_usd <= OPENAI_MONTHLY_CEILING_USD:
        raise AggregatorError("the monthly model budget must be within the kairos-llm ceiling")
    symbols = sorted({candidate.intent.symbol for candidate in candidates})
    request = {
        "op": "review",
        "routes": [candidate.route.model_dump(mode="json") for candidate in candidates],
        "symbols": symbols,
        "market": {symbol: market[symbol] for symbol in symbols if symbol in market},
        "closed_bars": {symbol: closed_bars[symbol] for symbol in symbols if symbol in closed_bars},
        "signals": signals,
        "macro": macro,
        "llm": {
            "openai_api_key": openai_key,
            "openai_base_url": config.openai_base_url,
            "ledger": str(ledger),
            "monthly_budget_microusd": int(round(monthly_budget_usd * 1_000_000)),
        },
    }
    answer = await asyncio.to_thread(runner or _run_worker, config, request)
    rows = answer.get("reviews")
    by_id = {candidate.id: candidate for candidate in candidates}
    if not isinstance(rows, list) or sorted(str(r.get("intent_id")) for r in rows) != sorted(by_id):
        raise AggregatorError("Aggregator worker did not answer for exactly the candidates it was given")
    records, contracts = [], {}
    for row in rows:
        candidate = by_id[str(row["intent_id"])]
        try:
            review = CandidateReviewV1.model_validate(row["review"])
        except (ValidationError, KeyError, TypeError) as exc:
            raise AggregatorError(f"bad review for {candidate.instrument}: {type(exc).__name__}") from exc
        if review.route.route_id != candidate.route.route_id or review.intent.intent_id != candidate.id:
            raise AggregatorError("a review belongs to another route than the one that was sent")
        records.append(_record(candidate, review, row.get("context")))
        contracts[candidate.id] = review
    return Reviews(records, [str(problem) for problem in answer.get("ingest_problems", [])], contracts)


def allowed_in_order(reviews: list[dict[str, Any]], candidates: list[Routed], max_picks: int) -> list[str]:
    """ALLOWed candidates, highest priority first, one per instrument."""
    by_id = {c.id: c for c in candidates}
    ranked = sorted(
        (r for r in reviews if r["decision"] == "ALLOW" and r["id"] in by_id),
        key=lambda r: (-int(r["priority"]), -by_id[r["id"]].strength, r["id"]),
    )
    picks, symbols = [], set()
    for review in ranked:
        symbol = by_id[review["id"]].instrument
        if symbol in symbols:
            continue
        symbols.add(symbol)
        picks.append(review["id"])
        if len(picks) >= max_picks:
            break
    return picks
