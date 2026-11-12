"""Bridge to the Kairos Aggregator's candidate-review rules.

Each routed candidate gets its own review: the model is asked with the Aggregator's own
prompt at the effort the route requests, and its answer is checked by the Aggregator's own
schema and guard. The result per candidate is ALLOW, VETO or DEFER, a priority 0-100 and
reason codes. Anything broken becomes DEFER, which ends that candidate.

The reviewer cannot change a side, a price, a stop, a size or propose a trade.

The review context is still assembled by ``aggregator_worker.py`` and the model is called
directly: ``DecisionContextV1``, ``CandidateReviewBrain`` and ``LLMGateway`` are the open
part of docs/ALIGNMENT.md, step 2.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .signals import Routed
from .text_scouts import model_extract

WORKER = Path(__file__).with_name("aggregator_worker.py")


class AggregatorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AggregatorConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-aggregator"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-aggregator"))
    core_source: Path = field(default_factory=lambda: sibling_checkout("kairos-core"))
    timeout_s: float = 60.0

    def check(self) -> None:
        if not self.python.exists():
            raise AggregatorError(f"Aggregator Python not found: {self.python}")
        for path, marker in (
            (self.source, "kairos_aggregator/candidate_review.py"),
            (self.core_source, "kairos_core/__init__.py"),
        ):
            if not (path / marker).exists():
                raise AggregatorError(f"source not found: {path / marker}")


def revisions(config: AggregatorConfig) -> dict[str, str]:
    return {"aggregator": git_revision(config.source)}


def _run_worker(config: AggregatorConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(
            config.python,
            WORKER,
            [str(config.source), str(config.core_source)],
            request,
            timeout_s=config.timeout_s,
        )
    except SiblingError as exc:
        raise AggregatorError(f"Aggregator: {exc}") from exc


async def review_candidates(
    session: aiohttp.ClientSession | None,
    config: AggregatorConfig,
    *,
    api_key: str,
    model: str,
    candidates: list[Routed],
    routing_text: dict[str, dict[str, Any]],
    signals: list[dict[str, Any]],
    market: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    context: dict[str, Any],
    now_ms: int,
    macro: Any = None,
    timeout_s: float = 90.0,
    runner=_run_worker,
    extract=model_extract,
    clock_ms=lambda: int(time.time() * 1000),
) -> list[dict[str, Any]]:
    """One Aggregator review per candidate. Raises ``AggregatorError`` only if the worker itself fails."""
    if not candidates:
        return []
    prepared = await asyncio.to_thread(
        runner,
        config,
        {
            "op": "prepare",
            "now_ms": now_ms,
            "candidates": [
                {
                    **candidate.record(),
                    "conflict_rationale": candidate.route.conflict_rationale,
                    "routing_text": routing_text.get(candidate.instrument, {}),
                }
                for candidate in candidates
            ],
            "signals": signals,
            "market": market,
            "portfolio": portfolio,
            "context": context,
            "macro": macro,
        },
    )
    system = prepared["system"]
    latency: dict[str, float] = {}
    finished: dict[str, int] = {}

    async def ask(item: dict[str, Any]):
        started = time.monotonic()
        answer = await extract(
            session,
            api_key=api_key,
            model=model,
            system=system,
            user=item["user"],
            effort=item["effort"],
            timeout_s=timeout_s,
        )
        latency[item["id"]] = time.monotonic() - started
        finished[item["id"]] = clock_ms()
        return answer

    answers = await asyncio.gather(*(ask(item) for item in prepared["prepared"]))
    checked = await asyncio.to_thread(
        runner,
        config,
        {
            "op": "validate",
            "signals": signals,
            "reviews": [
                {
                    "id": item["id"],
                    "side": item["side"],
                    "tier": item["tier"],
                    "routing_ids": item["routing_ids"],
                    "model_output": answer[0],
                }
                for item, answer in zip(prepared["prepared"], answers, strict=False)
            ],
        },
    )
    by_id = {review["id"]: review for review in checked["reviews"]}
    reviews = []
    for item, (_, usage, problem) in zip(prepared["prepared"], answers, strict=False):
        review = dict(by_id[item["id"]])
        review.update(
            tier=item["tier"],
            effort=item["effort"],
            text_score=item["text_score"],
            usage=usage,
            problem=problem,
            latency_s=round(latency[item["id"]], 2),
            # Recorded only: the route's review deadline is not enforced by this stand-in.
            route_deadline_missed=finished[item["id"]] > int(item["review_deadline_ms"]),
        )
        reviews.append(review)
    return reviews


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
