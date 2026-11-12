"""Bridge to the real Kairos Router classification and Aggregator candidate review.

Each candidate gets its own review, as in the Kairos design: the Router's rule decides
whether fresh news opposes the candidate's side (CONFLICT) or not (NORMAL), the model is
asked with the Aggregator's own prompt, and its answer is checked by the Aggregator's own
schema and guard. The result per candidate is ALLOW, VETO or DEFER, a priority 0-100 and
reason codes. Anything broken becomes DEFER, which ends that candidate.

The reviewer cannot change a side, a price, a stop, a size or propose a trade.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .candidates import Candidate
from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .text_scouts import model_extract

WORKER = Path(__file__).with_name("aggregator_worker.py")


class AggregatorError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AggregatorConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-aggregator"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-aggregator"))
    router_source: Path = field(default_factory=lambda: sibling_checkout("kairos-router"))
    core_source: Path = field(default_factory=lambda: sibling_checkout("kairos-core"))
    timeout_s: float = 60.0

    def check(self) -> None:
        if not self.python.exists():
            raise AggregatorError(f"Aggregator Python not found: {self.python}")
        for path, marker in (
            (self.source, "kairos_aggregator/candidate_review.py"),
            (self.router_source, "kairos_router/candidate.py"),
            (self.core_source, "kairos_core/__init__.py"),
        ):
            if not (path / marker).exists():
                raise AggregatorError(f"source not found: {path / marker}")


def revisions(config: AggregatorConfig) -> dict[str, str]:
    return {"aggregator": git_revision(config.source), "router": git_revision(config.router_source)}


def _run_worker(config: AggregatorConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(
            config.python,
            WORKER,
            [str(config.source), str(config.router_source), str(config.core_source)],
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
    candidates: list[Candidate],
    signals: list[dict[str, Any]],
    market: dict[str, dict[str, Any]],
    portfolio: dict[str, Any],
    context: dict[str, Any],
    now_ms: int,
    macro: Any = None,
    timeout_s: float = 90.0,
    runner=_run_worker,
    extract=model_extract,
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
            "candidates": [c.as_dict() for c in candidates],
            "signals": signals,
            "market": market,
            "portfolio": portfolio,
            "context": context,
            "macro": macro,
        },
    )
    system = prepared["system"]
    latency: dict[str, float] = {}

    async def ask(item: dict[str, Any]):
        if item.get("expired"):
            return None, {}, "candidate older than the Router's time-to-live"
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
        if item.get("expired"):
            review.update(decision="DEFER", priority=0, reason_codes=["CANDIDATE_EXPIRED"], reviewer="ROUTER")
        took = latency.get(item["id"])
        review.update(
            tier=item["tier"],
            effort=item["effort"],
            text_score=item["text_score"],
            usage=usage,
            problem=problem,
            age_s=item.get("age_s"),
            latency_s=round(took, 2) if took is not None else None,
            # Recorded only: the Router's 20-second review deadline is not enforced here.
            router_deadline_missed=bool(
                took is not None and took * 1000 > prepared.get("review_timeout_ms", 20_000)
            ),
        )
        reviews.append(review)
    return reviews


def allowed_in_order(reviews: list[dict[str, Any]], candidates: list[Candidate], max_picks: int) -> list[str]:
    """ALLOWed candidates, highest priority first, one per instrument."""
    by_id = {c.id: c for c in candidates}
    ranked = sorted(
        (r for r in reviews if r["decision"] == "ALLOW" and r["id"] in by_id),
        key=lambda r: (-int(r["priority"]), -by_id[r["id"]].score, r["id"]),
    )
    picks, symbols = [], set()
    for review in ranked:
        symbol = by_id[review["id"]].symbol
        if symbol in symbols:
            continue
        symbols.add(symbol)
        picks.append(review["id"])
        if len(picks) >= max_picks:
            break
    return picks
