"""Bridge to the real Kairos Router policy (repository ``kairos-router``).

Every strategy intent gets its ``CandidateRouteV1`` from the Router's own code, run by
``router_worker.py`` in the Router's environment. The route says whether fresh text
evidence opposes the candidate (CONFLICT) or not (NORMAL), fixes the review deadline and
names the exact evidence used. A candidate the Router rejects is dropped before any review.

The Router cannot change an intent: the route that comes back has to carry the same
intent identity that went in.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairos_core.contracts import CandidateRouteV1, StrategyIntentV1
from pydantic import ValidationError

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .signals import Routed
from .worker_session import WorkerSession

WORKER = Path(__file__).with_name("router_worker.py")
ROUTE_SOURCE = "kairos-autotrader"  # the process that ran the Router's policy
MAX_REVIEW_TIMEOUT_MS = 60_000  # the upper bound of RouterSettings.candidate_review_timeout_ms


class RouterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RouterConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-router"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-router"))
    timeout_s: float = 60.0

    def check(self) -> None:
        if not self.python.exists():
            raise RouterError(f"Router Python not found: {self.python}")
        if not (self.source / "kairos_router" / "candidate.py").exists():
            raise RouterError(f"Router source not found: {self.source}")


def router_revision(config: RouterConfig) -> str:
    return git_revision(config.source, "kairos_router")


def router_session(config: RouterConfig) -> WorkerSession:
    return WorkerSession(config.python, WORKER, [str(config.source)], timeout_s=config.timeout_s)


def _run_worker(config: RouterConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(config.python, WORKER, [str(config.source)], request, timeout_s=config.timeout_s)
    except SiblingError as exc:
        raise RouterError(f"Router: {exc}") from exc


@dataclass(frozen=True, slots=True)
class Routing:
    """Routed candidates, the ones the Router rejected, and the settings it used."""

    routed: list[Routed]
    rejected: list[dict[str, Any]]
    text: dict[str, dict[str, Any]]
    settings: dict[str, Any]


async def route_intents(
    config: RouterConfig,
    intents: dict[str, StrategyIntentV1],
    *,
    signals: list[dict[str, Any]],
    now_ms: int,
    review_timeout_ms: int,
    runner=None,
    session: WorkerSession | None = None,
) -> Routing:
    """Route the intent of every instrument. Raises ``RouterError`` if the worker fails."""
    if not intents:
        return Routing([], [], {}, {})
    by_id = {str(intent.intent_id): (instrument, intent) for instrument, intent in intents.items()}
    request = {
        "now_ms": now_ms,
        "review_timeout_ms": review_timeout_ms,
        "source": ROUTE_SOURCE,
        "symbols": sorted({intent.symbol for intent in intents.values()}),
        "signals": signals,
        "intents": [intent.model_dump(mode="json") for intent in intents.values()],
    }
    if session is None:
        answer = await asyncio.to_thread(runner or _run_worker, config, request)
    else:
        try:
            answer = await session.ask(request)
        except SiblingError as exc:
            raise RouterError(f"Router: {exc}") from exc
    rows = answer.get("routes")
    if not isinstance(rows, list) or {str(r.get("intent_id")) for r in rows} != set(by_id):
        raise RouterError("Router worker did not answer for exactly the intents it was given")
    routed: list[Routed] = []
    rejected: list[dict[str, Any]] = []
    text: dict[str, dict[str, Any]] = {}
    for row in rows:
        instrument, intent = by_id[str(row["intent_id"])]
        if "rejected" in row:
            rejected.append({"symbol": instrument, "id": intent.intent_id, "reason": str(row["rejected"])})
            continue
        try:
            route = CandidateRouteV1.model_validate(row["route"])
            if route.intent.intent_id != intent.intent_id:
                raise ValueError("the route carries another intent than the one that was sent")
            routed.append(Routed(instrument, route))
        except (ValidationError, KeyError, TypeError, ValueError) as exc:
            raise RouterError(f"bad route for {instrument}: {type(exc).__name__}: {str(exc)[:160]}") from exc
        text[instrument] = dict(row.get("text") or {})
    routed.sort(key=lambda item: (item.signal_time, item.id))
    return Routing(routed, rejected, text, dict(answer.get("settings") or {}))
