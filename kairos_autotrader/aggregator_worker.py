"""Reviews routed candidates with the real Kairos Aggregator, context store and model gateway.

Executed as a script by the Aggregator repository's own Python environment, with the
repository's unchanged code:

* ``CandidateContextStore``: every source is ingested with its receipt clock, and
  ``build`` freezes one ``DecisionContextV1`` per route (market, closed bars, the exact
  routed text, macro), with the store's own causality, freshness and identity rules;
* ``CandidateReviewBrain.review_with_context``: deadline and staleness checks, the
  Aggregator's prompt, one call through ``kairos_llm.BudgetedLLMGateway`` with the
  architecture-owned workload route (NORMAL or CONFLICT), the output schema, the conflict
  guard and a ``CandidateReviewV1`` with full model provenance. Anything that fails is a
  deterministic DEFER with the Aggregator's reason code.

The budget behind the gateway is ``FileUsageBudget`` below: the ``LLMUsageBudget`` protocol
of ``kairos-llm`` on a JSON ledger in the runner's runtime directory, in place of the
Postgres ledger of ``kairos-persistence``. Like that one, it reserves the worst case
before the call, commits the actual cost after it, and leaves a failed call's reservation
counted.

argv:   <aggregator source>
stdin:  {"op": "review",
         "routes": [<CandidateRouteV1 as JSON>, ...],
         "symbols": ["SOLUSDT", ...],
         "market": {"SOLUSDT": {"payload": <MarketSnapshot>, "received_at_ms": ...}},
         "closed_bars": {"SOLUSDT": [<cached closed bar>, ...]},
         "signals": [{"id", "event_time", "topic", "sentiment", "impact", "confidence",
                      "sources", "summary", "received_ms"}, ...],
         "macro": {"payload": <StrategicAllocation>, "received_at_ms": ...} | null,
         "llm": {"openai_api_key": "...", "openai_base_url": null, "ledger": "<path>",
                 "monthly_budget_microusd": 12000000}}
stdout: a marker line followed by
        {"reviews": [{"intent_id", "review": <CandidateReviewV1 as JSON>,
                      "context": {"context_id", "sources": {kind: availability or reason},
                                  "required": [...]} | null}],
         "ingest_problems": ["..."]}
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MARKER = "@@KAIROS-WORKER@@"
BAR_SOURCE = "kairos-quant-scouts"  # the producer named by Quant Scouts' parser


def _now_ms() -> int:
    return int(time.time() * 1000)


class FileUsageBudget:
    """``kairos_llm.LLMUsageBudget`` on a JSON ledger, one month per UTC calendar month."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("budget ledger is not an object")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(temp, self.path)

    @staticmethod
    def _month() -> str:
        return datetime.now(UTC).strftime("%Y-%m")

    async def reserve(
        self, *, provider: str, reservation_id: str, reserved_microusd: int, monthly_budget_microusd: int
    ) -> None:
        from kairos_llm.errors import LLMBudgetError

        async with self._lock:
            data = self._load()
            entries = data.setdefault(self._month(), {}).setdefault(provider, {})
            if reservation_id in entries:
                raise LLMBudgetError("reservation identity was reused")
            spent = sum(e["actual"] if e["actual"] is not None else e["reserved"] for e in entries.values())
            if spent + reserved_microusd > monthly_budget_microusd:
                raise LLMBudgetError("monthly model budget is exhausted")
            entries[reservation_id] = {"reserved": reserved_microusd, "actual": None, "at_ms": _now_ms()}
            self._save(data)

    async def commit(self, *, provider: str, reservation_id: str, actual_microusd: int) -> None:
        from kairos_llm.errors import LLMBudgetError

        async with self._lock:
            data = self._load()
            for month in data.values():
                entry = month.get(provider, {}).get(reservation_id)
                if entry is not None:
                    if entry["actual"] is not None or actual_microusd > entry["reserved"]:
                        raise LLMBudgetError("reservation cannot be committed")
                    entry["actual"] = actual_microusd
                    self._save(data)
                    return
            raise LLMBudgetError("reservation is unknown")


def _sentiment_payload(raw: dict) -> dict:
    from kairos_core.contracts import SentimentSignal
    from kairos_core.enums import ImpactDirection

    return SentimentSignal(
        message_id=str(raw["id"]),
        produced_at=datetime.fromisoformat(raw["event_time"]),
        source="text-scouts",
        topic=raw["topic"],
        sentiment=float(raw["sentiment"]),
        impact=ImpactDirection(raw["impact"]),
        confidence=float(raw["confidence"]),
        sources=list(raw.get("sources", [])),
        summary=str(raw.get("summary", "")),
    ).to_payload()


def _bar_payload(symbol: str, row: list) -> dict:
    from kairos_core.contracts import ClosedBarEventV1

    bar = ClosedBarEventV1(
        source=BAR_SOURCE,
        symbol=symbol,
        open_time_ms=int(row[0]),
        close_time_ms=int(row[0]) + 59_999,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        base_volume=float(row[5]),
        quote_volume=float(row[6]),
        taker_buy_base_volume=float(row[7]),
        taker_buy_quote_volume=float(row[8]),
    )
    if bar.bar_sha256 != row[9]:
        raise ValueError(f"cached bar {int(row[0])} does not hash to its recorded identity")
    return bar.to_payload()


def _ingest(store, request: dict) -> list[str]:
    """Every source with its receipt clock. A rejected source is reported, never substituted."""
    problems: list[str] = []

    def take(kind: str, payload: dict, received_at_ms: int, label: str) -> None:
        try:
            store.ingest(kind, payload, received_at_ms=int(received_at_ms))
        except Exception as exc:  # noqa: BLE001 - the store's own validation decides
            problems.append(f"{label}: {type(exc).__name__}: {str(exc)[:160]}")

    for symbol, item in (request.get("market") or {}).items():
        take("market", item["payload"], item["received_at_ms"], f"market {symbol}")
    # Cached bars get this process's receipt clock, as after a service restart.
    received = _now_ms()
    for symbol, rows in (request.get("closed_bars") or {}).items():
        for row in rows:
            try:
                payload = _bar_payload(symbol, row)
            except Exception as exc:  # noqa: BLE001 - reported, never substituted
                problems.append(f"closed bar {symbol}: {type(exc).__name__}: {str(exc)[:160]}")
                continue
            take("closed_bars", payload, received, f"closed bar {symbol}")
    for raw in request.get("signals") or []:
        try:
            payload = _sentiment_payload(raw)
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            problems.append(f"text {raw.get('id')}: {type(exc).__name__}")
            continue
        take("text", payload, raw["received_ms"], f"text {raw.get('id')}")
    macro = request.get("macro")
    if macro:
        take("macro", macro["payload"], macro["received_at_ms"], "macro")
    return problems


def _summary(context) -> dict:
    return {
        "context_id": context.context_id,
        "sources": {
            source.kind: source.availability if source.availability == "AVAILABLE" else source.reason_code
            for source in context.sources
        },
        "required": list(context.required_sources),
    }


async def review(request: dict, *, compact: bool = False) -> dict:
    from kairos_aggregator.candidate_review import CandidateReviewBrain, deterministic_defer
    from kairos_aggregator.config import AggregatorSettings
    from kairos_aggregator.decision_context import CandidateContextStore
    from kairos_core.contracts import CandidateRouteV1
    from kairos_llm import BudgetedLLMGateway, LLMGateway, LLMSettings, Provider

    started = time.monotonic()
    routes = [CandidateRouteV1.model_validate(raw) for raw in request["routes"]]
    validated = time.monotonic()
    settings = AggregatorSettings(_env_file=None, bus_backend="memory", trading_symbols=request["symbols"])
    store = CandidateContextStore(settings)
    problems = _ingest(store, request)
    ingested = time.monotonic()

    llm = request["llm"]
    deadline = min(route.review_deadline_ms for route in routes)
    # Leave time for contract validation, IPC and publication by the runner.
    finish_ms = deadline - 2_000
    timeout_s = max(0.001, min(55.0, (finish_ms - _now_ms()) / 1000))
    gateway = BudgetedLLMGateway(
        LLMGateway(
            settings=LLMSettings(
                _env_file=None,
                openai_api_key=llm["openai_api_key"],
                openai_base_url=llm.get("openai_base_url"),
                max_retries=0,
                request_timeout_s=timeout_s,
            )
        ),
        FileUsageBudget(Path(llm["ledger"])),
        monthly_budgets_microusd={
            Provider.OPENAI: int(llm["monthly_budget_microusd"]),
            Provider.DEEPSEEK: 1_000_000,
        },
    )
    brain = CandidateReviewBrain(gateway, source=settings.service_name, clock_ms=_now_ms)
    ready = time.monotonic()

    # Freeze every route from the same immutable evidence cut before any provider setup or call.
    # Lazy HTTP client construction and model prompt compilation can run before their first await.
    cut = _now_ms()
    contexts = []
    for route in routes:
        try:
            contexts.append(store.build(route, cutoff_ms=cut, captured_at_ms=cut))
        except (ValueError, KeyError, TypeError):
            contexts.append(None)
    captured = time.monotonic()

    def defer(route, reason):
        return deterministic_defer(
            route, reviewed_at_ms=_now_ms(), reason_code=reason, source=settings.service_name
        )

    async def one(route, context):
        if context is None:
            return defer(route, "DECISION_CONTEXT_INVALID")
        remaining = (finish_ms - _now_ms()) / 1000
        if remaining > 0:
            try:
                async with asyncio.timeout(remaining):
                    return await brain.review_with_context(route, context)
            except TimeoutError:
                pass
        return defer(route, "REVIEW_DEADLINE_EXCEEDED")

    try:
        results = await asyncio.gather(
            *(one(route, context) for route, context in zip(routes, contexts, strict=True))
        )
    finally:
        await gateway.close()
    reviews = []
    for route, result, context in zip(routes, results, contexts, strict=True):
        if result.intent != route.intent or result.route != route:
            raise ValueError("a review carries another route than the one that was sent")
        reviews.append(
            {
                "intent_id": route.intent.intent_id,
                "route_id": route.route_id,
                # The caller holds the exact route and intent verified above.
                "review": result.model_dump(mode="json", exclude={"route", "intent"} if compact else None),
                "context": _summary(context) if context is not None else None,
                "context_capture_at_ms": cut,
            }
        )
    return {
        "reviews": reviews,
        "ingest_problems": problems,
        "timing": {
            "validate_ms": round((validated - started) * 1000),
            "ingest_ms": round((ingested - validated) * 1000),
            "gateway_init_ms": round((ready - ingested) * 1000),
            "capture_ms": round((captured - ready) * 1000),
            "review_ms": round((time.monotonic() - captured) * 1000),
        },
    }


def main() -> int:
    from worker_session import serve_requests

    # argv[1] replaces this script's own directory on the import path, so nothing from the
    # runner package can shadow Aggregator imports.
    sys.path[0] = sys.argv[1]
    if len(sys.argv) > 2:
        sys.path.insert(1, sys.argv[2])  # the kairos-llm checkout, ahead of the installed copy
    if "--serve" in sys.argv[3:]:
        # Load libraries before advertising readiness. No key or provider call is needed.
        import platform

        import kairos_llm  # noqa: F401
        from kairos_aggregator import candidate_review, config, decision_context  # noqa: F401
        from openai.resources.responses import AsyncResponses  # noqa: F401

        # The SDK queries this on the first request. On Windows CPython may start a
        # shell to discover the OS version; its result is cached by the standard library.
        platform.platform()

        return serve_requests(lambda request: _handle(request, compact=True))
    result = _handle(json.load(sys.stdin), compact=False)
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


def _handle(request: dict, *, compact: bool) -> dict:
    secret = str((request.get("llm") or {}).get("openai_api_key") or "")
    try:
        if request.get("op") != "review":
            raise ValueError(f"unknown operation {request.get('op')!r}")
        return asyncio.run(review(request, compact=compact))
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        message = f"{type(exc).__name__}: {str(exc)[:400]}"
        return {"error": message.replace(secret, "[key]") if secret else message}


if __name__ == "__main__":
    raise SystemExit(main())
