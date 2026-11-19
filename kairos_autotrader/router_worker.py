"""Routes strategy intents with the real Kairos Router policy.

Executed as a script by the Router repository's own Python environment. It uses the
repository's unchanged code:

* ``SignalWindow``: confidence-calibrated aggregation of text evidence in the causal window
  that ends at the intent's decision, with the Router's own settings;
* ``CandidateRouterPolicy.build``: the ``CandidateRouteV1`` of one immutable intent, NORMAL
  or CONFLICT, with its review deadline and the exact evidence ids used.

Text evidence is bucketed as ``RouterService`` does it: a topic that is one of the traded
symbols is symbol-specific, everything else is market-wide, and symbol evidence wins when
there is any. The time checks of ``RouterService._candidate_rejection`` are restated below,
because that method belongs to the bus service, whose pinned release refuses LIVE routing
by configuration. Nothing here can change an intent.

argv:   <router source>
stdin:  {"now_ms": ..., "review_timeout_ms": 60000, "source": "kairos-autotrader",
         "symbols": ["SOLUSDT", ...],
         "signals": [{"id", "event_time", "topic", "sentiment", "impact", "confidence",
                      "sources", "summary"}, ...],
         "intents": [<StrategyIntentV1 as JSON>, ...]}
stdout: a marker line followed by
        {"routes": [{"intent_id": ..., "route": <CandidateRouteV1 as JSON>,
                     "text": {"bias", "score", "confidence"}}
                    | {"intent_id": ..., "rejected": "..."}],
         "settings": {"sentiment_ttl_s", "candidate_ttl_s", "review_timeout_ms"}}
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

MARKER = "@@KAIROS-WORKER@@"


def _sentiment(raw: dict):
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
    )


def _sentiments(rows: list) -> list:
    """Stored signals as SentimentSignal contracts. Malformed evidence is dropped, never guessed at."""
    signals = []
    for raw in rows:
        try:
            signal = _sentiment(raw)
        except Exception:  # noqa: BLE001 - the contract's own validation decides what is malformed
            signal = None
        if signal is not None:
            signals.append(signal)
    return signals


def _rejection(settings, intent, route, now_ms: int) -> str | None:
    """The time and symbol checks of ``RouterService._candidate_rejection``, in its order."""
    if not settings.symbol_allowed(intent.symbol):
        return "unsupported candidate symbol"
    if intent.decision_ts_ms > now_ms + int(settings.event_future_tolerance_s * 1_000):
        return "future-dated candidate"
    if intent.decision_ts_ms > now_ms:
        return "candidate postdates routing evaluation time"
    if intent.decision_ts_ms < now_ms - int(settings.candidate_ttl_s * 1_000):
        return "stale candidate"
    if now_ms >= intent.entry_expires_ts_ms:
        return "expired candidate"
    if now_ms >= route.review_deadline_ms:
        return "candidate review deadline has elapsed"
    return None


def route(request: dict) -> dict:
    from kairos_core.contracts import StrategyIntentV1
    from kairos_router.aggregation import SignalWindow
    from kairos_router.candidate import CandidateRouterPolicy
    from kairos_router.config import RouterSettings

    # No environment file: the settings are the Router's defaults plus the two values given.
    settings = RouterSettings(
        _env_file=None,
        trading_symbols=[str(symbol) for symbol in request["symbols"]],
        candidate_review_timeout_ms=int(request["review_timeout_ms"]),
    )
    window = SignalWindow(
        sentiment_ttl_s=settings.sentiment_ttl_s,
        deadband=settings.sentiment_deadband,
        min_confidence=settings.sentiment_min_confidence,
        max_evidence=settings.max_sentiment_evidence,
        max_points_per_topic=settings.processed_cache_size,
    )
    policy = CandidateRouterPolicy(
        source=str(request["source"]),
        review_timeout_ms=settings.candidate_review_timeout_ms,
        max_evidence_ids=settings.max_candidate_evidence_ids,
    )
    for signal in _sentiments(request.get("signals", [])):
        if window.has_message(signal.message_id):
            continue
        bucket = signal.topic.strip().upper() if settings.symbol_allowed(signal.topic) else "*"
        window.add_sentiment(
            bucket,
            message_id=signal.message_id,
            produced_at=signal.produced_at,
            sentiment=signal.sentiment,
            impact=signal.impact,
            confidence=signal.confidence,
        )
    now_ms = int(request["now_ms"])
    routes: list[dict] = []
    for raw in request["intents"]:
        intent = StrategyIntentV1.model_validate(raw)
        as_of = datetime.fromtimestamp(intent.decision_ts_ms / 1_000, tz=UTC)
        text = window.aggregate(intent.symbol, as_of=as_of)
        if not text.has_relevant_evidence:
            text = window.aggregate("*", as_of=as_of)
        built = policy.build(intent, text)
        rejection = _rejection(settings, intent, built, now_ms)
        if rejection is not None:
            routes.append({"intent_id": intent.intent_id, "rejected": rejection})
            continue
        routes.append(
            {
                "intent_id": intent.intent_id,
                "route": built.model_dump(mode="json"),
                "text": {
                    "bias": text.bias.value,
                    "score": round(text.score, 6),
                    "confidence": round(text.confidence, 6),
                },
            }
        )
    return {
        "routes": routes,
        "settings": {
            "sentiment_ttl_s": settings.sentiment_ttl_s,
            "candidate_ttl_s": settings.candidate_ttl_s,
            "review_timeout_ms": settings.candidate_review_timeout_ms,
        },
    }


def main() -> int:
    if "--serve" in sys.argv[2:]:
        from worker_session import serve_requests

        sys.path[0] = sys.argv[1]
        # Load the same policy, validators and settings before announcing readiness.
        route(
            {
                "symbols": ["BTCUSDT"],
                "review_timeout_ms": 60_000,
                "source": "kairos-autotrader",
                "now_ms": 0,
                "intents": [],
                "signals": [],
            }
        )
        return serve_requests(route)
    # argv[1] replaces this script's own directory on the import path, so nothing from the
    # runner package can shadow Router imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        result = route(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
