"""Runs the real Kairos Router classification and Aggregator candidate-review rules.

Executed as a script by the Aggregator repository's own Python environment. It uses the
repositories' unchanged code:

* Router: ``SignalWindow`` (confidence-calibrated text aggregation with the Router's own
  time-to-live, dead band and evidence cap) and ``classify_candidate`` (NORMAL or CONFLICT).
* Aggregator: the candidate-review system prompt, the ``CandidateReviewOutput`` schema
  (decision ALLOW / VETO / DEFER, priority, reason codes) and the guard that turns an
  ALLOW into DEFER when strongly adverse news caused a conflict route.

The strict ``StrategyIntentV1`` / ``DecisionContextV1`` contracts are not used: they are
bound to Binance minute bars and code fingerprints that an EVEDEX candidate does not have,
and filling them in would be a false claim. The review payload below carries the same
kinds of information and says what it is.

argv: <aggregator source> <router source> <kairos-core source>
stdin (prepare):  {"op": "prepare", "now_ms": ..., "candidates": [...], "signals": [...],
                   "market": {"<symbol>": {...}}, "portfolio": {...}, "context": {...}}
stdin (validate): {"op": "validate", "signals": [...], "reviews": [{"id", "side", "tier",
                   "routing_ids": [...], "model_output": {...} | null}]}
stdout: a marker line followed by JSON.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

MARKER = "@@KAIROS-WORKER@@"
EVIDENCE_LOOKBACK_MS = 6 * 3_600_000
MAX_EVIDENCE = 8
TEXT_BUCKET = "MARKET"  # Text Scouts evidence is market-wide, not per instrument


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


def prepare(request: dict) -> dict:
    from kairos_aggregator.candidate_review import CAUSAL_CANDIDATE_REVIEW_SYSTEM
    from kairos_aggregator.compiler import calibrated_sentiment
    from kairos_core.enums import CandidateReviewTier, Side
    from kairos_router.aggregation import SignalWindow
    from kairos_router.candidate import classify_candidate
    from kairos_router.config import RouterSettings

    defaults = {name: field.default for name, field in RouterSettings.model_fields.items()}
    candidate_ttl_ms = int(defaults["candidate_ttl_s"] * 1000)
    now_ms = int(request["now_ms"])
    as_of = datetime.fromtimestamp(now_ms / 1000, UTC)
    origins = {str(raw["id"]): raw.get("origin", "llm") for raw in request.get("signals", [])}
    sentiments = _sentiments(request.get("signals", []))
    window = SignalWindow(
        sentiment_ttl_s=defaults["sentiment_ttl_s"],
        deadband=defaults["sentiment_deadband"],
        min_confidence=defaults["sentiment_min_confidence"],
        max_evidence=defaults["max_sentiment_evidence"],
    )
    for signal in sentiments:
        window.add_sentiment(
            TEXT_BUCKET,
            message_id=signal.message_id,
            produced_at=signal.produced_at,
            sentiment=signal.sentiment,
            impact=signal.impact,
            confidence=signal.confidence,
        )
    text = window.aggregate(TEXT_BUCKET, as_of=as_of)

    recent = sorted(
        (
            s
            for s in sentiments
            if 0 <= now_ms - int(s.produced_at.timestamp() * 1000) <= EVIDENCE_LOOKBACK_MS
        ),
        key=lambda s: (-s.produced_at.timestamp(), s.message_id),
    )[:MAX_EVIDENCE]
    evidence = [
        {
            "message_id": s.message_id,
            "produced_at": s.produced_at.isoformat(),
            "age_minutes": int((now_ms - s.produced_at.timestamp() * 1000) // 60_000),
            "topic": s.topic,
            "sentiment": round(calibrated_sentiment(s), 6),
            "confidence": round(s.confidence, 6),
            "impact": s.impact.value,
            "summary": s.summary,
            "sources": sorted(set(s.sources))[:3],
            "extraction": "model" if origins.get(s.message_id) == "llm" else "local keyword fallback",
        }
        for s in recent
    ]
    market = request.get("market", {})
    prepared = []
    for candidate in request["candidates"]:
        side = Side(candidate["side"])
        tier = classify_candidate(side, text.bias)
        rationale = None
        if tier is CandidateReviewTier.CONFLICT:
            rationale = (
                f"strategy_side={side.value} text_side={text.bias.value} "
                f"text_score={text.score:.6f} text_confidence={text.confidence:.6f}"
            )
        symbol = candidate["symbol"]
        context = {
            "contract": "autotrader-review-context.v1 for EVEDEX; "
            "same kinds of content as DecisionContextV1, "
            "not that contract",
            "route": {"review_tier": tier.value, "conflict_rationale": rationale, "routed_at_ms": now_ms},
            "immutable_intent": {
                "intent_id": candidate["id"],
                "strategy_id": candidate["setup"],
                "symbol": symbol,
                "venue": "EVEDEX",
                "side": side.value,
                "decision_ts_ms": candidate["signal_time"],
                "reference_price": candidate["ref_price"],
                "signal_strength": candidate["score"],
                "exit_plan": {
                    "stop_price": candidate["stop"],
                    "target_price": candidate["take"],
                    "max_holding_minutes": candidate["timeout_minutes"],
                },
                "rationale": candidate.get("rationale", ""),
            },
            "market": market.get(symbol, "UNAVAILABLE"),
            "market_leader": market.get("BTCUSD", "UNAVAILABLE") if symbol != "BTCUSD" else "same instrument",
            "text": {
                "availability": "AVAILABLE" if evidence else "UNAVAILABLE",
                "routing_window_seconds": defaults["sentiment_ttl_s"],
                "routing": {
                    "bias": text.bias.value,
                    "calibrated_score": round(text.score, 6),
                    "confidence": round(text.confidence, 6),
                    "evidence_ids": list(text.sentiment_ids),
                },
                "evidence": evidence,
            },
            "macro": request.get("macro") or "UNAVAILABLE",
            "portfolio": request.get("portfolio", "UNAVAILABLE"),
            "other": request.get("context", {}),
        }
        prepared.append(
            {
                "id": candidate["id"],
                "side": side.value,
                "tier": tier.value,
                "effort": "high" if tier is CandidateReviewTier.CONFLICT else "medium",
                "routing_ids": list(text.sentiment_ids),
                "text_score": round(text.score, 6),
                # The Router drops a candidate older than its time-to-live before any review.
                "expired": now_ms - int(candidate["signal_time"]) > candidate_ttl_ms,
                "age_s": round((now_ms - int(candidate["signal_time"])) / 1000, 1),
                "user": json.dumps(
                    {"authority": "review_only", "decision_context": context},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        )
    return {
        "system": CAUSAL_CANDIDATE_REVIEW_SYSTEM,
        "prepared": prepared,
        "candidate_ttl_s": defaults["candidate_ttl_s"],
        "review_timeout_ms": defaults["candidate_review_timeout_ms"],
        "text": {"bias": text.bias.value, "score": round(text.score, 6), "evidence": len(evidence)},
    }


def validate(request: dict) -> dict:
    from kairos_aggregator.candidate_review import CandidateReviewOutput, materially_opposes_candidate
    from kairos_core.enums import CandidateReviewTier, ReviewDecision, Side

    by_id = {signal.message_id: signal for signal in _sentiments(request.get("signals", []))}
    reviews = []
    for item in request["reviews"]:
        try:
            output = CandidateReviewOutput.model_validate(item["model_output"])
        except Exception:  # noqa: BLE001 - the Aggregator's own fail-closed result
            reviews.append(
                {
                    "id": item["id"],
                    "decision": "DEFER",
                    "priority": 0,
                    "reason_codes": ["LLM_FAILURE"],
                    "reviewer": "DETERMINISTIC",
                    "guard": False,
                }
            )
            continue
        route = SimpleNamespace(
            review_tier=CandidateReviewTier(item["tier"]), intent=SimpleNamespace(side=Side(item["side"]))
        )
        used = [by_id[i] for i in item.get("routing_ids", []) if i in by_id]
        guard = output.decision is ReviewDecision.ALLOW and materially_opposes_candidate(route, used)
        if guard:
            output = CandidateReviewOutput(
                decision=ReviewDecision.DEFER,
                priority=0,
                reason_codes=tuple(sorted({*output.reason_codes, "CONFLICT_ALLOW_GUARD"})),
            )
        reviews.append(
            {
                "id": item["id"],
                "decision": output.decision.value,
                "priority": output.priority,
                "reason_codes": list(output.reason_codes),
                "reviewer": "LLM",
                "guard": guard,
            }
        )
    return {"reviews": reviews}


def main() -> int:
    sys.path[0] = sys.argv[1]
    sys.path.insert(1, sys.argv[2])
    sys.path.insert(2, sys.argv[3])
    request = json.load(sys.stdin)
    try:
        result = prepare(request) if request.get("op") == "prepare" else validate(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
