"""Runs the Kairos Aggregator's candidate-review rules for already routed candidates.

Executed as a script by the Aggregator repository's own Python environment. It uses the
repository's unchanged code: the candidate-review system prompt, the
``CandidateReviewOutput`` schema (decision ALLOW / VETO / DEFER, priority, reason codes)
and the guard that turns an ALLOW into DEFER when strongly adverse news caused a conflict
route. The route itself (tier, deadline, evidence ids) comes from the real Router policy
and is only read here.

Still a stand-in: the review payload below is assembled here and is not
``DecisionContextV1``, and the model call is made by the runner, not by ``LLMGateway``
through ``CandidateReviewBrain``. See docs/ALIGNMENT.md, step 2.

argv: <aggregator source> <kairos-core source>
stdin (prepare):  {"op": "prepare", "now_ms": ..., "candidates": [<signal record with
                   "conflict_rationale" and "routing_text">, ...], "signals": [...],
                   "market": {"<instrument>": {...}}, "portfolio": {...}, "context": {...}}
stdin (validate): {"op": "validate", "signals": [...], "reviews": [{"id", "side", "tier",
                   "routing_ids": [...], "model_output": {...} | null}]}
stdout: a marker line followed by JSON.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from types import SimpleNamespace

MARKER = "@@KAIROS-WORKER@@"
EVIDENCE_LOOKBACK_MS = 6 * 3_600_000
MAX_EVIDENCE = 8


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
    from kairos_core.enums import CandidateReviewTier

    now_ms = int(request["now_ms"])
    origins = {str(raw["id"]): raw.get("origin", "llm") for raw in request.get("signals", [])}
    sentiments = _sentiments(request.get("signals", []))
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
        tier = CandidateReviewTier(candidate["review_tier"])
        instrument = candidate["symbol"]
        routing_text = dict(candidate.get("routing_text") or {})
        context = {
            "contract": "autotrader-review-context.v1; same kinds of content as "
            "DecisionContextV1, not that contract",
            "route": {
                "route_id": candidate["route_id"],
                "review_tier": tier.value,
                "conflict_rationale": candidate.get("conflict_rationale"),
                "routed_at_ms": int(candidate["signal_time"]) - 1,
                "review_deadline_ms": candidate["review_deadline_ms"],
            },
            "immutable_intent": {
                "intent_id": candidate["id"],
                "strategy_id": candidate["strategy_id"],
                "strategy_revision": candidate["strategy_revision"],
                "setup": candidate["setup"],
                "symbol": candidate["signal_symbol"],
                "venue": candidate["signal_venue"],
                "execution_venue": "EVEDEX",
                "execution_instrument": instrument,
                "side": candidate["side"],
                "decision_ts_ms": int(candidate["signal_time"]) - 1,
                "reference_price": candidate["ref_price"],
                "signal_strength": candidate["score"],
                "exit_plan": {
                    "stop_price": candidate["stop"],
                    "target_price": candidate["take"],
                    "max_holding_minutes": candidate["timeout_minutes"],
                },
            },
            "market": market.get(instrument, "UNAVAILABLE"),
            "market_leader": market.get("BTCUSD", "UNAVAILABLE")
            if instrument != "BTCUSD"
            else "same instrument",
            "text": {
                "availability": "AVAILABLE" if evidence else "UNAVAILABLE",
                "routing": {**routing_text, "evidence_ids": list(candidate["text_evidence_ids"])},
                "evidence": evidence,
            },
            "macro": request.get("macro") or "UNAVAILABLE",
            "portfolio": request.get("portfolio", "UNAVAILABLE"),
            "other": request.get("context", {}),
        }
        prepared.append(
            {
                "id": candidate["id"],
                "side": candidate["side"],
                "tier": tier.value,
                "effort": "high" if tier is CandidateReviewTier.CONFLICT else "medium",
                "routing_ids": list(candidate["text_evidence_ids"]),
                "text_score": routing_text.get("score"),
                "review_deadline_ms": candidate["review_deadline_ms"],
                "user": json.dumps(
                    {"authority": "review_only", "decision_context": context},
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ),
            }
        )
    return {"system": CAUSAL_CANDIDATE_REVIEW_SYSTEM, "prepared": prepared, "evidence": len(evidence)}


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
    request = json.load(sys.stdin)
    try:
        result = prepare(request) if request.get("op") == "prepare" else validate(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
