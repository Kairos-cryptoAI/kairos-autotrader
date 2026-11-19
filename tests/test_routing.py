"""Offline tests of the bridge to the real Kairos Router policy.

``test_real_*`` start the real repository code (no network, no model); they are skipped
where the Router checkout or its environment is absent.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from support import QUARTER, route_of, strict_intent

from kairos_autotrader.routing import ROUTE_SOURCE, RouterConfig, RouterError, route_intents

DECISION = 1_794_400_000_000 // QUARTER * QUARTER - 1


def news(sid: str, *, impact: str, sentiment: float, confidence: float, minutes_before: float, topic="ETF"):
    produced = datetime.fromtimestamp((DECISION - minutes_before * 60_000) / 1000, UTC)
    return {
        "id": sid,
        "topic": topic,
        "sentiment": sentiment,
        "impact": impact,
        "confidence": confidence,
        "sources": ["https://example.com/a"],
        "summary": "s",
        "origin": "llm",
        "event_time": produced.isoformat(),
    }


def intents() -> dict:
    return {
        "SOLUSD": strict_intent(
            symbol="SOLUSDT", decision_ts_ms=DECISION, side="SHORT", stop=83.0, take=81.0
        ),
        "LINKUSD": strict_intent(
            symbol="LINKUSDT", decision_ts_ms=DECISION, ref_price=10.3, stop=10.18, take=10.48
        ),
    }


def route(runner, *, offered=None, **over):
    arguments = dict(signals=[], now_ms=DECISION + 15_000, review_timeout_ms=60_000, runner=runner)
    arguments.update(over)
    return asyncio.run(route_intents(RouterConfig(), offered or intents(), **arguments))


def test_routes_are_validated_contracts_around_the_same_intents():
    offered = intents()
    seen = []

    def runner(config, request):
        seen.append(request)
        rows = []
        for raw in request["intents"]:
            intent = next(i for i in offered.values() if i.intent_id == raw["intent_id"])
            if intent.symbol == "LINKUSDT":
                rows.append({"intent_id": intent.intent_id, "rejected": "stale candidate"})
            else:
                rows.append(
                    {
                        "intent_id": intent.intent_id,
                        "route": route_of(intent).model_dump(mode="json"),
                        "text": {"bias": "FLAT", "score": 0.0, "confidence": 0.0},
                    }
                )
        return {"routes": rows, "settings": {"candidate_ttl_s": 120.0}}

    routing = route(runner, offered=offered)
    request = seen[0]
    assert request["source"] == ROUTE_SOURCE and request["symbols"] == ["LINKUSDT", "SOLUSDT"]
    assert (request["now_ms"], request["review_timeout_ms"]) == (DECISION + 15_000, 60_000)
    (routed,) = routing.routed
    assert routed.instrument == "SOLUSD" and routed.intent == offered["SOLUSD"]
    assert routing.rejected == [
        {"symbol": "LINKUSD", "id": offered["LINKUSD"].intent_id, "reason": "stale candidate"}
    ]
    assert routing.text == {"SOLUSD": {"bias": "FLAT", "score": 0.0, "confidence": 0.0}}
    assert routing.settings == {"candidate_ttl_s": 120.0}
    assert (
        asyncio.run(route_intents(RouterConfig(), {}, signals=[], now_ms=1, review_timeout_ms=1)).routed == []
    )


@pytest.mark.parametrize(
    "answer",
    [
        lambda rows: {"routes": rows[:1]},
        lambda rows: {"routes": rows + [{"intent_id": "f" * 64, "rejected": "x"}]},
        lambda rows: {"routes": [{**rows[0], "route": {"review_tier": "NORMAL"}}, rows[1]]},
        lambda rows: {"routes": [{**rows[0], "route": rows[1]["route"]}, rows[1]]},
        lambda rows: {},
    ],
)
def test_an_answer_that_is_not_exactly_about_the_offered_intents_is_refused(answer):
    offered = intents()

    def runner(config, request):
        rows = [
            {"intent_id": i.intent_id, "route": route_of(i).model_dump(mode="json")} for i in offered.values()
        ]
        return answer(rows)

    with pytest.raises(RouterError):
        route(runner, offered=offered)


def real_config() -> RouterConfig:
    config = RouterConfig()
    try:
        config.check()
    except RouterError:
        pytest.skip("Router checkout or environment is not present")
    return config


def test_real_router_marks_a_conflict_only_for_fresh_confident_opposing_text():
    config = real_config()
    offered = intents()
    calm = route(None, offered=offered)
    assert calm.settings == {"sentiment_ttl_s": 600.0, "candidate_ttl_s": 120.0, "review_timeout_ms": 60_000}
    assert {(r.instrument, r.route.review_tier.value) for r in calm.routed} == {
        ("LINKUSD", "NORMAL"),
        ("SOLUSD", "NORMAL"),
    }
    for item in calm.routed:
        assert item.route.source == ROUTE_SOURCE and item.route.evidence_ids == ()
        assert item.route.routed_at_ms == DECISION and item.route.review_deadline_ms == DECISION + 60_000
        assert item.route.requested_reasoning_effort.value == "medium"

    bullish = news("n1", impact="bullish", sentiment=0.9, confidence=0.6, minutes_before=3)
    old = news("n2", impact="bullish", sentiment=0.9, confidence=0.9, minutes_before=45)
    weak = news("n3", impact="bullish", sentiment=0.9, confidence=0.1, minutes_before=2)
    later = news("n4", impact="bearish", sentiment=-0.9, confidence=0.9, minutes_before=-1)
    by_instrument = {
        r.instrument: r
        for r in asyncio.run(
            route_intents(
                config,
                offered,
                signals=[bullish, old, weak, later],
                now_ms=DECISION + 15_000,
                review_timeout_ms=30_000,
            )
        ).routed
    }
    short, long = by_instrument["SOLUSD"].route, by_instrument["LINKUSD"].route
    assert (short.review_tier.value, short.requested_reasoning_effort.value) == ("CONFLICT", "high")
    assert short.evidence_ids == ("n1",), "stale, weak and later text is not part of the route"
    assert short.conflict_rationale.startswith("strategy_side=SHORT text_side=LONG")
    assert short.review_deadline_ms == DECISION + 30_000
    assert long.review_tier.value == "NORMAL" and long.evidence_ids == ("n1",)
    assert short.intent == offered["SOLUSD"], "the Router returns the intent it was given, unchanged"

    specific = news(
        "n5", impact="bearish", sentiment=-0.9, confidence=0.9, minutes_before=1, topic="LINKUSDT"
    )
    by_instrument = {
        r.instrument: r.route
        for r in asyncio.run(
            route_intents(
                config,
                offered,
                signals=[bullish, specific],
                now_ms=DECISION + 15_000,
                review_timeout_ms=60_000,
            )
        ).routed
    }
    assert by_instrument["LINKUSD"].review_tier.value == "CONFLICT", "symbol evidence wins over market-wide"
    assert by_instrument["LINKUSD"].evidence_ids == ("n5",)
    assert by_instrument["SOLUSD"].evidence_ids == ("n1",)


def test_real_router_time_rules_reject_late_and_early_candidates():
    config = real_config()
    offered = intents()

    def reasons(now_ms: int, timeout_ms: int = 60_000) -> set[str]:
        routing = asyncio.run(
            route_intents(config, offered, signals=[], now_ms=now_ms, review_timeout_ms=timeout_ms)
        )
        assert routing.routed == [] and {r["symbol"] for r in routing.rejected} == {"SOLUSD", "LINKUSD"}
        return {r["reason"] for r in routing.rejected}

    assert reasons(DECISION + 121_000) == {"stale candidate"}
    assert reasons(DECISION + 60_000) == {"candidate review deadline has elapsed"}
    assert reasons(DECISION + 20_000, timeout_ms=20_000) == {"candidate review deadline has elapsed"}
    assert reasons(DECISION - 1_000) == {"candidate postdates routing evaluation time"}
    assert reasons(DECISION - 60_000) == {"future-dated candidate"}
    assert len(route(None, offered=offered, now_ms=DECISION + 59_999).routed) == 2


def test_real_router_ignores_pair_evidence_outside_the_candidate_batch():
    config = real_config()
    offered = {"SOLUSD": intents()["SOLUSD"]}
    foreign = news(
        "link", topic="LINKUSDT", impact="bullish", sentiment=0.9, confidence=0.9, minutes_before=1
    )
    market = news("market", topic="ETF", impact="bearish", sentiment=-0.9, confidence=0.8, minutes_before=2)
    (item,) = asyncio.run(
        route_intents(
            config, offered, signals=[foreign, market], now_ms=DECISION + 15_000, review_timeout_ms=60_000
        )
    ).routed
    assert item.route.evidence_ids == ("market",)
    assert item.route.review_tier.value == "NORMAL"


def test_real_warm_router_preserves_identity_deadlines_and_isolates_batches():
    from kairos_autotrader.routing import router_session

    config = real_config()
    offered = intents()
    common = dict(now_ms=DECISION + 15_000, review_timeout_ms=60_000)
    expected = asyncio.run(route_intents(config, offered, signals=[], **common))

    async def exercise():
        session = router_session(config)
        try:
            await session.warm()
            bullish = news("n1", impact="bullish", sentiment=0.9, confidence=0.8, minutes_before=1)
            first = await route_intents(config, offered, signals=[bullish], session=session, **common)
            assert any(item.route.review_tier.value == "CONFLICT" for item in first.routed)
            second = await route_intents(config, offered, signals=[], session=session, **common)
            assert second == expected
        finally:
            await session.stop()

    asyncio.run(exercise())
