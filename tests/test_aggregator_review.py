"""Offline tests of the bridge to the Kairos Aggregator review rules.

``test_real_*`` start the real repository code (no network, no model); they are skipped
where the checkout or the Aggregator environment is absent. The Router's classification
is tested in ``test_routing.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from support import routed

from kairos_autotrader import aggregator_review, brain, engine, signals
from kairos_autotrader.aggregator_review import (
    AggregatorConfig,
    AggregatorError,
    allowed_in_order,
    review_candidates,
)
from kairos_autotrader.engine import Config, PaperBroker, Store, Trader


def candidate(symbol: str = "SOLUSD", side: str = "SHORT", score: float = 0.4, **route) -> signals.Routed:
    stop, take = (83.9, 81.65) if side == "SHORT" else (82.1, 84.35)
    return routed(
        symbol=symbol, side=side, ref_price=83.0, stop=stop, take=take, atr=1.1, score=score, **route
    )


def news(sid: str, *, impact: str, sentiment: float, confidence: float, minutes_ago: float) -> dict:
    return {
        "id": sid,
        "topic": "ETF approval",
        "sentiment": sentiment,
        "impact": impact,
        "confidence": confidence,
        "sources": ["https://example.com/a"],
        "summary": "s",
        "origin": "llm",
        "event_time": (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat(),
    }


def real_config() -> AggregatorConfig:
    config = AggregatorConfig()
    try:
        config.check()
    except AggregatorError:
        pytest.skip("Aggregator / Router checkouts or environment are not present")
    return config


def prepare(config, candidates, signals, routing_text=None, macro=None):
    return aggregator_review._run_worker(
        config,
        {
            "op": "prepare",
            "now_ms": int(time.time() * 1000),
            "candidates": [
                {
                    **c.record(),
                    "conflict_rationale": c.route.conflict_rationale,
                    "routing_text": (routing_text or {}).get(c.instrument, {}),
                }
                for c in candidates
            ],
            "signals": signals,
            "market": {"SOLUSD": {"rsi": 41.0, "trend": "DOWN"}},
            "portfolio": {"open_positions": [], "free_slots": 3},
            "context": {},
            "macro": macro,
        },
    )


def test_real_review_context_is_built_around_the_route_it_was_given():
    config = real_config()
    fresh_bullish = news("n1", impact="bullish", sentiment=0.9, confidence=0.6, minutes_ago=3)
    conflicted = candidate(tier="CONFLICT", evidence_ids=("n1",))
    text = {"SOLUSD": {"bias": "LONG", "score": 0.54, "confidence": 0.6}}
    conflict = prepare(config, [conflicted], [fresh_bullish], text, macro={"regime": "BEAR"})
    item = conflict["prepared"][0]
    assert (item["id"], item["tier"], item["effort"]) == (conflicted.id, "CONFLICT", "high")
    assert item["routing_ids"] == ["n1"] and item["text_score"] == 0.54
    assert item["review_deadline_ms"] == conflicted.route.review_deadline_ms
    assert "ALLOW|VETO|DEFER" in conflict["system"] and "Never propose" in conflict["system"]
    context = json.loads(item["user"])
    assert context["authority"] == "review_only"
    review = context["decision_context"]
    assert review["route"] == {
        "route_id": conflicted.route.route_id,
        "review_tier": "CONFLICT",
        "conflict_rationale": conflicted.route.conflict_rationale,
        "routed_at_ms": conflicted.intent.decision_ts_ms,
        "review_deadline_ms": conflicted.route.review_deadline_ms,
    }
    intent = review["immutable_intent"]
    assert (intent["intent_id"], intent["symbol"], intent["venue"]) == (
        conflicted.id,
        "SOLUSDT",
        "BINANCE_UM",
    )
    assert (intent["execution_venue"], intent["execution_instrument"]) == ("EVEDEX", "SOLUSD")
    assert intent["side"] == "SHORT" and intent["strategy_id"] == conflicted.intent.strategy_id
    assert intent["exit_plan"] == {"stop_price": 83.9, "target_price": 81.65, "max_holding_minutes": 720}
    assert review["market"] == {"rsi": 41.0, "trend": "DOWN"} and review["macro"] == {"regime": "BEAR"}
    assert review["text"]["routing"] == {**text["SOLUSD"], "evidence_ids": ["n1"]}
    assert review["text"]["evidence"][0]["sentiment"] == pytest.approx(0.54), "calibrated by confidence"
    assert len(item["user"]) < 6_000, "the prompt carries no megabyte of input bar hashes"

    calm = prepare(config, [candidate(), candidate(side="LONG")], [])
    assert [(p["tier"], p["effort"], p["routing_ids"]) for p in calm["prepared"]] == [
        ("NORMAL", "medium", []),
        ("NORMAL", "medium", []),
    ]
    quiet = json.loads(calm["prepared"][0]["user"])["decision_context"]
    assert quiet["text"]["availability"] == "UNAVAILABLE" and quiet["macro"] == "UNAVAILABLE"
    assert quiet["route"]["conflict_rationale"] is None


def test_real_aggregator_validation_fails_closed_and_applies_its_conflict_guard():
    config = real_config()
    adverse = news("n1", impact="bullish", sentiment=1.0, confidence=0.9, minutes_ago=2)

    def review(cid, tier, output, routing=()):
        return {
            "id": cid,
            "side": "SHORT",
            "tier": tier,
            "routing_ids": list(routing),
            "model_output": output,
        }

    allow = {"decision": "ALLOW", "priority": 70, "reason_codes": ["TREND_ALIGNED"]}
    result = aggregator_review._run_worker(
        config,
        {
            "op": "validate",
            "signals": [adverse],
            "reviews": [
                review("ok", "NORMAL", allow),
                review("guarded", "CONFLICT", allow, ["n1"]),
                review(
                    "veto",
                    "NORMAL",
                    {"decision": "VETO", "priority": 0, "reason_codes": ["NEWS_INVALIDATES"]},
                ),
                review("moves-side", "NORMAL", {**allow, "side": "LONG"}),
                review("bad-code", "NORMAL", {**allow, "reason_codes": ["looks good"]}),
                review("no-answer", "NORMAL", None),
            ],
        },
    )
    by_id = {r["id"]: r for r in result["reviews"]}
    assert (by_id["ok"]["decision"], by_id["ok"]["priority"], by_id["ok"]["reviewer"]) == ("ALLOW", 70, "LLM")
    assert (
        by_id["guarded"]["decision"] == "DEFER" and "CONFLICT_ALLOW_GUARD" in by_id["guarded"]["reason_codes"]
    )
    assert by_id["guarded"]["priority"] == 0 and by_id["guarded"]["guard"] is True
    assert by_id["veto"]["decision"] == "VETO"
    for broken in ("moves-side", "bad-code", "no-answer"):
        assert by_id[broken] == {
            "id": broken,
            "decision": "DEFER",
            "priority": 0,
            "reason_codes": ["LLM_FAILURE"],
            "reviewer": "DETERMINISTIC",
            "guard": False,
        }, broken


def test_each_candidate_is_reviewed_once_with_its_own_effort():
    sol, link = candidate(tier="CONFLICT", evidence_ids=("n1",)), candidate(symbol="LINKUSD")
    asked = []
    clock = iter([sol.route.review_deadline_ms - 1, link.route.review_deadline_ms + 1])

    def runner(config, request):
        if request["op"] == "prepare":
            sent = request["candidates"]
            assert [c["id"] for c in sent] == [sol.id, link.id]
            assert sent[0]["conflict_rationale"] == sol.route.conflict_rationale
            assert sent[0]["routing_text"] == {"score": 0.5} and sent[1]["routing_text"] == {}
            assert sent[0]["route_id"] == sol.route.route_id and sent[0]["text_evidence_ids"] == ["n1"]
            return {
                "system": "SYS",
                "prepared": [
                    {
                        "id": sol.id,
                        "side": "SHORT",
                        "tier": "CONFLICT",
                        "effort": "high",
                        "routing_ids": ["n1"],
                        "text_score": 0.5,
                        "review_deadline_ms": sol.route.review_deadline_ms,
                        "user": "U-SOL",
                    },
                    {
                        "id": link.id,
                        "side": "SHORT",
                        "tier": "NORMAL",
                        "effort": "medium",
                        "routing_ids": [],
                        "text_score": None,
                        "review_deadline_ms": link.route.review_deadline_ms,
                        "user": "U-LINK",
                    },
                ],
            }
        assert [r["model_output"] for r in request["reviews"]] == [{"decision": "VETO"}, None]
        return {
            "reviews": [
                {
                    "id": sol.id,
                    "decision": "VETO",
                    "priority": 0,
                    "reason_codes": ["X"],
                    "reviewer": "LLM",
                    "guard": False,
                },
                {
                    "id": link.id,
                    "decision": "DEFER",
                    "priority": 0,
                    "reason_codes": ["LLM_FAILURE"],
                    "reviewer": "DETERMINISTIC",
                    "guard": False,
                },
            ]
        }

    async def extract(session, *, api_key, model, system, user, effort, timeout_s):
        asked.append((system, user, effort))
        return (
            ({"decision": "VETO"}, {"input_tokens": 500}, "") if user == "U-SOL" else (None, {}, "HTTP 500")
        )

    reviews = asyncio.run(
        review_candidates(
            None,
            AggregatorConfig(),
            api_key="k",
            model="m",
            candidates=[sol, link],
            routing_text={"SOLUSD": {"score": 0.5}},
            signals=[],
            market={},
            portfolio={},
            context={},
            now_ms=1,
            runner=runner,
            extract=extract,
            clock_ms=lambda: next(clock),
        )
    )
    assert sorted(asked) == [("SYS", "U-LINK", "medium"), ("SYS", "U-SOL", "high")]
    assert [(r["id"], r["decision"], r["tier"], r["problem"]) for r in reviews] == [
        (sol.id, "VETO", "CONFLICT", ""),
        (link.id, "DEFER", "NORMAL", "HTTP 500"),
    ]
    assert [r["route_deadline_missed"] for r in reviews] == [False, True], "recorded against the route"


def test_allowed_candidates_are_ranked_by_priority_one_per_instrument():
    a, b, c, d = (
        candidate(score=0.3),
        candidate(score=0.5),
        candidate(symbol="LINKUSD"),
        candidate(symbol="XRPUSD"),
    )
    reviews = [
        {"id": a.id, "decision": "ALLOW", "priority": 60},
        {"id": b.id, "decision": "ALLOW", "priority": 80},
        {"id": c.id, "decision": "ALLOW", "priority": 70},
        {"id": d.id, "decision": "VETO", "priority": 99},
        {"id": "GHOST", "decision": "ALLOW", "priority": 100},
    ]
    assert allowed_in_order(reviews, [a, b, c, d], 3) == [b.id, c.id]
    assert allowed_in_order(reviews, [a, b, c, d], 1) == [b.id]
    tie = [
        {"id": a.id, "decision": "ALLOW", "priority": 50},
        {"id": b.id, "decision": "ALLOW", "priority": 50},
    ]
    assert allowed_in_order(tie, [a, b], 1) == [b.id], "equal priority: the stronger signal first"


def trader_with_reviews(tmp_path, monkeypatch, *, mode: str, shadow: bool = True, reviews=None, fail=False):
    cfg = Config(review_mode=mode, review_shadow=shadow, use_text_scouts=False)
    cfg.validate()
    store = Store(tmp_path / "rt")
    trader = Trader(
        api=None,
        session=None,
        store=store,
        config=cfg,
        broker=PaperBroker(store, cfg, 0),
        fee=0,
        instruments={},
        openai_key="sk-test",
    )
    trader.aggregator, trader._aggregator_revisions = AggregatorConfig(), {"aggregator": "x"}
    calls = {"probability": 0, "aggregator": 0}

    async def fear_greed(session):
        return None

    async def ask_llm(session, **kwargs):
        calls["probability"] += 1
        first = kwargs["candidates"][0]
        return brain.Decision("TRADE", first.id, 0.56, "probability says go", "LLM", picks=[first.id])

    async def fake_reviews(session, config, **kwargs):
        calls["aggregator"] += 1
        if fail:
            raise AggregatorError("worker failed")
        return reviews

    monkeypatch.setattr(brain, "fetch_fear_greed", fear_greed)
    monkeypatch.setattr(brain, "ask_llm", ask_llm)
    monkeypatch.setattr(engine, "review_candidates", fake_reviews)
    return trader, calls


def review_row(cid, decision, priority):
    return {
        "id": cid,
        "decision": decision,
        "priority": priority,
        "reason_codes": ["CODE"],
        "tier": "NORMAL",
        "effort": "medium",
        "text_score": 0.0,
        "usage": {"input_tokens": 900, "output_tokens": 40},
        "problem": "",
    }


def events(trader) -> list[dict]:
    path = trader.store.directory / "journal.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_shadow_mode_trades_on_the_probability_review_and_records_the_aggregator(tmp_path, monkeypatch):
    sol, link = candidate(), candidate(symbol="LINKUSD")
    trader, calls = trader_with_reviews(
        tmp_path,
        monkeypatch,
        mode="probability",
        reviews=[review_row(sol.id, "VETO", 0), review_row(link.id, "ALLOW", 65)],
    )
    decision = asyncio.run(trader.decide([], [sol, link], 2))
    assert decision.source == "LLM" and decision.picks == [sol.id]
    assert calls == {"probability": 1, "aggregator": 1} and trader.store.state["day"]["llm_calls"] == 3
    names = [e["event"] for e in events(trader)]
    assert "aggregator_review" in names and "shadow_decision" in names
    shadow = next(e for e in events(trader) if e["event"] == "shadow_decision")["decision"]
    assert shadow["source"] == "AGGREGATOR" and shadow["picks"] == [link.id]


def test_aggregator_mode_trades_allowed_candidates_by_priority(tmp_path, monkeypatch):
    sol, link, xrp = (
        candidate(),
        candidate(symbol="LINKUSD"),
        candidate(symbol="XRPUSD"),
    )
    trader, calls = trader_with_reviews(
        tmp_path,
        monkeypatch,
        mode="aggregator",
        reviews=[
            review_row(sol.id, "ALLOW", 55),
            review_row(link.id, "ALLOW", 80),
            review_row(xrp.id, "DEFER", 0),
        ],
    )
    decision = asyncio.run(trader.decide([], [sol, link, xrp], 2))
    assert (
        decision.source == "AGGREGATOR" and decision.action == "TRADE" and decision.picks == [link.id, sol.id]
    )
    assert decision.confidence == 0.8 and "ALLOW 2, VETO 0, DEFER 1" in decision.reason
    assert {v["candidate_id"]: v["verdict"] for v in decision.verdicts}[xrp.id] == "DEFER"
    shadow = next(e for e in events(trader) if e["event"] == "shadow_decision")["decision"]
    assert shadow["source"] == "LLM", "the probability review is recorded as the shadow"

    alone, calls = trader_with_reviews(
        tmp_path / "b", monkeypatch, mode="aggregator", shadow=False, reviews=[review_row(sol.id, "VETO", 0)]
    )
    waited = asyncio.run(alone.decide([], [sol], 1))
    assert waited.action == "WAIT" and calls == {"probability": 0, "aggregator": 1}


def test_aggregator_failure_never_trades_in_its_own_mode_and_is_harmless_as_shadow(tmp_path, monkeypatch):
    sol = candidate()
    own, _ = trader_with_reviews(tmp_path / "a", monkeypatch, mode="aggregator", shadow=False, fail=True)
    assert asyncio.run(own.decide([], [sol], 1)).action == "WAIT"
    assert "aggregator_error" in [e["event"] for e in events(own)]
    shadow, _ = trader_with_reviews(tmp_path / "b", monkeypatch, mode="probability", fail=True)
    decision = asyncio.run(shadow.decide([], [sol], 1))
    assert decision.source == "LLM" and decision.action == "TRADE"


def test_unknown_review_mode_is_refused():
    with pytest.raises(engine.ConfigError):
        Config(review_mode="yolo").validate()
