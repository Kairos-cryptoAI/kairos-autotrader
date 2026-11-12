"""Offline tests of the bridge to the real Kairos Text Scouts layer.

``test_real_text_scouts_validation_*`` start the real repository code (no network);
they are skipped where the checkout or its environment is absent.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta

import pytest

from kairos_autotrader import brain, text_scouts
from kairos_autotrader.engine import Config, PaperBroker, Store, Trader
from kairos_autotrader.text_scouts import NewsDesk, TextScoutsConfig, TextScoutsError, model_extract


def item(key: str, title: str, url: str) -> dict:
    return {
        "key": key,
        "score": 6.0,
        "title": title,
        "body": "",
        "url": url,
        "source": "feed",
        "source_kind": "rss",
        "engagement": 0.0,
        "published_at": datetime.now(UTC).isoformat(),
        "provenance": [],
    }


def signal(sid: str, url: str, origin: str, *, minutes_ago: int = 5, impact: str = "bearish") -> dict:
    return {
        "id": sid,
        "topic": "Exchange hack",
        "sentiment": -0.7 if impact == "bearish" else 0.7,
        "impact": impact,
        "confidence": 0.6,
        "sources": [url],
        "summary": "s",
        "event_time": (datetime.now(UTC) - timedelta(minutes=minutes_ago)).isoformat(),
        "origin": origin,
    }


A = item("u:https://a.example/1", "Exchange hack drains bitcoin", "https://a.example/1")
B = item("u:https://b.example/2", "Fed rate decision looms", "https://b.example/2")


class Worker:
    """Stands in for the Text Scouts worker process."""

    def __init__(self, items, local=()):
        self.items, self.local, self.requests = list(items), list(local), []

    def __call__(self, config, request):
        self.requests.append(request)
        if request["op"] == "collect":
            offered = [i for i in self.items if i["key"] not in request["seen"]]
            return {
                "counts": {
                    "fetched": 9,
                    "timely": 3,
                    "unique": 3,
                    "unseen": len(offered),
                    "kept": len(offered),
                },
                "sources": {"gdelt": 4, "rss": 5},
                "items": offered,
                "system": "SYS",
                "user": "USER",
                "schema": {},
                "local_signals": self.local if offered else [],
                "dedup_window_s": 21600.0,
            }
        return {"signals": [signal("llm-1", "https://a.example/1", "llm")], "offered": 1}


def desk(tmp_path, worker, extract=None) -> NewsDesk:
    async def no_model(*args, **kwargs):
        raise AssertionError("the model must not be called")

    return NewsDesk(Store(tmp_path / "rt"), TextScoutsConfig(), runner=worker, extract=extract or no_model)


def journal(d: NewsDesk) -> list[dict]:
    path = d.store.directory / "journal.jsonl"
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    )


def test_model_answer_becomes_signals_and_settles_every_offered_item(tmp_path):
    calls = []

    async def extract(session, *, api_key, model, system, user):
        calls.append((api_key, model, system, user))
        return {"signals": []}, {"input_tokens": 400, "output_tokens": 60}, ""

    worker = Worker([A, B])
    d = desk(tmp_path, worker, extract)
    first = asyncio.run(d.poll(object(), openai_key="sk-test", model="m", max_llm_calls=10))
    assert calls == [("sk-test", "m", "SYS", "USER")], "Text Scouts' own prompt is sent unchanged"
    assert first["origin"] == "llm" and [s["id"] for s in first["signals"]] == ["llm-1"]
    assert worker.requests[1]["op"] == "signals" and worker.requests[1]["model_output"] == {"signals": []}
    state = d.store.state["news"]
    assert set(state["seen"]) == {A["key"], B["key"]}, "an abstention is a decision too"
    assert state["day"]["llm_calls"] == 1 and journal(d)[-1]["event"] == "news_poll"
    second = asyncio.run(d.poll(object(), openai_key="sk-test", model="m", max_llm_calls=10))
    assert worker.requests[2]["seen"] == sorted([A["key"], B["key"]])
    assert second["signals"] == [] and len(calls) == 1, "nothing new, so no second model call"


def test_model_failure_falls_back_to_local_evidence_and_retries_the_rest(tmp_path):
    async def broken(session, **kwargs):
        return None, {}, "model call failed: HTTP 500"

    d = desk(tmp_path, Worker([A, B], local=[signal("loc-1", "https://a.example/1", "local")]), broken)
    summary = asyncio.run(d.poll(object(), openai_key="sk-test", model="m", max_llm_calls=10))
    assert summary["origin"] == "local" and summary["problem"].endswith("HTTP 500")
    assert [s["origin"] for s in summary["signals"]] == ["local"]
    assert set(d.store.state["news"]["seen"]) == {A["key"]}, "the item without evidence is offered again"


def test_without_a_key_or_past_the_daily_cap_no_model_is_called(tmp_path):
    local = [signal("loc-1", "https://a.example/1", "local")]
    no_key = asyncio.run(
        desk(tmp_path / "a", Worker([A], local)).poll(None, openai_key=None, model="m", max_llm_calls=10)
    )
    capped = asyncio.run(
        desk(tmp_path / "b", Worker([A], local)).poll(
            object(), openai_key="sk-test", model="m", max_llm_calls=0
        )
    )
    assert no_key["origin"] == capped["origin"] == "local"


def test_worker_failure_is_an_error_not_a_silent_empty_result(tmp_path):
    def failing(config, request):
        raise TextScoutsError("worker failed")

    with pytest.raises(TextScoutsError):
        asyncio.run(desk(tmp_path, failing).poll(None, openai_key=None, model="m", max_llm_calls=0))


def test_context_serves_recent_signals_newest_first_with_source_domains(tmp_path):
    d = desk(tmp_path, Worker([]))
    now_ms = int(time.time() * 1000)
    d.store.state["news"] = {
        "seen": {},
        "signals": [
            {**signal("old", "https://www.reuters.com/x", "llm", minutes_ago=500), "received_ms": now_ms},
            {
                **signal("mid", "https://www.reuters.com/markets/y?utm=1", "llm", minutes_ago=90),
                "received_ms": now_ms,
            },
            {
                **signal("new", "https://cointelegraph.com/z", "local", minutes_ago=4, impact="bullish"),
                "received_ms": now_ms,
            },
        ],
    }
    rows = d.context(now_ms)
    assert [(r["impact"], r["origin"], r["sources"]) for r in rows] == [
        ("bullish", "local", ["cointelegraph.com"]),
        ("bearish", "llm", ["reuters.com"]),
    ]
    assert rows[0]["age_minutes"] in (3, 4) and rows[1]["age_minutes"] in (89, 90)


class Response:
    def __init__(self, status, payload):
        self.status, self._payload = status, payload

    async def text(self):
        return json.dumps(self._payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class Session:
    def __init__(self, responses):
        self.responses, self.bodies = list(responses), []

    def post(self, url, **kwargs):
        self.bodies.append(dict(kwargs["json"]))
        return self.responses.pop(0)


def completed(text: str) -> dict:
    return {
        "status": "completed",
        "usage": {"input_tokens": 300, "output_tokens": 40},
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
    }


def test_extraction_call_retries_without_json_mode_and_reads_fenced_json():
    text_scouts.NO_JSON_MODE.clear()
    session = Session(
        [Response(400, {"error": "json mode"}), Response(200, completed('```json\n{"signals": []}\n```'))]
    )
    output, usage, problem = asyncio.run(
        model_extract(session, api_key="sk-test", model="m", system="S", user="U")
    )
    assert output == {"signals": []} and usage == {"input_tokens": 300, "output_tokens": 40}
    assert "text" in session.bodies[0] and "text" not in session.bodies[1] and "JSON mode" in problem
    assert session.bodies[0]["instructions"] == "S" and session.bodies[0]["input"] == "U"
    failed = asyncio.run(
        model_extract(
            Session([Response(200, completed("no json here"))]), api_key="k", model="m", system="S", user="U"
        )
    )
    assert failed[0] is None and "unusable" in failed[2]


def test_reviewer_payload_carries_news_evidence_or_says_the_layer_is_off():
    on = brain.build_payload([], [], trades_today=0, text_signals=[{"topic": "CPI", "impact": "bearish"}])
    off = brain.build_payload([], [], trades_today=0)
    assert on["text_scouts_signals"] == [{"topic": "CPI", "impact": "bearish"}]
    assert off["text_scouts_signals"] == "news layer is off" and "text_scouts_signals" in brain.SYSTEM


def test_news_failure_never_reaches_the_trading_loop(tmp_path):
    cfg = Config(use_llm=False)
    store = Store(tmp_path / "paper")
    trader = Trader(
        api=None,
        session=None,
        store=store,
        config=cfg,
        broker=PaperBroker(store, cfg, 0),
        fee=0,
        instruments={},
        openai_key=None,
    )

    def failing(config, request):
        raise TextScoutsError("feeds unreachable")

    trader.news = NewsDesk(store, TextScoutsConfig(), runner=failing)
    asyncio.run(trader.news_tick())
    events = [
        json.loads(line)["event"] for line in (store.directory / "journal.jsonl").read_text().splitlines()
    ]
    assert events == ["news_error"]
    off = Trader(
        api=None,
        session=None,
        store=store,
        config=Config(use_llm=False, use_text_scouts=False),
        broker=PaperBroker(store, cfg, 0),
        fee=0,
        instruments={},
        openai_key=None,
    )
    assert off.news is None
    asyncio.run(off.news_tick())


def real_config() -> TextScoutsConfig:
    config = TextScoutsConfig()
    try:
        config.check()
    except TextScoutsError:
        pytest.skip("Text Scouts checkout or environment is not present")
    return config


def test_real_text_scouts_validation_keeps_supported_signals_and_caps_confidence():
    config = real_config()
    output = {
        "signals": [
            {
                "topic": "Exchange hack",
                "sentiment": -0.8,
                "impact": "bearish",
                "confidence": 0.95,
                "summary": "funds drained",
                "item_ids": [1],
            },
            {
                "topic": "Invented",
                "sentiment": 0.5,
                "impact": "bullish",
                "confidence": 0.9,
                "summary": "x",
                "item_ids": [7],
            },
            {
                "topic": "Contradiction",
                "sentiment": 0.6,
                "impact": "bearish",
                "confidence": 0.9,
                "summary": "x",
                "item_ids": [2],
            },
        ]
    }
    result = text_scouts._run_worker(config, {"op": "signals", "items": [A, B], "model_output": output})
    assert result["offered"] == 3 and len(result["signals"]) == 1
    kept = result["signals"][0]
    assert kept["topic"] == "Exchange hack" and kept["impact"] == "bearish" and kept["origin"] == "llm"
    assert kept["confidence"] == pytest.approx(0.65), "a single source is capped by Text Scouts' own rule"
    assert kept["sources"] == ["https://a.example/1"]


def test_real_text_scouts_validation_rejects_output_outside_its_schema():
    config = real_config()
    with pytest.raises(TextScoutsError):
        text_scouts._run_worker(
            config, {"op": "signals", "items": [A], "model_output": {"signals": "all bullish"}}
        )
