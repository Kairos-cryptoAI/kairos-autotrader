"""Runs the real Kairos Text Scouts pipeline stages and prints the result as JSON.

Executed as a script by the Text Scouts repository's own Python environment. It calls
the repository's unchanged code: sources, normaliser, freshness bound, deduplicator,
local relevance filter, the sentiment prompt and schema, the deterministic transform
from model output to signals and the local keyword fallback. The one thing it does not
do is call a model: the caller does that with the prompt returned here and hands the
model's answer back for validation.

argv: <text-scouts source dir> <kairos-core source dir>
stdin (collect): {"op": "collect", "seen": ["<dedup key>", ...], "feeds": ["https://...", ...]}
stdin (signals): {"op": "signals", "items": [<items from collect>], "model_output": {...}}
stdout: one line starting with the marker below, followed by JSON. Library log lines
        may precede it and are ignored by the caller.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime

MARKER = "@@KAIROS-WORKER@@"


def _item_out(item, key: str, score: float) -> dict:
    return {
        "key": key,
        "score": score,
        "title": item.title,
        "body": item.body,
        "url": item.url,
        "source": item.source,
        "source_kind": item.source_kind,
        "engagement": item.engagement,
        "published_at": item.published_at.isoformat(),
        "provenance": list(item.provenance),
    }


def _item_in(raw: dict):
    from kairos_text.models import NewsItem

    return NewsItem(
        title=raw["title"],
        body=raw.get("body", ""),
        url=raw.get("url", ""),
        source=raw.get("source", ""),
        source_kind=raw.get("source_kind", ""),
        engagement=float(raw.get("engagement", 0.0)),
        published_at=datetime.fromisoformat(raw["published_at"]),
        timestamp_is_estimated=False,
        provenance=tuple(raw.get("provenance", ())),
    )


def _signal_out(signal, origin: str) -> dict:
    return {
        "id": signal.message_id,
        "topic": signal.topic,
        "sentiment": signal.sentiment,
        "impact": signal.impact.value,
        "confidence": signal.confidence,
        "sources": list(signal.sources),
        "summary": signal.summary,
        "event_time": signal.produced_at.isoformat(),
        "origin": origin,
    }


async def collect(request: dict) -> dict:
    from kairos_text.config import TextSettings
    from kairos_text.dedup import EventDeduplicator, dedup_key
    from kairos_text.filter import LocalRelevanceFilter
    from kairos_text.freshness import EventFreshnessFilter
    from kairos_text.local import local_sentiment
    from kairos_text.normalize import EventNormalizer
    from kairos_text.prompts import SENTIMENT_SYSTEM
    from kairos_text.schemas import SentimentBatch
    from kairos_text.sentiment import SentimentExtractor
    from kairos_text.sources.gdelt import GDELTSource
    from kairos_text.sources.rss import RSSSource

    defaults = {name: field.default for name, field in TextSettings.model_fields.items()}
    feeds = request.get("feeds") or defaults["rss_feeds"]
    sources = [
        GDELTSource(
            query=defaults["gdelt_query"],
            timespan=defaults["gdelt_timespan"],
            max_records=defaults["gdelt_max_records"],
        ),
        RSSSource(feeds),
    ]
    per_source: dict[str, int | str] = {}
    raw = []
    for source in sources:
        try:
            fetched = await source.fetch()
        except Exception as exc:  # noqa: BLE001 - one source must not take the others down
            per_source[source.name] = f"failed: {type(exc).__name__}"
            continue
        per_source[source.name] = len(fetched)
        raw.extend(fetched)

    normalizer = EventNormalizer()
    freshness = EventFreshnessFilter(
        defaults["max_event_age_s"],
        defaults["max_future_skew_s"],
        allow_estimated_timestamps=defaults["allow_estimated_timestamps"],
    )
    dedup = EventDeduplicator(defaults["dedup_window_s"])
    relevance = LocalRelevanceFilter(threshold=defaults["relevance_threshold"], top_k=defaults["top_k"])

    timely = freshness.select(normalizer.normalize(raw))
    candidates = dedup.collapse_batch(timely)
    seen = set(request.get("seen") or ())
    unseen = [item for item in candidates if dedup_key(item) not in seen]
    relevant = relevance.select(unseen)

    extractor = SentimentExtractor(gateway=None)
    return {
        "counts": {
            "fetched": len(raw),
            "timely": len(timely),
            "unique": len(candidates),
            "unseen": len(unseen),
            "kept": len(relevant),
        },
        "sources": per_source,
        "items": [_item_out(item, dedup_key(item), relevance.score(item)) for item in relevant],
        "system": SENTIMENT_SYSTEM,
        "user": extractor._format_batch(relevant) if relevant else "",
        "schema": SentimentBatch.model_json_schema(),
        "local_signals": [_signal_out(s, "local") for s in local_sentiment(relevant)],
        "dedup_window_s": defaults["dedup_window_s"],
    }


def signals(request: dict) -> dict:
    from kairos_text.schemas import SentimentBatch
    from kairos_text.sentiment import SentimentExtractor

    items = [_item_in(raw) for raw in request["items"]]
    batch = SentimentBatch.model_validate(request["model_output"])
    extractor = SentimentExtractor(gateway=None)
    return {
        "signals": [_signal_out(s, "llm") for s in extractor._signals_from_batch(items, batch)],
        "offered": len(batch.signals),
    }


def main() -> int:
    # Replace this script's own directory on the import path with the two source trees.
    sys.path[0] = sys.argv[1]
    sys.path.insert(1, sys.argv[2])
    request = json.load(sys.stdin)
    try:
        result = asyncio.run(collect(request)) if request.get("op") == "collect" else signals(request)
    except Exception as exc:  # noqa: BLE001 - reported to the caller as data
        result = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
