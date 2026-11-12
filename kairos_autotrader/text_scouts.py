"""Bridge to the real Kairos Text Scouts layer (repository ``kairos-text-scouts``).

Every poll runs the repository's own pipeline through ``text_scouts_worker.py``: official
sources (GDELT and RSS), normalisation, the 30-minute freshness bound, deduplication and
the local relevance filter. The few items that survive are sent to the model with Text
Scouts' own prompt, and the answer goes back through Text Scouts' own validation, which
turns it into typed sentiment signals. Without a model key, or when the model fails,
Text Scouts' local keyword fallback is used instead and marked as such.

The signals are evidence for the reviewer. They never create a trade, a side or an order.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .brain import FEEDS, OPENAI_URL, extract_output_text
from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python

WORKER = Path(__file__).with_name("text_scouts_worker.py")
KEEP_SIGNALS_HOURS = 12


class TextScoutsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TextScoutsConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-text-scouts"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-text-scouts"))
    core_source: Path = field(default_factory=lambda: sibling_checkout("kairos-core"))
    feeds: tuple[str, ...] = tuple(url for _, url in FEEDS)
    timeout_s: float = 150.0

    def check(self) -> None:
        if not self.python.exists():
            raise TextScoutsError(f"Text Scouts Python not found: {self.python}")
        if not (self.source / "kairos_text" / "service.py").exists():
            raise TextScoutsError(f"Text Scouts source not found: {self.source}")
        if not (self.core_source / "kairos_core" / "__init__.py").exists():
            raise TextScoutsError(f"kairos-core source not found: {self.core_source}")


def revision(source: Path) -> str:
    return git_revision(source)


def _run_worker(config: TextScoutsConfig, request: dict[str, Any]) -> dict[str, Any]:
    try:
        return run_worker(
            config.python,
            WORKER,
            [str(config.source), str(config.core_source)],
            request,
            timeout_s=config.timeout_s,
        )
    except SiblingError as exc:
        raise TextScoutsError(f"Text Scouts: {exc}") from exc


def _loose_json(text: str) -> Any:
    """The model's JSON object, tolerating a code fence or stray text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the model answer")
    return json.loads(text[start : end + 1])


NO_JSON_MODE: set[str] = set()  # models that refused JSON mode in this process


async def model_extract(
    session: aiohttp.ClientSession,
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    timeout_s: float = 60.0,
    effort: str = "low",
) -> tuple[Any, dict[str, int], str]:
    """One JSON-answer model call. Returns (parsed answer or None, token usage, problem)."""
    body: dict[str, Any] = {
        "model": model,
        "instructions": system,
        "input": user,
        "reasoning": {"effort": effort},
        "max_output_tokens": 3000 if effort == "low" else 8000,
        "store": False,
    }
    if model not in NO_JSON_MODE:
        body["text"] = {"format": {"type": "json_object"}}
    problem = ""
    for attempt in range(2):
        try:
            async with session.post(
                OPENAI_URL,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=timeout_s),
            ) as response:
                status, text = response.status, await response.text()
        except (TimeoutError, aiohttp.ClientError) as exc:
            return None, {}, f"model call failed: {type(exc).__name__}"
        if status == 400 and attempt == 0 and "text" in body:
            # Some models refuse JSON mode for this prompt shape; the prompt itself asks for JSON.
            body.pop("text")
            NO_JSON_MODE.add(model)
            problem = "retried without JSON mode"
            continue
        if status != 200:
            return None, {}, f"model call failed: HTTP {status} {text[:160]}"
        try:
            raw = json.loads(text)
            usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
            tokens = {
                k: int(usage[k]) for k in ("input_tokens", "output_tokens") if isinstance(usage.get(k), int)
            }
            return _loose_json(extract_output_text(raw)), tokens, problem
        except ValueError as exc:
            return None, {}, f"model answer unusable: {str(exc)[:120]}"
    return None, {}, problem or "model call failed"


def _domain(ref: str) -> str:
    host = urlsplit(ref).netloc.lower()
    return host[4:] if host.startswith("www.") else (host or ref[:60])


class NewsDesk:
    """Keeps Text Scouts running inside the trader: polls, remembers what was seen, serves context."""

    def __init__(
        self, store: Any, config: TextScoutsConfig | None = None, *, runner=_run_worker, extract=model_extract
    ) -> None:
        self.store, self.config = store, config or TextScoutsConfig()
        self._runner, self._extract = runner, extract
        self.revision = revision(self.config.source)

    def _state(self) -> dict[str, Any]:
        news = self.store.state.setdefault("news", {})
        news.setdefault("seen", {})
        news.setdefault("signals", [])
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if news.get("day", {}).get("date") != today:
            news["day"] = {"date": today, "llm_calls": 0, "polls": 0}
        return news

    async def poll(
        self, session: aiohttp.ClientSession | None, *, openai_key: str | None, model: str, max_llm_calls: int
    ) -> dict[str, Any]:
        """One Text Scouts cycle. Returns a short summary; raises ``TextScoutsError`` if the worker fails."""
        news = self._state()
        now_ms = int(time.time() * 1000)
        collected = await asyncio.to_thread(
            self._runner,
            self.config,
            {"op": "collect", "seen": sorted(news["seen"]), "feeds": list(self.config.feeds)},
        )
        news["day"]["polls"] += 1
        window_ms = int(float(collected.get("dedup_window_s", 21_600.0)) * 1000)
        for key in [k for k, moment in news["seen"].items() if moment < now_ms - window_ms]:
            del news["seen"][key]
        items = collected["items"]
        signals: list[dict[str, Any]] = []
        origin, problem, usage = "none", "", {}
        if items:
            answered = False
            if openai_key and session is not None and news["day"]["llm_calls"] < max_llm_calls:
                news["day"]["llm_calls"] += 1
                self.store.save()
                output, usage, problem = await self._extract(
                    session,
                    api_key=openai_key,
                    model=model,
                    system=collected["system"],
                    user=collected["user"],
                )
                if output is not None:
                    try:
                        checked = await asyncio.to_thread(
                            self._runner,
                            self.config,
                            {"op": "signals", "items": items, "model_output": output},
                        )
                        signals, origin, answered = checked["signals"], "llm", True
                    except TextScoutsError as exc:
                        problem = str(exc)[:200]
            if not answered:
                # Text Scouts' degraded mode: coarse keyword evidence, clearly marked.
                signals, origin = collected["local_signals"], "local"
            # A valid model answer settles these items, also when it abstains. After a model
            # failure only items that produced local evidence are settled; the rest are
            # offered again on the next poll, as the Text Scouts service does.
            settled = {ref for s in signals for ref in s["sources"]}
            for item in items:
                refs = {item.get("url")} | set(item.get("provenance") or ())
                if answered or settled & refs:
                    news["seen"][item["key"]] = now_ms
        known = {s["id"] for s in news["signals"]}
        fresh = [s for s in signals if s["id"] not in known]
        for signal in fresh:
            news["signals"].append({**signal, "received_ms": now_ms})
        news["signals"] = [
            s for s in news["signals"] if s["received_ms"] >= now_ms - KEEP_SIGNALS_HOURS * 3_600_000
        ]
        self.store.save()
        summary = {
            "counts": collected["counts"],
            "sources": collected["sources"],
            "origin": origin,
            "kept": [i["title"][:160] for i in items],
            "signals": fresh,
            "usage": usage,
            "problem": problem,
        }
        if items or problem:
            self.store.journal("news_poll", revision=self.revision, **summary)
        return summary

    def raw_signals(self) -> list[dict[str, Any]]:
        """Stored signals with their identities and times, as the Router and Aggregator need them."""
        return [dict(signal) for signal in self._state()["signals"]]

    def context(self, now_ms: int, lookback_hours: float = 6.0) -> list[dict[str, Any]]:
        """Recent signals for the reviewer, newest first."""
        rows = []
        for signal in self._state()["signals"]:
            try:
                event_ms = int(datetime.fromisoformat(signal["event_time"]).timestamp() * 1000)
            except (KeyError, ValueError):
                event_ms = signal["received_ms"]
            age = (now_ms - event_ms) // 60_000
            if 0 <= age <= lookback_hours * 60:
                rows.append(
                    {
                        "topic": signal["topic"],
                        "impact": signal["impact"],
                        "sentiment": round(signal["sentiment"], 2),
                        "confidence": round(signal["confidence"], 2),
                        "age_minutes": int(age),
                        "summary": str(signal.get("summary", ""))[:300],
                        "sources": sorted({_domain(ref) for ref in signal.get("sources", [])})[:5],
                        "origin": signal["origin"],
                    }
                )
        rows.sort(key=lambda row: row["age_minutes"])
        return rows[:25]
