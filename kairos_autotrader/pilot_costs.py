"""Read model usage attributable to a UTC pilot window from the local runner records."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("expected an ISO timestamp")
    moment = datetime.fromisoformat(value)
    if moment.utcoffset() is None:
        raise ValueError("expected an offset-aware timestamp")
    return moment.astimezone(UTC)


def _unsigned(value: Any) -> bool:
    return type(value) is int and value >= 0


def _aggregator(path: Path, start: datetime, end: datetime, issues: set[str]) -> tuple[Decimal, int, bool]:
    try:
        with path.open(encoding="utf-8-sig") as stream:
            ledger = json.load(stream)
    except FileNotFoundError:
        issues.add("aggregator_ledger_missing")
        return Decimal(0), 0, False
    except (OSError, ValueError):
        issues.add("aggregator_ledger_unreadable")
        return Decimal(0), 0, False
    if not isinstance(ledger, dict):
        issues.add("aggregator_ledger_invalid")
        return Decimal(0), 0, False

    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    unit = timedelta(microseconds=1)
    first_us, last_us = (start - epoch) // unit, (end - epoch) // unit
    confirmed: dict[str, int | None] = {}
    ambiguous: set[str] = set()
    for providers in ledger.values():
        if not isinstance(providers, dict):
            issues.add("aggregator_ledger_invalid")
            continue
        for reservations in providers.values():
            if not isinstance(reservations, dict):
                issues.add("aggregator_ledger_invalid")
                continue
            for key, entry in reservations.items():
                if not isinstance(entry, dict) or not _unsigned(entry.get("at_ms")):
                    issues.add("aggregator_ledger_invalid")
                    continue
                if not first_us <= entry["at_ms"] * 1000 <= last_us:
                    continue
                reserved, actual = entry.get("reserved"), entry.get("actual")
                if (
                    not _unsigned(reserved)
                    or "actual" not in entry
                    or (actual is not None and (not _unsigned(actual) or actual > reserved))
                ):
                    issues.add("aggregator_ledger_invalid")
                    ambiguous.add(key)
                    continue
                if key in confirmed:
                    issues.add("aggregator_duplicate_reservation")
                    if confirmed[key] != actual:
                        ambiguous.add(key)
                confirmed[key] = actual
    for key in ambiguous:
        confirmed.pop(key, None)
    micros = sum(value for value in confirmed.values() if value is not None)
    pending = sum(value is None for value in confirmed.values())
    return Decimal(micros).scaleb(-6), pending, not issues and not pending


def _usage_record(row: dict[str, Any]) -> tuple[str, Any] | None:
    event = row.get("event")
    if event == "news_error":
        return "text_scouts", None
    if event == "macro_error":
        return "macro", None
    if event == "news_poll":
        # A poll with no paid call can still return the local keyword evidence.
        if not row.get("usage") and row.get("origin") in ("local", "none") and not row.get("problem"):
            return None
        return "text_scouts", row.get("usage")
    if event == "macro_allocation":
        # MacroDesk emits the usage. The runner also emits an allocation summary for the same call.
        if "usage" in row or "trigger" in row:
            return "macro", row.get("usage")
    if event in ("decision", "shadow_decision"):
        decision = row.get("decision")
        if not isinstance(decision, dict):
            return "shadow_probability" if event == "shadow_decision" else "probability", None
        if decision.get("source") == "LLM":
            name = "shadow_probability" if event == "shadow_decision" else "probability"
            return name, decision.get("usage")
    return None


def _tokens(
    path: Path, start: datetime, end: datetime, issues: set[str]
) -> tuple[dict[str, dict[str, int]], bool]:
    sources = ("macro", "text_scouts", "probability", "shadow_probability")
    counts = {source: {"input_tokens": 0, "output_tokens": 0} for source in sources}
    complete = True
    try:
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("invalid journal record")
                    when = _utc(row.get("ts"))
                except (ValueError, OverflowError):
                    complete = False
                    issues.add("journal_invalid_record")
                    continue
                if not start <= when <= end:
                    continue
                record = _usage_record(row)
                if record is None:
                    continue
                source, usage = record
                if not isinstance(usage, dict) or not all(_unsigned(usage.get(k)) for k in counts[source]):
                    complete = False
                    issues.add(f"{source}_usage_missing")
                    continue
                for key in counts[source]:
                    counts[source][key] += usage[key]
    except FileNotFoundError:
        issues.add("journal_missing")
        complete = False
    except OSError:
        issues.add("journal_unreadable")
        complete = False
    return counts, complete


def pilot_model_usage(directory: Path, *, started_at: str, until: str) -> dict[str, Any]:
    """Count confirmed reservation costs and provider-reported tokens in an inclusive UTC interval.

    Aggregator reservations are dated at dispatch. Direct model calls are dated by the runner's
    completion event. Those events omit cache details and can lack usage after a failed response.
    """
    start, end = _utc(started_at), _utc(until)
    if start > end:
        raise ValueError("pilot cost window ends before its start")
    issues: set[str] = set()
    cost, pending, ledger_complete = _aggregator(directory / "llm-budget.json", start, end, issues)
    tokens, journal_complete = _tokens(directory / "journal.jsonl", start, end, issues)
    return {
        "known_aggregator_usd": cost,
        "pending_reservations": pending,
        "aggregator_complete": ledger_complete,
        "journal_complete": journal_complete,
        "complete_billing": False,
        "tokens": tokens,
        "issues": tuple(sorted(issues)),
    }
