import json
from decimal import Decimal

import pytest

from kairos_autotrader.pilot_costs import pilot_model_usage

START = "2026-11-19T00:00:00Z"
END = "2026-11-19T03:00:01+03:00"
STAMP = 1795046400000


def read(path):
    return pilot_model_usage(path, started_at=START, until=END)


def test_costs_and_direct_tokens_in_exact_window(tmp_path):
    def charge(at, actual):
        return {"at_ms": at, "reserved": 100_000, "actual": actual}

    ledger = {
        "2026-11": {
            "openai": {
                "before": charge(STAMP - 1, 99),
                "start": charge(STAMP, 30_000),
                "pending": charge(STAMP + 999, None),
                "end": charge(STAMP + 1000, 1),
                "later": charge(STAMP + 1001, 99),
            }
        }
    }
    (tmp_path / "llm-budget.json").write_text(json.dumps(ledger))
    usage = {"input_tokens": 20, "output_tokens": 2}
    rows = [
        {"event": "macro_allocation", "usage": usage},
        {"event": "macro_allocation", "allocation": {"regime": "CHOP"}},
        {"event": "news_poll", "usage": usage, "origin": "llm"},
        {"event": "news_poll", "usage": {}, "origin": "none"},
        {"event": "decision", "decision": {"source": "LLM", "usage": usage}},
        {"event": "shadow_decision", "decision": {"source": "LLM", "usage": usage}},
        {"event": "shadow_decision", "decision": {"source": "AGGREGATOR", "usage": usage}},
        {"event": "aggregator_review", "reviews": [{"model": {"cost_usd": 0.03}}]},
        {"event": "macro_allocation", "usage": usage, "ts": "2026-11-19T00:00:01.000001Z"},
    ]
    (tmp_path / "journal.jsonl").write_text("\n".join(json.dumps({"ts": END, **row}) for row in rows))
    result = read(tmp_path)
    assert result["known_aggregator_usd"] == Decimal("0.030001")
    assert result["pending_reservations"] == 1 and not result["aggregator_complete"]
    assert result["journal_complete"] and not result["complete_billing"]
    assert result["tokens"] == {
        k: usage for k in ("macro", "text_scouts", "probability", "shadow_probability")
    }


def test_missing_corrupt_and_partial_evidence_keep_known_amounts(tmp_path):
    empty = read(tmp_path)
    assert set(empty["issues"]) == {"aggregator_ledger_missing", "journal_missing"}
    assert not empty["aggregator_complete"] and not empty["journal_complete"]
    ledger_path = tmp_path / "llm-budget.json"
    ledger_path.write_bytes(b"not json")
    assert "aggregator_ledger_unreadable" in read(tmp_path)["issues"]
    entry = {"at_ms": STAMP, "reserved": 2, "actual": 1}
    ledger_path.write_text(
        json.dumps(
            {
                "this": {"api": {"a": entry, "b": entry, "c": None}},
                "other": {"api": {"a": {**entry, "actual": 2}}},
            }
        )
    )
    rows = [
        json.dumps({"ts": START, "event": "decision", "decision": {"source": "LLM", "usage": {}}}),
        json.dumps({"ts": START, "event": "news_poll", "usage": {"input_tokens": True, "output_tokens": 2}}),
        "not json",
        json.dumps({"ts": START, "event": "macro_error"}),
        json.dumps(
            {"ts": START, "event": "macro_allocation", "usage": {"input_tokens": 4, "output_tokens": 3}}
        ),
    ]
    (tmp_path / "journal.jsonl").write_text("\n".join(rows))
    result = read(tmp_path)
    assert result["known_aggregator_usd"] == Decimal("0.000001")
    assert not result["aggregator_complete"] and not result["journal_complete"]
    assert set(result["issues"]) == {
        "aggregator_duplicate_reservation",
        "aggregator_ledger_invalid",
        "journal_invalid_record",
        "macro_usage_missing",
        "probability_usage_missing",
        "text_scouts_usage_missing",
    }
    assert result["tokens"]["macro"] == {"input_tokens": 4, "output_tokens": 3}


def test_empty_window_and_parameter_validation(tmp_path):
    (tmp_path / "llm-budget.json").write_text("{}")
    (tmp_path / "journal.jsonl").write_text("")
    result = read(tmp_path)
    assert result["aggregator_complete"] and result["journal_complete"] and not result["complete_billing"]
    assert result["known_aggregator_usd"] == 0
    for bad_start in ("yesterday", "2026-11-19T00:00:00", "2026-11-20T00:00:00Z"):
        with pytest.raises(ValueError):
            pilot_model_usage(tmp_path, started_at=bad_start, until=END)
