"""Handing approved entries to the Execution Engine and following the session."""

from __future__ import annotations

import asyncio
import json
import sys
import time

import pytest

from kairos_autotrader import brain
from kairos_autotrader.engine import Config, ConfigError
from tests.test_autotrader import FakeVenue, candidate, journal_of, live_trader


def _fake_execution(tmp_path, monkeypatch, *, outcome: str, delay_s: float = 0.0):
    """A stand-in ``kairos_execution.production_session_cli`` on the child's import path."""
    package = tmp_path / "fake-execution" / "kairos_execution"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "production_setup.py").write_text("print('{\"ok\": true}')\n", encoding="utf-8")
    (package / "production_session_cli.py").write_text(
        "import json, sys, time\n"
        f"time.sleep({delay_s})\n"
        "args = sys.argv\n"
        "candidate = json.loads(open(args[args.index('--candidate') + 1], encoding='utf-8').read())\n"
        "assert set(candidate) == {'review', 'allocation', 'system_mode', 'per_trade_risk_fraction'}\n"
        f"print(json.dumps({{'outcome': {outcome!r}, 'trade_id': 'a' * 64, 'detail': 'synthetic'}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake-execution"))
    config = tmp_path / "session.json"
    config.write_text("{}", encoding="utf-8")
    return dict(
        executor="execution_engine", execution_python=sys.executable, execution_session_config=str(config)
    )


def _wait_for(trader, *, seconds=20.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if not trader._follow_execution():
            return
        time.sleep(0.2)
    raise AssertionError("the session did not finish")


def _real_review(monkeypatch):
    """The Risk stand-in approves; the review handed over is a real ``CandidateReviewV1``."""
    from kairos_autotrader import engine
    from tests.support import approve_entries
    from tests.test_risk import review

    approve_entries(monkeypatch, quantity=0.2)
    reviewed = review()
    monkeypatch.setattr(engine.Trader, "_review_of", lambda self, candidate: reviewed)
    return reviewed


def test_execution_engine_executor_needs_both_paths():
    with pytest.raises(ConfigError, match="execution_python"):
        Config(executor="execution_engine").validate()
    with pytest.raises(ConfigError, match="runner or execution_engine"):
        Config(executor="other").validate()


def test_an_approved_entry_is_handed_over_and_nothing_else_opens_meanwhile(tmp_path, monkeypatch):
    reviewed = _real_review(monkeypatch)
    venue = FakeVenue()
    trader = live_trader(
        tmp_path, venue, **_fake_execution(tmp_path, monkeypatch, outcome="CLOSED_AND_ACCOUNTED", delay_s=1.0)
    )
    asyncio.run(trader.start())
    entry = candidate()
    asyncio.run(trader.try_enter(entry, brain.control_decision([entry])))
    assert not venue.market_bodies, "the runner places no order itself"
    session = trader.store.state["execution_session"]
    assert session["candidate"] == entry.id
    handed = json.loads(open(session["candidate_file"], encoding="utf-8").read())
    assert handed["review"] == reviewed.model_dump(mode="json") and handed["system_mode"] == "NORMAL"
    assert trader._follow_execution() is True, "while the session runs no new entry is considered"
    _wait_for(trader)
    finished = [e for e in journal_of(trader) if e["event"] == "execution_session_finished"]
    assert finished and finished[-1]["outcome"] == "CLOSED_AND_ACCOUNTED"
    assert "execution_session" not in trader.store.state and not trader.store.state.get("halted")


def test_an_unresolved_session_halts_new_entries(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue, **_fake_execution(tmp_path, monkeypatch, outcome="UNRESOLVED"))
    asyncio.run(trader.start())
    entry = candidate()
    asyncio.run(trader.try_enter(entry, brain.control_decision([entry])))
    _wait_for(trader)
    assert "Execution Engine" in str(trader.store.state.get("halted"))


def test_paper_mode_keeps_the_runner_as_executor(tmp_path, monkeypatch):
    from tests.test_autotrader import paper_trader

    cfg = Config(**{"use_llm": False, **_fake_execution(tmp_path, monkeypatch, outcome="REFUSED")})
    trader = paper_trader(tmp_path, FakeVenue(), cfg)
    assert trader.executions is None


def test_report_reads_the_execution_engines_closed_trades(tmp_path, monkeypatch, capsys):
    from kairos_autotrader.execution_session import closed_trades

    package = tmp_path / "fake-report" / "kairos_execution"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "production_setup.py").write_text("print('{\"ok\": true}')\n", encoding="utf-8")
    row = {
        "settled_at": "2026-11-14T10:00:00+00:00",
        "venue_symbol": "SOLUSD",
        "side": "BUY",
        "exit_reason": "TARGET",
        "net_pnl_usd": "0.31",
        "r_multiple": "1.2",
        "fees_usd": "0.02",
        "funding_cost_usd": "0",
    }
    (package / "production_report.py").write_text(
        "import json, sys\n"
        "assert sys.argv[sys.argv.index('--manifest') + 1].endswith('manifest.json')\n"
        f"print(json.dumps({row!r}))\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake-report"))
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"manifest": str(tmp_path / "manifest.json")}), encoding="utf-8")
    assert closed_trades(sys.executable, session) == [row]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "executor": "execution_engine",
                "execution_python": sys.executable,
                "execution_session_config": str(session),
            }
        ),
        encoding="utf-8",
    )
    from kairos_autotrader.__main__ import _report_execution

    _report_execution(config)
    out = capsys.readouterr().out
    assert "сделок закрыто 1" in out and "+1.200 R" in out and "SOLUSD BUY TARGET" in out


def test_learning_outcomes_join_runner_and_execution_trades_with_their_regime(tmp_path, monkeypatch):
    package = tmp_path / "fake-ledger" / "kairos_execution"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "production_setup.py").write_text("print('{\"ok\": true}')\n", encoding="utf-8")
    row = {
        "version": "kairos.execution-closed-trade.v1",
        "strategy_id": "hourly_trend_candidates_v1",
        "symbol": "SOLUSDT",
        "decided_at_ms": 1_800_000_000_000,
        "r_multiple": "-1.0200",
    }
    (package / "production_report.py").write_text(
        f"import json\nprint(json.dumps({row!r}))\n", encoding="utf-8"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake-ledger"))
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"manifest": str(tmp_path / "manifest.json")}), encoding="utf-8")
    trader = live_trader(
        tmp_path,
        FakeVenue(),
        executor="execution_engine",
        execution_python=sys.executable,
        execution_session_config=str(session),
    )
    for record in (
        {"event": "macro_allocation", "regime": "BEAR", "produced_at": "2027-01-01T00:00:00+00:00"},
        {"event": "macro_allocation", "regime": "BULL", "produced_at": "2027-01-15T00:00:00+00:00"},
        {
            "event": "trade_closed",
            "strategy_id": "hourly_trend_candidates_v1",
            "signal_symbol": "XRPUSDT",
            "regime": "CHOP",
            "decided_at_ms": 5,
            "r_multiple": "1.5",
        },
        {"event": "trade_closed", "pnl_usd": "0.1"},
    ):
        trader.store.journal(record.pop("event"), **record)
    outcomes = trader.learning_outcomes()
    assert {(o["symbol"], o["regime"], o["r_multiple"]) for o in outcomes} == {
        ("XRPUSDT", "CHOP", "1.5"),
        ("SOLUSDT", "BULL", "-1.0200"),
    }


def test_replayed_signals_join_the_ledger_as_half_weight_evidence(tmp_path):
    trader = live_trader(tmp_path, FakeVenue())
    replay = {
        "strategy_id": "hourly_trend_candidates_v1",
        "symbol": "SOLUSDT",
        "regime": "UNKNOWN",
        "r_multiple": "-0.31",
        "decided_at_ms": 1_000,
        "source": "REPLAY",
    }
    (trader.store.directory / "learning-replay.jsonl").write_text(
        json.dumps(replay) + "\n" + json.dumps({**replay, "source": "LIVE"}) + "\nnot json\n",
        encoding="utf-8",
    )
    assert trader.learning_outcomes() == [replay]


def test_adaptive_strategy_is_observed_measured_and_never_traded(tmp_path, monkeypatch):
    import asyncio as aio

    from kairos_autotrader import engine as engine_module

    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    start = 1_800_000_000_000
    intent = {
        "intent_id": "a" * 64,
        "side": "LONG",
        "reference_price": 100.0,
        "decision_ts_ms": start,
        "exit_plan": {"stop_price": 99.0, "target_price": 102.0, "max_holding_ms": 10 * 60_000},
    }

    async def fake_scan(config, series, slot, **kwargs):
        return {
            "strategy_id": "adaptive_pullback_range_v1",
            "results": {"SOLUSDT": {"status": "INTENT", "regime": "BULL", "intent": intent}},
        }

    monkeypatch.setattr(engine_module, "adaptive_shadow_scan", fake_scan)
    rows = [
        [start + i * 60_000, 100.0, 100.5 if i < 3 else 102.5, 99.5, 100.2, 1, 1, 1, 1, "x"]
        for i in range(15)
    ]
    trader.engine = object()  # the scan itself is stubbed; only its presence matters here
    aio.run(trader._observe_adaptive({"SOLUSD": rows[:2]}, start))
    assert trader.store.state["shadow_pending"][0]["id"] == "a" * 64
    assert not venue.market_bodies, "an observed strategy never places an order"
    trader._measure_shadow({"SOLUSD": rows})
    assert trader.store.state["shadow_pending"] == []
    outcomes = trader.learning_outcomes()
    assert len(outcomes) == 1 and outcomes[0]["source"] == "REPLAY"
    assert outcomes[0]["strategy_id"] == "adaptive_pullback_range_v1" and float(outcomes[0]["r_multiple"]) > 1


def test_stale_readiness_receipts_stop_the_entry_before_any_session(tmp_path, monkeypatch):
    _real_review(monkeypatch)
    venue = FakeVenue()
    trader = live_trader(
        tmp_path, venue, **_fake_execution(tmp_path, monkeypatch, outcome="CLOSED_AND_ACCOUNTED")
    )
    refusing = json.dumps({"ok": False, "error": "facts are missing"})
    (tmp_path / "fake-execution" / "kairos_execution" / "production_setup.py").write_text(
        f"import sys\nprint({refusing!r})\nsys.exit(1)\n", encoding="utf-8"
    )
    asyncio.run(trader.start())
    entry = candidate()
    asyncio.run(trader.try_enter(entry, brain.control_decision([entry])))
    assert "execution_session" not in trader.store.state and not venue.market_bodies
    skipped = [e for e in journal_of(trader) if e["event"] == "entry_skipped"][-1]
    assert "receipts could not be refreshed" in skipped["reason"] and "facts are missing" in skipped["reason"]


def test_a_briefly_locked_heartbeat_is_written_on_retry(tmp_path, monkeypatch):
    from pathlib import Path

    from tests.test_autotrader import paper_trader

    trader = paper_trader(tmp_path, FakeVenue(), Config(use_llm=False))
    original = Path.replace
    failures = []

    def flaky(self, target):
        if len(failures) < 2:
            failures.append(1)
            raise PermissionError(5, "Access is denied")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    trader._heartbeat("2026-11-16T00:00:00+00:00")
    assert json.loads((trader.store.directory / "heartbeat.json").read_text())["started_at"].startswith(
        "2026-11-16"
    )
    journal = trader.store.directory / "journal.jsonl"
    assert not journal.exists() or not [e for e in journal_of(trader) if e["event"] == "heartbeat_error"]
