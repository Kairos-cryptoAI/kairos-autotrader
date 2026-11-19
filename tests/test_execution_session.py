"""Handing approved entries to the Execution Engine and following the session."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from kairos_autotrader.engine import Config, ConfigError
from tests.test_autotrader import FakeVenue, journal_of, live_trader


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
    with pytest.raises(ConfigError, match="execution_portfolio"):
        Config(executor="other").validate()


@pytest.mark.parametrize("config", [{}, {"executor": "runner"}, {"execution_profile": "missing.json"}])
def test_live_cli_without_verified_profile_cannot_read_keys_or_touch_the_venue(tmp_path, config, monkeypatch):
    from types import SimpleNamespace

    from kairos_autotrader import __main__ as cli

    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        cli, "load_secrets", lambda *a, **kw: pytest.fail("read credentials without a profile")
    )
    args = SimpleNamespace(config=path, runtime_dir=tmp_path / "runtime")
    with pytest.raises(ConfigError, match="profile"):
        asyncio.run(cli.build_trader(args, None, "live"))
    assert not args.runtime_dir.exists()


def test_close_report_survives_a_crash_on_either_side_of_the_journal_append(tmp_path):
    from kairos_autotrader.engine import Store

    for already_appended in (False, True):
        store = Store(tmp_path / str(already_appended))
        row = {"id": "order", "close_report_id": "live:order", "trade_id": "first"}
        store.state["pending_close_report"] = row
        store.save()
        if already_appended:
            store.journal("trade_closed", **row)
        restored = Store(store.directory)
        restored.publish_close(recovering=True)
        assert "pending_close_report" not in Store(store.directory).state
        rows = (store.directory / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        assert [json.loads(s)["trade_id"] for s in rows] == ["first"]


@pytest.mark.parametrize("receipt", [None, "partial", '{"outcome":"UNRESOLVED"}'])
def test_unresolved_legacy_handoff_blocks_all_new_entries_and_writes(tmp_path, receipt):
    from kairos_autotrader.engine import NeedsOperator

    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    folder = trader.store.directory / "execution"
    folder.mkdir()
    path = folder / "old.result.json"
    if receipt is not None:
        path.write_text(receipt, encoding="utf-8")
    trader.store.state["execution_session"] = {"result_file": str(path), "candidate": "old"}
    with pytest.raises(NeedsOperator, match="legacy"):
        asyncio.run(trader.start(new_session=True))
    assert trader.store.state["execution_session"]["candidate"] == "old"
    assert not venue.market_bodies and not venue.close_bodies


def test_legacy_terminal_receipt_is_archived_once_without_restarting_its_session(tmp_path):
    venue = FakeVenue()
    trader = live_trader(tmp_path, venue)
    folder = trader.store.directory / "execution"
    folder.mkdir()
    path = folder / "old.result.json"
    path.write_text('{"outcome":"CLOSED_AND_ACCOUNTED","trade_id":"old-trade"}', encoding="utf-8")
    trader.store.state["execution_session"] = {"result_file": str(path), "candidate": "old"}
    asyncio.run(trader.start())
    asyncio.run(trader.start())
    (event,) = [r for r in journal_of(trader) if r["event"] == "execution_session_finished"]
    assert event["candidate"] == "old" and event["trade_id"] == "old-trade"
    assert not trader.store.state.get("execution_session") and not venue.market_bodies


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
        "trade_id": "engine-trade",
        "exit_origin": "system",
        "exit_reason": "TARGET",
        "settled_at": "2027-01-16T12:00:00+00:00",
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
        {
            "event": "execution_session_finished",
            "outcome": "CLOSED_AND_ACCOUNTED",
            "trade_id": "engine-trade",
            "candidate": "engine-candidate",
        },
        {"event": "macro_allocation", "regime": "BEAR", "produced_at": "2027-01-01T00:00:00+00:00"},
        {"event": "macro_allocation", "regime": "BULL", "produced_at": "2027-01-15T00:00:00+00:00"},
        {
            "event": "trade_closed",
            "strategy_id": "hourly_trend_candidates_v1",
            "signal_symbol": "XRPUSDT",
            "regime": "CHOP",
            "decided_at_ms": 5,
            "r_multiple": "1.5",
            "mode": "live",
            "trade_id": "portfolio-trade",
            "candidate_id": "portfolio-candidate",
            "exit_origin": "system",
            "pnl_basis": "execution_fills",
            "pnl_exact": True,
        },
        {"event": "trade_closed", "pnl_usd": "0.1"},
    ):
        trader.store.journal(record.pop("event"), **record)
    outcomes = trader.learning_outcomes()
    assert {(o["symbol"], o["regime"], o["r_multiple"]) for o in outcomes} == {
        ("XRPUSDT", "CHOP", "1.5"),
        ("SOLUSDT", "BULL", "-1.0200"),
    }


def test_only_attributed_automatic_live_closes_can_change_live_risk(tmp_path):
    from kairos_autotrader.engine import collect_learning_outcomes
    from kairos_autotrader.outcomes import recorded_closes

    old = {
        "event": "trade_closed",
        "strategy_id": "s",
        "signal_symbol": "SOLUSDT",
        "decided_at_ms": 5,
        "r_multiple": "3",
        "pnl_exact": True,
    }
    exact = {
        **old,
        "mode": "live",
        "trade_id": "verified",
        "candidate_id": "c",
        "exit_origin": "system",
        "pnl_basis": "execution_fills",
    }
    events = [
        old,
        {"event": "session_start", "mode": "paper"},
        old,
        {**exact, "mode": "paper"},
        {"event": "session_start", "mode": "live"},
        {**old, "reason": "VENUE_EXIT"},
        {**old, "pnl_exact": False},
        {**exact, "exit_origin": "operator", "trade_id": "manual"},
        {**exact, "exit_origin": "unknown", "trade_id": "unknown"},
        {**exact, "candidate_id": None, "trade_id": "orphan"},
        exact,
        exact,
    ]
    (tmp_path / "journal.jsonl").write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")
    outcomes = collect_learning_outcomes(tmp_path, Config())
    assert len(outcomes) == 1 and outcomes[0]["source"] == "LIVE"
    closes = recorded_closes(events)
    assert [c.pnl_basis for c in closes[:5]] == [
        "unknown",
        "paper_fills",
        "execution_fills",
        "account_change",
        "price_estimate",
    ]
    assert closes[3].exit_origin == "unknown"


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
