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
    (package / "production_session_cli.py").write_text(
        "import json, sys, time\n"
        f"time.sleep({delay_s})\n"
        "args = sys.argv\n"
        "candidate = json.loads(open(args[args.index('--candidate') + 1], encoding='utf-8').read())\n"
        "assert set(candidate) == {'review', 'allocation', 'system_mode'}\n"
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
