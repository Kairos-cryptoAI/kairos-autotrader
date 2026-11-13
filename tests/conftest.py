"""Shared test setup: a test never starts a sibling process unless it says so."""

from __future__ import annotations

import pytest

from kairos_autotrader import engine, system_engine


@pytest.fixture(autouse=True)
def no_signal_chain_by_default(monkeypatch):
    """A ``Trader`` built in a test finds no signal chain, wherever the tests run.

    On a workstation with every sibling checkout present the trader would otherwise
    download bars at start. Tests of the chain wire stand-ins and set ``trader.engine``
    themselves; the ``test_real_*`` tests build their own configs.
    """

    class Absent(system_engine.EngineConfig):
        def check_hourly_trend(self) -> None:
            raise system_engine.EngineError("tests do not start sibling processes by default")

    monkeypatch.setattr(engine, "EngineConfig", Absent)
