"""A failed market-data observation cannot become a decision for this or the next candidate."""

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from pydantic import ValidationError
from test_risk import LIMITS, SCOPE, _fake_session, review

from kairos_autotrader import risk_worker
from kairos_autotrader.risk import RiskError, evaluate_entry


@pytest.mark.parametrize("symbol", ["BTCUSDT", "AAVEUSDT"])
async def test_error_envelope_reaches_entry_caller_and_next_request_stays_aligned(
    tmp_path, monkeypatch, symbol
):
    session = _fake_session(tmp_path, monkeypatch)
    script = tmp_path / "fake_risk_worker.py"
    script.write_text("""
import json, sys
def say(message):
    print("@@KAIROS-WORKER@@" + json.dumps(message), flush=True)
say({"ready": True})
for line in sys.stdin:
    request = json.loads(line)
    say({"need_account": True})
    account = json.loads(sys.stdin.readline())["account"]
    if request["review"]["intent"]["symbol"] != "SOLUSDT":
        say({"need_account": True})
        fresh = json.loads(sys.stdin.readline())["account"]
        assert fresh["seq"] > account["seq"]
        say({"error": "EVEDEX public order book timed out"})
    else:
        say({"refused": "minimum lot exceeds budget"})
""")
    reads = []

    async def account():
        reads.append(1)
        return {"seq": len(reads)}

    async def check(instrument):
        return await evaluate_entry(
            session.config,
            review=review(symbol=instrument),
            allocation=None,
            account=account,
            scope=SCOPE,
            limits=LIMITS,
            taker_fee_bps=4,
            assessed_notional_usd=25,
            session=session,
        )

    try:
        with pytest.raises(RiskError, match="EVEDEX public order book timed out"):
            await check(symbol)
        process = session._process
        outcome = await check("SOLUSDT")
        assert not outcome.approved and outcome.reasons == ("minimum lot exceeds budget",)
        assert session._process is process and len(reads) == 3
    finally:
        await session.stop()


@pytest.mark.parametrize("succeeds", [True, False])
async def test_transient_public_book_error_recollects_both_inputs_once(monkeypatch, succeeds):
    request = {
        "review": {"intent": {"symbol": "BTCUSDT"}},
        "scope": {"instruments": ["BTCUSD"]},
        "venue": {"taker_fee_bps": 4.5, "assessed_notional_usd": 25},
    }
    fail = risk_worker.TemporaryVenueBookError("EVEDEX BTCUSD order book timed out")
    measure = AsyncMock(side_effect=[fail, "new books" if succeeds else fail])
    reads = Mock(side_effect=[{"account": {"seq": 1}}, {"account": {"seq": 2}}])
    ask = Mock()
    decide = Mock(return_value={"refused": "normal risk refusal"})
    monkeypatch.setattr(risk_worker, "venue_symbol_for", lambda _: "BTCUSD")
    monkeypatch.setattr(risk_worker, "_measure", measure)
    monkeypatch.setattr(risk_worker, "_say", ask)
    monkeypatch.setattr(risk_worker, "_listen", reads)
    monkeypatch.setattr(risk_worker, "decide", decide)
    if succeeds:
        assert await risk_worker._serve_one(request) == {"refused": "normal risk refusal"}
        decide.assert_called_once_with({**request, "account": {"seq": 2}}, "new books")
    else:
        with pytest.raises(risk_worker.TemporaryVenueBookError, match="EVEDEX BTCUSD"):
            await risk_worker._serve_one(request)
        decide.assert_not_called()
    assert measure.await_count == reads.call_count == ask.call_count == 2


@pytest.mark.parametrize(
    "http_status,retry_after,temporary", [(503, None, True), (503, "30", False), (429, None, False)]
)
async def test_public_book_failures_identify_exchange_without_bypassing_quota(
    monkeypatch, http_status, retry_after, temporary
):
    class Context:
        def __init__(self, url=None):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, url, **kwargs):
            return Context(url)

        def raise_for_status(self):
            if self.url == risk_worker.BINANCE_DEPTH:
                raise aiohttp.ClientResponseError(
                    Mock(),
                    (),
                    status=http_status,
                    headers={"Retry-After": retry_after} if retry_after else {},
                )

        async def json(self, **kwargs):
            return {}

    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: Context())
    monkeypatch.setitem(
        sys.modules,
        "kairos_quant.venue_comparison",
        SimpleNamespace(parse_binance_book=Mock(), parse_evedex_book=Mock()),
    )
    with pytest.raises(
        risk_worker.VenueBookError, match=f"Binance AAVEUSDT order book returned HTTP {http_status}"
    ) as exc:
        await risk_worker._measure("AAVEUSDT", "AAVEUSD", taker_fee_bps=4, assessed=25)
    assert isinstance(exc.value, risk_worker.TemporaryVenueBookError) is temporary


def test_worker_contract_error_names_field_without_account_input():
    from kairos_core.contracts import StrategicAllocation

    with pytest.raises(ValidationError) as error:
        StrategicAllocation.model_validate({"private": "private-account-material"})
    result = risk_worker._worker_error(error.value)["error"]
    assert "StrategicAllocation" in result and "source" in result and "missing" in result
    assert "private-account-material" not in result
