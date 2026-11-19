import asyncio
import sys
from pathlib import Path

import pytest

from kairos_autotrader import worker_session
from kairos_autotrader.sibling import SiblingError


async def test_worker_restarts_without_replaying_a_dispatched_or_cancelled_request(tmp_path):
    script = tmp_path / "worker.py"
    log = tmp_path / "requests"
    script.write_text(
        """import os, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from worker_session import serve_requests
def handle(request):
    op = request["op"]
    with Path(sys.argv[2]).open("a") as stream:
        print(op, file=stream)
    if op == "exit":
        os._exit(1)
    if op == "hang":
        time.sleep(30)
    return {"op": op, "pad": "x" * 200000}
serve_requests(handle)
""",
        encoding="utf-8",
    )
    client = worker_session.WorkerSession(
        Path(sys.executable), script, [str(Path(worker_session.__file__).parent), str(log)], timeout_s=10
    )
    try:
        await client.warm()
        with pytest.raises(SiblingError):
            await client.ask({"op": "exit"})
        assert len((await client.ask({"op": "restart"}))["pad"]) == 200000
        interrupted = asyncio.create_task(client.ask({"op": "hang"}))
        async with asyncio.timeout(10):
            while "hang" not in log.read_text():
                await asyncio.sleep(0.01)
        interrupted.cancel()
        with pytest.raises(asyncio.CancelledError):
            await interrupted
        assert (await client.ask({"op": "again"}))["op"] == "again"
        assert log.read_text().splitlines() == ["exit", "restart", "hang", "again"]
    finally:
        await client.stop()
