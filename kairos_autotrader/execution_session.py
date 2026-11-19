"""Read accounting and final receipts from earlier single-trade Execution Engine sessions."""

from __future__ import annotations

import json
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from kairos_execution.portfolio_broker import NeedsOperator


def closed_trades(python: Path, session_config: Path) -> list[dict[str, Any]]:
    """The Execution Engine's closed trades from its own database (``production_report``)."""
    manifest = json.loads(Path(session_config).read_text(encoding="utf-8"))["manifest"]
    # The operator selects the interpreter. The report module is fixed and the manifest is one argument.
    completed = subprocess.run(  # nosec B603
        [str(python), "-m", "kairos_execution.production_report", "--manifest", str(manifest)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip()[:300] or "report failed")
    return [json.loads(line) for line in completed.stdout.decode().splitlines() if line.strip()]


def legacy_session_result(store: Any) -> dict[str, Any] | None:
    """Only a terminal receipt releases the old handoff. Never restart its child."""
    if "execution_session" not in store.state:
        return None
    session = store.state["execution_session"]
    if not isinstance(session, dict) or not session.get("candidate") or not session.get("result_file"):
        raise NeedsOperator("legacy execution session has no usable handoff record")
    result = Path(session["result_file"]).resolve()
    if not result.is_relative_to((store.directory / "execution").resolve()):
        raise NeedsOperator("legacy execution receipt is outside its runtime")
    try:
        answer = json.loads(result.read_text(encoding="utf-8").strip().splitlines()[-1])
    except (OSError, IndexError, ValueError) as exc:
        raise NeedsOperator("legacy execution session has no terminal receipt") from exc
    if not isinstance(answer, dict) or (
        answer.get("outcome") != "REFUSED"
        and not (answer.get("outcome") == "CLOSED_AND_ACCOUNTED" and answer.get("trade_id"))
    ):
        raise NeedsOperator("legacy execution outcome is unresolved")
    return {**answer, "candidate": session["candidate"], "exit_requested": session.get("exit_requested")}
