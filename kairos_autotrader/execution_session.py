"""Hand an approved entry to the Kairos Execution Engine and follow its trade session.

With ``executor = "execution_engine"`` the runner does not place the order itself. It
writes the candidate (the Aggregator's ``CandidateReviewV1``, the Macro allocation and
the system mode) to a file and starts ``kairos_execution.production_session_cli`` in the
Execution Engine's own environment. That session arms the trade automatically within
the owner's signed limits, measures the venue fresh, lets the system's Risk Manager
decide again, enters with a bounded LIMIT_GTC order, protects, exits and reconciles.

The session's one-line JSON result is written to a file next to the candidate, so a
restarted runner still finds it. While a session runs, the runner opens nothing else:
the production path holds one position at a time.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINAL_OK = {"CLOSED_AND_ACCOUNTED", "REFUSED"}


@dataclass(frozen=True)
class ExecutionSettings:
    python: Path
    session_config: Path
    max_session_s: float


def closed_trades(python: Path, session_config: Path) -> list[dict[str, Any]]:
    """The Execution Engine's closed trades from its own database (``production_report``)."""
    manifest = json.loads(Path(session_config).read_text(encoding="utf-8"))["manifest"]
    completed = subprocess.run(  # noqa: S603 - fixed module of the owner's configured environment
        [str(python), "-m", "kairos_execution.production_report", "--manifest", str(manifest)],
        capture_output=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip()[:300] or "report failed")
    return [json.loads(line) for line in completed.stdout.decode().splitlines() if line.strip()]


class ExecutionSessions:
    """The one running Execution Engine session, persisted in the runner's state."""

    def __init__(self, settings: ExecutionSettings, store: Any) -> None:
        self.settings = settings
        self.store = store
        self._process: subprocess.Popen[bytes] | None = None

    def active(self) -> dict[str, Any] | None:
        session = self.store.state.get("execution_session")
        return session if isinstance(session, dict) else None

    def start(
        self,
        *,
        candidate_id: str,
        review: dict[str, Any],
        allocation: dict[str, Any] | None,
        per_trade_risk_fraction: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.active() is not None:
            raise RuntimeError("an Execution Engine session is already running")
        directory = Path(self.store.directory) / "execution"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{int(time.time() * 1000)}-{candidate_id[:16]}"
        candidate = directory / f"{stem}.candidate.json"
        result = directory / f"{stem}.result.json"
        log = directory / f"{stem}.stderr.log"
        exit_request = directory / f"{stem}.exit-request"
        candidate.write_text(
            json.dumps(
                {
                    "review": review,
                    "allocation": allocation,
                    "system_mode": "NORMAL",
                    "per_trade_risk_fraction": per_trade_risk_fraction,
                }
            ),
            encoding="utf-8",
        )
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        with result.open("wb") as stdout, log.open("wb") as stderr:
            self._process = subprocess.Popen(  # noqa: S603 - fixed module of the owner's configured environment
                [
                    str(self.settings.python),
                    "-m",
                    "kairos_execution.production_session_cli",
                    "--config",
                    str(self.settings.session_config),
                    "--candidate",
                    str(candidate),
                    "--exit-request",
                    str(exit_request),
                ],
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                creationflags=flags,
            )
        session = {
            "candidate": candidate_id,
            "pid": self._process.pid,
            "started_ms": int(time.time() * 1000),
            "candidate_file": str(candidate),
            "result_file": str(result),
            "exit_request_file": str(exit_request),
            **(context or {}),
        }
        self.store.state["execution_session"] = session
        self.store.save()
        return session

    def request_exit(self, reason: str) -> bool:
        """Ask the running session to close its position early; True when newly asked."""
        session = self.active()
        if session is None or session.get("exit_requested"):
            return False
        path = Path(session["exit_request_file"])
        partial = path.with_suffix(".partial")
        partial.write_text(reason.strip()[:200] + "\n", encoding="utf-8")
        partial.replace(path)
        session["exit_requested"] = reason.strip()[:200]
        self.store.save()
        return True

    def poll(self) -> dict[str, Any] | None:
        """The session's final answer once it exists; the session is then cleared."""
        session = self.active()
        if session is None:
            return None
        path = Path(session["result_file"])
        text = path.read_text(encoding="utf-8", errors="replace").strip() if path.exists() else ""
        answer: dict[str, Any] | None = None
        if text:
            try:
                parsed = json.loads(text.splitlines()[-1])
                answer = parsed if isinstance(parsed, dict) else None
            except ValueError:
                answer = None
            if answer is None:
                answer = {"outcome": "UNRESOLVED", "detail": "the session's answer is not one JSON object"}
        elif self._process is not None and self._process.poll() is not None:
            answer = {
                "outcome": "UNRESOLVED",
                "detail": f"the session exited ({self._process.returncode}) without an answer",
            }
        elif time.time() * 1000 - int(session["started_ms"]) > self.settings.max_session_s * 1000:
            answer = {
                "outcome": "UNRESOLVED",
                "detail": "the session gave no answer within its maximum duration",
            }
        if answer is None:
            return None
        self.store.state.pop("execution_session", None)
        self.store.save()
        self._process = None
        return {
            **answer,
            "candidate": session["candidate"],
            "exit_requested": session.get("exit_requested"),
            "plan": session.get("plan"),
        }
