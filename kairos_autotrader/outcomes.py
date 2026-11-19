"""Interpret recorded closes without attributing an account balance change to a strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RecordedClose:
    row: dict[str, Any]
    mode: str
    pnl_basis: str
    exit_origin: str


def recorded_closes(records: list[dict[str, Any]]) -> list[RecordedClose]:
    """Older closes lack provenance. Their preceding session identifies only the mode."""
    closes = []
    session_mode = "unknown"
    for row in records:
        if row.get("event") == "session_start":
            session_mode = row.get("mode", "unknown")
        if row.get("event") != "trade_closed":
            continue
        mode = row.get("mode", session_mode)
        if mode not in ("live", "paper") or session_mode not in ("unknown", mode):
            mode = "unknown"
        basis = row.get("pnl_basis")
        if basis is None:
            if mode == "paper" and row.get("pnl_exact") is True:
                basis = "paper_fills"
            elif row.get("pnl_exact") is False:
                basis = "price_estimate"
            elif mode == "live" and row.get("pnl_exact") is True:
                basis = "account_change"
            else:
                basis = "unknown"
        origin = row.get("exit_origin", "operator" if row.get("reason") == "OPERATOR_CLOSE" else "unknown")
        closes.append(RecordedClose(row, mode, basis, origin))
    return closes
