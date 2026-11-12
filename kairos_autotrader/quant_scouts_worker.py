"""Turns Binance USD-M kline pages into closed bars with the real Quant Scouts parser.

Executed as a script by the Quant Scouts repository's own Python environment. It uses the
repository's unchanged ``kairos_quant.long_gap_recovery.parse_page``, which accepts only a
complete, ordered, gap-free page and builds one strict ``ClosedBarEventV1`` per kline. It
does no network or file access and imports nothing from the runner.

argv:   <quant-scouts source>
stdin:  {"pages": [{"symbol": "SOLUSDT", "start": <open ms>, "end": <exclusive ms>,
                    "rows": [<kline as Binance returns it>, ...]}, ...]}
stdout: a marker line followed by
        {"pages": [{"symbol", "start", "end",
                    "bars": [[open_time_ms, open, high, low, close, base_volume, quote_volume,
                              taker_buy_base_volume, taker_buy_quote_volume, bar_sha256], ...]}
                   | {"symbol", "start", "end", "error": "..."}]}
"""

from __future__ import annotations

import json
import sys

MARKER = "@@KAIROS-WORKER@@"


def _page(parse_page, page: dict) -> dict:
    head = {"symbol": page.get("symbol"), "start": page.get("start"), "end": page.get("end")}
    try:
        bars = parse_page(
            page["rows"], symbol=str(page["symbol"]), start=int(page["start"]), end=int(page["end"])
        )
    except Exception as exc:  # noqa: BLE001 - one bad page must not hide the others
        return {**head, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {
        **head,
        "bars": [
            [
                bar.open_time_ms,
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.base_volume,
                bar.quote_volume,
                bar.taker_buy_base_volume,
                bar.taker_buy_quote_volume,
                bar.bar_sha256,
            ]
            for bar in bars
        ],
    }


def main() -> int:
    # argv[1] replaces this script's own directory on the import path, so nothing from the
    # runner package can shadow Quant Scouts imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        from kairos_quant.long_gap_recovery import parse_page

        answer: dict = {"pages": [_page(parse_page, page) for page in request["pages"]]}
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
