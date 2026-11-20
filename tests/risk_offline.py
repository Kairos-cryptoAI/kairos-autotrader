"""Runs ``risk_worker.decide`` offline: real Quant Scouts and Risk Manager code, no network.

Executed by the Quant Scouts environment, like the worker itself:

argv:  <quant-scouts source> <risk-manager source> <runner package directory>
stdin: {"request": <worker request>, "books": {"binance": {...}, "evedex": {...}}}
stdout: the worker's answer as JSON
"""

from __future__ import annotations

import importlib.util
import json
import sys


def main() -> int:
    sys.path[0] = sys.argv[1]
    sys.path.insert(1, sys.argv[2])
    spec = importlib.util.spec_from_file_location("risk_worker", f"{sys.argv[3]}/risk_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)  # type: ignore[union-attr]

    from kairos_quant.venue_comparison import BookLevel, BookSnapshot

    payload = json.load(sys.stdin)
    request, books = payload["request"], payload["books"]
    if payload.get("refusal_only"):
        # Only a request the worker refuses before measuring may come here: nothing is fetched.
        import asyncio

        answer = asyncio.run(worker.evaluate(request))
        assert "refused" in answer, "a measurement would need the network"
        sys.stdout.write(json.dumps(answer))
        return 0
    # Both synthetic sources describe one observation. Do not age their clocks by the
    # time it takes the test subprocess to import its dependencies on a cold machine.
    now = payload["sampled_at_ms"]
    worker._now_ms = lambda: now

    lead = int(payload.get("exchange_clock_lead_ms", 0))

    def book(source: str, raw: dict) -> BookSnapshot:
        return BookSnapshot(
            source=source,
            symbol=raw["symbol"],
            timestamp_ms=now + lead - int(raw.get("age_ms", 100)),
            bids=tuple(BookLevel(price, quantity) for price, quantity in raw["bids"]),
            asks=tuple(BookLevel(price, quantity) for price, quantity in raw["asks"]),
            latency_ms=50.0,
        )

    # The worker's own measurement path, which times it on the venues' clock.
    venue = worker.venue_quality(
        book("binance", books["binance"]),
        book("evedex", books["evedex"]),
        books["evedex"]["symbol"],
        latency_ms=50,
        taker_fee_bps=request["venue"]["taker_fee_bps"],
        assessed=request["venue"]["assessed_notional_usd"],
    )
    try:
        answer = worker.decide(request, venue)
    except Exception as exc:  # noqa: BLE001 - the test reads it
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
