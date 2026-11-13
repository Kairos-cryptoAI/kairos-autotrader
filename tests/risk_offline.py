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
import time


def main() -> int:
    sys.path[0] = sys.argv[1]
    sys.path.insert(1, sys.argv[2])
    spec = importlib.util.spec_from_file_location("risk_worker", f"{sys.argv[3]}/risk_worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)  # type: ignore[union-attr]

    from kairos_core.enums import EvedexProfile
    from kairos_quant.runtime_venue_gate import VenueGatePolicy, build_venue_quality
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
    now = int(time.time() * 1000)

    def book(source: str, raw: dict) -> BookSnapshot:
        return BookSnapshot(
            source=source,
            symbol=raw["symbol"],
            timestamp_ms=now - int(raw.get("age_ms", 100)),
            bids=tuple(BookLevel(price, quantity) for price, quantity in raw["bids"]),
            asks=tuple(BookLevel(price, quantity) for price, quantity in raw["asks"]),
            latency_ms=50.0,
        )

    venue = build_venue_quality(
        binance=book("binance", books["binance"]),
        evedex=book("evedex", books["evedex"]),
        venue_symbol=books["evedex"]["symbol"],
        observed_at_ms=now,
        latency_ms=50,
        policy=VenueGatePolicy(
            assessed_notional_usd=request["venue"]["assessed_notional_usd"],
            taker_fee_bps=request["venue"]["taker_fee_bps"],
        ),
        profile=EvedexProfile.PROD,
        source="kairos-quant-scouts",
    )
    try:
        answer = worker.decide(request, venue)
    except Exception as exc:  # noqa: BLE001 - the test reads it
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
