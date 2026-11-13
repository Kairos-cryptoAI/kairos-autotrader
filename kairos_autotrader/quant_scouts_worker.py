"""Runs the real Quant Scouts code for the runner: closed bars and market snapshots.

Executed as a script by the Quant Scouts repository's own Python environment. It imports
nothing from the runner.

Closed bars (``"op": "pages"`` or no op). The repository's unchanged
``kairos_quant.long_gap_recovery.parse_page`` accepts only a complete, ordered, gap-free page
and builds one strict ``ClosedBarEventV1`` per kline. No network or file access:

stdin:  {"pages": [{"symbol": "SOLUSDT", "start": <open ms>, "end": <exclusive ms>,
                    "rows": [<kline as Binance returns it>, ...]}, ...]}
stdout: {"pages": [{"symbol", "start", "end",
                    "bars": [[open_time_ms, open, high, low, close, base_volume, quote_volume,
                              taker_buy_base_volume, taker_buy_quote_volume, bar_sha256], ...]}
                   | {"symbol", "start", "end", "error": "..."}]}

Market snapshots (``"op": "snapshot"``). The repository's ``BinanceFuturesCollector`` reads
funding and the one-hour open-interest window from Binance's public REST endpoints with its
own validation; the order book is one public REST depth page, checked for freshness with
the service's ``book_stale_after_s``; ``SnapshotBuilder`` computes the indicators from the
closed bars given and builds the ``MarketSnapshot``, as the service does. The service's
liquidation totals come from a stream this process does not have; they stay at the
builder's zero default:

stdin:  {"op": "snapshot", "bars": {"SOLUSDT": [<cached closed bar>, ...]}}
stdout: {"snapshots": {"SOLUSDT": <MarketSnapshot payload> | {"error": "..."}}}
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

MARKER = "@@KAIROS-WORKER@@"
DEPTH_LIMIT = 20


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


async def _depth(session, rest_base: str, symbol: str) -> dict:
    url = f"{rest_base}/fapi/v1/depth"
    async with session.get(url, params={"symbol": symbol, "limit": str(DEPTH_LIMIT)}) as response:
        if response.status != 200:
            raise ValueError(f"depth answered HTTP {response.status}")
        body = await response.json()
    if not isinstance(body, dict):
        raise ValueError("depth answer is not an object")
    return body


async def _snapshots(request: dict) -> dict:
    import aiohttp
    from kairos_quant.collectors.binance_ws import BinanceFuturesCollector
    from kairos_quant.config import QuantSettings
    from kairos_quant.orderbook import normalize_order_book
    from kairos_quant.snapshot import SnapshotBuilder

    bars = request["bars"]
    symbols = sorted(bars)
    settings = QuantSettings(_env_file=None, trading_symbols=symbols)
    collector = BinanceFuturesCollector(
        symbols,
        settings.binance_ws_base,
        settings.binance_rest_base,
        max_exchange_future_skew_s=settings.maximum_binance_future_skew_ms / 1_000,
    )
    timeout = aiohttp.ClientTimeout(total=15)
    async with aiohttp.ClientSession(trust_env=True, timeout=timeout) as session:
        # Derivatives first: the collector reads open interest one symbol after another. The
        # books are read last, so that each one is as fresh as possible when it is built.
        await collector.refresh_funding(session)
        await collector.refresh_open_interest(session)
        depths = await asyncio.gather(
            *(_depth(session, settings.binance_rest_base, symbol) for symbol in symbols),
            return_exceptions=True,
        )
    builder = SnapshotBuilder(settings.service_name, settings.price_window, settings.depth_levels)
    snapshots: dict[str, dict] = {}
    for symbol, depth in zip(symbols, depths, strict=True):
        try:
            if isinstance(depth, BaseException):
                raise ValueError(f"order book unavailable ({type(depth).__name__})")
            event_ms = collector._integer(depth.get("E"))
            age_s = time.time() - event_ms / 1_000 if event_ms else -1.0
            if (
                event_ms is None
                or not -collector._max_exchange_future_skew_s <= age_s <= settings.book_stale_after_s
            ):
                raise ValueError("order book is stale or from the future")
            book_bids, book_asks = normalize_order_book(
                collector._levels(depth.get("bids")), collector._levels(depth.get("asks"))
            )
            if not collector.is_funding_fresh(symbol, settings.derivatives_stale_after_s):
                raise ValueError("funding rate is not fresh")
            if not collector.is_open_interest_fresh(symbol, settings.derivatives_stale_after_s):
                raise ValueError("open interest is not fresh")
            rows = bars[symbol][-settings.price_window :]
            if not rows:
                raise ValueError("no closed bars")
            for row in rows:
                builder.push_candle(symbol, high=float(row[2]), low=float(row[3]), close=float(row[4]))
            key = symbol.lower()
            snapshot = builder.build(
                symbol,
                bids=book_bids,
                asks=book_asks,
                funding_rate=collector.funding[key],
                open_interest=collector.open_interest[key],
                oi_change_pct_1h=collector.oi_change_pct_1h[key],
                volume_usd=float(rows[-1][6]),  # the collector's own rule: the last closed bar
            )
            snapshots[symbol] = snapshot.to_payload()
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not hide the others
            snapshots[symbol] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    return {"snapshots": snapshots}


def main() -> int:
    # argv[1] replaces this script's own directory on the import path, so nothing from the
    # runner package can shadow Quant Scouts imports.
    sys.path[0] = sys.argv[1]
    request = json.load(sys.stdin)
    try:
        if request.get("op") == "snapshot":
            answer: dict = asyncio.run(_snapshots(request))
        else:
            from kairos_quant.long_gap_recovery import parse_page

            answer = {"pages": [_page(parse_page, page) for page in request["pages"]]}
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
