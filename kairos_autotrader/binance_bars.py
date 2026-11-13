"""Closed Binance USD-M one-minute bars: the signal data of the Kairos design.

The runner only moves bytes: it downloads public kline pages and keeps a cache on disk.
What a page means is decided by the real Quant Scouts parser, run by
``quant_scouts_worker.py`` in that repository's own environment. It accepts only complete,
ordered, gap-free pages, so the cache is one contiguous series of strict closed bars, each
with the ``bar_sha256`` the ``ClosedBarEventV1`` contract gives it.

Public market data only: no key, no account, no orders.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python

MINUTE_MS = 60_000
BINANCE_REST = "https://fapi.binance.com"
KLINES_PATH = "/fapi/v1/klines"
PAGE_BARS = 499  # the largest page in Binance's second-cheapest request-weight band
OVERLAP_BARS = 5  # already cached bars that every update downloads again and compares
WORKER_BARS = 40_000  # bars handed to one parser process
WORKER = Path(__file__).with_name("quant_scouts_worker.py")

# A cached bar: open time, open, high, low, close, base volume, quote volume,
# taker-buy base volume, taker-buy quote volume, bar_sha256.
Bar = list[Any]
FetchPage = Callable[[str, int, int], Awaitable[object]]


class BarsError(RuntimeError):
    """Closed bars of one symbol could not be brought up to date."""


@dataclass(frozen=True, slots=True)
class QuantConfig:
    python: Path = field(default_factory=lambda: sibling_python("kairos-quant-scouts"))
    source: Path = field(default_factory=lambda: sibling_checkout("kairos-quant-scouts"))
    timeout_s: float = 180.0

    def check(self) -> None:
        if not self.python.exists():
            raise BarsError(f"Quant Scouts Python not found: {self.python}")
        if not (self.source / "kairos_quant" / "long_gap_recovery.py").exists():
            raise BarsError(f"Quant Scouts source not found: {self.source}")


def quant_revision(config: QuantConfig) -> str:
    return git_revision(config.source, "kairos_quant")


def _run_parser(config: QuantConfig, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        answer = run_worker(
            config.python, WORKER, [str(config.source)], {"pages": pages}, timeout_s=config.timeout_s
        )
        parsed = answer["pages"]
    except SiblingError as exc:
        raise BarsError(f"Quant Scouts: {exc}") from exc
    except (KeyError, TypeError) as exc:
        raise BarsError("Quant Scouts worker returned malformed output") from exc
    if not isinstance(parsed, list) or len(parsed) != len(pages):
        raise BarsError("Quant Scouts worker returned malformed output")
    return parsed


def http_fetcher(session: aiohttp.ClientSession, *, timeout_s: float = 20.0) -> FetchPage:
    """Download one page of closed one-minute klines, ``[start_ms, end_ms)``."""

    async def fetch(symbol: str, start_ms: int, end_ms: int) -> object:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "startTime": str(start_ms),
            "endTime": str(end_ms - 1),
            "limit": str((end_ms - start_ms) // MINUTE_MS),
        }
        async with session.get(
            BINANCE_REST + KLINES_PATH, params=params, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            if response.status != 200:
                raise BarsError(f"Binance answered HTTP {response.status}")
            return json.loads(await response.text())

    return fetch


def contiguous(bars: list[Bar]) -> bool:
    return all(int(b[0]) - int(a[0]) == MINUTE_MS for a, b in zip(bars, bars[1:], strict=False))


@dataclass(slots=True)
class ClosedBarStore:
    """One contiguous series of strict closed bars per Binance symbol, cached on disk."""

    directory: Path
    _bars: dict[str, list[Bar]] = field(default_factory=dict)

    def _path(self, symbol: str) -> Path:
        return self.directory / f"{symbol}.1m.json"

    def _load(self, symbol: str) -> list[Bar]:
        if symbol not in self._bars:
            path = self._path(symbol)
            try:
                bars = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            except (ValueError, OSError):
                bars = []
            usable = isinstance(bars, list) and all(isinstance(b, list) and len(b) == 10 for b in bars)
            self._bars[symbol] = bars if usable and contiguous(bars) else []
        return self._bars[symbol]

    def _save(self, symbol: str, bars: list[Bar]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        temp = self._path(symbol).with_suffix(".tmp")
        temp.write_text(json.dumps(bars, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, self._path(symbol))

    def _plan(self, symbol: str, first_ms: int, end_ms: int) -> tuple[list[Bar], list[tuple[int, int]]]:
        """The cached bars that stay and the pages to download, oldest first."""
        kept = [bar for bar in self._load(symbol) if first_ms <= int(bar[0]) < end_ms]
        if kept and int(kept[0][0]) != first_ms:
            kept = []  # the cache does not reach back to the window's first bar: rebuild it
        start = max(first_ms, int(kept[-1][0]) - (OVERLAP_BARS - 1) * MINUTE_MS) if kept else first_ms
        pages = []
        while start < end_ms:
            stop = min(end_ms, start + PAGE_BARS * MINUTE_MS)
            pages.append((start, stop))
            start = stop
        return kept, pages

    @staticmethod
    def _merge(symbol: str, kept: list[Bar], fresh: list[Bar]) -> list[Bar]:
        """Cached bars plus freshly parsed ones; the overlap has to be identical."""
        known = {int(bar[0]): bar for bar in kept[-OVERLAP_BARS:]}
        merged = list(kept)
        for bar in fresh:
            cached = known.get(int(bar[0]))
            if cached is not None:
                if cached != bar:
                    raise BarsError(
                        f"{symbol}: the venue changed a closed bar at {int(bar[0])}; "
                        "the cache is rebuilt on the next update"
                    )
            elif not merged or int(bar[0]) == int(merged[-1][0]) + MINUTE_MS:
                merged.append(bar)
            else:
                raise BarsError(f"{symbol}: closed bars are not contiguous at {int(bar[0])}")
        return merged

    async def update(
        self,
        config: QuantConfig,
        symbols: list[str],
        now_ms: int,
        history_ms: int,
        *,
        fetch: FetchPage,
        parser: Callable[[QuantConfig, list[dict[str, Any]]], list[dict[str, Any]]] | None = None,
        concurrency: int = 4,
    ) -> tuple[dict[str, list[Bar]], dict[str, str]]:
        """Bring every symbol up to the last closed minute.

        Returns the bars of ``[end - history, end)`` per symbol, and the reason for every
        symbol that could not be completed. A symbol with a problem keeps its old cache,
        except after a changed closed bar, which drops it.
        """
        end_ms = now_ms // MINUTE_MS * MINUTE_MS
        first_ms = end_ms - history_ms // MINUTE_MS * MINUTE_MS
        gate = asyncio.Semaphore(concurrency)
        problems: dict[str, str] = {}
        plans: dict[str, tuple[list[Bar], list[tuple[int, int]]]] = {}
        downloaded: dict[str, list[dict[str, Any]]] = {}

        async def download(symbol: str) -> None:
            kept, spans = self._plan(symbol, first_ms, end_ms)
            plans[symbol] = (kept, spans)
            pages = []
            try:
                for start, stop in spans:
                    async with gate:
                        rows = await fetch(symbol, start, stop)
                    pages.append({"symbol": symbol, "start": start, "end": stop, "rows": rows})
            except (TimeoutError, aiohttp.ClientError, BarsError, ValueError) as exc:
                problems[symbol] = f"{type(exc).__name__}: {str(exc)[:160]}"
                return
            downloaded[symbol] = pages

        await asyncio.gather(*(download(symbol) for symbol in symbols))

        batches: list[list[dict[str, Any]]] = [[]]
        size = 0
        for symbol in symbols:
            for page in downloaded.get(symbol, []):
                count = (page["end"] - page["start"]) // MINUTE_MS
                if batches[-1] and size + count > WORKER_BARS:
                    batches.append([])
                    size = 0
                batches[-1].append(page)
                size += count
        parsed: dict[str, list[dict[str, Any]]] = {symbol: [] for symbol in downloaded}
        run = parser or _run_parser
        for batch in batches:
            if not batch:
                continue
            try:
                answers = await asyncio.to_thread(run, config, batch)
            except BarsError as exc:
                for page in batch:
                    problems.setdefault(page["symbol"], str(exc)[:200])
                continue
            for answer in answers:
                parsed[str(answer.get("symbol"))].append(answer)

        result: dict[str, list[Bar]] = {}
        for symbol in symbols:
            if symbol in problems or symbol not in downloaded:
                continue
            kept, spans = plans[symbol]
            fresh: list[Bar] = []
            answers = parsed[symbol]
            bad = next((a for a in answers if "error" in a or not isinstance(a.get("bars"), list)), None)
            if bad is not None or len(answers) != len(spans):
                problems[symbol] = str((bad or {}).get("error", "parser returned too few pages"))[:200]
                continue
            for answer in sorted(answers, key=lambda a: int(a["start"])):
                fresh.extend(answer["bars"])
            try:
                merged = self._merge(symbol, kept, fresh)
            except BarsError as exc:
                problems[symbol] = str(exc)
                if "changed a closed bar" in str(exc):
                    self._bars[symbol] = []
                    try:
                        await asyncio.to_thread(self._save, symbol, [])
                    except OSError:
                        problems[symbol] += "; the cache file could not be cleared"
                continue
            if not merged or int(merged[0][0]) != first_ms or int(merged[-1][0]) != end_ms - MINUTE_MS:
                problems[symbol] = "closed bars do not cover the whole window"
                continue
            try:
                await asyncio.to_thread(self._save, symbol, merged)
            except OSError as exc:
                # Without a durable cache the next update could not compare its overlap.
                problems[symbol] = f"closed-bar cache could not be written: {type(exc).__name__}"
                continue
            self._bars[symbol] = merged
            result[symbol] = merged
        return result, problems
