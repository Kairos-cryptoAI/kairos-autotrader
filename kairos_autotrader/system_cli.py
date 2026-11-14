"""Commands that run the real Kairos Strategy Engine next to the trader, without orders or keys.

``engine-replay``  measures the engine's sleeves and the live rules on history, with costs.
``shadow``         records the engine's new intents as they appear, for later evaluation.

Signals are handled as the journal's signal records (``signals.outcome`` reads them).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from bisect import bisect_left
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from .binance_bars import ClosedBarStore, QuantConfig, http_fetcher
from .engine import BAR_MS, Config, ConfigError, Store, say
from .evedex import Evedex
from .management import compare_policies, policies_for, write_policy
from .signals import outcome, signal_symbol
from .system_engine import (
    DAY_MS,
    EXCLUDED_SLEEVES,
    MINUTE_MS,
    EngineConfig,
    MinuteStore,
    collect,
    engine_revision,
    hourly_trend_history,
    to_candles,
)
from .text_scouts import NewsDesk

SHADOW_MODE = "engine-shadow"
LEARNING_REPLAY_FILE = "learning-replay.jsonl"
LIVE_RULES = "боевые правила (hourly_trend, сигнал по свечам Binance), скан раз в час"
LIVE_RULES_QUARTER = "боевые правила (hourly_trend, сигнал по свечам Binance), скан каждые 15 минут"


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%d.%m")


def evaluate(records: list[dict[str, Any]], candles: list, cost_fraction: float) -> list[dict[str, Any]]:
    """Outcome of each signal record on the minute bars that followed it. Unfinished ones are left out."""
    times = [c.t for c in candles]
    rows = []
    for record in records:
        start = bisect_left(times, record["signal_time"])
        later = candles[start : start + record["timeout_minutes"] + 2]
        result = outcome(record, later, cost_fraction=cost_fraction)
        if result["r"] is None:
            continue
        risk = abs(record["ref_price"] - record["stop"])
        rows.append(
            {
                "id": record["id"],
                "setup": record["strategy_id"],
                "symbol": record["symbol"],
                "side": record["side"],
                "signal_time": record["signal_time"],
                "reason": result["reason"],
                "r": result["r"],
                "cost_r": cost_fraction * record["ref_price"] / risk,
                "stop_pct": 100 * risk / record["ref_price"],
            }
        )
    return rows


def summarise(name: str, rows: list[dict[str, Any]], days: float) -> dict[str, Any] | None:
    if not rows:
        say(f"  {name}: завершённых сигналов нет")
        return None
    values = [r["r"] for r in rows]
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values) - 1)
    error = (variance / len(values)) ** 0.5
    exits = Counter(r["reason"] for r in rows)
    cost = sum(r["cost_r"] for r in rows) / len(rows)
    stop = sorted(r["stop_pct"] for r in rows)[len(rows) // 2]
    first, last = min(r["signal_time"] for r in rows), max(r["signal_time"] for r in rows)
    say(
        f"  {name}: {len(rows)} сигналов ({len(rows) / days:.1f} в день, {_day(first)}–{_day(last)}), "
        f"средний результат {mean:+.3f} R ± {error:.3f}, в плюс {sum(v > 0 for v in values)}, "
        f"комиссии {cost:.2f} R на сделку, стоп {stop:.2f}%, "
        f"выходы: тейк {exits.get('TAKE', 0)} / стоп {exits.get('STOP', 0)} / срок {exits.get('TIMEOUT', 0)}"
    )
    return {
        "name": name,
        "signals": len(rows),
        "per_day": round(len(rows) / days, 2),
        "mean_r": round(mean, 4),
        "standard_error": round(error, 4),
        "wins": sum(v > 0 for v in values),
        "mean_cost_r": round(cost, 4),
        "median_stop_pct": round(stop, 3),
        "exits": dict(exits),
    }


async def cmd_engine_replay(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    engine = EngineConfig(window_days=args.days)
    engine.check()
    revision = engine_revision(engine.source)
    symbols = args.symbols or config.symbols
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    store = MinuteStore(args.runtime_dir / "minute-cache")
    say(
        f"Strategy Engine: {engine.source}, ревизия {revision[:12]}. "
        f"История: {args.days} дней минутных свечей EVEDEX и Binance, "
        f"{len(symbols)} инструментов."
    )
    quant = QuantConfig()
    quant.check()
    closed = ClosedBarStore(args.runtime_dir / "closed-bars")
    async with aiohttp.ClientSession(trust_env=True) as session:
        api = Evedex(session)
        fee = float(await api.taker_fee())
        cost_fraction = 2 * (fee + config.slippage_fraction)
        now_ms = int(time.time() * 1000)
        candidates, problems, bars = await collect(api, store, engine, symbols, now_ms)
        signal_bars, bar_problems = await closed.update(
            quant,
            [signal_symbol(symbol) for symbol in symbols],
            now_ms,
            args.days * DAY_MS,
            fetch=http_fetcher(session),
        )
    problems.extend(f"{symbol}: {reason}" for symbol, reason in bar_problems.items())
    for problem in problems:
        say(f"  замечание: {problem}")
    # The rules that trade live: the signal on Binance bars, as in the live scan, the
    # outcome on the EVEDEX minute bars the order would have met.
    live, live_problems = hourly_trend_history(
        engine,
        {
            symbol: signal_bars[signal_symbol(symbol)]
            for symbol in symbols
            if signal_symbol(symbol) in signal_bars
        },
        reward_risk=config.reward_risk,
        timeout_minutes=config.timeout_minutes,
    )
    problems.extend(live_problems)
    for problem in live_problems:
        say(f"  замечание: {problem}")
    rows: list[dict[str, Any]] = []
    own: list[dict[str, Any]] = []
    own_quarter: list[dict[str, Any]] = []
    candles_by_symbol: dict[str, list[Any]] = {}
    for symbol in symbols:
        candles = to_candles(bars.get(symbol, []))
        if not candles:
            continue
        candles_by_symbol[symbol] = candles
        rows.extend(evaluate([c for c in candidates if c["symbol"] == symbol], candles, cost_fraction))
        quarter = [c for c in live if c["symbol"] == symbol]
        own_quarter.extend(evaluate(quarter, candles, cost_fraction))
        own.extend(evaluate([c for c in quarter if c["signal_time"] % BAR_MS == 0], candles, cost_fraction))
    say(
        f"Результат каждого сигнала по минутным свечам, с комиссиями и проскальзыванием "
        f"({cost_fraction * 100:.3f}% на круг). R = один риск на сделку."
    )
    summary = []
    for sleeve in engine.sleeves:
        warmup = 34.0 if sleeve == "regime_aligned_right_tail_v1" else 3.0
        summary.append(
            summarise(sleeve, [r for r in rows if r["setup"] == sleeve], max(1.0, args.days - warmup))
        )
    summary.append(summarise(LIVE_RULES, own, max(1.0, args.days - 12.5)))
    summary.append(summarise(LIVE_RULES_QUARTER, own_quarter, max(1.0, args.days - 12.5)))
    say("Не запускались: " + "; ".join(f"{name} ({reason})" for name, reason in EXCLUDED_SLEEVES.items()))
    # Position management (break-even, trailing, partial exits) against the plan, paired on
    # the live rules' signals; the chosen policy, if any is proven, is used by the runner.
    management = compare_policies(live, candles_by_symbol, cost_fraction, policies_for(config.reward_risk))
    policy = write_policy(args.runtime_dir, management, now_ms=now_ms, reward_risk=config.reward_risk)
    say("Сопровождение позиции против плана (те же сигналы, ошибка с учётом пересечений):")
    for item in management:
        say(
            f"  {item['policy']['name']}: {item['mean_r']:+.3f} R (план {item['plan_mean_r']:+.3f} R), "
            f"выигрыш {item['gain_r']:+.3f} ± {item['gain_error_r']:.3f} R, "
            f"нижняя граница {item['gain_lower_r']:+.3f} R"
        )
    say(
        "Выбрано: "
        + (
            policy["policy"]["name"]
            if policy["policy"]
            else "ничего — ни одно сопровождение не доказало преимущества, позиции идут по плану"
        )
        + "."
    )
    say(
        "Это условный результат по истории, не сделки. "
        "Погрешность ± показывает, отличим ли результат от нуля."
    )
    report = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_revision": revision,
        "days": args.days,
        "symbols": symbols,
        "cost_fraction": cost_fraction,
        "summary": [s for s in summary if s],
        "management": management,
        "problems": problems,
        "signals": rows,
    }
    path = args.runtime_dir / "engine-replay.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    say(f"Подробности записаны в {path}.")
    # The live rules' replayed signals feed the learning ledger as REPLAY evidence (half weight).
    # No Macro decision exists for history, so their regime is UNKNOWN.
    replay = args.runtime_dir / LEARNING_REPLAY_FILE
    replay.write_text(
        "".join(
            json.dumps(
                {
                    "strategy_id": row["setup"],
                    "symbol": signal_symbol(row["symbol"]),
                    "regime": "UNKNOWN",
                    "r_multiple": str(row["r"]),
                    "decided_at_ms": int(row["signal_time"]),
                    "source": "REPLAY",
                }
            )
            + "\n"
            for row in own_quarter
        ),
        encoding="utf-8",
    )
    say(f"Повторы истории для журнала обучения ({len(own_quarter)} сигналов) записаны в {replay}.")
    return 0


async def cmd_shadow(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    engine = EngineConfig()
    engine.check()
    store = Store(args.runtime_dir)
    state = store.state
    if state.get("mode") not in (None, SHADOW_MODE):
        raise ConfigError("this runtime directory belongs to another mode; use a separate directory")
    revision = engine_revision(engine.source)
    now_ms = int(time.time() * 1000)
    state["mode"] = SHADOW_MODE
    state.setdefault("started_ms", now_ms)
    state.setdefault("seen", {})
    if state.get("engine_revision") != revision:
        state["engine_revision"] = revision
        store.journal(
            "shadow_start",
            engine_revision=revision,
            sleeves=list(engine.sleeves),
            symbols=config.symbols,
            excluded=EXCLUDED_SLEEVES,
        )
    store.save()
    minute = MinuteStore(args.runtime_dir / "minute-cache")
    say(
        f"Теневой режим Strategy Engine (ревизия {revision[:12]}). Заявок нет, ключи не нужны. "
        f"Сигналы пишутся в {args.runtime_dir}. Остановить: Ctrl+C или файл STOP."
    )
    cycles = 0
    reported: list[str] = []
    async with aiohttp.ClientSession(trust_env=True) as session:
        api = Evedex(session)
        while args.max_cycles is None or cycles < args.max_cycles:
            cycles += 1
            now_ms = int(time.time() * 1000)
            try:
                candidates, problems, _ = await collect(
                    api, minute, engine, config.symbols, now_ms, since_ms=state["started_ms"]
                )
            except (TimeoutError, aiohttp.ClientError) as exc:
                store.journal("error", kind=type(exc).__name__, message=str(exc)[:300])
                say(f"Ошибка связи ({type(exc).__name__}); повтор в следующем цикле.")
                candidates, problems = [], []
            seen = state["seen"]
            fresh = [c for c in candidates if c["id"] not in seen]
            for candidate in fresh:
                seen[candidate["id"]] = candidate["signal_time"]
            for key in [
                k for k, moment in seen.items() if moment < now_ms - (engine.window_days + 1) * DAY_MS
            ]:
                del seen[key]
            if fresh or problems != reported:
                reported = problems  # a standing remark is written once, not every cycle
                store.journal(
                    "scan",
                    bar=now_ms // MINUTE_MS * MINUTE_MS,
                    source="strategy_engine",
                    engine_revision=revision,
                    features=[],
                    candidates=fresh,
                    skipped=problems,
                )
            store.save()
            if fresh:
                say(
                    "Новые сигналы движка: "
                    + ", ".join(f"{c['symbol']} {c['side']} {c['setup']}" for c in fresh)
                )
            elif cycles == 1 or cycles % 12 == 0:
                say(
                    f"Новых сигналов нет. Всего записано: {len(seen)}."
                    + (f" Замечания: {'; '.join(problems)[:200]}" if problems else "")
                )
            if store.flag("STOP"):
                say("Найден файл STOP. Останавливаюсь.")
                return 0
            if args.max_cycles is not None and cycles >= args.max_cycles:
                break
            # Wake shortly after the next five-minute boundary, when the engine's 5m bars close.
            pause = args.interval - (time.time() % args.interval) + 20
            await asyncio.sleep(pause)
    return 0


async def cmd_news(args: argparse.Namespace) -> int:
    """One Text Scouts poll without a model key: shows what the news layer sees right now."""
    import tempfile

    store = Store(args.runtime_dir or Path(tempfile.mkdtemp(prefix="kairos-news-")))
    desk = NewsDesk(store)
    desk.config.check()
    say(
        f"Text Scouts: {desk.config.source}, ревизия {desk.revision[:12]}. "
        f"Источники: GDELT и {len(desk.config.feeds)} RSS-лент."
    )
    async with aiohttp.ClientSession(trust_env=True) as session:
        summary = await desk.poll(session, openai_key=None, model="", max_llm_calls=0)
    counts = summary["counts"]
    say(
        f"Получено {counts['fetched']}, свежих (до 30 минут) {counts['timely']}, "
        f"без повторов {counts['unique']}, "
        f"значимых {counts['kept']}. По источникам: {summary['sources']}."
    )
    for title in summary["kept"]:
        say(f"  значимая новость: {title}")
    for signal in summary["signals"]:
        say(
            f"  сигнал: {signal['topic']} — {signal['impact']}, уверенность {signal['confidence']:.2f} "
            f"({'модель' if signal['origin'] == 'llm' else 'грубая оценка без модели'})"
        )
    if not summary["signals"]:
        say(
            "Сигналов нет. Без ключа модели работает только грубая оценка по ключевым словам; "
            "в боевом режиме значимые новости разбирает модель."
        )
    return 0


def add_parsers(sub: argparse._SubParsersAction) -> dict[str, Any]:
    replay = sub.add_parser(
        "engine-replay", help="measure the real Strategy Engine on EVEDEX history; no orders"
    )
    replay.add_argument("--runtime-dir", type=Path, required=True)
    replay.add_argument("--config", type=Path)
    replay.add_argument("--days", type=int, default=45)
    replay.add_argument("--symbols", nargs="*")

    shadow = sub.add_parser("shadow", help="record the Strategy Engine's new intents; no orders, no keys")
    shadow.add_argument("--runtime-dir", type=Path, required=True)
    shadow.add_argument("--config", type=Path)
    shadow.add_argument("--interval", type=int, default=300)
    shadow.add_argument("--max-cycles", type=int, help=argparse.SUPPRESS)
    news = sub.add_parser("news", help="one Text Scouts poll without a model key; no orders")
    news.add_argument("--runtime-dir", type=Path)
    return {"engine-replay": cmd_engine_replay, "shadow": cmd_shadow, "news": cmd_news}
