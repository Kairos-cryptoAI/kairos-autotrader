"""Command line: check, scan, run and report.

Secrets are read from files at start and never printed or written anywhere.
Real orders need ``run --mode live --confirm-real-orders`` typed by the operator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path

import aiohttp

from . import system_cli
from .engine import Config, ConfigError, LiveBroker, PaperBroker, Store, Trader, say
from .evedex import Evedex, EvedexError, OrderSigner, Rejected, new_order_id
from .signals import outcome
from .system_engine import EngineError
from .text_scouts import TextScoutsError

LABELLED = re.compile(r"^\s*[^:=]{1,80}?\s*[:=]\s*(\S.*?)\s*$")


class SecretError(ValueError):
    pass


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def _value(line: str) -> str:
    match = LABELLED.match(line)
    return match.group(1) if match else line.strip()


def secret_from(keys_file: Path | None, line: int | None, single_file: Path | None, what: str) -> str | None:
    """A secret from its own file, or from ``label: value`` line N of the keys file."""
    if single_file is not None:
        value = single_file.read_text(encoding="utf-8-sig").strip()
    elif keys_file is not None and line is not None:
        lines = _lines(keys_file)
        if not 1 <= line <= len(lines):
            raise SecretError(f"{what}: line {line} is outside the keys file")
        value = _value(lines[line - 1])
    else:
        return None
    if not value or any(ch.isspace() for ch in value):
        raise SecretError(f"{what}: the value is empty or contains spaces")
    return value


def find_openai_line(keys_file: Path) -> int | None:
    """The single line whose label mentions OpenAI, if there is exactly one."""
    hits = [
        i
        for i, line in enumerate(_lines(keys_file), 1)
        if re.search(r"openai", re.split(r"[:=]", line, maxsplit=1)[0], re.I)
    ]
    return hits[0] if len(hits) == 1 else None


def load_secrets(args: argparse.Namespace, *, need_trading: bool) -> dict[str, str | None]:
    keys_file = args.keys_file
    api_key = secret_from(keys_file, args.api_key_line, args.api_key_file, "EVEDEX API key")
    wallet = secret_from(keys_file, args.wallet_key_line, args.wallet_key_file, "wallet key")
    openai_line = args.openai_key_line
    if openai_line is None and keys_file is not None and args.openai_key_file is None:
        openai_line = find_openai_line(keys_file)
    openai = secret_from(keys_file, openai_line, args.openai_key_file, "OpenAI key")
    if wallet is not None and not re.fullmatch(r"(0x)?[0-9a-fA-F]{64}", wallet):
        raise SecretError("wallet key: the selected line does not look like a 64-hex private key")
    if openai is not None and not openai.startswith("sk-"):
        raise SecretError("OpenAI key: the selected line does not look like an OpenAI key")
    if need_trading and (api_key is None or wallet is None):
        raise SecretError("live mode needs both the EVEDEX API key and the wallet key")
    return {"api_key": api_key, "wallet": wallet, "openai": openai}


def add_secret_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--keys-file", type=Path, help="file with 'label: value' lines")
    parser.add_argument("--api-key-line", type=int, help="line number of the EVEDEX API key")
    parser.add_argument("--wallet-key-line", type=int, help="line number of the wallet private key")
    parser.add_argument(
        "--openai-key-line", type=int, help="line number of the OpenAI key (auto-detected by label)"
    )
    parser.add_argument("--api-key-file", type=Path, help="file that contains only the EVEDEX API key")
    parser.add_argument("--wallet-key-file", type=Path, help="file that contains only the wallet private key")
    parser.add_argument("--openai-key-file", type=Path, help="file that contains only the OpenAI key")


def self_test_signing() -> str:
    """Sign sample payloads with a throwaway key to prove the signing stack works here."""
    from eth_account import Account

    throwaway = Account.create().key.hex()
    signer = OrderSigner(throwaway, 42161)
    body = signer.market_order(order_id=new_order_id(), instrument="SOLUSD", side="BUY", cash=Decimal("10"))
    signer.close_order(order_id=new_order_id(), instrument="SOLUSD", quantity=Decimal("0.1"))
    signer.tpsl(instrument="SOLUSD", kind="stop-loss", side="BUY", quantity=Decimal(0), price=Decimal("80"))
    return "ok" if len(body["signature"]) == 132 else "unexpected signature length"


async def cmd_check(args: argparse.Namespace) -> int:
    config = Config.load(args.config)
    secrets = load_secrets(args, need_trading=False)
    async with aiohttp.ClientSession(trust_env=True) as session:
        api = Evedex(session, secrets["api_key"])
        instruments = await api.instruments()
        fee = await api.taker_fee()
        say(f"Публичные данные: {len(instruments)} инструментов, комиссия тейкера {fee * 100:.4f}%.")
        offset = (int(time.time() * 1000) - await api.server_time_ms()) / 1000
        say(
            f"Часы компьютера относительно биржи: {offset:+.1f} с."
            + (
                " ВНИМАНИЕ: расхождение больше 5 с, синхронизируйте время Windows, "
                "иначе входы будут пропускаться."
                if abs(offset) > 5
                else ""
            )
        )
        for symbol in config.symbols:
            inst = instruments.get(symbol)
            if inst is None:
                say(f"  {symbol}: НЕТ на бирже")
                continue
            book = await api.book(symbol)
            spread = float((book.ask - book.bid) / book.ask) * 1e4
            lot_usd = max(inst.min_quantity * book.ask, inst.min_volume)
            share = Decimal(str(config.capital_basis_usd)) / config.max_open_positions
            notes = []
            if spread > config.max_spread_bps:
                notes.append("спред сейчас шире допустимого")
            if lot_usd > share:
                notes.append("мин. лот больше доли капитала на позицию: только как фон для нейросети")
            say(
                f"  {symbol}: bid {book.bid} / ask {book.ask}, спред {spread:.1f} б.п., "
                f"мин. лот {lot_usd:.2f} USDT, "
                f"торги {'открыты' if inst.tradable else 'ЗАКРЫТЫ'}"
                + (f" ({'; '.join(notes)})" if notes else "")
            )
        say(f"Подпись заявок (проверка на одноразовом ключе): {self_test_signing()}.")
        say(f"Ключ нейросети: {'найден' if secrets['openai'] else 'не задан (режим без нейросети)'}.")
        if secrets["api_key"]:
            try:
                me = await api.me()
                funding, available = await api.balance()
                positions = await api.positions()
                orders = await api.open_orders()
                say(
                    f"Счёт доступен. Баланс {funding} USDT, свободно {available} USDT, "
                    f"позиций {len(positions)}, открытых заявок {len(orders)}, "
                    f"margin call: {me.get('marginCall')}."
                )
            except Rejected as exc:
                say(f"Биржа отказала в доступе к счёту: HTTP {exc.status} {exc.label or exc.body[:200]}")
                return 2
        else:
            say("Ключ биржи не задан: счёт не проверялся.")
        if secrets["wallet"]:
            say(f"Ключ кошелька читается, адрес {OrderSigner(secrets['wallet'], 42161).address}.")
    return 0


async def build_trader(args: argparse.Namespace, session: aiohttp.ClientSession, mode: str) -> Trader:
    config = Config.load(args.config)
    if getattr(args, "no_llm", False):
        config.use_llm = False
    secrets = load_secrets(args, need_trading=mode == "live")
    store = Store(args.runtime_dir)
    api = Evedex(session, secrets["api_key"])
    instruments = await api.instruments()
    fee = await api.taker_fee()
    if mode == "live":
        if secrets["wallet"] is None:
            raise SecretError("live mode needs the wallet key")
        broker: PaperBroker | LiveBroker = LiveBroker(api, store, config, secrets["wallet"])
    else:
        broker = PaperBroker(store, config, fee)
    if config.use_llm and not secrets["openai"]:
        say("Ключ нейросети не задан: решения принимает только алгоритм.")
    return Trader(
        api=api,
        session=session,
        store=store,
        config=config,
        broker=broker,
        fee=fee,
        instruments=instruments,
        openai_key=secrets["openai"] if config.use_llm else None,
    )


async def cmd_run(args: argparse.Namespace) -> int:
    if args.mode == "live" and not args.confirm_real_orders:
        say("Для настоящих заявок добавьте флаг --confirm-real-orders.")
        return 2
    async with aiohttp.ClientSession(trust_env=True) as session:
        trader = await build_trader(args, session, args.mode)
        if args.new_session:
            trader.store.state.pop("session", None)
            trader.store.state.pop("halted", None)
        say("Остановить входы: создайте файл STOP в папке запуска. Закрыть позицию сейчас: файл CLOSE.")
        try:
            await trader.run(max_steps=args.max_steps)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            left = trader.trades()
            if left:
                say(
                    f"ВНИМАНИЕ: открытых позиций осталось: {len(left)} "
                    f"({', '.join(t['symbol'] for t in left)}). Стоп и тейк стоят на бирже; "
                    "запустите программу снова, чтобы она продолжила их вести."
                )
    return 0


async def cmd_scan(args: argparse.Namespace) -> int:
    async with aiohttp.ClientSession(trust_env=True) as session:
        args.runtime_dir = args.runtime_dir or Path(tempfile.mkdtemp(prefix="kairos-scan-"))
        trader = await build_trader(args, session, "paper")
        features, candidates = await trader.scan(int(time.time() * 1000))
        for f in features:
            say(
                f"{f.symbol}: цена {f.price:g}, тренд 4ч {f.trend}, RSI {f.rsi:.0f}, "
                f"ATR {f.atr_pct:.2f}%, за 24ч {f.change_24h_pct:+.2f}%"
            )
        if not candidates:
            say("Сигналов на последнем закрытом баре нет.")
            return 0
        for c in candidates:
            say(
                f"СИГНАЛ {c.instrument} {c.side} {c.setup}: цена Binance {c.reference_price:g}, "
                f"стоп {c.stop:g}, тейк {c.take:g}, обзор {c.route.review_tier.value}"
            )
        decision = await trader.decide(features, candidates, trader.config.max_open_positions)
        say(
            f"Решение ({decision.source}): {decision.action} {', '.join(decision.picks)} "
            f"| {decision.reason[:300]}"
        )
        for verdict in decision.verdicts:
            say(
                f"  {verdict['candidate_id']}: {verdict['verdict']} {verdict['confidence']:.2f} "
                f"— {verdict['reason'][:200]}"
            )
    return 0


def _read_journal(directory: Path) -> list[dict]:
    path = directory / "journal.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _report_execution(config_path: Path | None) -> None:
    """The Execution Engine's closed trades, when it is the executor: its database is the journal."""
    if config_path is None:
        return
    config = Config.load(config_path)
    if config.executor != "execution_engine":
        return
    from .execution_session import closed_trades

    try:
        trades = closed_trades(Path(config.execution_python), Path(config.execution_session_config))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        say(f"Журнал Execution Engine недоступен: {str(exc)[:200]}")
        return
    net = sum(Decimal(t["net_pnl_usd"]) for t in trades)
    r_values = [Decimal(t["r_multiple"]) for t in trades if t.get("r_multiple") is not None]
    wins = sum(Decimal(t["net_pnl_usd"]) > 0 for t in trades)
    say(
        f"Execution Engine (единый журнал): сделок закрыто {len(trades)}, в плюс {wins}, "
        f"итог {net:+.4f} USDT"
        + (f", средний результат {sum(r_values) / len(r_values):+.3f} R." if r_values else ".")
    )
    for t in trades:
        say(
            f"  {t['settled_at'][:19]} {t['venue_symbol']} {t['side']} {t['exit_reason']}: "
            f"{Decimal(t['net_pnl_usd']):+.4f} USDT ({t['r_multiple']} R; комиссии {t['fees_usd']}, "
            f"финансирование {t['funding_cost_usd']})"
        )


async def cmd_report(args: argparse.Namespace) -> int:
    _report_execution(getattr(args, "config", None))
    records = _read_journal(args.runtime_dir)
    if not records:
        say("Журнал пуст.")
        return 0
    closed = [r for r in records if r["event"] == "trade_closed"]
    decisions = [r for r in records if r["event"] == "decision"]
    total = sum(Decimal(r["pnl_usd"]) for r in closed)
    wins = sum(Decimal(r["pnl_usd"]) > 0 for r in closed)
    say(f"Сделок закрыто: {len(closed)}, из них в плюс: {wins}. Итог: {total:+.4f} USDT.")
    for r in closed:
        say(
            f"  {r['ts'][:19]} {r['symbol']} {r['side']} {r['reason']}: {Decimal(r['pnl_usd']):+.4f} USDT "
            f"(решение: {r.get('source')}{'' if r.get('pnl_exact', True) else ', оценка по ценам'})"
        )
    equities = [r for r in records if r["event"] in ("session_start", "trade_closed") and r.get("equity")]
    if len(equities) > 1:
        say(
            f"Баланс по бирже: {Decimal(equities[0]['equity']):.4f} "
            f"-> {Decimal(equities[-1]['equity']):.4f} USDT."
        )
    actions = Counter((d["decision"]["source"], d["decision"]["action"]) for d in decisions)
    say(
        "Решения: " + ", ".join(f"{src} {act}: {n}" for (src, act), n in sorted(actions.items()))
        if actions
        else "Решений пока нет."
    )
    tokens_in = sum(d["decision"].get("usage", {}).get("input_tokens", 0) for d in decisions)
    tokens_out = sum(d["decision"].get("usage", {}).get("output_tokens", 0) for d in decisions)
    if tokens_in or tokens_out:
        say(f"Нейросеть: {tokens_in} входных и {tokens_out} выходных токенов.")
    ledger = args.runtime_dir / "llm-budget.json"
    if ledger.exists():
        for month, providers in sorted(json.loads(ledger.read_text(encoding="utf-8")).items()):
            for provider, entries in sorted(providers.items()):
                spent = sum(
                    e["actual"] if e["actual"] is not None else e["reserved"] for e in entries.values()
                )
                open_ = sum(1 for e in entries.values() if e["actual"] is None)
                say(
                    f"Обзор Aggregator, {month}, {provider}: {len(entries)} вызовов, учтено "
                    f"{spent / 1e6:.4f} USD"
                    + (f" (из них {open_} без ответа, учтены по резерву)" if open_ else "")
                )
    if len(closed) < 30:
        say("Сделок меньше 30: по такому числу нельзя судить, зарабатывает система или ей везёт.")
    if not args.outcomes:
        return 0

    # What would every signal have done, whether or not it was traded.
    candidates: dict[str, dict] = {}
    for record in records:
        if record["event"] == "scan":
            for raw in record.get("candidates", []):
                candidates[raw["id"]] = raw

    def picks(decision: dict) -> list[str]:
        return decision.get("picks") or ([decision["candidate_id"]] if decision.get("candidate_id") else [])

    chosen = {
        cid
        for d in decisions
        if d["decision"]["action"] == "TRADE" and d["decision"]["source"] == "LLM"
        for cid in picks(d["decision"])
    }
    control = {
        cid
        for d in decisions
        for cid in (d.get("control_picks") or ([d["control_pick"]] if d.get("control_pick") else []))
    }
    reviewed = {
        cid for d in decisions if d["decision"]["source"] == "LLM" for cid in d.get("candidate_ids", [])
    }
    async with aiohttp.ClientSession(trust_env=True) as session:
        api = Evedex(session)
        fee = float(await api.taker_fee())
        results: dict[str, float] = {}
        for cid, candidate in candidates.items():
            start = int(candidate["signal_time"])
            end = start + (int(candidate["timeout_minutes"]) + 10) * 60_000
            bars = await api.candles(candidate["symbol"], "5m", start, end)
            result = outcome(candidate, bars, cost_fraction=2 * (fee + 0.0002))
            if result["r"] is not None:
                results[cid] = result["r"]

    def line(name: str, ids: set[str]) -> None:
        values = [results[i] for i in ids if i in results]
        if values:
            say(
                f"  {name}: {len(values)} сигналов, средний результат {sum(values) / len(values):+.3f} R, "
                f"в плюс {sum(v > 0 for v in values)}"
            )
        else:
            say(f"  {name}: завершённых сигналов нет")

    say("Условный результат сигналов по свечам (R = один риск на сделку, с комиссиями):")
    line("все сигналы алгоритма", set(candidates))
    line("выбор алгоритма без нейросети", control)
    line("нейросеть разрешила", chosen)
    line("нейросеть отклонила", reviewed - chosen)
    reviewed_by_aggregator: dict[str, set[str]] = {"ALLOW": set(), "VETO": set(), "DEFER": set()}
    for record in records:
        if record["event"] == "aggregator_review":
            for review in record.get("reviews", []):
                reviewed_by_aggregator.setdefault(review["decision"], set()).add(review["id"])
    if any(reviewed_by_aggregator.values()):
        line("Aggregator разрешил (ALLOW)", reviewed_by_aggregator["ALLOW"])
        line("Aggregator отклонил (VETO)", reviewed_by_aggregator["VETO"])
        line("Aggregator отложил (DEFER)", reviewed_by_aggregator["DEFER"])
    by_risk: dict[bool, set[str]] = {True: set(), False: set()}
    refusals: Counter = Counter()
    for record in records:
        if record["event"] == "risk_review":  # versions before 0.7.0: shadow verdicts
            for verdict in record.get("decisions", []):
                by_risk[bool(verdict.get("approved"))].add(verdict["id"])
                refusals.update(verdict.get("rejection_reasons") or [])
        elif record["event"] == "risk_decision":  # the Risk Manager's decision on an entry
            by_risk[bool(record.get("approved"))].add(record["candidate"])
            refusals.update(
                record.get("rejection_reasons") or ([record["refused"]] if "refused" in record else [])
            )
    if by_risk[True] or by_risk[False]:
        line("Risk Manager одобрил", by_risk[True] - by_risk[False])
        line("Risk Manager отклонил", by_risk[False])
        if refusals:
            say(
                "  причины отказов Risk Manager: "
                + ", ".join(f"{name} ×{n}" for name, n in refusals.most_common(6))
            )
    allocations = [r for r in records if r["event"] == "macro_allocation"]
    if allocations:
        last = allocations[-1]["allocation"]
        say(
            f"  Macro Strategist, последнее решение ({allocations[-1]['ts'][:16]}): режим {last['regime']}, "
            f"резерв {last['stable_reserve_pct']:.0%}, доли {last['strategy_weights']}, "
            f"источник: {last.get('origin')}"
        )
    setups = sorted({c["setup"] for c in candidates.values()})
    if len(setups) > 1:
        for setup in setups:
            line(f"правило {setup}", {cid for cid, c in candidates.items() if c["setup"] == setup})
    traded = {r.get("candidate_id") for r in closed}
    line("реально сторгованные", {cid for cid in traded if cid})
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="kairos_autotrader")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser(
        "check", help="read-only check of market data, signing and (with keys) the account"
    )
    check.add_argument("--config", type=Path)
    add_secret_args(check)

    scan = sub.add_parser("scan", help="one scan of the market; no orders")
    scan.add_argument("--config", type=Path)
    scan.add_argument("--runtime-dir", type=Path)
    scan.add_argument("--no-llm", action="store_true")
    add_secret_args(scan)

    run = sub.add_parser("run", help="trade in a loop")
    run.add_argument("--mode", choices=("paper", "live"), required=True)
    run.add_argument("--runtime-dir", type=Path, required=True)
    run.add_argument("--config", type=Path)
    run.add_argument("--no-llm", action="store_true", help="algorithm only, for comparison")
    run.add_argument("--confirm-real-orders", action="store_true")
    run.add_argument(
        "--new-session", action="store_true", help="reset the session loss baseline and any halt"
    )
    run.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    add_secret_args(run)

    report = sub.add_parser("report", help="summary of the journal")
    report.add_argument("--runtime-dir", type=Path, required=True)
    report.add_argument("--outcomes", action="store_true", help="also evaluate every signal on later candles")
    report.add_argument("--config", type=Path, help="with executor execution_engine: its closed trades too")

    extra = system_cli.add_parsers(sub)

    args = parser.parse_args(argv)
    command = {"check": cmd_check, "scan": cmd_scan, "run": cmd_run, "report": cmd_report, **extra}[
        args.command
    ]
    try:
        return asyncio.run(command(args))
    except KeyboardInterrupt:
        return 130
    except (ConfigError, SecretError, EvedexError, EngineError, TextScoutsError, OSError) as exc:
        say(f"Остановлено: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
