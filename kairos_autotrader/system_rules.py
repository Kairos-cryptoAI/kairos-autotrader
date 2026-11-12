"""Bridge to the real Kairos Macro Strategist and Risk Manager.

Macro: once a day, and again after a price crash, a model is asked with the Macro
Strategist's own prompt for a strategic allocation: market regime (BULL, BEAR or CHOP),
the share of capital kept in reserve, a weight per strategy and a leverage ceiling. The
answer is checked by the Macro Strategist's own schema; anything broken becomes its
defensive allocation (CHOP, everything in reserve), which permits no new trades.

Risk: every candidate is put through the Risk Manager's rules: regime vetoes and strategy
weight from the allocation, total open risk, daily drawdown, loss-at-stop sizing.

Neither can change a candidate's side or exits, and neither places orders.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

from .sibling import SiblingError, git_revision, run_worker, sibling_checkout, sibling_python
from .text_scouts import model_extract

WORKER = Path(__file__).with_name("system_rules_worker.py")
DAY_MS = 86_400_000
MACRO_UNITS = {
    "price": "quote_currency_per_base_unit",
    "volume_24h_usd": "USD",
    "funding_rate_pct": "percent_per_interval",
    "open_interest_usd": "USD",
    "rsi": "index_0_100",
    "atr_pct": "percent_of_price",
    "change_1h_pct": "percent",
    "change_24h_pct": "percent",
    "change_7d_pct": "percent",
    "quant_bias": "categorical_side_from_4h_trend",
}


class RulesError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RulesConfig:
    macro_python: Path = field(default_factory=lambda: sibling_python("kairos-macro-strategist"))
    macro_source: Path = field(default_factory=lambda: sibling_checkout("kairos-macro-strategist"))
    llm_source: Path = field(default_factory=lambda: sibling_checkout("kairos-llm"))
    core_source: Path = field(default_factory=lambda: sibling_checkout("kairos-core"))
    risk_python: Path = field(default_factory=lambda: sibling_python("kairos-risk-manager"))
    risk_source: Path = field(default_factory=lambda: sibling_checkout("kairos-risk-manager"))
    timeout_s: float = 60.0

    def check_macro(self) -> None:
        for path in (
            self.macro_python,
            self.macro_source / "kairos_macro" / "strategist.py",
            self.llm_source / "kairos_llm" / "__init__.py",
            self.core_source / "kairos_core" / "__init__.py",
        ):
            if not path.exists():
                raise RulesError(f"Macro Strategist part not found: {path}")

    def check_risk(self) -> None:
        for path in (self.risk_python, self.risk_source / "kairos_risk" / "paper.py"):
            if not path.exists():
                raise RulesError(f"Risk Manager part not found: {path}")


def revisions(config: RulesConfig) -> dict[str, str]:
    return {"macro": git_revision(config.macro_source), "risk": git_revision(config.risk_source)}


def _run_worker(config: RulesConfig, kind: str, request: dict[str, Any]) -> dict[str, Any]:
    if kind == "macro":
        python = config.macro_python
        arguments = ["macro", str(config.macro_source), str(config.core_source), str(config.llm_source)]
    else:
        python, arguments = config.risk_python, ["risk", str(config.risk_source)]
    try:
        return run_worker(python, WORKER, arguments, request, timeout_s=config.timeout_s)
    except SiblingError as exc:
        raise RulesError(f"{kind}: {exc}") from exc


def bias_of(trend: str) -> str:
    return {"UP": "LONG", "DOWN": "SHORT"}.get(trend, "FLAT")


class MacroDesk:
    """Keeps one current strategic allocation: daily, and again after a price shock."""

    def __init__(
        self,
        store: Any,
        config: RulesConfig,
        strategy_ids: tuple[str, ...],
        *,
        runner=_run_worker,
        extract=model_extract,
    ) -> None:
        self.store, self.config, self.strategy_ids = store, config, tuple(sorted(strategy_ids))
        self._runner, self._extract = runner, extract

    def _state(self) -> dict[str, Any]:
        return self.store.state.setdefault("macro", {})

    def allocation(self, now_ms: int | None = None) -> dict[str, Any] | None:
        """The stored allocation, or None. The Risk Manager applies its own age limit."""
        return self._state().get("allocation")

    def summary(self) -> dict[str, Any] | str:
        allocation = self.allocation()
        if not allocation:
            return "UNAVAILABLE"
        return {
            key: allocation.get(key)
            for key in (
                "regime",
                "stable_reserve_pct",
                "strategy_weights",
                "max_gross_leverage",
                "triggered_by",
                "rationale",
                "produced_at",
                "origin",
            )
        }

    async def refresh(
        self,
        session: aiohttp.ClientSession | None,
        *,
        openai_key: str | None,
        model: str,
        effort: str,
        market: dict[str, dict[str, Any]],
        portfolio: dict[str, Any],
        performance: dict[str, Any],
        now_ms: int | None = None,
    ) -> dict[str, Any] | None:
        """Run the strategist if an allocation is due. Returns the new allocation or None."""
        if not openai_key or session is None or not market:
            return None
        now_ms = now_ms or int(time.time() * 1000)
        state = self._state()
        request = {
            "op": "prepare",
            "allowed_strategy_ids": list(self.strategy_ids),
            "portfolio": portfolio,
            "performance": performance,
            "symbols": sorted(market),
            "units": MACRO_UNITS,
            "biases": {symbol: values.get("quant_bias", "FLAT") for symbol, values in market.items()},
            "market_values": market,
            "changes_1h_pct": {s: v["change_1h_pct"] for s, v in market.items() if "change_1h_pct" in v},
        }
        due = not state.get("allocation") or now_ms - state.get("last_run_ms", 0) >= DAY_MS
        trigger = {"kind": "schedule"} if due else None
        # The shock check needs the detector, so the prepare call is made either way; it is local and cheap.
        request["trigger"] = trigger or {}
        prepared = await asyncio.to_thread(self._runner, self.config, "macro", request)
        if trigger is None and prepared["shocks"]:
            cooldown_ms = int(float(prepared["shock_cooldown_s"]) * 1000)
            if now_ms - state.get("last_shock_ms", 0) >= cooldown_ms:
                trigger = {"kind": "shock_event", "shocks": prepared["shocks"]}
                request["trigger"] = trigger
                prepared = await asyncio.to_thread(self._runner, self.config, "macro", request)
                state["last_shock_ms"] = now_ms
        if trigger is None:
            return None
        state["last_run_ms"] = now_ms
        self.store.save()
        output, usage, problem = await self._extract(
            session,
            api_key=openai_key,
            model=model,
            system=prepared["system"],
            user=prepared["user"],
            effort=effort,
            timeout_s=180.0,
        )
        checked = await asyncio.to_thread(
            self._runner,
            self.config,
            "macro",
            {
                "op": "validate",
                "model_output": output,
                "allowed_strategy_ids": prepared["allowed_strategy_ids"],
                "trigger": trigger["kind"],
            },
        )
        allocation = checked["allocation"]
        state["allocation"] = allocation
        self.store.save()
        self.store.journal(
            "macro_allocation",
            allocation=allocation,
            trigger=trigger,
            regime_hint=prepared["regime_hint"],
            regime_evidence=prepared["regime_evidence"],
            usage=usage,
            problem=problem,
        )
        return allocation


async def risk_review(
    config: RulesConfig,
    *,
    candidates: list[dict[str, Any]],
    account: dict[str, Any],
    allocation: dict[str, Any] | None,
    now_ms: int,
    runner=_run_worker,
) -> dict[str, Any]:
    """The Risk Manager's decision for each candidate. Raises ``RulesError`` if the worker fails."""
    if not candidates:
        return {"decisions": []}
    return await asyncio.to_thread(
        runner,
        config,
        "risk",
        {
            "op": "evaluate",
            "now_ms": now_ms,
            "candidates": candidates,
            "account": account,
            "allocation": allocation,
        },
    )
