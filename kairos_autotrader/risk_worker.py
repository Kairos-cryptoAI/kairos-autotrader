"""Decides one entry with the real Kairos Risk Manager (``LiveRiskEvaluator``).

Executed as a script by the Quant Scouts repository's own Python environment, with the
Risk Manager's source on the import path: the venue measurement needs Quant Scouts' code
(and its numpy), the decision needs the Risk Manager's, and both lock the same
``kairos-core``. The two must run in one process because the evaluator accepts a venue
measurement only while it is at most two seconds old.

Unchanged code used:

* Quant Scouts: ``parse_binance_book``, ``parse_evedex_book``, ``build_venue_quality`` with
  the EVEDEX PROD profile and its closed ``PROD_SYMBOL_MAP`` — one ``VenueQualityV1`` from
  the two public order books, measured here, just before the decision;
* Risk Manager: ``LiveRiskEvaluator.evaluate`` with ``LiveRiskScopeV1`` and
  ``LiveRiskLimitsV1``: every admission gate, the sizing and the ``RiskTradeDecisionV1``.

The two verifiers the evaluator requires are this runner's, registered by it as the
composer of a LIVE deployment:

* ``RunnerAccountVerifier`` turns the account evidence the runner read from EVEDEX with
  its authenticated session (balance, positions with their protective stops, open orders)
  and its own trade ledger into ``AccountSnapshotV2`` and the verified metrics. It refuses
  stale or future evidence, a resting order, a pending entry and every position without
  Kairos lineage. Its trust boundary is the runner process: the evidence arrives on this
  process's standard input from the runner that read it.
* ``RunnerInputVerifier`` vouches for the venue measurement made in this process and for
  the review and allocation as validated contracts.

argv:   <quant-scouts source> <risk-manager source>
stdin:  {"review": <CandidateReviewV1>, "allocation": <StrategicAllocation> | null,
         "account": {...}, "scope": {...}, "limits": {...},
         "venue": {"taker_fee_bps", "assessed_notional_usd"}}
stdout: a marker line followed by
        {"evaluation": <LiveRiskEvaluationV1>} | {"refused": "..."}
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from typing import Any

MARKER = "@@KAIROS-WORKER@@"
BINANCE_DEPTH = "https://fapi.binance.com/fapi/v1/depth"
EVEDEX_DEPTH = "https://exchange-api.evedex.com/api/market/{symbol}/deep"
MAX_EVIDENCE_AGE_MS = 5_000


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _measure(signal_symbol: str, venue_symbol: str, *, taker_fee_bps: float, assessed: float):
    """One PROD ``VenueQualityV1`` from both public books, fetched now."""
    import aiohttp
    from kairos_core.enums import EvedexProfile
    from kairos_quant.runtime_venue_gate import VenueGatePolicy, build_venue_quality
    from kairos_quant.venue_comparison import parse_binance_book, parse_evedex_book

    async with aiohttp.ClientSession(trust_env=True, timeout=aiohttp.ClientTimeout(total=10)) as session:

        async def get(url: str, params: dict | None = None):
            started = time.perf_counter()
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                payload = await response.json(content_type=None)
            return payload, (time.perf_counter() - started) * 1000

        (evedex_raw, evedex_ms), (binance_raw, binance_ms) = await asyncio.gather(
            get(EVEDEX_DEPTH.format(symbol=venue_symbol)),
            get(BINANCE_DEPTH, {"symbol": signal_symbol, "limit": "100"}),
        )
    evedex = parse_evedex_book(venue_symbol, evedex_raw, latency_ms=evedex_ms)
    binance = parse_binance_book(signal_symbol, binance_raw, latency_ms=binance_ms)
    return build_venue_quality(
        binance=binance,
        evedex=evedex,
        venue_symbol=venue_symbol,
        observed_at_ms=_now_ms(),
        latency_ms=math.ceil(max(evedex_ms, binance_ms)),
        policy=VenueGatePolicy(assessed_notional_usd=assessed, taker_fee_bps=taker_fee_bps),
        profile=EvedexProfile.PROD,
        source="kairos-quant-scouts",
    )


class RunnerAccountVerifier:
    """Account evidence read by the runner, as ``VerifiedLiveAccountV1``."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id

    def verify(self, raw_evidence: object, *, source_id: str, expected_scope):
        from kairos_core.contracts import AccountSnapshotV2, ExitPlanV1, PositionSnapshotV2, canonical_sha256
        from kairos_core.enums import EvedexProfile, Side, TradeLifecycleState, TradingMode
        from kairos_risk.live import VerifiedLiveAccountV1

        if source_id != self.source_id or not isinstance(raw_evidence, dict):
            raise ValueError("account evidence comes from another source")
        evidence = raw_evidence
        captured = int(evidence["captured_at_ms"])
        age = _now_ms() - captured
        if not 0 <= age <= MAX_EVIDENCE_AGE_MS:
            raise ValueError("account evidence is stale or from the future")
        if evidence.get("pending_entry"):
            raise ValueError("an entry is pending; its exposure is not represented")
        if evidence.get("open_orders"):
            raise ValueError("resting orders without Kairos lineage are on the account")
        if evidence["account_id"] != expected_scope.account_id:
            raise ValueError("account evidence belongs to another account")

        positions, open_risk, notional, strategy_notional, unrealized = [], 0.0, 0.0, 0.0, 0.0
        for raw in evidence["positions"]:
            trade = raw.get("trade")
            if not trade:
                raise ValueError(f"position {raw['venue_symbol']} has no Kairos lineage")
            side = Side(raw["side"])
            quantity, entry, mark = (
                float(raw["quantity"]),
                float(raw["entry_price"]),
                float(raw["mark_price"]),
            )
            sign = 1.0 if side is Side.LONG else -1.0
            exit_plan = ExitPlanV1(
                stop_price=float(trade["stop"]),
                target_price=float(trade["take"]),
                max_holding_ms=int(trade["max_holding_ms"]),
            )
            stop_id = raw.get("stop_order_id")
            pnl = sign * (mark - entry) * quantity
            positions.append(
                PositionSnapshotV2(
                    venue_symbol=raw["venue_symbol"],
                    side=side,
                    signed_quantity=sign * quantity,
                    entry_price=entry,
                    mark_price=mark,
                    leverage=1.0,
                    unrealized_pnl_usd=pnl,
                    strategy_id=trade["strategy_id"],
                    strategy_revision=trade["strategy_revision"],
                    intent_id=trade["intent_id"],
                    risk_decision_id=trade["risk_decision_id"],
                    trade_id=trade["risk_trade_id"],
                    lifecycle_state=TradeLifecycleState.ACTIVE if stop_id else TradeLifecycleState.PROTECTING,
                    entry_client_order_id=trade["entry_order_id"],
                    stop_client_order_id=stop_id,
                    first_fill_at_ms=int(trade["first_fill_ms"]),
                    timeout_at_ms=int(trade["first_fill_ms"]) + int(trade["max_holding_ms"]),
                    exit_plan=exit_plan,
                )
            )
            open_risk += quantity * abs(entry - exit_plan.stop_price)
            notional += quantity * mark
            unrealized += pnl
            if trade["strategy_id"] == expected_scope.strategy_id:
                strategy_notional += quantity * mark
        realized = float(evidence["daily_realized_pnl_usd"])
        # Unrealized profit is never counted as equity; an unrealized loss always is.
        equity = float(evidence["balance_usd"]) + min(0.0, unrealized)
        account = AccountSnapshotV2(
            source=self.source_id,
            trading_mode=TradingMode.LIVE,
            evedex_profile=EvedexProfile.PROD,
            account_id=evidence["account_id"],
            equity_usd=equity,
            available_balance_usd=float(evidence["available_usd"]),
            margin_used_usd=max(0.0, float(evidence["balance_usd"]) - float(evidence["available_usd"])),
            durable_day_start_equity_usd=float(evidence["day_start_equity_usd"]),
            durable_peak_equity_usd=max(float(evidence["peak_equity_usd"]), equity),
            daily_realized_pnl_usd=realized,
            unrealized_pnl_usd=unrealized,
            total_open_risk_usd=open_risk,
            positions=tuple(positions),
            open_orders=(),
            captured_at_ms=captured,
            reconciliation_seq=int(evidence["reconciliation_seq"]),
            reconciled=True,
            reconciliation_detail=str(evidence.get("detail", ""))[:2_048],
        )
        return VerifiedLiveAccountV1(
            source_id=self.source_id,
            scope_sha256=expected_scope.scope_sha256,
            evidence_sha256=canonical_sha256(evidence),
            as_of_ms=captured,
            account=account,
            aggregate_open_risk_usd=open_risk,
            aggregate_open_notional_usd=notional,
            reserved_open_risk_usd=0.0,
            reserved_open_notional_usd=0.0,
            strategy_open_notional_usd=strategy_notional,
            strategy_reserved_notional_usd=0.0,
            daily_adverse_loss_usd=max(0.0, -realized) + max(0.0, -unrealized),
        )


class RunnerInputVerifier:
    """Vouches for the venue measurement made in this process and the validated inputs."""

    SOURCE_ID = "kairos-autotrader-inputs"

    def __init__(self, venue) -> None:
        self.venue = venue

    def verify(self, raw_proof, *, expected_scope, review, venue, allocation, system_mode, decided_at_ms):
        from kairos_core.contracts import canonical_sha256
        from kairos_risk.live import VerifiedLiveInputsV1

        if venue.measurement_id != self.venue.measurement_id or raw_proof != self.venue.measurement_id:
            raise ValueError("the venue measurement was not made by this process")
        return VerifiedLiveInputsV1(
            source_id=self.SOURCE_ID,
            scope_sha256=expected_scope.scope_sha256,
            evidence_sha256=canonical_sha256(
                {
                    "review": review.review_id,
                    "venue": venue.measurement_id,
                    "allocation": allocation is not None,
                }
            ),
            review_sha256=canonical_sha256(review.model_dump(mode="json")),
            venue_sha256=canonical_sha256(venue.model_dump(mode="json")),
            allocation_sha256=canonical_sha256(allocation.model_dump(mode="json")) if allocation else None,
            system_mode=system_mode,
            observed_at_ms=venue.observed_at_ms,
        )


def venue_symbol_for(signal_symbol: str) -> str | None:
    """The EVEDEX PROD instrument Quant Scouts approves for a Binance signal symbol."""
    from kairos_quant.runtime_venue_gate import PROD_SYMBOL_MAP

    return PROD_SYMBOL_MAP.get(signal_symbol)


def decide(request: dict, venue) -> dict:
    """The Risk Manager's decision on the request with a venue measurement made by this process."""
    from kairos_core.contracts import CandidateReviewV1, StrategicAllocation, canonical_sha256
    from kairos_core.enums import SystemMode
    from kairos_quant.runtime_venue_gate import PROD_SYMBOL_MAP
    from kairos_risk.live import LiveRiskEvaluator, LiveRiskLimitsV1, LiveRiskScopeV1

    review = CandidateReviewV1.model_validate(request["review"])
    allocation = (
        StrategicAllocation.model_validate(request["allocation"]) if request.get("allocation") else None
    )
    scope_in, account = request["scope"], request["account"]
    limits = LiveRiskLimitsV1(**request["limits"])
    instruments = set(scope_in["instruments"])
    scope = LiveRiskScopeV1(
        context_sha256=canonical_sha256(scope_in["context"]),
        source_set_sha256=canonical_sha256(scope_in["sources"]),
        policy_sha256=canonical_sha256(scope_in["policy"]),
        risk_limits_sha256=canonical_sha256(limits.model_dump(mode="json")),
        # The account the runner trades, never the one the evidence names.
        account_id=scope_in["account_id"],
        strategy_id=review.intent.strategy_id,
        strategy_revision=review.intent.strategy_revision,
        approved_prod_symbols=tuple((s, v) for s, v in PROD_SYMBOL_MAP.items() if v in instruments),
    )
    evaluator = LiveRiskEvaluator(
        scope,
        limits,
        account_verifiers={account["source_id"]: RunnerAccountVerifier(account["source_id"])},
        input_verifier=RunnerInputVerifier(venue),
    )
    result = evaluator.evaluate(
        account,
        source_id=account["source_id"],
        review=review,
        venue=venue,
        allocation=allocation,
        decided_at_ms=_now_ms(),
        system_mode=SystemMode.NORMAL,
        raw_inputs_evidence=venue.measurement_id,
    )
    return {"evaluation": result.model_dump(mode="json")}


async def evaluate(request: dict) -> dict:
    signal_symbol = request["review"]["intent"]["symbol"]
    venue_symbol = venue_symbol_for(signal_symbol)
    if venue_symbol is None or venue_symbol not in set(request["scope"]["instruments"]):
        return {"refused": f"{signal_symbol} is outside Quant Scouts' approved PROD symbol map"}
    venue = await _measure(
        signal_symbol,
        venue_symbol,
        taker_fee_bps=float(request["venue"]["taker_fee_bps"]),
        assessed=float(request["venue"]["assessed_notional_usd"]),
    )
    return decide(request, venue)


def main() -> int:
    # argv[1] replaces this script's own directory on the import path, so nothing from the
    # runner package can shadow the system's imports.
    sys.path[0] = sys.argv[1]
    sys.path.insert(1, sys.argv[2])
    request = json.load(sys.stdin)
    try:
        answer: dict[str, Any] = asyncio.run(evaluate(request))
    except Exception as exc:  # noqa: BLE001 - reported to the caller, which fails closed
        answer = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}
    sys.stdout.write("\n" + MARKER + json.dumps(answer, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
