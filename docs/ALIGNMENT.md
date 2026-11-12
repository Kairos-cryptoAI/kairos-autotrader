# Alignment with the Kairos repositories — 2026-11-12

This file records, repository by repository, what the live runner uses unchanged, what
it currently re-states or duplicates, and the order in which the duplicates are removed.
It is a map of work, not a readiness claim. Revisions are the `main` heads on this date.

## Correction of an earlier reading

`StrategyIntentV1.venue` is `Literal["BINANCE_UM"]`
(`kairos-core/kairos_core/contracts/strategy.py:181`). This is the design, not an
obstacle: in Kairos the signal is computed on complete Binance USD-M one-minute bars and
the order is placed on EVEDEX, with `VenueQualityV1` measuring the basis, spread and depth
between the two. The contracts already carry the real venue: `RiskTradeDecisionV1` and
`AccountSnapshotV2` accept `TradingMode.LIVE` with `EvedexProfile.PROD`
(`contracts/paper.py:41`, `:305`). No core contract has to change for EVEDEX execution.

What rejects a real venue today is service configuration, not the contracts:
`kairos-risk-manager/kairos_risk/config.py` and
`kairos-execution-engine/kairos_execution/config.py:87` raise on `TradingMode.LIVE`, and
`kairos-deploy/docker-compose.live.yml` is a deliberately inert profile.

## Repository map

| repository (revision) | used unchanged today | re-stated or duplicated in the runner | target |
| --- | --- | --- | --- |
| `kairos-core` (`ff3af75`) | `SentimentSignal`, `StrategicAllocation`, enums — inside workers only | `strategy.Candidate` stands in for `StrategyIntentV1`; review context stands in for `DecisionContextV1`; no `CandidateRouteV1`, `CandidateReviewV1`, `RiskTradeDecisionV1` | the runner passes the real contracts between layers |
| `kairos-strategy-engine` (`9ff13e7`) | `registry.generate_sleeve_intents` in shadow, on EVEDEX 1m bars | `strategy.analyse` (two hourly rules) is not a registered sleeve | rules become a sleeve; intents come from `runtime.generate_runtime_strategy_intents` on Binance bars |
| `kairos-quant-scouts` (`647c8b5`) | nothing | market data is read from EVEDEX by `evedex.py`; no `ClosedBarEventV1`; spread and age checks stand in for `VenueQualityV1` | closed Binance bars and `VenueQualityV1` from this repository's collectors |
| `kairos-text-scouts` (`a8dbd88`) | full pipeline except the provider call | the model call is made by `text_scouts.model_extract`, not `kairos-llm` | provider call through the gateway |
| `kairos-router` (`a373e62`) | `SignalWindow`, `classify_candidate`, time-to-live value | `CandidateRouterPolicy.build` is not called (needs `StrategyIntentV1`) | real `CandidateRouteV1` |
| `kairos-aggregator` (`8f96df4`) | review prompt, `CandidateReviewOutput`, `materially_opposes_candidate` | `CandidateReviewBrain.review_with_context` and `CandidateContextStore` are not called; context is assembled in `aggregator_worker.py` | real `DecisionContextV1` and `CandidateReviewV1` |
| `kairos-macro-strategist` (`1f0d262`) | prompt, `build_macro_context`, `AllocationOutput`, defensive fallback, `ShockDetector` | regime hint re-stated (bound method of the service); scheduling is the runner's | unchanged apart from the gateway |
| `kairos-risk-manager` (`d8c031e`) | `RiskSettings`, `_allocation_gate`, `_size_quantity`, `_economics`, `_position_stop_risk` of the PAPER pipeline | account gate re-stated; `kairos_risk/live.py` (`LiveRiskEvaluator`, the pure LIVE evaluation with injected account and input verifiers) is **not used** | `LiveRiskEvaluator.evaluate` with verifiers backed by the runner's account reads |
| `kairos-llm` (`d91566b`) | nothing | `brain.ask_llm` and `text_scouts.model_extract` call the provider directly; no workload routing, budget reservation or call evidence | `LLMGateway.complete` per workload |
| `kairos-execution-engine` (`1e8afe6`) | nothing; `evedex.py` signing is fixture-tested against the same pinned SDK | whole order path: `evedex.py`, `engine.LiveBroker`, file state | API-key order transport contributed as an adapter; the runner keeps only orchestration |
| `kairos-persistence` (`506628f`) | nothing | `engine.Store` (state file and journal) | durable journal for venue effects |
| `kairos-backtest` (`da12378`) | nothing | `system_cli.evaluate` replays candidates on minute bars | replay through the shared generators |
| `kairos-deploy` (`3684dd8`) | nothing | `start-live.cmd` | a runner profile next to the existing ones |
| `kairos` (`f545141`) | conventions: `CONTRIBUTING.md`, toolchain, checks | — | manifest entry, architecture section |

## Rules of `CONTRIBUTING.md` the runner does not meet yet

1. *Do not bypass Risk Manager validation.* With `risk_mode="autotrader"` (the default) an
   entry is sized by `engine.size_position` alone. `risk_mode="kairos"` requires Risk
   approval; it becomes the default once `LiveRiskEvaluator` replaces the re-stated gate.
2. *Do not copy a `kairos-core` message definition into a service.* Nothing is copied, but
   local shapes stand in for five contracts (see the table).
3. *External venue mutations must be prepared in the execution journal before submission.*
   `LiveBroker.enter` records `pending_entry` in the state file first, which serves the
   same purpose for one process but is not the execution journal.
4. *Provider and model selection belong to `kairos-llm`.* The runner selects the model.

## Order of removal

Dependency order, as `CONTRIBUTING.md` prescribes: contracts first, consumers after.

1. **Signals.** Closed Binance USD-M 1m bars (Quant Scouts collector) to the Strategy
   Engine; the runner's two rules are contributed as a sleeve in `kairos-strategy-engine`
   with parity fixtures; `strategy.analyse` and `strategy.Candidate` are deleted.
2. **Routing and review.** `CandidateRouterPolicy.build`, `CandidateContextStore.build`
   and `CandidateReviewBrain.review_with_context`; `aggregator_worker.py`'s own context
   assembly is deleted. Provider calls move to `LLMGateway`.
3. **Risk.** `LiveRiskEvaluator` with account and instrument verifiers; the re-stated
   account gate in `system_rules_worker.py` and `engine.size_position` are deleted.
4. **Execution.** The runner executes a `RiskTradeDecisionV1`; the API-key transport is
   proposed to `kairos-execution-engine`.
5. **State.** The execution journal replaces `pending_entry`.

Each step lands with its tests and removes the code it replaces in the same change.

## Instrument universe

Kairos trades five symbols, signal `BTCUSDT`…`XRPUSDT` mapped to EVEDEX instruments. The
runner's configured list has ten EVEDEX instruments chosen by spread. Step 1 narrows the
traded set to instruments that exist on both venues; instruments without a Binance
reference stay observation-only.
