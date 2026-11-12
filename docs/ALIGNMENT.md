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
| `kairos-core` (`ff3af75`) | a pinned dependency of the runner: `StrategyIntentV1` and `CandidateRouteV1` are the objects passed between scan, review and entry (`signals.Routed`); `ClosedBarEventV1`, `SentimentSignal`, `StrategicAllocation`, enums inside workers | review context stands in for `DecisionContextV1`; no `CandidateReviewV1`, `RiskTradeDecisionV1`, `MarketSnapshot`, `VenueQualityV1` | the remaining contracts, steps 2b-4 |
| `kairos-strategy-engine` (`7096bfc`, branch `hourly-trend-strict-adapter-20261112`) | `hourly_trend.evaluate_hourly_trend_closed_bars` produces every live candidate as a `StrategyIntentV1` on closed Binance bars, through the shared `_strict_intent`, pinned by `hourly_trend_source_identity`; `candle_to_closed_bar` rebuilds each cached bar and its hash is compared; `registry.generate_sleeve_intents` on EVEDEX bars in shadow | no rule code. The package is isolated, not registered | registration is a separate source-pinned step (it changes eleven fingerprints) |
| `kairos-quant-scouts` (`647c8b5`) | `long_gap_recovery.parse_page` accepts every Binance kline page and gives each bar its `ClosedBarEventV1` identity | pages come from public REST through the runner's cache (`binance_bars.py`), not from the collector's durable stream; no `MarketSnapshot`; the entry drift check stands in for `VenueQualityV1` | the collector's stream, `MarketSnapshot` and `VenueQualityV1` |
| `kairos-text-scouts` (`a8dbd88`) | full pipeline except the provider call | the model call is made by `text_scouts.model_extract`, not `kairos-llm` | provider call through the gateway |
| `kairos-router` (`a373e62`) | `SignalWindow` and `CandidateRouterPolicy.build` with `RouterSettings` defaults: every candidate is a real `CandidateRouteV1` | the symbol bucketing of text and the time checks of `RouterService._candidate_rejection` are restated in `router_worker.py` (the pinned service refuses LIVE by configuration) | the service path once LIVE routing exists upstream |
| `kairos-aggregator` (`8f96df4`) | review prompt, `CandidateReviewOutput`, `materially_opposes_candidate` | `CandidateReviewBrain.review_with_context` and `CandidateContextStore` are not called; context is assembled in `aggregator_worker.py` from the real route | real `DecisionContextV1` and `CandidateReviewV1` (step 2b) |
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
2. *Do not copy a `kairos-core` message definition into a service.* Nothing is copied.
   `StrategyIntentV1` and `CandidateRouteV1` are used as they are; local shapes still
   stand in for the review context, the review and the risk decision (see the table).
3. *External venue mutations must be prepared in the execution journal before submission.*
   `LiveBroker.enter` records `pending_entry` in the state file first, which serves the
   same purpose for one process but is not the execution journal.
4. *Provider and model selection belong to `kairos-llm`.* The runner selects the model.

## Order of removal

Dependency order, as `CONTRIBUTING.md` prescribes: contracts first, consumers after.

1. **Signals.**
   - Done 2026-11-12: the runner's two rules are `kairos_strategy.hourly_trend` in
     `kairos-strategy-engine` (`dc601e4`); `strategy.py` with its indicators, bar builder
     and `analyse` is deleted; `Trader.scan` and `engine-replay` call the engine. Parity:
     a frozen fixture in the engine repository (169 candidates, 1525 decisions), and on
     22 days of real EVEDEX history for ten instruments the engine path on minute bars
     gave the same 1231 candidates, with equal prices, as the old path on the venue's
     15-minute candles.
   - Done 2026-11-12 with step 2a: the signal source is closed Binance USD-M 1m bars.
     Open: they come from public REST klines accepted by the Quant Scouts parser, not
     from the collector's durable stream.
   - Measured 2026-11-12 before any switch (36 days, ten instruments, outcome on EVEDEX
     minute bars, round-trip cost 0.13%): signal on Binance bars 2167 candidates,
     -0.049 R ± 0.024; signal on EVEDEX bars 2142 candidates, -0.080 R ± 0.024; 93% of
     the candidates coincide. The source of the bars does not change the picture: the
     switch is an alignment with the design, not a source of edge.
2. **Routing and review.**
   - 2a, done 2026-11-12: `kairos_strategy.hourly_trend` has a strict adapter in the
     engine (`7096bfc`), the runner depends on `kairos-core`, and a candidate is a
     `CandidateRouteV1` from `CandidateRouterPolicy.build` around the engine's
     `StrategyIntentV1`. `candidates.Candidate`, `fill_quiet_minutes` and the
     classification inside `aggregator_worker.py` are deleted. Checked on live public
     data: at the 2026-11-12 14:00 UTC decision the strict intents of five instruments
     equalled the research candidates of the same rules price for price, the Router gave
     NORMAL routes without text and CONFLICT with opposing text, and rejected a late
     candidate as stale. The running trader, on EVEDEX bars, had the same five sides and
     setups at that decision.
   - 2b, open: `CandidateContextStore.build` and `CandidateReviewBrain.review_with_context`
     through `BudgetedLLMGateway`; `aggregator_worker.py`'s own context assembly and the
     direct provider calls are deleted. Found while preparing it:
     - an hourly_trend intent declares 30 240 input bar hashes (about 2 MB), and
       `compile_decision_context` renders the whole intent identity into the prompt. The
       same holds for the 48 060-bar window of `regime_aligned_right_tail_v1`. The prompt
       has to bound that list upstream, in `kairos-aggregator`, before a model can
       review such a candidate;
     - `DecisionContextV1` requires a `MarketSnapshot` of the signal symbol received
       before the decision, so the runner needs Quant Scouts' snapshot builder on
       Binance order-book data;
     - `BudgetedLLMGateway` needs a durable `LLMUsageBudget`; the system's one is
       Postgres-backed (`kairos-persistence`);
     - the review has to finish before the route's deadline, at most 60 s after the
       decision (`review_timeout_s`); model answers took 11 s at the median and up to
       30 s in the live journal, and the scan itself about 12 s;
     - in 30 live scans the Aggregator's own prompt answered ALLOW 3, VETO 15 and
       DEFER 90 times. Whether it becomes the deciding reviewer is the operator's
       decision (`review_mode`).
3. **Risk.** `LiveRiskEvaluator` with account and instrument verifiers; the re-stated
   account gate in `system_rules_worker.py` and `engine.size_position` are deleted.
4. **Execution.** The runner executes a `RiskTradeDecisionV1`; the API-key transport is
   proposed to `kairos-execution-engine`.
5. **State.** The execution journal replaces `pending_entry`.

Each step lands with its tests and removes the code it replaces in the same change.

## Instrument universe

Kairos trades five symbols, signal `BTCUSDT`…`XRPUSDT` mapped to EVEDEX instruments. The
runner's configured list has ten EVEDEX instruments chosen by spread; all ten have a
Binance USD-M contract, found by the rule `XXXUSD` → `XXXUSDT` (`signals.signal_symbol`).
An instrument whose Binance bars cannot be completed is skipped for that scan and never
traded on guessed data.
