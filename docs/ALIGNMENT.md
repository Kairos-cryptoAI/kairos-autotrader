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
| `kairos-core` (`ff3af75`) | a pinned dependency of the runner: `StrategyIntentV1` and `CandidateRouteV1` are the objects passed between scan, review and entry (`signals.Routed`); every Aggregator review is a validated `CandidateReviewV1` on a `DecisionContextV1` with `MarketSnapshot`, `ClosedBarEventV1`, `SentimentSignal` and `StrategicAllocation` sources | no `RiskTradeDecisionV1`, `VenueQualityV1` | steps 3-4 |
| `kairos-strategy-engine` (`7096bfc`, branch `hourly-trend-strict-adapter-20261112`) | `hourly_trend.evaluate_hourly_trend_closed_bars` produces every live candidate as a `StrategyIntentV1` on closed Binance bars, through the shared `_strict_intent`, pinned by `hourly_trend_source_identity`; `candle_to_closed_bar` rebuilds each cached bar and its hash is compared; `registry.generate_sleeve_intents` on EVEDEX bars in shadow | no rule code. The package is isolated, not registered | registration is a separate source-pinned step (it changes eleven fingerprints) |
| `kairos-quant-scouts` (`647c8b5`) | `long_gap_recovery.parse_page` accepts every Binance kline page and gives each bar its `ClosedBarEventV1` identity; `BinanceFuturesCollector` reads funding and the open-interest hour, `SnapshotBuilder` builds the `MarketSnapshot` before every decision | pages and the order book come from public REST, not from the collector's streams: liquidation totals stay at zero; the entry drift check stands in for `VenueQualityV1` | the collector's stream and `VenueQualityV1` |
| `kairos-text-scouts` (`a8dbd88`) | full pipeline except the provider call | the model call is made by `text_scouts.model_extract`, not `kairos-llm` | provider call through the gateway |
| `kairos-router` (`a373e62`) | `SignalWindow` and `CandidateRouterPolicy.build` with `RouterSettings` defaults: every candidate is a real `CandidateRouteV1` | the symbol bucketing of text and the time checks of `RouterService._candidate_rejection` are restated in `router_worker.py` (the pinned service refuses LIVE by configuration) | the service path once LIVE routing exists upstream |
| `kairos-aggregator` (`34963c2`, branch `bounded-review-prompt-20261113`) | `CandidateContextStore` freezes the `DecisionContextV1`, `CandidateReviewBrain.review_with_context` reviews it and returns the `CandidateReviewV1`; `compile_decision_context` bounds a long input declaration in the prompt | the review runs in a worker per scan, not in `CandidateReviewService` on the bus: receipt clocks of cached bars are the worker's, as after a service restart | the service once the runner publishes to the bus |
| `kairos-macro-strategist` (`1f0d262`) | prompt, `build_macro_context`, `AllocationOutput`, defensive fallback, `ShockDetector` | regime hint re-stated (bound method of the service); scheduling is the runner's | unchanged apart from the gateway |
| `kairos-risk-manager` (`d8c031e`) | `RiskSettings`, `_allocation_gate`, `_size_quantity`, `_economics`, `_position_stop_risk` of the PAPER pipeline | account gate re-stated; `kairos_risk/live.py` (`LiveRiskEvaluator`, the pure LIVE evaluation with injected account and input verifiers) is **not used** | `LiveRiskEvaluator.evaluate` with verifiers backed by the runner's account reads |
| `kairos-llm` (`d91566b`) | the Aggregator review calls `BudgetedLLMGateway.complete` with the architecture-owned workload routes (`gpt-6-luna` medium, `gpt-6.1-sol` high on CONFLICT) and the registered price table | the budget is `FileUsageBudget` in `aggregator_worker.py`, a JSON ledger per runtime directory in place of the Postgres ledger; `brain.ask_llm`, `text_scouts.model_extract` and the Macro call still call the provider directly | the other three calls through the gateway |
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
   Intent, route, context and review are the contracts themselves; a local shape still
   stands in for the risk decision (see the table).
3. *External venue mutations must be prepared in the execution journal before submission.*
   `LiveBroker.enter` records `pending_entry` in the state file first, which serves the
   same purpose for one process but is not the execution journal.
4. *Provider and model selection belong to `kairos-llm`.* True for the Aggregator review.
   The runner's own probability review, Text Scouts' extraction and the Macro call still
   select `llm_model` themselves.

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
   - 2b, done 2026-11-13: every Aggregator review is `CandidateReviewBrain.review_with_context`
     on a `DecisionContextV1` from `CandidateContextStore.build`, through
     `BudgetedLLMGateway`. The runner's own review context, prompt choice and model call
     for this review are deleted. What it took:
     - `kairos-aggregator` `34963c2`: a declaration of more than 1 440 input bars (an
       hourly_trend intent declares 30 240, about 2 MB) is rendered into the prompt as its
       count and its first and last hash; shorter intents get byte-identical prompts;
     - a `MarketSnapshot` per signal symbol is built by Quant Scouts' collector and
       snapshot builder 10-55 s before each boundary, so that it exists before the
       decision and lives (120 s) until the review ends;
     - the newest 60 closed bars the rules read, the routed Text Scouts signals with their
       receipt clocks and the Macro allocation (with its receipt clock, from this version
       on) are the other sources; the store decides what is available;
     - a monthly budget per runtime directory (`review_budget_usd`, at most the 12 USD
       kairos-llm registers), reserved before and committed after each call.
     Checked: against a local stand-in for the provider, with the real worker, store,
     brain, gateway and OpenAI SDK, a NORMAL route got ALLOW from `gpt-6-luna` and a
     CONFLICT route VETO from `gpt-6.1-sol`, both with provenance and committed budget,
     and the 30 240-bar intent reached the model bounded, under 80 KB. Without market
     evidence, with a snapshot produced after the decision, with a forged bar or without
     budget the review is a DEFER and the model is never asked.
     On live public data, paper broker, the same local stand-in: for the 2026-11-13
     03:15 UTC decision ten snapshots were built 46 s before the boundary, the scan ended
     25 s after it with four routed candidates, and the four reviews (prompts of about
     67 KB each, 60 closed bars included) ended 30 s before the route deadline with the
     budget committed. The ALLOW answers came from the stand-in, not from a model.
   - Open: the review still runs in a worker per scan, not in `CandidateReviewService` on
     the bus; three other model calls do not use the gateway; in 30 live scans before 2b
     the Aggregator's prompt answered ALLOW 3, VETO 15 and DEFER 90 times, so making it
     the deciding reviewer (`review_mode="aggregator"`) would cut the number of trades.
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
