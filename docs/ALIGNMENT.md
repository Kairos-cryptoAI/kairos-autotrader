# Alignment with the Kairos repositories — 2026-11-12, updated 2026-11-14

This file records, repository by repository, what the live runner uses unchanged, what
it currently re-states or duplicates, and the order in which the duplicates are removed.
It is a map of work, not a readiness claim. Revisions are the `main` heads on this date.
Updated 2026-11-14: step 4, and the plan beyond the removal of duplicates, which follows
the owner's end goal in `kairos/docs/PHILOSOPHY.md` (branch `docs/philosophy-20261114`).

## Portfolio execution update, 2026-11-19

The current runner delegates all live orders to `kairos_execution.portfolio_broker`.
The Execution Engine owns the portfolio journal, venue transport, account authorization
and settlement. The runner passes each Risk decision to that broker. The single-trade
session and direct runner order paths described in the dated tables below are historical.

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
| `kairos-strategy-engine` (`dc46fde`, branch `hourly-trend-candidates-20261112`; exits on the venue tick since `dc46fde`) | `hourly_trend.evaluate_hourly_trend_closed_bars` produces every live candidate as a `StrategyIntentV1` on closed Binance bars, through the shared `_strict_intent`, pinned by `hourly_trend_source_identity`; `candle_to_closed_bar` rebuilds each cached bar and its hash is compared; `registry.generate_sleeve_intents` on EVEDEX bars in shadow | no rule code. The package is isolated, not registered | registration is a separate source-pinned step (it changes eleven fingerprints) |
| `kairos-quant-scouts` (`38d0c91`, branch `prod-symbol-map-ten-20261113`; `venue_now` since `38d0c91`) | `long_gap_recovery.parse_page` accepts every Binance kline page and gives each bar its `ClosedBarEventV1` identity; `BinanceFuturesCollector` reads funding and the open-interest hour, `SnapshotBuilder` builds the `MarketSnapshot` before every decision | pages and the order book come from public REST, not from the collector's streams: liquidation totals stay at zero; the entry drift check stands in for `VenueQualityV1` | the collector's stream and `VenueQualityV1` |
| `kairos-text-scouts` (`a8dbd88`) | full pipeline except the provider call | the model call is made by `text_scouts.model_extract`, not `kairos-llm` | provider call through the gateway |
| `kairos-router` (`a373e62`) | `SignalWindow` and `CandidateRouterPolicy.build` with `RouterSettings` defaults: every candidate is a real `CandidateRouteV1` | the symbol bucketing of text and the time checks of `RouterService._candidate_rejection` are restated in `router_worker.py` (the pinned service refuses LIVE by configuration) | the service path once LIVE routing exists upstream |
| `kairos-aggregator` (`34963c2`, branch `bounded-review-prompt-20261113`) | `CandidateContextStore` freezes the `DecisionContextV1`, `CandidateReviewBrain.review_with_context` reviews it and returns the `CandidateReviewV1`; `compile_decision_context` bounds a long input declaration in the prompt | the review runs in a worker per scan, not in `CandidateReviewService` on the bus: receipt clocks of cached bars are the worker's, as after a service restart | the service once the runner publishes to the bus |
| `kairos-macro-strategist` (`1f0d262`) | prompt, `build_macro_context`, `AllocationOutput`, defensive fallback, `ShockDetector` | regime hint re-stated (bound method of the service); scheduling is the runner's | unchanged apart from the gateway |
| `kairos-risk-manager` (`d8c031e`) | `LiveRiskEvaluator.evaluate` decides every entry: review, Macro allocation, venue quality, account and daily limits, sizing, `RiskTradeDecisionV1` | in the runner's own path the account and input verifiers are the runner's (`risk_worker.py`) with `require_instrument_rules=False`; in the Execution Engine path the verifiers are the system's (`production_rest_sources`) with instrument rules required, now that exits are on the tick | the Execution Engine path only |
| `kairos-llm` (`d91566b`) | the Aggregator review calls `BudgetedLLMGateway.complete` with the architecture-owned workload routes (`gpt-6-luna` medium, `gpt-6.2-sol` high on CONFLICT) and the registered price table | the budget is `FileUsageBudget` in `aggregator_worker.py`, a JSON ledger per runtime directory in place of the Postgres ledger; `brain.ask_llm`, `text_scouts.model_extract` and the Macro call still call the provider directly | the other three calls through the gateway |
| `kairos-execution-engine` (`39cacba`, branch `live-runner-execution-20261113`) | with `executor="execution_engine"`: the one-trade session (`production_session_cli`) runs the whole production path; not yet usable live (see step 4) | with the default `executor="runner"`: the whole order path, `evedex.py`, `engine.LiveBroker`, file state | the Execution Engine path as the only executor; `evedex.py` and `LiveBroker` deleted |
| `kairos-persistence` (`506628f`) | nothing | `engine.Store` (state file and journal) | durable journal for venue effects |
| `kairos-backtest` (`da12378`) | nothing | `system_cli.evaluate` replays candidates on minute bars | replay through the shared generators |
| `kairos-deploy` (`3684dd8`) | nothing | `start-live.cmd` | a runner profile next to the existing ones |
| `kairos` (`f545141`) | conventions: `CONTRIBUTING.md`, toolchain, checks | — | manifest entry, architecture section |

## Rules of `CONTRIBUTING.md` the runner does not meet yet

1. *Do not bypass Risk Manager validation.* Met from 0.7.0: no entry without an approved
   `RiskTradeDecisionV1`, and the runner trades its quantity or less.
2. *Do not copy a `kairos-core` message definition into a service.* Nothing is copied.
   Intent, route, context, review and risk decision are the contracts themselves.
3. *External venue mutations must be prepared in the execution journal before submission.*
   Met with `executor="execution_engine"`, where every mutation is a journaled effect of
   the Execution Engine. With the default `executor="runner"`, `LiveBroker.enter` records
   `pending_entry` in the state file first, which is not the execution journal.
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
     CONFLICT route VETO from `gpt-6.2-sol`, both with provenance and committed budget,
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
3. **Risk.** Done 2026-11-13: every entry is `LiveRiskEvaluator.evaluate` on the
   Aggregator's `CandidateReviewV1`, the Macro allocation, a PROD `VenueQualityV1` measured
   by Quant Scouts' code just before the decision, and account evidence; the runner trades
   the decision's quantity floored to the venue lot, or nothing. Deleted:
   `engine.size_position`, the re-stated account gate and sizing in
   `system_rules_worker.py`, the shadow risk review, `risk_mode` and `risk_shadow`. The
   Macro Strategist now allocates to `hourly_trend_candidates_v1` itself, and the
   Aggregator review is the deciding reviewer by default, because the evaluator admits
   only an ALLOW `CandidateReviewV1`. Found on the way:
   - the scope's account id was first taken from the evidence, so a foreign account could
     not be told apart; it is now the runner's own account;
   - Quant Scouts' PROD symbol map had five symbols, so HYPE, DOGE, LINK, SUI, AAVE and ZEC
     could get no venue measurement and could not be traded. At the owner's decision the
     map was extended in `kairos-quant-scouts` `6bd59a3` (branch
     `prod-symbol-map-ten-20261113`) to all ten configured instruments; on live public
     books all ten were allowed, basis within 2.1 bps, spread within 5.1 bps;
   - the exit prices of an intent are Binance prices, off the EVEDEX tick, so the
     authenticated instrument-rules mode cannot be used yet.
   Checked on live public data, paper ledger, local model stand-in and a stated test
   allocation (BEAR, weight for the strategy): at the 2026-11-13 15:45 UTC decision four
   candidates were ALLOWed by the stand-in; the Risk Manager refused AAVE and ZEC (outside
   the PROD map) and approved XRP short with 7.264 XRP, worst-case loss 0.25 USD, on a
   venue measurement with basis -3.0 bps and a book 0.5 s old; the runner entered 7.2 XRP
   (lot step 0.1) 43 s after the boundary. Each decision took 1.3-2.6 s. Tests run the real
   evaluator offline on synthetic books for the approval and sixteen refusal cases.
4. **Execution.** Built 2026-11-13/14, not yet usable live. With `executor="execution_engine"`
   in live mode the runner hands the Aggregator review and the Macro allocation to
   `kairos_execution.production_session_cli` and opens nothing else until the session ends;
   the session runs the system's production path for one trade. What it took, by
   repository:
   - `kairos-execution-engine`: the reviewed eleven-pair PROD universe shared by custody,
     lifecycle and capture (`58f569b`); one-command provisioning of the isolated database
     (`deff9a6`); `drain_events` polling instead of durably blocking entries, and the
     transport preflight at start (`8b855bf`); EVEDEX PROD REST sources for account,
     inputs with instrument rules, exits, cancellation and terminal V2 evidence under a
     documented single-position trust model (`edbbf1e`, `docs/REST-SOURCE-TRUST-MODEL.md`);
     owner tools for the arm key, signed arms and evidence receipts (`dc87851`); Telegram
     single-host alerts and their evidence facts (`b8ae425`, `cffc20f`); the automatically
     armed one-trade session, whose arm carries exactly the owner's signed limits
     (`a404651`), its command line (`36c4cf0`) and a fresh venue measurement inside it
     (`39cacba`);
   - `kairos-strategy-engine` `dc46fde`: exits placed on the execution venue's tick, so the
     Risk Manager's instrument-rules mode can be required;
   - `kairos-quant-scouts` `38d0c91`: an on-demand PROD venue measurement, because a
     `VenueQualityV1` lives five seconds;
   - the runner: the tick is passed to the rules (`7b94d81`), a heartbeat for the alert
     monitor (`0dccc1c`), the hand-over to the Execution Engine (`8d38294`).
   Then, on 2026-11-14: three defects that blocked every real REST trade were fixed
   (`4a8c9a4`: the GTC owner binding, the cancellation's Risk decision id, decimals in
   venue JSON); an operator latch (arm, kill, resume), an entry bounded by the two-second
   quote window and an end-to-end stand (`964e49e`); the owner drills for recovery,
   operator control, limits, custody and account pairing, whose facts the system's own
   interpreters accept (`496794b`, `4ae1d24`).
   Checked: the production graph with the REST sources passes every assembly check; owner
   arms and receipts are accepted by the system's verifiers; a fresh measurement comes from
   live public books; a Telegram test message was delivered. On a provisioned database
   against a venue double that speaks EVEDEX's REST and signs with a real EIP-712 key
   (`tests/rest_stand.py`), one armed session enters with `LIMIT_GTC`, places both
   protections, exits by take-profit, by stop-loss or by an early exit, reconciles and has
   its terminal accounting accepted (`KAIROS_REST_STAND=1`, three trades, about 50 s).
   Open before live use, all the owner's: the encrypted wallet keystore and a single-key
   API file, the seven signed technical-canary receipts (one needs an independent security
   reviewer), running the drills on the real database, and `session.json`. The production
   path holds one position at a time and admits only a bounded `LIMIT_GTC` entry.
5. **State.** Done for the Execution Engine path 2026-11-14: its database is the journal of
   closed trades (`kairos_execution.production_report`, `efeeb12`), `report --config`
   reads them from there (`214e330`) and the learning ledger joins them with the runner's
   own. The runner's `Store` keeps orchestration state. `evedex.py` and `LiveBroker` are
   deleted once the Execution Engine path has traded live.

Each step lands with its tests and removes the code it replaces in the same change.

## Beyond the removal: toward the owner's end goal

`kairos/docs/PHILOSOPHY.md` asks for an adaptive trader that decides capital, risk,
leverage, instruments and position management itself inside a fixed owner constitution,
analyses continuously and learns from its own trades, using a professional's tools only
where they measurably help. Steps 6-11 were built on 2026-11-14, each in the repository
that owns the role, with tests, without loosening a constitution limit. "Measured" below
is a replay of the live rules on 30 days of EVEDEX minute history (2 173 signals, costs
included); overlapping signals of one instrument and day are one cluster in the error.

6. **Learning ledger.** `kairos_risk.learning` (`4fb3425`, `034b006`): every closed trade in
   R, per strategy x instrument x regime with fallbacks, evidence weighted (live 1, replay
   0.5), a lower bound at 95% (99% for one instrument). The runner feeds it its own and the
   Execution Engine's closed trades (`050fc71`) and replayed live-rule signals as replay
   evidence (`6214fb9`); `report --ledger` prints it (`8cefbeb`). Measured: the live rules
   have no edge overall (-0.017 +- 0.024 R); per instrument only SOLUSDT is proven (mean
   +0.43 R, lower +0.23 R).
7. **Adaptive risk and capital.** `kairos_risk.adaptive_risk` (`4fb3425`): the share of the
   0.25% ceiling grows only with a proven lower bound, is half without evidence, zero with
   a proven non-positive one, and shrinks with drawdown, a low review priority and venue
   cost. The runner applies it to every entry (`050fc71`); the Execution session takes the
   fraction a candidate carries and can only lower the owner's limit (`56038f1`).
8. **Strategies and instruments.** `kairos_strategy.adaptive` runs beside `hourly_trend` in
   its DRY_RUN observation mode; its intents are measured on later bars and enter the
   ledger as replay evidence, and it trades only after the owner lifts that gate and the
   ledger proves it (`8cefbeb`). Instruments are chosen by the ledger's instrument cells:
   a proven non-positive cell gets zero risk.
9. **Continuous regime and early exits.** The Macro Strategist re-judges the regime every
   `macro_refresh_hours` (6) and after shocks. A newer regime that opposes an open position
   closes it early: the runner's broker at market, an Execution session through its
   reviewed exit path on an `--exit-request` (`e7e2257`, `7dbe72e`, `b58b693`). Each early
   exit is measured against holding the plan; after 20, a negative total switches them off.
   Text Scouts events do not close positions yet.
10. **Stop management.** `kairos_autotrader.management` (`157c3e3`) compares break-even and
    trailing policies with the plan on the same signals and writes
    `management-policy.json`; a policy is used only if its gain is proven at 99% over at
    least 200 signals. Measured: none is (best, trailing from 1.0 R by 0.5 R: +0.016 +-
    0.022 R), so positions keep the plan. With a proven policy, a stop only moves toward
    less risk and the new stop is confirmed on the venue before the old one is cancelled.
    The Execution Engine path keeps its hard plan stop and gets a soft managed stop through
    the early-exit request; moving the hard stop there needs changes to its exit-risk,
    cancellation and reconciliation code and to the REST trust model, under the security
    review.
11. **Partial exits, scaling in, leverage.** Partial exits are management policies (half at
    a profit, then break-even), in lots of the venue's step leaving at least the minimum
    lot; measured, not proven. Adding to an open winner is measured too and not proven
    (`7038f67`: from +0.5 R, 124 add-ons, -0.082 +- 0.093 R), so the runner keeps one
    position per instrument; add-ons on EVEDEX's netted positions would need per-leg stops
    and takes, built only once adding is proven. `adaptive_leverage` (`8cce3d2`) decides
    leverage by the same evidence and never above the constitution, which is 1x for LIVE;
    every Risk decision records it.

## Instrument universe

Kairos was designed for five symbols, signal `BTCUSDT`…`XRPUSDT` mapped to EVEDEX
instruments. The runner's configured list has ten EVEDEX instruments chosen by spread, and
Quant Scouts' PROD map now approves all ten (`6bd59a3`); the DEV, PAPER and simulation
universes of the other repositories keep the original five. All ten have a
Binance USD-M contract, found by the rule `XXXUSD` → `XXXUSDT` (`signals.signal_symbol`).
An instrument whose Binance bars cannot be completed is skipped for that scan and never
traded on guessed data.
