# kairos-autotrader

**Live runner.** One process that runs the Kairos layers on EVEDEX with real orders. It
exists because no deployment profile of the service stack reaches a real venue: the
pinned Execution, Risk and Router releases reject `LIVE` by configuration and PAPER is
restricted to the EVEDEX DEV profile. The runner is the place where a reviewed candidate
becomes one real, protected, 1x market order and is managed to its exit.

As in the Kairos design, the signal is computed on closed Binance USD-M one-minute bars
and the order is placed on EVEDEX. Between the layers the runner passes the Kairos
contracts themselves (`kairos-core`, pinned): `StrategyIntentV1` from the Strategy Engine
and `CandidateRouteV1` from the Router.

It is not a claim of readiness or profitability. The meta-repository's readiness flags
stay `false`; nothing here changes them.

## What runs where

The runner does not copy code from the layer repositories. Each layer is executed
unchanged, in that repository's own locked environment, by a small worker script that
exchanges JSON with the runner ([`kairos_autotrader/sibling.py`](kairos_autotrader/sibling.py)).

| layer | repository | used by the runner | mode |
| --- | --- | --- | --- |
| 1B Text Scouts | `kairos-text-scouts` | sources, normaliser, freshness bound, deduplicator, relevance filter, sentiment prompt, schema and signal transform, local fallback | live: evidence for reviewers |
| 1A Quant Scouts | `kairos-quant-scouts` | `long_gap_recovery.parse_page`: a Binance kline page becomes strict `ClosedBarEventV1` bars only if it is complete, ordered and gap-free | live: every signal bar passes it |
| 2 Router | `kairos-router` | `SignalWindow` text aggregation, `CandidateRouterPolicy.build` → `CandidateRouteV1` (tier, review deadline, exact evidence ids) | live: every candidate is routed; a rejected one is dropped |
| 3 Aggregator | `kairos-aggregator` | `CandidateContextStore` → `DecisionContextV1`, `CandidateReviewBrain.review_with_context` → `CandidateReviewV1`, model calls through `kairos-llm`'s `BudgetedLLMGateway` | shadow by default (`review_mode`) |
| 4 Macro Strategist | `kairos-macro-strategist` | allocation prompt, context assembler, `AllocationOutput`, defensive fallback, shock detector | live: one allocation a day |
| 5 Risk Manager | `kairos-risk-manager` | `LiveRiskEvaluator`: every admission gate, the sizing and the `RiskTradeDecisionV1`, on a PROD `VenueQualityV1` measured by Quant Scouts' `build_venue_quality` just before the decision | live: decides every entry; the runner trades its quantity or less |
| Strategy Engine | `kairos-strategy-engine` | `kairos_strategy.hourly_trend` through its strict adapter: closed Binance bars in, `StrategyIntentV1` out, pinned by the package's own source identity; registered sleeve generators | live: every candidate comes from it; registered sleeves in shadow only (`shadow`, `engine-replay`) |

Owned by the runner itself:

- `signals.py` — a routed candidate is the Router's `CandidateRouteV1` (with the
  engine's `StrategyIntentV1` inside) plus the EVEDEX instrument it is traded on. Side,
  stop, target and timeout are read from the contract every time; journal records keep
  identities and a few readable fields, because one intent is about 2 MB.
- `binance_bars.py` — downloads public Binance kline pages and keeps the closed-bar
  cache; each update re-downloads the last cached minutes and refuses a changed bar.
- `system_engine.py` — the bridge to the engine and the pin. Every scan (`scan_minutes`,
  four times an hour) asks the engine; an engine whose source differs from the pinned one
  is refused, and so is an intent that does not declare exactly the cached bars.
- `routing.py` — the bridge to the Router; a route around another intent is refused.
- `brain.py` — a single probability review of all candidates (the default reviewer).
- `evedex.py` — EVEDEX REST client and EIP-712 order signing, byte-compatible with
  `@evedex/exchange-bot-sdk` 1.2.11 (fixture-tested).
- `engine.py` — the loop: scan, decide, size, enter with exchange-side stop and target,
  manage, account; hard ceilings as code constants; file-based state and journal.

## Known deviations from the Kairos contracts

These are tracked, not hidden. See
[`docs/ALIGNMENT.md`](docs/ALIGNMENT.md) for the file-level map and the order of removal.

- The default reviewer (`brain.py`, `review_mode="probability"`) is the runner's own and
  has no counterpart in the design; it, Text Scouts' extraction and the Macro call still
  call the provider directly. The Aggregator review runs in a worker per scan, not in
  `CandidateReviewService` on the bus, and its budget is a JSON ledger
  (`<runtime-dir>/llm-budget.json`), not the Postgres ledger of `kairos-persistence`.
- Routes come from the Router's pure policy. The time checks of
  `RouterService._candidate_rejection` are restated in `router_worker.py`, because the
  pinned service refuses LIVE routing by configuration.
- Closed bars come from Binance's public REST klines through the Quant Scouts parser, not
  from the Quant Scouts collector's durable stream. The `MarketSnapshot` for a review is
  built from REST data, so its liquidation totals stay at zero. There is no
  `VenueQualityV1`; the entry check bounds the distance between the EVEDEX price and the
  Binance reference instead.
- `LiveRiskEvaluator` needs a registered account verifier and input verifier; the
  Risk Manager ships none. The runner registers its own (`risk_worker.py`): the account
  evidence is what it read from EVEDEX with its session plus its trade ledger, and the
  trust boundary is the runner process. In paper mode the evidence is the paper ledger,
  so the decision's LIVE/PROD labels describe the evaluated policy, not a real account.
- The evaluator runs with `require_instrument_rules=False`: the intent's exit prices are
  Binance prices and are not on EVEDEX's price tick, which the authenticated-rules mode
  rejects. The runner floors the quantity to the venue's lot and refuses a size below
  the minimum lot instead.
- Only instruments in Quant Scouts' PROD symbol map (BTC, ETH, SOL, BNB, XRP) can be
  measured for `VenueQualityV1`, so only they can be traded; signals on the other
  configured instruments are recorded and refused.
- State is two files, not the transactional inbox/outbox and execution journal.
- The candidate rules are an isolated research package of the Strategy Engine, not a
  registered or promoted sleeve.

## Local development

```sh
uv sync --locked
uv run --locked python -m kairos_autotrader check
uv run --locked python -m kairos_autotrader scan --no-llm
uv run --locked python -m kairos_autotrader run --mode paper --no-llm --runtime-dir <dir>
```

Layer workers need the sibling repositories checked out next to this one, each with its
own `uv sync --locked` environment. A missing sibling disables that layer with a message;
tests that start real sibling code are skipped where it is absent.

`kairos-aggregator` must contain the bounded review prompt (commit `34963c2`, branch
`bounded-review-prompt-20261113`, until it is on `main`); without it every Aggregator
review of an hourly_trend candidate fails at the provider and ends as DEFER.
`kairos-strategy-engine` must contain the strict `kairos_strategy.hourly_trend` adapter
(commit `7096bfc`, branch `hourly-trend-strict-adapter-20261112`, until it is on `main`).
Without it, with a different source of that package, or without the `kairos-quant-scouts`
and `kairos-router` environments, the runner manages open positions and opens nothing.
The first start downloads three weeks of Binance minute bars per instrument (about a
minute and a half for ten instruments) into `<runtime-dir>/closed-bars`.

Real orders require `run --mode live --confirm-real-orders` and key files supplied by the
operator. Keys are read at start and are never printed, logged or written. The operator
guide (Russian) is [`docs/OPERATOR-GUIDE-RU.md`](docs/OPERATOR-GUIDE-RU.md).

## Checks

```sh
make all
```

runs Ruff, Ruff format, mypy, Bandit, the network-free test suite and the package build,
the same gates as every Kairos Python repository.

## Safety boundaries

- Hard ceilings are constants in `engine.py`: capital basis 100 USDT, risk per trade
  1 USDT, session loss 10 USDT, 12 trades a day, 3 open positions, leverage 1x.
- Writes to the venue are never retried. A 4xx is a refusal with no effect; a timeout or
  5xx is an unknown outcome that is reconciled from positions before anything else.
- An entry without an accepted exchange stop is closed at once and entries halt.
- No withdraw or transfer schema exists in the signer.
- Models receive typed evidence only. They can allow or veto an offered candidate; they
  cannot change a side, a price, a stop or a size, and cannot propose a trade.
