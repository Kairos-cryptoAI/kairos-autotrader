# kairos-autotrader

**Live runner.** One process that runs the Kairos layers on EVEDEX with real orders. It
exists because no deployment profile of the service stack reaches a real venue: the
pinned Execution and Risk releases reject `LIVE` at startup, PAPER is restricted to the
EVEDEX DEV profile, and the strict Strategy Parity contracts are bound to Binance minute
bars. The runner is the place where a reviewed candidate becomes one real, protected,
1x market order and is managed to its exit.

It is not a claim of readiness or profitability. The meta-repository's readiness flags
stay `false`; nothing here changes them.

## What runs where

The runner does not copy code from the layer repositories. Each layer is executed
unchanged, in that repository's own locked environment, by a small worker script that
exchanges JSON with the runner ([`kairos_autotrader/sibling.py`](kairos_autotrader/sibling.py)).

| layer | repository | used by the runner | mode |
| --- | --- | --- | --- |
| 1B Text Scouts | `kairos-text-scouts` | sources, normaliser, freshness bound, deduplicator, relevance filter, sentiment prompt, schema and signal transform, local fallback | live: evidence for reviewers |
| 2 Router | `kairos-router` | `SignalWindow` text aggregation, `classify_candidate`, candidate time-to-live | live inside the Aggregator review |
| 3 Aggregator | `kairos-aggregator` | candidate-review prompt, `CandidateReviewOutput`, conflict guard | shadow by default (`review_mode`) |
| 4 Macro Strategist | `kairos-macro-strategist` | allocation prompt, context assembler, `AllocationOutput`, defensive fallback, shock detector | live: one allocation a day |
| 5 Risk Manager | `kairos-risk-manager` | `RiskSettings` limits, allocation gate, loss-at-stop sizing, economics | shadow by default (`risk_mode`) |
| Strategy Engine | `kairos-strategy-engine` | registered sleeve generators on EVEDEX 1m bars | shadow only (`shadow`, `engine-replay`) |

Owned by the runner itself:

- `strategy.py` — two candidate rules on one-hour bars (trend pullback, breakout) with fixed
  exits. The hourly bars are rebuilt from 15-minute candles on every scan (`scan_minutes`),
  so the rules run four times an hour on the same timeframe.
- `brain.py` — a single probability review of all candidates (the default reviewer).
- `evedex.py` — EVEDEX REST client and EIP-712 order signing, byte-compatible with
  `@evedex/exchange-bot-sdk` 1.2.11 (fixture-tested).
- `engine.py` — the loop: scan, decide, size, enter with exchange-side stop and target,
  manage, account; hard ceilings as code constants; file-based state and journal.

## Known deviations from the Kairos contracts

These are tracked, not hidden. See
[`docs/ALIGNMENT.md`](docs/ALIGNMENT.md) for the file-level map and the order of removal.

- Candidates are a local dataclass, not `StrategyIntentV1`: that contract admits only
  `venue="BINANCE_UM"` and requires strategy code/config/input fingerprints.
- The review context is a local JSON shape, not `DecisionContextV1`.
- The Risk Manager's account gate is re-stated with the same formulas and settings; the
  original needs an `AccountSnapshotV2` that only a PAPER DEV account produces.
- With the default `risk_mode="autotrader"` an entry does not require Risk Manager
  approval. `CONTRIBUTING.md` in the meta-repository forbids bypassing Risk validation;
  `risk_mode="kairos"` restores it and is the intended end state.
- State is two files, not the transactional inbox/outbox and execution journal.
- The two rules in `strategy.py` are not a registered Strategy Engine sleeve.

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
