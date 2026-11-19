# kairos-autotrader

The Kairos runner evaluates strategies on closed Binance USD-M minute bars, reviews
their signals, and manages positions on EVEDEX. Paper and live runs use the same
strategy, review and risk policies. Live orders require explicit confirmation.
The [operator guide](docs/OPERATOR-GUIDE-RU.md) covers configuration, position
management and the historical studies in Russian. The
[Kairos project](https://github.com/Kairos-cryptoAI/kairos) maintains the service-stack
manifest, release evidence and readiness statements.

This revision blocks new EVEDEX production entries. A native full-position
take-profit increased an existing short when it triggered. The runner records
the reason at startup. Paper trading, account reconciliation, cancellation and
the dedicated position-close endpoint remain available.

## From a closed bar to a position

| Stage | What the runner uses |
| --- | --- |
| Market data | Quant Scouts validates public Binance minute bars. The runner caches them in `closed-bars`, rechecks the most recent five and records revisions. Text Scouts supplies dated news evidence. |
| Strategy | Strategy Engine emits a `StrategyIntentV1` with the side, stop, target, expiry and bar identities. The runner checks the strategy source and the bars it supplied. |
| Review | Router assigns the tier, evidence and deadline. Aggregator assembles the context and returns ALLOW, VETO or DEFER through the LLM gateway. The default profile also records the runner's probability review for comparison. |
| Allocation and risk | Macro Strategist allocates capital by regime. Quant Scouts measures EVEDEX and Binance venue quality. Risk Manager requires an Aggregator ALLOW, usable allocation and account evidence, then determines size and leverage within its limits. |
| Execution | Paper mode simulates fills. Every live entry goes to the Execution Engine portfolio broker with a validated Risk decision, quantity, leverage, stop and target. Execution journals each request and places a limit order bounded by the approved price. |

Workers exchange JSON with the runner and load the corresponding sibling checkouts.
The runner keeps its own `state.json`, `journal.jsonl` and Aggregator budget ledger.
The probability reviewer, Macro and Text Scouts model calls have separate accounting
from the Aggregator gateway. The [alignment notes](docs/ALIGNMENT.md) trace the
implementation to its Kairos contracts.

## Profiles and limits

The [example](config.example.json) selects 23 instruments, a capital basis of 100 USD,
a 0.25 USD per-trade risk cap, a 5 USD session loss stop and a 40 USD monthly
Aggregator budget. With no config file the runner selects ten instruments and a
12 USD review budget and `hourly_trend_candidates_v1` for entries. The example
enables both the hourly and right-tail rules with a shared portfolio risk budget.

The `regime_aligned_right_tail_v1` strategy has a default decision interval of 24 hours. Supported settings for
`right_tail_interval_hours` are 4, 6, 8, 12 and 24. Hourly features are still
calculated for review when only right-tail is selected. A strategy can use its
own fixed `management_policies` override. The example supplies no override.

The runner accepts up to 150,000 USD of configured capital. The dollar risk cap can
be set up to 1% of that basis, and the session loss stop up to 5%. Risk Manager
further limits the risk at the stop to 0.25% of equity per trade and 1% across open
positions. It may reduce size or refuse an entry. The runner permits up to ten open
positions and forty entries per day. The signed 100 USD profile limits the pilot to
twelve entries per day. Risk Manager sets leverage within the venue and policy limits,
up to 30×. See [the configuration and validation](kairos_autotrader/engine.py).

## Run from a checkout

Use Python 3.11 or later and the uv version pinned in [pyproject.toml](pyproject.toml).
Check out the required sibling repositories under the same parent directory and run
`uv sync --locked` in each Python checkout used by a worker. For a wheel installation
set `KAIROS_WORKSPACE_ROOT` to that parent directory. Run these commands from this
repository. Supply your own key files outside the checkout wherever a command uses
`/path/to/`. Use separate directories for paper, shadow and live state.

```sh
uv sync --locked
uv run --locked kairos-autotrader check --api-key-file /path/to/evedex-api-key --wallet-key-file /path/to/wallet-key
uv run --locked kairos-autotrader scan --no-llm --config config.example.json
uv run --locked kairos-autotrader shadow --runtime-dir ../runtime/shadow
uv run --locked kairos-autotrader engine-replay --runtime-dir ../runtime/replay --days 45
uv run --locked kairos-autotrader run --mode paper --config config.example.json --runtime-dir ../runtime/paper --openai-key-file /path/to/openai-key
uv run --locked kairos-autotrader report --runtime-dir ../runtime/paper --outcomes
```

`check` reads market and account data. `scan --no-llm` inspects candidates without
model calls. A paper run with model review uses live market data and simulates orders.
For real orders, configure `execution_profile` with the path to the operator-signed
Execution portfolio profile. Relative paths start at the configuration file. The profile
binds the venue account, wallet, instrument universe, size and loss limits, entry
deadline, runner directory and persistent Execution journal. Then run

```sh
uv run --locked kairos-autotrader run --mode live --confirm-real-orders --config config.json --runtime-dir ../runtime/live --api-key-file /path/to/evedex-api-key --wallet-key-file /path/to/wallet-key --openai-key-file /path/to/openai-key
uv run --locked kairos-autotrader report --config config.json --runtime-dir ../runtime/live --ledger
```

A `STOP` file in the runtime directory stops new entries and lets the runner manage
open positions to exit. A `CLOSE` file requests immediate closure of owned positions.
The signed deadline stops new entries while open trades continue to settlement.
The runner exits after a fresh observation confirms a flat account and a settled journal.
`--new-session` starts a new reporting period and preserves live loss controls.
The historical `runner` executor name selects the portfolio broker. Configuration
for the retired `execution_engine` single-trade path is supported by the report.

## Reading trade results

The report groups closes by live or paper mode and by the source of their PnL. Each
row shows the recorded exit initiator. With the signed profile the report also reads
the account-wide pilot balance, net transfers, conservative loss and counts of open
positions, closed trades and attributed outcomes from the Execution journal.
It records the first observation at or after the deadline, its delay and the first
fully settled observation. The report deducts settled Aggregator fees over that interval
and identifies the incomplete coverage of external model charges.
A disappearing position has unknown origin.
For earlier live trades the balance delta includes any manual trading, transfers and
funding during the trade. Overlapping positions have price estimates. The learning
ledger admits a live result only with attributed execution accounting, a system exit
and an identifiable trade and candidate. Simulated replay evidence retains its own
source. Compare observed account balances separately from strategy PnL.

## Development checks

Run `make all` for Ruff lint and format, mypy, Bandit, pytest and the package build.
Tests that start sibling workers require those checkouts and their environments.
The [Makefile](Makefile) lists the equivalent uv commands for platforms without Make.
