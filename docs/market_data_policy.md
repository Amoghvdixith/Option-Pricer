# Market-data policy

## Sources and labels

| Label | Source | Exercise style | Use |
|---|---|---|---|
| `SYNTHETIC` | `market.synthetic_european_chain` (seeded SSVI generator; fictional timestamps) | European | default for forwards, smiles, surface, arbitrage diagnostics, tests |
| user snapshot | CSV + `.meta.json` loaded with `market.load_snapshot` | must be stated per row | as labelled |
| `MARKET_DELAYED` | `market.fetch_yfinance_chain` (optional; network; explicit call only) | **American** (US equity/ETF options) | descriptive BSM IVs only |

Every row carries `data_label`, `source` and `exercise_style`. Snapshots record the retrieval time,
the spot timestamp, expiries, currency, settlement, the expiry-time assumption and caveats in the
JSON metadata. Delayed data is never described as live or synchronised.

## What was actually downloaded

`scripts/fetch_snapshot.py` was run once in this build (2026-11-24 11:51 UTC). It saved
`data/snapshots/SPY_20261124T115114Z.csv` (1,739 contracts, 6 nearest expiries). The download ran at
about 06:51 New York time, **before the US market opened**. 99.1% of contracts had a zero bid, and
cleaning left 7 two-sided quotes (`results/tables/market_snapshot_cleaning_summary.csv`). The snapshot is
therefore kept only as a labelled record. No market smile figure was produced from it, and last-trade
prices were **not** used as a substitute for quotes. The surface results in this repository are
**synthetic**. To get a usable snapshot, run the fetch script during US trading hours.

## Cleaning rules (`market.clean_chain`)

Rows are never dropped. Each gets `accepted`, `reasons` and `flags`.

| Rule | Condition | Notes |
|---|---|---|
| `missing_quote` | bid or ask NaN | |
| `negative_quote` | bid or ask < 0 | |
| `crossed_market` | bid > ask | |
| `zero_bid` | bid == 0 | policy `reject` (default) excludes it from IV work; `flag` keeps it and tags it. Zero-bid contracts are **not** declared worthless |
| `wide_spread` | ask - bid > 50% of mid and > 0.05 | thresholds in `configs/market_data.yaml` |
| `stale_quote` | quote older than 900 s vs the spot timestamp | only when per-quote timestamps exist (yfinance gives none; its last-trade time is kept separately) |
| `short_expiry` | < 7 calendar days | |
| `bound_violation` | ask below the lower bound or bid above the upper bound | European bounds for European rows; American bounds (intrinsic, S, K) for American rows |
| `low_liquidity` | volume / open interest below optional thresholds | off by default |

On the synthetic chain, all 12 injected defects are caught
(`results/tables/quote_cleaning_injected_defects.csv`). The other rejections are genuine
zero-bid or wide-spread deep-OTM quotes created by tick rounding.

## Exercise-style safeguards

US equity and ETF options (e.g. SPY) are American-style and physically settled, and their underlyings
pay discrete dividends. Therefore:

* `surface.implied_forward(s)`, `build_smiles` and `arbitrage_diagnostics` **refuse** American rows
  unless `approximate=True`. Results are then labelled `APPROXIMATE_from_american_quotes`.
* `implied_vol(..., style="american")` refuses unless `approximate_american=True`; the result is then
  labelled `european_bsm_approximation_of_american_quote`.
* No exact forward or dividend yield is inferred from American prices. European no-arbitrage
  conditions are never certified from American-chain BSM IVs.
* Quote timing is a further limitation: delayed bid/ask, spot and rates are not synchronised.

Rigorous European validation runs on the synthetic European dataset, where the truth is known.

## Forward estimation

For each expiry, a weighted least-squares fit of `C - P = DF F - DF K` uses the 25 matched, accepted
strike pairs nearest the money (weights `1/(spread_C + spread_P)^2`). It reports `F`, `DF`, their
standard errors from the WLS covariance (delta method for `F`), implied rate, implied carry and
implied yield, plus the residual dispersion. These quantities are *implied carry*, not a dividend
forecast. On the synthetic chain the true `F` and `DF` lie within 2.2 reported standard errors (`results/tables/implied_forwards.csv`).
`DF` is the harder parameter because the strike range near the money is narrow.
