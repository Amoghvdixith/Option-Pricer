# Delta-hedging accounting

`hedging.hedge_paths` simulates a **self-financing** hedge of one short European option.

## Ledger

| When | Cash-flow entries |
|---|---|
| t0 | + premium `V0 = BSM(sigma_implied)`; buy `Delta_0` shares (computed with `sigma_hedge`, default `sigma_implied`): `-Delta_0 S_0`; cost `-c |Delta_0| S_0` |
| each interval | interest `cash (e^{r dt} - 1)`; dividends `q Delta_k dt (S_k + S_{k+1})/2` (trapezoid approximation of the continuous yield) |
| rebalance at `t_{k+1} < T` | `-(Delta_{k+1} - Delta_k) S_{k+1}`; cost `-c |Delta_{k+1} - Delta_k| S_{k+1}`; delta uses remaining maturity `T - t_{k+1}` |
| T | pay the payoff; sell the remaining shares `+Delta S_T`; cost `-c |Delta| S_T` |

Terminal net P&L is the final cash balance in time-T currency. No capital is added after t0. The audit
example (`results/tables/hedging_accounting_audit_example.csv`) lists every entry, with the option
marked at `sigma_implied` and the marked portfolio value `cash + Delta S - option` at each date.

## Checks (tests/test_hedging.py)

* **Reconciliation.** The external flows (premium, trades, costs, dividends, payoff, liquidation)
  carried forward to T at `r` equal final cash to about 1e-14 (1.9e-14 in the audit figure).
* **Discounted-gains identity** (q = 0, no costs):
  `e^{-rT} PnL = V0 - e^{-rT} payoff + sum_k Delta_k (e^{-r t_{k+1}} S_{k+1} - e^{-r t_k} S_k)`. This
  is an independent formula and matches to 1e-10.
* **Costs.** The initial cost equals `c Delta_0 S_0` and the final cost equals `c |Delta_last| S_T`. The
  P&L with costs equals the P&L without costs minus the future value of all costs on the same path.

## Experimental design

* Paths are simulated **exactly** under P at the union of all rebalance grids (4, 21, 252, 1000 per
  year), so every frequency hedges the same paths (common random numbers).
* Defaults: `S0 = K = 100`, `T = 1`, `r = 3%`, `q = 0` (simplest accounting), `mu = 7%`,
  `sigma_implied = sigma_hedge = sigma_realised = 20%`, no costs. A `q = 2%` run is reported separately
  (`hedging_frequency_summary_with_dividends.csv`).
* The reported statistics are mean, SD and SE of the mean across paths, quantiles, mean cost and turnover.

## Interpretation

* For small steps, the hedged short-option P&L is approximately
  `sum_k 1/2 Gamma_k S_k^2 (sigma_imp^2 dt - (dS/S)^2)` (gamma-weighted variance difference). In the run
  with realised vol 25%, 252 rebalances/yr and 3,000 paths, its correlation with the actual P&L is
  0.996. It is still an approximation: the RMS difference is 0.094, against a P&L SD of 1.0.
* Short gamma loses on average when realised volatility exceeds implied volatility, and gains when it is
  below. The sign and roughly linear shape are visible in `16_mean_pnl_vs_vol_mismatch.png`. This is a
  property of the averages, not a guarantee for any single path.
* Hedging error falls as rebalancing increases (SD 3.25 -> 1.47 -> 0.44 -> 0.22 for 4/21/252/1000
  per year in the fast run). The ratio `SD x sqrt(n)` stays roughly constant (6.5-7.0), which is consistent
  with the usual `1/sqrt(n)` heuristic. No closed-form hedging-error formula is hard-coded; the guide's
  Derman-Kamal-style expression is treated only as a scaling heuristic.
* With proportional costs, the mean P&L falls as frequency rises and the RMS P&L has an interior
  minimum in the fast run: 504 rebalances/yr at 5 bp (0.542, almost flat from 252 to 504), 252/yr at
  10 bp (0.791) and 126/yr at 20 bp (1.184), on a grid of 4, 12, 21, 52, 126, 252, 504 and 1000.
  These are sample results on a coarse frequency grid, not universal optima.
