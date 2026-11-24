# Methodology

## Model and conventions

* **Dynamics (pricing).** Under the risk-neutral measure Q the underlying follows GBM,
  `dS/S = (r - q) dt + sigma dW`, with constant continuously compounded rate `r`, continuous
  dividend yield `q` and volatility `sigma`. A European claim is worth `e^{-rT} E_Q[payoff(S_T)]`.
* **Dynamics (hedging simulation).** Under the physical measure P, `dS/S = (mu - q) dt + sigma_real dW`,
  where `mu` is the total expected return including dividends. Pricing and hedging still use Q-model
  quantities (implied or hedge volatility).
* **Time.** ACT/365F year fractions (`optpricer.conventions.year_fraction`); volatility is annualised on
  the same calendar clock. Trading-day (252) counts appear only as labels for rebalancing frequencies.
* **Units.** Greeks are stored in raw mathematical units; screen units come only from
  `conventions.to_display_units` (see `greek_conventions.md`).

## Code layout

| Module | Responsibility |
|---|---|
| `instruments.py` | immutable, validated `Option` (kind, style, K, T) and `MarketData` (S, r, q, sigma) |
| `conventions.py` | year fractions, discounting, display-unit conversions |
| `bsm.py` | vectorised BSM prices, Greeks, bounds, parity helpers |
| `binomial.py` | CRR, CRR averaging, Leisen-Reimer trees; lattice Greeks; exercise boundary |
| `pde.py` | log-space theta-scheme (explicit/implicit/CN + Rannacher), Thomas solver, PSOR LCP solver |
| `mc.py` | exact terminal-GBM Monte Carlo with SE/CI; optional `mcsim` cross-check |
| `implied_vol.py` | bracketed, safeguarded Newton + Brent IV solver with diagnostics |
| `market.py` | synthetic European chains, snapshot I/O, cleaning policy, optional yfinance download |
| `surface.py` | parity-implied forwards, smiles, total-variance surface, arbitrage diagnostics, density |
| `hedging.py` | self-financing delta-hedge simulator with audit ledger |
| `api.py` / `cli.py` | unified `price()` entry point and command line |
| `experiments/` | drivers that write figures, tables and `run_metadata.json` |

The Streamlit app (`app/streamlit_app.py`) and the notebooks only call these functions.

## Method/style support

| Method | European | American |
|---|---|---|
| BSM closed form | yes | refused (`UnsupportedMethodError`) |
| Terminal Monte Carlo | yes | refused (Longstaff-Schwartz not implemented) |
| CRR / CRR averaged | yes | yes |
| Leisen-Reimer (odd n) | yes | yes |
| Explicit FD | yes (stability-checked) | yes (projection) |
| Implicit / Crank-Nicolson FD | yes | yes (PSOR) |

So American options have **two independent families** of methods (lattices and PDE), not four.

## Reproducibility

* Every random draw comes from an explicit `numpy.random.Generator` seeded through the configs.
* `configs/*.yaml` hold all grids, budgets, tolerances and seeds, with `fast` and `research` sections.
* `scripts/run_all.py` writes `results/run_metadata.json`: package versions, platform, full
  configuration, seeds, per-stage runtimes and data provenance.
* Every figure in `results/figures` has its numbers in `results/tables`.
