# Limitations

This is an educational research library. It is **not production software** and not investment advice.

## Model

* Constant `r`, `q`, `sigma`; a continuous dividend yield only. There are no discrete dividends, no term
  structures and no stochastic or local volatility. Real equity options with discrete dividends need a
  different model, and the American call early-exercise results here assume a continuous yield.
* GBM dynamics. No jumps, so smile and skew are not explained by the pricing model. The synthetic smile
  is an input from SSVI.
* Hedging simulations are frictionless apart from proportional costs: no bid/ask on the option, no
  market impact, no financing spread, and no discrete dividend dates.

## Numerical

* Accuracy numbers are **measured for the stated parameters** (`results/tables`); they are not
  guarantees for other regimes. Very short maturities, very high volatility, deep ITM/OTM strikes and
  negative-rate cases can need much finer grids.
* LR trees require odd `n` and have no theta. American LR convergence is roughly first order and
  irregular.
* The PDE uses a uniform log grid with Dirichlet boundaries. There is no non-uniform or adaptive grid
  and no penalty method. PSOR with fixed `omega` can be slow on very fine grids: a 3200x3200 American
  solve took about 174k PSOR iterations.
* Explicit-scheme stability checks assume constant coefficients (exact here, since the model is
  constant-coefficient in log-space).
* Monte Carlo is European and terminal-payoff only, with no Longstaff-Schwartz. Its CIs are normal
  approximations.
* The IV solver is per-quote and loops in Python. It is not Jaeckel's "Let's be rational" and is slower
  for very large chains.
* Tree, PDE and bump Greeks are approximations. Bump Greeks can be inaccurate near expiry and deep OTM.

## Data

* All surface, smile and arbitrage results in `results/` come from a **synthetic** European chain.
* The one delayed SPY snapshot saved in `data/snapshots` was downloaded before the US market opened.
  99% of its contracts had zero bids, and it was not used for any result. Even a good yfinance snapshot
  is delayed, unsynchronised and American-style, and its BSM IVs are descriptive only.
* No risk-free curve is downloaded. Rates are explicit inputs or parity-implied (European synthetic
  data only).

## Surface

* `TotalVarianceSurface` is piecewise linear with flat extrapolation. It is not an arbitrage-free
  parameterisation. SVI/SABR calibration and Dupire local volatility are **not implemented**.
* Arbitrage checks run on a finite grid only. Passing them is not proof of a globally arbitrage-free
  surface.
* Breeden-Litzenberger densities from noisy quotes are unreliable without smoothing.

## Engineering

* Tested on Windows with Python 3.14.4 only (the package declares >= 3.11 but was not run on other
  versions here).
* Only the fast experiment mode was executed for the committed results. `--mode research` exists but was
  not run for this report.
* The app was smoke-tested with `streamlit.testing.AppTest` and one headless start with a single manual
  click-through of the pricing tab in a local browser. It is not a full UI test suite.
