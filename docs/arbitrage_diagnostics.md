# Arbitrage diagnostics and their limits

`surface.arbitrage_diagnostics(chain, forwards, smiles)` runs **finite-grid** static checks on
European quotes. Puts are converted to call-equivalents through parity, `C = P + DF (F - K)`, using
the estimated `F` and `DF`. When both a call and a put exist at a strike, the tighter quote is used.

| Check | Condition on call prices `C(K)` | "Executable" when |
|---|---|---|
| bounds | `DF max(F-K, 0) <= C <= DF F` | the whole bid/ask lies outside the bound |
| monotonicity | `C(K_{i+1}) <= C(K_i)` | `bid(K_{i+1}) > ask(K_i)` |
| vertical spread | slope `s_i = (C_{i+1}-C_i)/(K_{i+1}-K_i)` in `[-DF, 0]` | `bid(K_i) - ask(K_{i+1}) > DF (K_{i+1}-K_i)` |
| convexity | slopes non-decreasing; butterfly weights `w1=(K3-K2)/(K3-K1)`, `w3=(K2-K1)/(K3-K1)` (non-uniform strikes) | `w1 ask(K1) + w3 ask(K3) - bid(K2) < 0` |
| calendar | total variance `w(k, T)` non-decreasing in `T` at equal log-forward moneyness `k` | later ask-IV variance < earlier bid-IV variance (an IV-space proxy, not a priced trade) |

The calendar check compares **forward-normalised** quantities, valid under deterministic,
proportional carry. Raw call prices at different maturities are never compared, because the
carry differs.

## Distinctions kept explicit

* **Mid-price inconsistency**: violated using mids. This is common with noisy or tick-rounded quotes.
* **Executable at the quotes**: tradeable at the bid/ask shown, still ignoring fees, size, borrow,
  timing and early-exercise risk.
* **Finite-grid diagnostics** versus **a globally arbitrage-free continuous surface**. Passing every
  check on the quoted strikes does *not* prove the interpolated surface is arbitrage-free.
  `TotalVarianceSurface` is linear in `k` within a slice and linear in `T` across slices. It is a
  visual and diagnostic tool, not an arbitrage-free fit (SVI/SABR are **not** implemented).

## Results (synthetic, `results/tables/arbitrage_diagnostic_report.csv`)

| Dataset | bounds | monotonicity | vertical | convexity | calendar | executable |
|---|---|---|---|---|---|---|
| true SSVI prices (no noise) | 0 | 0 | 0 | 0 | 0 | 0 |
| noisy synthetic quotes after cleaning | 12 | 10 | 1 | 36 | 0 | 0 |
| true prices + injected violations | 0 | 1 | 1 | 2 | 38 | 4 of 4 price checks; 0 calendar |

The true-surface row confirms the checks do not raise false alarms on an SSVI surface that meets
Gatheral-Jacquier's sufficient conditions. The noisy-quote row shows mid inconsistencies from quote
noise, tick rounding and parity conversion with an estimated `DF`; none is tradeable at the quotes.
The injected row shows detection on a non-uniform strike grid. One call was bumped by +0.4
(butterfly), one was raised above a lower strike (monotonicity), and the fourth expiry's mid total
variance was halved (calendar). The calendar defect is flagged at 38 points of the common-`k` comparison grid.
It is not "executable" because only the mid variance was altered, so the bid/ask IV band still
overlaps the earlier slice.

## Risk-neutral density

`risk_neutral_density` applies Breeden-Litzenberger, `q(K) = (1/DF) d2C/dK2`, with second divided
differences on a non-uniform grid. On true prices the density is non-negative and integrates close to
1 within the strike range (a unit test checks this). On noisy mids it oscillates and goes negative
(`results/tables/risk_neutral_density.csv`). Strike second derivatives amplify noise, and the result
depends heavily on interpolation and smoothing.
