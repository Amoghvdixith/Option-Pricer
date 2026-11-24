# Interview notes (tied to this code and its observed results)

Each answer points to the code that implements it and the measured result in `results/`. All numbers
are from the fast-mode run recorded in `results/run_metadata.json`.

**1. Derive the BSM PDE.** Hold the option and short `Delta = dV/dS` shares. Ito's lemma removes the
`dW` term, so the portfolio is riskless and earns `r`. Shares pay yield `q`, giving
`V_t + 1/2 sigma^2 S^2 V_SS + (r - q) S V_S - r V = 0`. In code this is
`validation.pde_identity_residual`, which holds to about 2e-16 (relative to max(S, K)) on 2,000 random
cases when theta is in raw per-year units. With per-day display theta it fails by more than 1
(`tests/test_greeks.py`).

**2. Why not exercise an American call early without dividends? When is a put exercised?** Exercising
gets `S - K`. Holding is worth at least `S - K e^{-rT}`, which exceeds `S - K` when `r >= 0`, plus
insurance value. `results/tables/american_call_rate_regimes.csv` shows equality to machine precision
at r = 5% and r = 0. At r = -2% the American call is worth 1.25 more, and with q = 4% it is worth 2.01
more. A deep ITM put is exercised early to earn interest on K sooner. The boundary S*(t) rises
towards K at expiry (`figures/06_american_put_exercise_boundary.png`). At r = 0, q = 0 the
early-exercise premium is zero (`figures/05_...`).

**3. Gamma and ATM near expiry.** `gamma = e^{-qT} phi(d1)/(S sigma sqrt T)`. The `1/sqrt T` term blows
up at the money as T -> 0, while away from the money phi(d1) -> 0. See the T = 0.1 curves in
`figures/07_greeks_vs_spot.png`. At expiry the library returns NaN at the kink rather than a fake
number.

**4. Theta-gamma trade-off.** For a delta-hedged position at the implied vol, the PDE gives
`theta ~= -1/2 sigma^2 S^2 gamma` (+ carry terms). Long gamma pays theta. Per step, the hedged P&L is
approximately `1/2 Gamma S^2 (sigma_imp^2 dt - (dS/S)^2)`. In `hedging.gamma_pnl_approximation`, that
approximation has correlation 0.996 with the simulated P&L (realised 25%, implied 20%, 252
rebalances/yr). It is still only an approximation: the RMS gap is 0.094 against a P&L SD of 1.0.

**5. Binomial oscillation and fixes.** The strike falls in different positions relative to the
terminal nodes as n changes parity. CRR has an observed order of about 1.0 and oscillates. Averaging n
and n+1 damps the oscillation. Leisen-Reimer centres the tree on the strike using Peizer-Pratt
inversion, giving an observed order of 1.99 for European calls (`tables/tree_convergence_orders.csv`).
For American puts LR loses that advantage: fitted order about 1.1.

**6. Why CN, and why Rannacher?** CN is second order in time and unconditionally linearly stable. The
non-smooth payoff excites high-frequency modes that CN damps poorly. Measured CN time errors at coarse
N are large and irregular (about 0.18 at N=10). Two implicit half-step pairs fix this (0.008 at N=10,
clean order 2.02). Without them a test shows a worse gamma near the strike. Separately, the explicit
scheme is rejected when the von Neumann amplification exceeds 1. The forced demonstration with
max|g| = 1.20 prints a "price" of -1129.

**7. Equity skew and the risk-neutral distribution.** Downside puts are expensive, so the risk-neutral
density has a fatter left tail than the lognormal. The synthetic SSVI surface uses rho = -0.6 to
produce that shape (`figures/11_smiles_across_expiries.png`). Breeden-Litzenberger
(`surface.risk_neutral_density`) recovers a non-negative density from clean prices. From noisy mids it
goes negative at 7 strikes, a reminder that second derivatives amplify noise.

**8. Short and delta-hedged, realised above implied: what happens?** You lose on average, roughly
in proportion to the variance gap. The simulation gives mean P&L of -1.19 +- 0.01 at +3 vol points and
-4.01 +- 0.03 at +10 vol points, versus +3.78 at -10 vol points (premium 9.41,
`tables/hedging_vol_mismatch.csv`). Discrete hedging adds noise that falls roughly as `1/sqrt(n)`: SD
3.25, 1.47, 0.44 and 0.22 at 4, 21, 252 and 1000 rebalances per year on the same paths. With 20 bp
costs, the RMS P&L is lowest near 126 rebalances per year in this run.

## Talking points about engineering choices

* **Method/style safeguards.** `api.price` refuses BSM/MC for American options, and the surface tools
  refuse American quotes unless the caller explicitly asks for a labelled approximation.
* **Honest numerics.** PSOR reports iterations and complementarity residuals (max 1.4e-10 across 40
  American cases) and raises on non-convergence. The IV solver distinguishes "no time value"
  (35 of 168 grid cases) from a genuine solution. Monte Carlo SEs come from independent units,
  including antithetic pairs.
* **Things I found while building it.** pandas 3 silently turned mixed-precision ISO timestamps into
  NaT, which hid stale quotes until a test caught it. Parity-implied discount factors from 10 near-ATM
  pairs were too noisy; using 25 pairs plus reported standard errors fixed that. A first-pass "compare"
  command aborted because the explicit scheme was (correctly) unstable at the default grid; per-method
  error reporting fixed that.
* **What I would add next.** Longstaff-Schwartz for American MC, discrete dividends, SVI calibration
  with no-arbitrage constraints, and Jaeckel's IV algorithm.
