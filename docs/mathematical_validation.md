# Mathematical validation and corrections to the build guide

The project followed the "05 Options Pricing Library" build guide. Benchmarks were recomputed from the
exact stated inputs, and every claim was checked against what the code actually produces. This file
records what was confirmed, what was corrected and why.

## Benchmarks recomputed

| Guide value | Inputs | Recomputed (unrounded) | Verdict |
|---|---|---|---|
| Hull: C = 4.76, P = 0.81 | S=42, K=40, r=10%, sigma=20%, T=0.5, q=0 | C = 4.759422, P = 0.808599 | correct after rounding; tests use unrounded values, cross-checked with an independent `math.erf` implementation |
| BS call 10.4506 | S=K=100, T=1, r=5%, q=0, sigma=20% | 10.450584 | correct to 4 dp |
| BS put 5.5735 | same | 5.573526 | correct to 4 dp |
| (added) Haug generalised BSM put 4.0870 | S=75, K=70, T=0.5, r=10%, b=5% (q=5%), sigma=35% | 4.087 (within 5e-5) | confirms the dividend-yield path |
| (added) American put about 6.0903 | S=K=100, T=1, r=5%, q=0, sigma=20% | 6.09034 +- 3.5e-5 (LR n=8001; gap to LR n=4001 and CN 3200x3200) | reference value used here |

## Corrections and qualifications

1. **"Put-call parity holds independently of the model."** Parity `C - P = S e^{-qT} - K e^{-rT}` is
   model-free only for **European** options on the same strike and expiry, with the stated carry (here
   a continuous yield) and frictionless markets. American options satisfy only inequalities
   (`bsm.american_parity_bounds`, stated for `r, q >= 0`). The library never applies the equality to
   American quotes.
2. **"An American call on a non-dividend stock should equal the European call."** True for `q = 0`
   **and `r >= 0`**. The tests and `results/tables/american_call_rate_regimes.csv` show exact equality
   at r = 5% and r = 0. At r = -2% the American call is worth 1.25 more (S=130, K=100, T=2). With
   q = 4%, r = 3% it is worth 2.01 more.
3. **"Price European and American options four independent ways."** Closed-form BSM and
   terminal-payoff Monte Carlo price **European** options only. American options are priced by two
   families, trees (CRR, LR) and PDE (explicit/implicit/CN with PSOR). The library refuses BSM and MC for
   American options. Longstaff-Schwartz is not implemented. The guide's CV wording ("4 pricing methods
   ... agreement within 1e-3") should not be used as is.
4. **"Smax approximately 4K" boundary.** A fixed multiple of the strike ignores volatility and maturity.
   The domain here is `+- n_sd * sigma sqrt(T)` (plus drift) around `ln S` and `ln K`, and the
   truncation error was measured: about 0.14 at n_sd = 1, 5e-3 at n_sd = 1.5, and flat below 1e-4 beyond
   n_sd = 3 for the ATM case.
5. **American call boundary.** The guide's European call boundary
   `S_max e^{-q(T-t)} - K e^{-r(T-t)}` is not valid for an American call with dividends, where deep ITM
   exercise gives `S - K`. The American boundary uses `max(S - K, S e^{-q tau} - K e^{-r tau})`. Similarly,
   the American put lower boundary is `max(K - S, K e^{-r tau} - S e^{-q tau})`.
6. **Explicit stability "dt <= dx^2 / sigma^2".** That is a simplified diffusion-only rule. For the
   actual log-space operator the library computes the von Neumann amplification factor, including
   drift and discounting, plus a stricter monotonicity condition (`|nu| dx <= sigma^2`,
   `dt (sigma^2/dx^2 + r) <= 1`). Unstable explicit grids are rejected by default.
7. **"CN accuracy O(dt^2, dx^2), unconditionally stable"; "< 1e-3 on a 400x400 grid; observed order
   approximately 2".** CN is linearly stable but can oscillate. Without Rannacher start-up, the measured
   CN time error is large and irregular at coarse N (about 0.18 at N=10 versus 0.008 with Rannacher). With
   Rannacher, cell-averaged payoffs and S on a node, the measured orders are 1.99 (space) and 2.02 (time).
   On the 120-case European grid the 400x400 CN maximum error was 8.0e-4. That is a measured result for
   these parameters, not a guarantee for other regimes.
8. **"CRR converges O(1/n) ... LR converges at O(1/n^2)"; "CRR/LR < 1e-3 at stated n".** For smooth
   European cases the measured orders were CRR 1.02-1.16 and LR 1.99. Yet CRR n=1000 exceeded 1e-3 in
   44 of 120 European cases (max 5.4e-3, high-vol two-year options). For the **American put**, LR showed
   roughly first-order, irregular convergence (fitted 1.10) and errors up to 5e-3 at n=201. No
   uniform `O(1/n^2)` claim is made.
9. **"Greeks vs bump-and-revalue: relative error < 1e-4" and "PDE identity to 1e-10".** Both hold for
   moderate parameters with tolerances that combine absolute and relative terms. The identity holds to
   about 2e-16 relative to `max(S, K)` on 2,000 random cases, **in raw annual units only**: substituting
   per-day theta breaks it by more than 1. Bump estimates are not uniformly accurate. Near expiry
   (T = 0.002) or deep OTM, central differences have relative errors up to 100%
   (`results/tables/greek_validation_bump.csv`).
10. **"IV round trip recovers sigma to 1e-8 for sigma in [0.05, 1.5], moneyness 0.5-2."** Not achievable
    where vega is numerically zero. In the fast round-trip grid (168 cases, T in {0.1, 1}), 35 cases
    have **no time value at double precision** (status `zero_time_value`). The solver reports them
    rather than inventing an IV. Where the solver's identifiability flag is off, sigma is recovered to
    5.4e-10. Where a finite IV is returned, the price is reproduced to 7e-11.
11. **"A unique root exists inside the no-arbitrage bounds" with a fixed bracket [1e-4, 5].** Uniqueness
    holds, but a fixed bracket can miss valid high-vol solutions. The solver expands the bracket up to
    `sigma_max` and reports `exceeds_sigma_max` otherwise. A test recovers sigma = 8.
12. **Guide code `implied_vol`** raises on non-convergence and has no bound checks or diagnostics.
    **Guide code `crr`** applies the American max at every level, including the root, which is fine, but
    it gives no Greeks or boundary and does not check `0 < p < 1`. Both were reimplemented.
13. **Market data.** The guide imputes forwards and dividend yields from SPY/AAPL put-call parity, builds
    an "arbitrage-checked" surface from BSM IVs, and drops zero bids as worthless. SPY and AAPL options
    are **American** (with discrete dividends). Parity is then only an inequality, and European
    no-arbitrage conditions cannot be certified from their BSM IVs. This project runs the rigorous
    European pipeline on a labelled synthetic European chain. American quotes produce only labelled,
    descriptive IVs. Zero bids are rejected or flagged by policy and are not assumed worthless.
14. **"Calendar: total variance non-decreasing in T".** Valid at equal **forward** moneyness under
    deterministic proportional carry; the check is implemented that way. Raw prices are not compared
    across maturities.
15. **"Butterfly: C(K-h) - 2C(K) + C(K+h) >= 0".** This requires a uniform grid. The library uses
    slope differences and non-uniform butterfly weights.
16. **Delta-hedging "hedging error SD proportional to sigma sqrt(pi/4) vega / sqrt(n) (Derman-Kamal)".**
    This is not hard-coded. The simulation measures `SD x sqrt(n)` of 6.5-7.0 across 4-1000 rebalances,
    consistent with `1/sqrt(n)` scaling only. The gamma P&L formula is presented as an approximation
    (correlation 0.996 with actual P&L, RMS gap 0.094).
17. **"Deploy free on Streamlit Community Cloud"; "v1.0 tag"; "CI green"; GitHub layout.** Excluded by
    the project brief: no deployment, git, CI workflows or `.github/`. The app is local only.
18. **"Live" option chains.** yfinance data are delayed and unsynchronised and are labelled
    `MARKET_DELAYED`.
