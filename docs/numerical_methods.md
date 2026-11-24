# Numerical methods

## Binomial trees (`binomial.py`)

* **Node order.** At step `i`, node `j = 0..i` counts up-moves: `S_{i,j} = S u^j d^{i-j}`, in ascending
  price order. Backward induction `V_i = e^{-r dt} (p V_{i+1}[1:] + (1-p) V_{i+1}[:-1])` is vectorised
  across nodes, with a Python loop over time only. A hand-written two-step tree is checked against the
  vectorised code node by node in `tests/test_trees.py`.
* **CRR.** `u = e^{sigma sqrt(dt)}`, `d = 1/u`, `p = (e^{(r-q)dt} - d)/(u - d)`. If `p` falls outside (0, 1),
  `ProbabilityError` is raised (this happens when `|r - q| sqrt(dt)` is large relative to `sigma`).
* **Leisen-Reimer.** Peizer-Pratt inversion method 2, `h(z) = 1/2 + sign(z)/2 sqrt(1 - exp(-(z/(n+1/3+0.1/(n+1)))^2 (n+1/6)))`,
  with `p = h(d2)`, `p' = h(d1)`, `u = e^{(r-q)dt} p'/p` and `d = (e^{(r-q)dt} - p u)/(1-p)`. The tests check
  the martingale condition `p u + (1-p) d = e^{(r-q)dt}`. **Odd n only**: an even n raises an error and is
  never rounded silently.
* **Degenerate inputs.** `T = 0` gives the intrinsic value. `sigma = 0` follows the deterministic
  forward path: European = discounted terminal payoff, American = max over grid dates. `S = 0` uses the
  same deterministic path.
* **Greeks.** Delta comes from the two step-1 nodes and gamma from the three step-2 nodes. Theta comes
  from `(V_{2,1} - V_{0,0})/(2 dt)` for CRR only, where `u d = 1` puts `S_{2,1} = S`. LR theta is NaN by
  design.
* **American boundary.** At each step this records the highest exercised node (put) or the lowest (call).
* **Observed convergence** (`results/tables/tree_convergence_orders.csv`, European, fast mode): CRR
  is about first order (fitted slope about 1.0), LR about second order (about 2.0). Averaging CRR over
  `n, n+1` removes most of the odd/even oscillation for the ATM case, but its fitted slope is irregular
  (about 0.5-0.9 depending on the case). For the American put, both CRR and LR show roughly first-order,
  non-smooth convergence; LR has no `O(1/n^2)` guarantee there.

## Finite differences (`pde.py`)

* **Variables.** `x = ln S`, `tau = T - t`. Operator `L V = 1/2 sigma^2 V_xx + nu V_x - r V`, with
  `nu = r - q - sigma^2/2`, centred differences:
  `a = sigma^2/(2dx^2) - nu/(2dx)`, `b = -sigma^2/dx^2 - r`, `c = sigma^2/(2dx^2) + nu/(2dx)`.
* **Theta-scheme.** `(I - theta h L) V^{k+1} = (I + (1-theta) h L) V^k` plus Dirichlet terms on both
  sides (new boundary values on the left, old values on the right).
* **Domain.** Half-width `n_sd * sigma sqrt(T) + |nu| T` around both `ln S` and `ln K` (default
  `n_sd = 6`), so it scales with volatility and maturity. The grid is shifted so `ln S` is exactly a
  node, which removes interpolation error. `results/tables/pde_domain_sensitivity.csv` shows truncation
  error of about 0.14 at `n_sd = 1` and below 1e-4 from about `n_sd = 3`.
* **Initial condition.** By default this is the exact cell average of the payoff over each log-cell
  ("cell-averaged payoff"); `smooth_payoff=False` uses point values. Both give second-order spatial
  convergence when S = K lies on a node. Cell averaging reduces the error constant (about 6x in the
  fast run).
* **Boundaries.** European call: `0` and `S e^{-q tau} - K e^{-r tau}`. European put:
  `K e^{-r tau} - S e^{-q tau}` and `0`. American put lower boundary: `max(K - S, K e^{-r tau} - S e^{-q tau})`.
  American call upper boundary: `max(S - K, S e^{-q tau} - K e^{-r tau})`. The American call therefore does
  not reuse the European asymptote, which would be wrong when `q > 0`.
* **Linear solver.** `scipy.linalg.solve_banded` by default. The package's own `thomas()` is tested against
  `numpy.linalg.solve` on random diagonally dominant systems and gives identical PDE prices.
* **Explicit stability.** See `pde.explicit_stability`. Von Neumann: the amplification factor
  `g(phi) = 1 + h[b + (a+c) cos phi + i (c-a) sin phi]` must satisfy `|g| <= 1 + max(0,-r) h`.
  Monotonicity (sufficient, stricter): `a, c >= 0` and `1 + h b >= 0`. An unstable configuration raises
  `UnstableSchemeError` with a suggested `n_time`. `allow_unstable=True` exists only for the
  demonstration in `results/figures/04_explicit_instability.png`.
* **Rannacher.** For CN, the first `rannacher_steps` (default 2) steps are replaced by two implicit
  half-steps each. Without it, the time-convergence plot shows the CN error is large and irregular at
  coarse `N`. With it, the error is smooth and second order. A test checks that Rannacher reduces the
  gamma error near the strike. Linear stability of CN does not guarantee positivity or smooth Greeks.
* **Error separation.** Spatial errors are measured against the closed form at fine `N`. Temporal errors
  are measured against a same-grid solution with 4x the finest `N`, so they exclude spatial error.
  Domain (boundary) error is measured by varying `n_sd` at constant `dx`. Interpolation error is zero
  by construction (spot on a node).

## American exercise: LCP and PSOR

Each implicit/CN step solves `A V >= rhs`, `V >= g`, `(A V - rhs) . (V - g) = 0`. PSOR runs Gauss-Seidel
sweeps with over-relaxation `omega` (default 1.2) and projects onto `V >= g` **inside every sweep**. It
uses red-black ordering so each half-sweep is vectorised. The iteration stops when the max-norm update
is below `tol` (default 1e-10). Non-convergence raises `ConvergenceError`; `allow_nonconvergence=True`
returns the result but flags `psor_converged = False`. The diagnostics report total and maximum
iterations, the obstacle violation `max(0, -min(V - g))` and the complementarity residual
`max |min(V - g, A V - rhs)|` across all steps.

This is a genuine LCP iteration, not a single post-solve `max(V, payoff)`. The only place a single
projection is used is the explicit scheme, where it is the exact solution of that step's LCP.

## Monte Carlo (`mc.py`)

Exact terminal sampling `S_T = S exp((r - q - sigma^2/2)T + sigma sqrt(T) Z)`. The standard error comes
from independent units: single payoffs, or antithetic pair averages. Moments are combined across chunks
with Chan's parallel update, so a chunked run equals an unchunked run with the same stream to about
1e-12. The CI is a normal approximation. There is no time-discretisation bias, so the MC error is purely
sampling error. MC confidence intervals are never used to judge tree/PDE accuracy.

The optional cross-check calls the sibling `mcsim.pricing.engine.price_mc` with `GBM(measure="Q")` and
one time step. With the same seed it reproduces this package's estimate exactly (both apply the same
exact transform to the same normal stream). With a different seed the two estimates agree within their
combined SE (`results/tables/mc_crosscheck_mcsim.csv`).

## Implied volatility (`implied_vol.py`)

Bounds first, then bracket expansion (doubling up to `sigma_max`, default 20), then safeguarded Newton
from Brenner-Subrahmanyam with raw vega, with a bisection fallback when a step leaves the bracket.
If needed, `brentq` finishes the job. Price tolerance (absolute 1e-12 + relative 1e-12) and volatility
tolerance (bracket width 1e-14) are separate. Statuses: `converged`, `converged_bracket`,
`zero_time_value` (IV = 0 boundary), `at_or_above_upper_bound` (IV -> infinity), `below_lower_bound`,
`exceeds_sigma_max`, `expired`, `degenerate_spot`. `poorly_identified` is set when
`(price tolerance + 1e-15 max(S, K)) / vega > 1e-8`.
