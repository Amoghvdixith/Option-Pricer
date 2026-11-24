# Greek conventions

All Greeks are computed and stored in **raw mathematical units**. Screen units are an explicit,
separate conversion (`optpricer.conventions.to_display_units`).

| Greek | Raw definition (stored) | Display conversion | Display unit |
|---|---|---|---|
| delta | dV/dS | x1 | per 1 unit of underlying |
| gamma | d2V/dS2 | x1 | delta per 1 unit of underlying |
| vega | dV/dsigma | x0.01 | per 1 vol point (sigma + 0.01) |
| theta | dV/dt, calendar time, expiry date fixed (= -dV/dT) | x1/365 | per calendar day |
| rho | dV/dr | x0.01 | per 1 rate percentage point |
| psi | dV/dq | x0.01 | per 1 dividend-yield percentage point |
| vanna | d2V/(dS dsigma) | x0.01 | change in delta per 1 vol point |
| volga (vomma) | d2V/dsigma2 | x1e-4 | change in raw vega x 0.01 per vol point, i.e. per (1 vol point)^2 |

**Theta** is calendar-time decay. Holding the expiry date fixed, one calendar day reduces the time to
expiry `T` by 1/365, so `theta_raw = -dV/dT` and `theta_display = theta_raw / 365`. Theta is not
the derivative with respect to remaining maturity.

## Formulas (continuous dividend yield)

With `d1 = [ln(S/K) + (r - q + sigma^2/2)T]/(sigma sqrt T)`, `d2 = d1 - sigma sqrt T` and `phi` the
standard normal pdf:

* call delta `e^{-qT} N(d1)`; put delta `e^{-qT}(N(d1) - 1)`
* gamma `e^{-qT} phi(d1)/(S sigma sqrt T)`; vega `S e^{-qT} phi(d1) sqrt T` (same for calls and puts)
* call theta `-S e^{-qT} phi(d1) sigma/(2 sqrt T) - r K e^{-rT} N(d2) + q S e^{-qT} N(d1)`
* put theta `-S e^{-qT} phi(d1) sigma/(2 sqrt T) + r K e^{-rT} N(-d2) - q S e^{-qT} N(-d1)`
* call rho `K T e^{-rT} N(d2)`; put rho `-K T e^{-rT} N(-d2)`
* vanna `-e^{-qT} phi(d1) d2/sigma`; volga `vega d1 d2/sigma`

## Validation (see `tests/test_greeks.py`, `results/figures/08_greek_validation_errors.png`)

* **Central finite differences** with `h = 1e-4`. Tolerances combine absolute and relative terms
  (`|a - b| <= abs_tol + rel_tol |b|`), because some Greeks are close to zero. First-order Greeks use
  rel 1e-6; second-order Greeks (gamma, volga, vanna) use rel 1e-4.
* **Bump-size sweep.** Large bumps show truncation error and tiny bumps show round-off. The best bump
  differs by Greek. Second derivatives degrade fastest as `h` shrinks.
* **Call-put relationships.** `delta_c - delta_p = e^{-qT}`; gamma, vega, vanna and volga are equal;
  `rho_c - rho_p = K T e^{-rT}`; `theta_c - theta_p = q S e^{-qT} - r K e^{-rT}`.
* **BSM PDE identity in raw annual units.** `theta + 1/2 sigma^2 S^2 gamma + (r - q) S delta - r V = 0`.
  On 2,000 random parameter sets the scaled residual is below 1e-10. A test also shows the identity
  fails by more than 1 if per-day display theta is substituted.
* **Known values.** S=K=100, T=1, r=5%, q=0, sigma=20%: delta 0.636831, gamma 0.018762, vega 37.5240
  (0.3752 per vol point), theta -6.4140 per year (-0.01757 per day), rho 53.2325.

## Edge cases

At `T = 0` or `sigma = 0` the Greeks are the one-sided limits (e.g. delta 0 or `e^{-qT}`, gamma 0). At
the payoff kink of the deterministic limit (`F = K`), delta, gamma, vega, vanna, volga, theta, rho and
psi are **NaN** because the derivative is undefined. Finite-difference estimates are not equally
accurate everywhere: near expiry and deep OTM, the bump errors in
`results/tables/greek_validation_bump.csv` are much larger than at the money.
