"""Black-Scholes-Merton implied volatility with diagnostics.

``implied_vol`` solves ``BSM(sigma) = price`` for one European option:

1. **Style check.** American quotes are refused unless ``approximate_american=True``.
   The result is then labelled ``european_bsm_approximation_of_american_quote``.
2. **Pre-checks.** ``T == 0`` returns status 'expired'. The price is compared with the
   model-free European bounds ``[lower, upper]``:
   below lower -> 'below_lower_bound'; within ``price_tol`` of lower -> 'zero_time_value'
   (IV = 0, a boundary limit rather than a finite interior solution); at or above upper ->
   'at_or_above_upper_bound' (IV = +inf limit).
3. **Bracketing.** The BSM price is strictly increasing in sigma for T > 0 and S > 0. Start
   from ``[0, initial_upper]`` and double the upper end until it brackets the price or
   reaches ``sigma_max``; if even ``sigma_max`` is too low, return status 'exceeds_sigma_max'.
4. **Safeguarded Newton.** Start from Brenner-Subrahmanyam ``sqrt(2 pi / T) * C / S``
   (clipped into the bracket). Use raw vega (dV/dsigma). Fall back to bisection when the
   Newton step leaves the bracket or vega is tiny; the bracket shrinks every iteration.
5. **Brent fallback** (``scipy.optimize.brentq``) on the final bracket if Newton/bisection
   has not met tolerance within ``max_iter``.

Convergence is reached when ``|price residual| <= price_tol_abs + price_tol_rel * price``
or when the bracket width is below ``vol_tol``.

Identifiability: near zero vega, many volatilities reproduce the price to machine
precision. The result therefore reports ``vega`` and ``vol_uncertainty = price_resolution / vega``
(price resolution = solver tolerance + ``1e-15 * max(S, K)`` floating-point resolution)
and sets ``poorly_identified`` when that exceeds ``identifiability_threshold``. For such cases
the solver promises price reproduction, not recovery of a specific sigma.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import bsm
from .errors import InputError, UnsupportedMethodError
from .instruments import normalise_kind, normalise_style


@dataclass
class IVResult:
    """Implied volatility and solver diagnostics."""

    iv: float
    status: str
    converged: bool
    iterations: int
    method: str
    price_residual: float
    vega: float
    vol_uncertainty: float
    poorly_identified: bool
    reason: str = ""
    label: str = "european_bsm"

    def to_dict(self) -> dict:
        return asdict(self)


def brenner_subrahmanyam(price: float, S: float, T: float) -> float:
    """Near-the-money starting value ``sqrt(2 pi / T) * price / S``."""
    return math.sqrt(2.0 * math.pi / T) * price / S


def implied_vol(price: float, S: float, K: float, T: float, r: float, q: float, kind: str = "call",
                style: str = "european", *, price_tol_abs: float = 1e-12, price_tol_rel: float = 1e-12,
                vol_tol: float = 1e-14, initial_upper: float = 2.0, sigma_max: float = 20.0,
                max_iter: int = 100, vega_floor: float = 1e-12,
                identifiability_threshold: float = 1e-8, approximate_american: bool = False) -> IVResult:
    """Implied volatility of one European option quote (see the module docstring)."""
    kind = normalise_kind(kind)
    style = normalise_style(style)
    label = "european_bsm"
    if style == "american":
        if not approximate_american:
            raise UnsupportedMethodError(
                "BSM implied volatility is defined for European options. For an American quote pass "
                "approximate_american=True to get a labelled descriptive approximation.")
        label = "european_bsm_approximation_of_american_quote"
    vals = [float(v) for v in (price, S, K, T, r, q)]
    if not all(math.isfinite(v) for v in vals):
        raise InputError("implied_vol inputs must be finite")
    price, S, K, T, r, q = vals
    if K <= 0 or S < 0 or T < 0:
        raise InputError("need K > 0, S >= 0, T >= 0")
    nan = math.nan

    def out(iv, status, conv, it, method, resid, vega=nan, reason=""):
        # price resolution: solver tolerance plus floating-point resolution of a BSM price of scale S
        res_ = price_tol_abs + price_tol_rel * abs(price) + 1e-15 * max(S, K)
        unc = res_ / vega if vega and vega > 0 else math.inf
        poor = bool(not math.isfinite(unc) or unc > identifiability_threshold)
        return IVResult(iv, status, conv, it, method, resid, vega, unc, poor, reason, label)

    if T == 0:
        return out(nan, "expired", False, 0, "none", nan, reason="T == 0: no time value to invert")
    if S == 0:
        return out(nan, "degenerate_spot", False, 0, "none", nan, reason="S == 0: price independent of sigma")
    lo_b, hi_b = (float(x) for x in bsm.european_bounds(S, K, T, r, q, kind))
    ptol = price_tol_abs + price_tol_rel * abs(price)
    if price < lo_b - ptol:
        return out(nan, "below_lower_bound", False, 0, "none", price - lo_b,
                   reason=f"price {price:.6g} below no-arbitrage lower bound {lo_b:.6g}")
    if price <= lo_b + ptol:
        return out(0.0, "zero_time_value", True, 0, "bound", price - lo_b, 0.0,
                   reason="price equals intrinsic forward value; IV = 0 is a boundary limit")
    if price >= hi_b - ptol:
        return out(math.inf, "at_or_above_upper_bound", False, 0, "none", price - hi_b,
                   reason=f"price {price:.6g} at/above upper bound {hi_b:.6g}; IV -> infinity")

    def f(s: float) -> float:
        return float(bsm.price(S, K, T, r, q, s, kind)) - price

    lo, hi = 0.0, float(initial_upper)
    while f(hi) < 0:
        if hi >= sigma_max:
            return out(nan, "exceeds_sigma_max", False, 0, "bracket", f(sigma_max),
                       reason=f"price requires sigma > sigma_max={sigma_max}")
        lo = hi
        hi = min(2.0 * hi, sigma_max)

    sigma = brenner_subrahmanyam(price, S, T)
    if not (lo < sigma < hi):
        sigma = 0.5 * (lo + hi)
    method = "newton"
    used_bisect = False
    for it in range(1, max_iter + 1):
        diff = f(sigma)
        v = float(bsm.vega(S, K, T, r, q, sigma))
        if abs(diff) <= ptol:
            m = method + ("+bisection" if used_bisect else "")
            return out(sigma, "converged", True, it, m, diff, v)
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        if hi - lo < vol_tol:
            m = method + ("+bisection" if used_bisect else "")
            return out(sigma, "converged_bracket", True, it, m, diff, v,
                       reason="bracket width below vol_tol; price tolerance not reached (limited by precision)")
        step = sigma - diff / v if v > vega_floor else math.nan
        if math.isfinite(step) and lo < step < hi:
            sigma = step
        else:
            sigma = 0.5 * (lo + hi)
            used_bisect = True

    # Brent fallback on the current bracket
    try:
        s_b, info = brentq(f, lo, hi, xtol=vol_tol, rtol=4 * np.finfo(float).eps, maxiter=500, full_output=True)
    except ValueError as exc:  # pragma: no cover - bracket guaranteed by construction
        return out(nan, "failed", False, max_iter, "brent", nan, reason=str(exc))
    diff = f(s_b)
    v = float(bsm.vega(S, K, T, r, q, s_b))
    ok = bool(info.converged)
    return out(s_b, "converged" if ok and abs(diff) <= ptol else ("converged_bracket" if ok else "failed"),
               ok, max_iter + info.iterations, "newton+brent", diff, v,
               reason="" if ok else "Brent did not converge")


def implied_vol_array(prices, S, K, T, r, q, kinds="call", style: str = "european", **kwargs) -> pd.DataFrame:
    """Vectorised wrapper: broadcasts the inputs and returns one row of diagnostics per quote.

    The scalar solver runs per element (loops over the broadcast inputs); this keeps
    per-quote diagnostics exact. Chains of a few thousand quotes take well under a second
    to a few seconds.
    """
    arrs = np.broadcast_arrays(np.asarray(prices, float), np.asarray(S, float), np.asarray(K, float),
                               np.asarray(T, float), np.asarray(r, float), np.asarray(q, float),
                               np.asarray(kinds))
    flat = [a.ravel() for a in arrs]
    rows = []
    for p, s, k, t, rr, qq, kd in zip(*flat):
        res = implied_vol(p, s, k, t, rr, qq, str(kd), style, **kwargs)
        rows.append(res.to_dict())
    return pd.DataFrame(rows)
