"""Black-Scholes-Merton prices and analytical Greeks for European options.

Model: GBM under the risk-neutral measure with constant ``r``, continuous dividend yield ``q``
and volatility ``sigma``. All functions are vectorised: ``S, K, T, r, q, sigma`` may be any
arrays that broadcast together, and ``kind`` may be one string or an array of
'call'/'put' strings.

Numerical treatment
-------------------
Prices use the forward/discount form ``V = DF * (F N(d1) - K N(d2))`` with ``F = S e^{(r-q)T}``.
The following cases are handled **explicitly**, with no division by zero followed by
warning suppression:

* ``sigma * sqrt(T) == 0`` (expiry or zero volatility): the deterministic limit
  ``DF * max(F - K, 0)`` (call) or ``DF * max(K - F, 0)`` (put). At ``T == 0`` this is the
  intrinsic value.
* ``S == 0``: the call is worth 0 and the put ``K e^{-rT}``. The process is absorbed at 0.

Greeks in the degenerate cases are the one-sided limits of the formulas. Where the
payoff kink makes a Greek undefined (``F == K`` with ``sigma sqrt(T) == 0``), gamma, vega,
vanna and volga are NaN, and delta is NaN as well because the left and right limits differ.

Greeks are returned in raw units (see :mod:`optpricer.conventions`). Theta is the
calendar-time derivative ``dV/dt = -dV/dT``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .errors import InputError

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _npdf(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):  # exp(-inf) -> 0 is the correct limit, not an error
        return np.exp(-0.5 * x * x) / _SQRT_2PI


def _is_call(kind, shape) -> np.ndarray:
    """Boolean array: True for calls. Validates every entry."""
    k = np.asarray(kind)
    if k.dtype.kind not in ("U", "S", "O"):
        raise InputError("kind must be 'call'/'put' string(s)")
    low = np.char.lower(k.astype(str))
    ok = np.isin(low, ["call", "put", "c", "p"])
    if not np.all(ok):
        bad = np.unique(k[~ok]) if k.ndim else k
        raise InputError(f"kind must be 'call' or 'put', got {bad!r}")
    return np.broadcast_to(np.isin(low, ["call", "c"]), shape)


def _prepare(S, K, T, r, q, sigma):
    arrs = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (S, K, T, r, q, sigma)))
    S, K, T, r, q, sigma = (np.array(a, dtype=float) for a in arrs)
    for name, a in zip(("S", "K", "T", "r", "q", "sigma"), (S, K, T, r, q, sigma)):
        if not np.all(np.isfinite(a)):
            raise InputError(f"{name} must be finite")
    if np.any(K <= 0):
        raise InputError("strike K must be positive")
    if np.any(S < 0):
        raise InputError("spot S must be non-negative")
    if np.any(T < 0):
        raise InputError("time to expiry T must be non-negative")
    if np.any(sigma < 0):
        raise InputError("volatility sigma must be non-negative")
    return S, K, T, r, q, sigma


def _core(S, K, T, r, q, sigma):
    """Shared intermediate quantities. Returns a dict of arrays."""
    sd = sigma * np.sqrt(T)
    df_r = np.exp(-r * T)
    df_q = np.exp(-q * T)
    F = S * np.exp((r - q) * T)
    regular = (sd > 0) & (S > 0)
    sd_safe = np.where(regular, sd, 1.0)
    S_safe = np.where(S > 0, S, 1.0)
    d1_reg = (np.log(S_safe / K) + (r - q) * T + 0.5 * sd_safe**2) / sd_safe
    # deterministic limit: d1 = d2 = +inf if F > K, -inf if F < K, nan at the kink F == K
    d_lim = np.where(F > K, np.inf, np.where(F < K, -np.inf, np.nan))
    d1 = np.where(regular, d1_reg, d_lim)
    d2 = np.where(regular, d1_reg - sd_safe, d_lim)
    return dict(sd=sd, df_r=df_r, df_q=df_q, F=F, regular=regular, d1=d1, d2=d2,
                sd_safe=sd_safe, S_safe=S_safe, kink=(~regular) & (F == K))


def price(S, K, T, r, q, sigma, kind="call") -> np.ndarray | float:
    """European BSM price with continuous dividend yield.

    Returns a NumPy scalar (0-d -> float) or an array with the broadcast shape.
    """
    S, K, T, r, q, sigma = _prepare(S, K, T, r, q, sigma)
    c = _core(S, K, T, r, q, sigma)
    is_call = _is_call(kind, S.shape)
    F, df_r = c["F"], c["df_r"]
    # In the kink case (F == K, deterministic) both N terms are nan; the limit value is 0.
    d1 = np.where(c["kink"], 0.0, c["d1"])
    d2 = np.where(c["kink"], 0.0, c["d2"])
    call = df_r * (F * ndtr(d1) - K * ndtr(d2))
    put = df_r * (K * ndtr(-d2) - F * ndtr(-d1))
    # Deterministic and kink cases: use the exact intrinsic-forward form (no cancellation).
    det = ~c["regular"]
    call = np.where(det, df_r * np.maximum(F - K, 0.0), call)
    put = np.where(det, df_r * np.maximum(K - F, 0.0), put)
    out = np.where(is_call, call, put)
    # floor tiny negative round-off; prices are never negative in this model
    out = np.maximum(out, 0.0)
    return out if out.ndim else float(out)


def greeks(S, K, T, r, q, sigma, kind="call") -> dict[str, np.ndarray]:
    """All analytical Greeks in raw units.

    Returns a dict with keys ``price, delta, gamma, vega, theta, rho, psi, vanna, volga``.
    ``theta`` is the calendar-time derivative (per year). ``psi = dV/dq``.
    """
    S, K, T, r, q, sigma = _prepare(S, K, T, r, q, sigma)
    c = _core(S, K, T, r, q, sigma)
    is_call = _is_call(kind, S.shape)
    d1, d2, df_r, df_q = c["d1"], c["d2"], c["df_r"], c["df_q"]
    reg, kink = c["regular"], c["kink"]
    sd_safe, S_safe = c["sd_safe"], c["S_safe"]
    sqrtT = np.sqrt(T)
    sig_safe = np.where(sigma > 0, sigma, 1.0)
    sqrtT_safe = np.where(T > 0, sqrtT, 1.0)

    Nd1, Nd2 = ndtr(d1), ndtr(d2)
    Nmd1, Nmd2 = ndtr(-d1), ndtr(-d2)
    pdf1 = np.where(reg, _npdf(np.where(reg, d1, 0.0)), 0.0)

    delta_c = df_q * Nd1
    delta_p = df_q * (Nd1 - 1.0)
    gamma = np.where(reg, df_q * pdf1 / (S_safe * sd_safe), 0.0)
    vega = np.where(reg, S * df_q * pdf1 * sqrtT, 0.0)
    decay = np.where(reg, -S * df_q * pdf1 * sigma / (2.0 * sqrtT_safe), 0.0)
    theta_c = decay - r * K * df_r * Nd2 + q * S * df_q * Nd1
    theta_p = decay + r * K * df_r * Nmd2 - q * S * df_q * Nmd1
    rho_c = K * T * df_r * Nd2
    rho_p = -K * T * df_r * Nmd2
    psi_c = -S * T * df_q * Nd1
    psi_p = S * T * df_q * Nmd1
    d1s = np.where(reg, d1, 0.0)
    d2s = np.where(reg, d2, 0.0)
    vanna = np.where(reg, -df_q * pdf1 * d2s / sig_safe, 0.0)
    volga = np.where(reg, vega * d1s * d2s / sig_safe, 0.0)

    out = {
        "price": np.asarray(price(S, K, T, r, q, sigma, np.where(is_call, "call", "put"))),
        "delta": np.where(is_call, delta_c, delta_p),
        "gamma": gamma,
        "vega": vega,
        "theta": np.where(is_call, theta_c, theta_p),
        "rho": np.where(is_call, rho_c, rho_p),
        "psi": np.where(is_call, psi_c, psi_p),
        "vanna": vanna,
        "volga": volga,
    }
    # Undefined at the payoff kink of the deterministic limit.
    for key in ("delta", "gamma", "vega", "vanna", "volga", "theta", "rho", "psi"):
        out[key] = np.where(kink, np.nan, out[key])
    return out


def _single(name):
    def f(S, K, T, r, q, sigma, kind="call"):
        g = greeks(S, K, T, r, q, sigma, kind)[name]
        return g if g.ndim else float(g)

    f.__name__ = name
    f.__doc__ = f"Analytical BSM {name} in raw units (see :func:`greeks`)."
    return f


delta = _single("delta")
gamma = _single("gamma")
vega = _single("vega")
theta = _single("theta")
rho = _single("rho")
psi = _single("psi")
vanna = _single("vanna")
volga = _single("volga")


def european_bounds(S, K, T, r, q, kind="call") -> tuple[np.ndarray, np.ndarray]:
    """Model-free no-arbitrage bounds for a European option with continuous yield.

    call: ``max(S e^{-qT} - K e^{-rT}, 0) <= C <= S e^{-qT}``
    put:  ``max(K e^{-rT} - S e^{-qT}, 0) <= P <= K e^{-rT}``
    """
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(S, K, T, r, q))
    is_call = _is_call(kind, S.shape)
    pv_s = S * np.exp(-q * T)
    pv_k = K * np.exp(-r * T)
    lo = np.where(is_call, np.maximum(pv_s - pv_k, 0.0), np.maximum(pv_k - pv_s, 0.0))
    hi = np.where(is_call, pv_s, pv_k)
    return lo, hi


def american_bounds(S, K, T, r, q, kind="call") -> tuple[np.ndarray, np.ndarray]:
    """Simple no-arbitrage bounds for an American option (continuous yield).

    lower: ``max(intrinsic, European lower bound)``; upper: ``S`` (call) or ``K`` (put).
    These are inequalities, not a parity relation.
    """
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(S, K, T, r, q))
    is_call = _is_call(kind, S.shape)
    lo_e, _ = european_bounds(S, K, T, r, q, kind)
    intrinsic = np.where(is_call, np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    lo = np.maximum(lo_e, intrinsic)
    hi = np.where(is_call, S, np.maximum(K, K * np.exp(-r * T)))
    return lo, hi


def parity_residual(call, put, S, K, T, r, q) -> np.ndarray:
    """European put-call parity residual ``C - P - (S e^{-qT} - K e^{-rT})``.

    Holds as an equality only for European options on the same strike and expiry, under
    continuous yield ``q``. For American options use :func:`american_bounds` /
    :func:`american_parity_bounds` instead.
    """
    return np.asarray(call) - np.asarray(put) - (np.asarray(S) * np.exp(-np.asarray(q) * T)
                                                 - np.asarray(K) * np.exp(-np.asarray(r) * T))


def american_parity_bounds(S, K, T, r, q) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on ``C_am - P_am`` with continuous dividend yield ``q`` and ``r >= 0``, ``q >= 0``.

    ``S e^{-qT} - K <= C_am - P_am <= S - K e^{-rT}`` (Hull's result, generalised to a
    continuous yield). This needs non-negative ``r`` and ``q``; outside that regime the
    function raises instead of returning a bound that may be wrong.
    """
    S, K, T, r, q = (np.asarray(x, dtype=float) for x in np.broadcast_arrays(S, K, T, r, q))
    if np.any(r < 0) or np.any(q < 0):
        raise InputError("american_parity_bounds is stated here only for r >= 0 and q >= 0")
    return S * np.exp(-q * T) - K, S - K * np.exp(-r * T)
