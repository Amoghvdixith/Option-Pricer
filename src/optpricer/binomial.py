"""Recombining binomial trees: Cox-Ross-Rubinstein and Leisen-Reimer.

Node convention
---------------
Time step ``i = 0..n`` (``t_i = i * dt``, ``dt = T/n``). Node ``j = 0..i`` is the number of
up-moves, so ``S_{i,j} = S u^j d^(i-j)`` and nodes are in **ascending** price order within
each step (``u > d``). Backward induction is vectorised over nodes:

    V_i = disc * (p * V_{i+1}[1:] + (1 - p) * V_{i+1}[:-1])

For American options, ``V_i = max(V_i, payoff(S_i))`` at every step ``i < n``. The
exercise boundary is recorded as the highest (put) or lowest (call) node at which
immediate exercise is strictly optimal and the intrinsic value is positive.

Risk-neutral probabilities must lie strictly inside (0, 1); otherwise a
:class:`~optpricer.errors.ProbabilityError` is raised. For CRR this fails when
``|r - q| sqrt(dt)`` is large compared with ``sigma``.

Leisen-Reimer (1996) uses Peizer-Pratt inversion method 2. The construction is
defined for **odd** ``n``. Even ``n`` raises an error; it is never silently rounded.

Degenerate cases (handled separately, without the lattice):
``T == 0``: intrinsic value. ``sigma == 0``: deterministic forward path ``S e^{(r-q)t}``;
European = discounted terminal payoff, American = max over the n+1 grid dates of the
discounted payoff.

Greek conventions (tree):
* delta from the two step-1 nodes, gamma from the three step-2 nodes (ratio of
  delta differences over half the outer node spacing).
* theta only for CRR, where ``u d = 1`` places the middle step-2 node exactly at ``S``:
  ``theta = (V_{2,1} - V_{0,0}) / (2 dt)``, calendar time per year. LR has ``u d != 1``, so the
  same difference would mix a spot move into theta; it is reported as NaN.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .errors import InputError, ProbabilityError
from .instruments import MarketData, Option


@dataclass
class TreeResult:
    """Price and diagnostics from a binomial tree."""

    price: float
    method: str
    n: int
    delta: float = math.nan
    gamma: float = math.nan
    theta: float = math.nan
    u: float = math.nan
    d: float = math.nan
    p: float = math.nan
    runtime: float = 0.0
    boundary_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    boundary_S: np.ndarray = field(default_factory=lambda: np.empty(0))
    notes: str = ""


def _payoff(S: np.ndarray, K: float, is_call: bool) -> np.ndarray:
    return np.maximum(S - K, 0.0) if is_call else np.maximum(K - S, 0.0)


def crr_parameters(T: float, n: int, r: float, q: float, sigma: float) -> tuple[float, float, float]:
    """CRR ``u = exp(sigma sqrt(dt))``, ``d = 1/u``, ``p = (e^{(r-q)dt} - d) / (u - d)``."""
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    return u, d, p


def peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion, method 2 (Leisen & Reimer 1996, eq. for h^{-1}).

    ``h(z) = 1/2 + sign(z)/2 * sqrt(1 - exp(-(z/(n + 1/3 + 0.1/(n+1)))^2 * (n + 1/6)))``
    """
    a = z / (n + 1.0 / 3.0 + 0.1 / (n + 1.0))
    val = 0.5 + math.copysign(0.5, z) * math.sqrt(max(0.0, 1.0 - math.exp(-a * a * (n + 1.0 / 6.0))))
    return val


def lr_parameters(S: float, K: float, T: float, n: int, r: float, q: float, sigma: float
                  ) -> tuple[float, float, float]:
    """Leisen-Reimer ``u, d, p`` for odd ``n`` (Peizer-Pratt method 2)."""
    if n % 2 == 0:
        raise InputError(f"Leisen-Reimer requires an odd number of steps; got n={n}. "
                         "Pass an odd n explicitly (the library does not round it).")
    dt = T / n
    sd = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sd
    d2 = d1 - sd
    p = peizer_pratt(d2, n)
    p_bar = peizer_pratt(d1, n)
    growth = math.exp((r - q) * dt)
    u = growth * p_bar / p
    d = (growth - p * u) / (1.0 - p)
    return u, d, p


def _deterministic(opt: Option, mkt: MarketData, n: int, method: str) -> TreeResult:
    t0 = time.perf_counter()
    if opt.T == 0:
        val = float(opt.payoff(mkt.S))
        return TreeResult(val, method, n, runtime=time.perf_counter() - t0, notes="T == 0: intrinsic value")
    times = np.linspace(0.0, opt.T, n + 1)
    path = mkt.S * np.exp((mkt.r - mkt.q) * times)
    disc_pay = np.exp(-mkt.r * times) * opt.payoff(path)
    val = float(disc_pay[-1] if not opt.is_american else disc_pay.max())
    return TreeResult(val, method, n, runtime=time.perf_counter() - t0,
                      notes="sigma == 0: deterministic forward path (no lattice)")


def _induct(opt: Option, S0: float, r: float, T: float, n: int, u: float, d: float, p: float,
            method: str, want_theta: bool, record_boundary: bool) -> TreeResult:
    if not (0.0 < p < 1.0) or not np.isfinite(p):
        raise ProbabilityError(
            f"{method}: risk-neutral probability p={p:.6g} is outside (0, 1) for n={n}; "
            "increase n or check r, q, sigma.")
    t0 = time.perf_counter()
    dt = T / n
    disc = math.exp(-r * dt)
    K, is_call, american = opt.K, opt.is_call, opt.is_american
    j = np.arange(n + 1)
    S = S0 * u**j * d ** (n - j)
    V = _payoff(S, K, is_call)
    level1 = level2 = None
    b_t: list[float] = []
    b_S: list[float] = []
    for i in range(n - 1, -1, -1):
        S = S[:i + 1] * (1.0 / d)  # S_{i,j} = S_{i+1,j} / d  (same j, one fewer down-move)
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if american:
            ex = _payoff(S, K, is_call)
            if record_boundary:
                exercised = (ex > V) & (ex > 0)
                if exercised.any():
                    b_t.append(i * dt)
                    b_S.append(float(np.max(S[exercised]) if not is_call else np.min(S[exercised])))
            V = np.maximum(V, ex)
        if i == 2:
            level2 = (S.copy(), V.copy())
        elif i == 1:
            level1 = (S.copy(), V.copy())
    res = TreeResult(float(V[0]), method, n, u=u, d=d, p=p)
    if level1 is not None:
        (S1, V1) = level1
        res.delta = float((V1[1] - V1[0]) / (S1[1] - S1[0]))
    if level2 is not None:
        (S2, V2) = level2
        du = (V2[2] - V2[1]) / (S2[2] - S2[1])
        dd = (V2[1] - V2[0]) / (S2[1] - S2[0])
        res.gamma = float((du - dd) / (0.5 * (S2[2] - S2[0])))
        if want_theta:
            res.theta = float((V2[1] - V[0]) / (2.0 * dt))
    res.runtime = time.perf_counter() - t0
    if record_boundary:
        res.boundary_t = np.asarray(b_t[::-1])
        res.boundary_S = np.asarray(b_S[::-1])
    return res


def _check_n(n: int) -> int:
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise InputError(f"number of steps n must be a positive integer, got {n!r}")
    return int(n)


def crr(opt: Option, mkt: MarketData, n: int = 500, record_boundary: bool = False) -> TreeResult:
    """Cox-Ross-Rubinstein tree price, with delta/gamma/theta from the lattice."""
    n = _check_n(n)
    if opt.T == 0 or mkt.sigma == 0:
        return _deterministic(opt, mkt, n, "crr")
    if mkt.S == 0:
        return _deterministic(opt, mkt.with_(sigma=0.0), n, "crr")
    u, d, p = crr_parameters(opt.T, n, mkt.r, mkt.q, mkt.sigma)
    return _induct(opt, mkt.S, mkt.r, opt.T, n, u, d, p, "crr", True, record_boundary)


def crr_averaged(opt: Option, mkt: MarketData, n: int = 500) -> TreeResult:
    """Average of CRR prices at ``n`` and ``n + 1`` steps (damps odd/even oscillation)."""
    a, b = crr(opt, mkt, n), crr(opt, mkt, n + 1)
    out = TreeResult(0.5 * (a.price + b.price), "crr_avg", n,
                     delta=0.5 * (a.delta + b.delta), gamma=0.5 * (a.gamma + b.gamma),
                     theta=0.5 * (a.theta + b.theta), runtime=a.runtime + b.runtime,
                     notes=f"mean of CRR n={n} and n={n + 1}")
    return out


def leisen_reimer(opt: Option, mkt: MarketData, n: int = 201, record_boundary: bool = False) -> TreeResult:
    """Leisen-Reimer tree (odd ``n`` only), with delta/gamma; theta is NaN (``u d != 1``)."""
    n = _check_n(n)
    if n % 2 == 0:
        raise InputError(f"Leisen-Reimer requires an odd number of steps; got n={n}. "
                         "Pass an odd n explicitly (the library does not round it).")
    if opt.T == 0 or mkt.sigma == 0:
        return _deterministic(opt, mkt, n, "lr")
    if mkt.S == 0:
        return _deterministic(opt, mkt.with_(sigma=0.0), n, "lr")
    u, d, p = lr_parameters(mkt.S, opt.K, opt.T, n, mkt.r, mkt.q, mkt.sigma)
    res = _induct(opt, mkt.S, mkt.r, opt.T, n, u, d, p, "lr", False, record_boundary)
    res.notes = "theta not reported: LR has u*d != 1, so no step-2 node sits at S"
    return res


def tree_greeks(opt: Option, mkt: MarketData, n: int = 501, method: str = "crr") -> dict[str, float]:
    """Delta, gamma and (CRR only) theta read off the lattice."""
    res = crr(opt, mkt, n) if method == "crr" else leisen_reimer(opt, mkt, n)
    return {"price": res.price, "delta": res.delta, "gamma": res.gamma, "theta": res.theta}
