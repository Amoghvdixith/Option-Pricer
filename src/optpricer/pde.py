r"""Finite-difference solvers for the Black-Scholes PDE in log-spot.

Formulation
-----------
With ``x = ln S`` and time to expiry ``tau = T - t``, the BSM PDE becomes the
constant-coefficient problem

    V_tau = 1/2 sigma^2 V_xx + nu V_x - r V,      nu = r - q - sigma^2 / 2,

with ``V(x, 0) = payoff(e^x)``. On a uniform grid ``x_0 < ... < x_M`` with spacing ``dx``, the
centred operator is ``(L V)_i = a V_{i-1} + b V_i + c V_{i+1}``, where

    a = sigma^2/(2 dx^2) - nu/(2 dx),  b = -sigma^2/dx^2 - r,  c = sigma^2/(2 dx^2) + nu/(2 dx).

The theta-scheme from ``tau_k`` to ``tau_{k+1} = tau_k + h`` is

    (I - theta h L) V^{k+1} = (I + (1 - theta) h L) V^k   (+ Dirichlet boundary terms)

with ``theta = 0`` explicit, ``1`` implicit (backward Euler) and ``1/2`` Crank-Nicolson.
*Rannacher start-up*: the first ``rannacher_steps`` CN steps are each replaced by two
implicit half-steps (``h/2``). This damps the high-frequency error introduced by the
payoff kink. Linear (von Neumann) stability of CN does **not** guarantee positivity or
smooth Greeks.

Domain
------
The log-domain is centred on both ``ln S`` and ``ln K``, with half-width
``n_sd * sigma sqrt(T) + |nu| T`` (default ``n_sd = 6``). It therefore scales with
volatility and maturity instead of a fixed ``S_max = 4K``. The grid is shifted so that
``ln S`` is exactly a node, which removes interpolation error at the requested spot.
Truncation sensitivity is measured by varying ``n_sd`` (see tests and
``scripts/run_convergence.py``).

Boundaries (Dirichlet, style-appropriate, valid as ``S -> 0`` / ``S -> inf``)
---------------------------------------------------------------------------
* European call:  low 0;  high ``S e^{-q tau} - K e^{-r tau}``
* European put:   low ``K e^{-r tau} - S e^{-q tau}``;  high 0
* American call:  low 0;  high ``max(S - K, S e^{-q tau} - K e^{-r tau})``
* American put:   low ``max(K - S, K e^{-r tau} - S e^{-q tau})``;  high 0

Stability of the explicit scheme
--------------------------------
Two checks are derived for the actual discretisation and recorded:
* *von Neumann*: the amplification factor
  ``g(phi) = 1 + h [b + (a + c) cos phi + i (c - a) sin phi]`` must satisfy
  ``|g| <= 1 + max(0, -r) h`` for all ``phi``. It is evaluated on a dense grid of ``phi``.
* *monotonicity / positivity* (sufficient, stricter): ``a >= 0``, ``c >= 0`` (equivalently
  ``|nu| dx <= sigma^2``) and ``1 + h b >= 0`` (``h (sigma^2/dx^2 + r) <= 1``).
An explicit configuration that fails von Neumann raises :class:`UnstableSchemeError`
unless ``allow_unstable=True``. A failed monotonicity check is reported in the diagnostics.
The guide's single rule ``dt <= dx^2/sigma^2`` is roughly the von Neumann
condition for small ``r`` and ``nu``; it is not a universal sufficient condition.

American options
----------------
Each implicit/CN step is a linear complementarity problem

    A V >= rhs,   V >= g,   (A V - rhs) . (V - g) = 0.

It is solved by projected SOR (red-black ordering, vectorised) with tolerance ``tol``
on the max-norm update, relaxation ``omega`` and an iteration cap. If any step fails to
converge, :class:`ConvergenceError` is raised unless ``allow_nonconvergence=True``. After
each step the obstacle residual ``min(V - g)`` and the complementarity residual
``max |min(V - g, A V - rhs)|`` are recorded. The explicit scheme needs no linear solve,
so for it the projection ``max(V, g)`` is the exact LCP solution of that step.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded

from .errors import ConvergenceError, InputError, UnstableSchemeError
from .instruments import MarketData, Option

SCHEMES = {"explicit": 0.0, "implicit": 1.0, "cn": 0.5}


# --------------------------------------------------------------------------- linear algebra
def thomas(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system with the Thomas algorithm (no pivoting).

    ``lower[i]`` multiplies ``x[i-1]`` in row ``i`` (``lower[0]`` is ignored) and
    ``upper[i]`` multiplies ``x[i+1]`` (``upper[-1]`` is ignored). Stability needs a
    diagonally dominant or otherwise well-conditioned matrix; the implicit and CN
    matrices here are diagonally dominant whenever ``a, c >= 0`` and ``r >= 0``.
    """
    n = len(diag)
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if not (len(lower) == len(upper) == len(rhs) == n):
        raise InputError("thomas: all bands and rhs must have the same length")
    cp = np.empty(n)
    dp = np.empty(n)
    denom = diag[0]
    if denom == 0:
        raise ZeroDivisionError("thomas: zero pivot at row 0")
    cp[0] = upper[0] / denom
    dp[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i] * cp[i - 1]
        if denom == 0:
            raise ZeroDivisionError(f"thomas: zero pivot at row {i}")
        cp[i] = upper[i] / denom
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / denom
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def _banded(lower: float, diag: float, upper: float, rhs: np.ndarray) -> np.ndarray:
    n = len(rhs)
    ab = np.zeros((3, n))
    ab[0, 1:] = upper
    ab[1, :] = diag
    ab[2, :-1] = lower
    return solve_banded((1, 1), ab, rhs)


def psor(lower: float, diag: float, upper: float, rhs: np.ndarray, obstacle: np.ndarray,
         x0: np.ndarray, left: float, right: float, omega: float = 1.2, tol: float = 1e-10,
         max_iter: int = 10_000) -> tuple[np.ndarray, int, bool]:
    """Projected SOR (red-black ordering) for one tridiagonal LCP.

    Interior unknowns ``x`` (length ``m``) satisfy
    ``lower*x[i-1] + diag*x[i] + upper*x[i+1] >= rhs[i]``, ``x >= obstacle`` and complementarity,
    with fixed boundary values ``left`` (= x[-1]) and ``right`` (= x[m]).
    Returns ``(x, iterations, converged)``. Convergence: max-norm update < ``tol``.
    """
    if not (0.0 < omega < 2.0):
        raise InputError(f"PSOR relaxation omega must lie in (0, 2), got {omega}")
    m = len(rhs)
    w = np.empty(m + 2)
    w[0], w[-1] = left, right
    w[1:-1] = np.maximum(x0, obstacle)
    g = obstacle
    red = np.arange(1, m + 1, 2)
    black = np.arange(2, m + 1, 2)
    for it in range(1, max_iter + 1):
        prev = w[1:-1].copy()
        for idx in (red, black):
            gs = (rhs[idx - 1] - lower * w[idx - 1] - upper * w[idx + 1]) / diag
            w[idx] = np.maximum(g[idx - 1], w[idx] + omega * (gs - w[idx]))
        if np.max(np.abs(w[1:-1] - prev)) < tol:
            return w[1:-1].copy(), it, True
    return w[1:-1].copy(), max_iter, False


# --------------------------------------------------------------------------- results
@dataclass
class PDEResult:
    """Price, grid Greeks and solver diagnostics."""

    price: float
    scheme: str
    n_space: int
    n_time: int
    delta: float = math.nan
    gamma: float = math.nan
    theta: float = math.nan
    runtime: float = 0.0
    diagnostics: dict = field(default_factory=dict)
    boundary_t: np.ndarray = field(default_factory=lambda: np.empty(0))
    boundary_S: np.ndarray = field(default_factory=lambda: np.empty(0))
    S_grid: np.ndarray | None = None
    V_final: np.ndarray | None = None
    tau_grid: np.ndarray | None = None
    V_history: np.ndarray | None = None


def explicit_stability(a: float, b: float, c: float, h: float, r: float, n_phi: int = 4001) -> dict:
    """Von Neumann and monotonicity diagnostics for the explicit log-space scheme."""
    phi = np.linspace(0.0, np.pi, n_phi)
    g = 1.0 + h * (b + (a + c) * np.cos(phi) + 1j * (c - a) * np.sin(phi))
    amp = float(np.max(np.abs(g)))
    allowed = 1.0 + max(0.0, -r) * h + 1e-12
    return {
        "von_neumann_max_amplification": amp,
        "von_neumann_allowed": allowed,
        "von_neumann_stable": amp <= allowed,
        "monotone": bool(a >= 0 and c >= 0 and 1.0 + h * b >= 0),
        "coef_a": a, "coef_c": c, "one_plus_h_b": 1.0 + h * b,
    }


def _boundaries(opt: Option, S_lo: float, S_hi: float, r: float, q: float, tau: float) -> tuple[float, float]:
    K = opt.K
    dfr, dfq = math.exp(-r * tau), math.exp(-q * tau)
    if opt.is_call:
        hi = S_hi * dfq - K * dfr
        if opt.is_american:
            hi = max(S_hi - K, hi)
        return 0.0, hi
    lo = K * dfr - S_lo * dfq
    if opt.is_american:
        lo = max(K - S_lo, lo)
    return lo, 0.0


def _cell_average_payoff(x: np.ndarray, dx: float, K: float, is_call: bool) -> np.ndarray:
    """Exact average of the payoff over [x_i - dx/2, x_i + dx/2] in log-space."""
    a = x - 0.5 * dx
    b = x + 0.5 * dx
    lk = math.log(K)
    if is_call:
        lo = np.maximum(a, lk)
        val = np.where(lo < b, (np.exp(b) - np.exp(lo)) - K * (b - lo), 0.0)
    else:
        hi = np.minimum(b, lk)
        val = np.where(a < hi, K * (hi - a) - (np.exp(hi) - np.exp(a)), 0.0)
    return val / dx


def _deterministic(opt: Option, mkt: MarketData, n_time: int, scheme: str) -> PDEResult:
    if opt.T == 0:
        return PDEResult(float(opt.payoff(mkt.S)), scheme, 0, 0, diagnostics={"note": "T == 0: intrinsic"})
    times = np.linspace(0.0, opt.T, n_time + 1)
    path = mkt.S * np.exp((mkt.r - mkt.q) * times)
    disc = np.exp(-mkt.r * times) * opt.payoff(path)
    val = float(disc.max() if opt.is_american else disc[-1])
    return PDEResult(val, scheme, 0, n_time, diagnostics={"note": "degenerate (sigma=0 or S=0): deterministic path"})


def fd_price(opt: Option, mkt: MarketData, scheme: str = "cn", n_space: int = 400, n_time: int = 400, *,
             rannacher_steps: int = 2, n_sd: float = 6.0, smooth_payoff: bool = True,
             linear_solver: str = "banded", omega: float = 1.2, tol: float = 1e-10,
             max_iter: int = 10_000, allow_unstable: bool = False, allow_nonconvergence: bool = False,
             return_grid: bool = False) -> PDEResult:
    """Price an option with the log-space theta-scheme.

    Parameters
    ----------
    scheme : 'explicit', 'implicit' or 'cn'.
    n_space, n_time : number of space intervals ``M`` and time steps ``N``.
    rannacher_steps : CN only. Number of initial CN steps replaced by two implicit half-steps.
    n_sd : domain half-width in standard deviations ``sigma sqrt(T)`` (plus drift).
    smooth_payoff : use the exact cell average of the payoff as the initial condition
        (restores the second-order behaviour that a kink between nodes destroys).
    linear_solver : 'banded' (``scipy.linalg.solve_banded``) or 'thomas' (pure Python,
        used for validation). Both solve the same tridiagonal system.
    omega, tol, max_iter : PSOR parameters (American only).
    allow_unstable : permit an explicit configuration that fails von Neumann (for the
        instability demonstration only).
    allow_nonconvergence : return a PSOR result that missed tolerance instead of raising;
        the diagnostics then show ``psor_converged = False``.
    return_grid : also return the spot grid, final values and the full time history.
    """
    if scheme not in SCHEMES:
        raise InputError(f"scheme must be one of {sorted(SCHEMES)}, got {scheme!r}")
    if linear_solver not in ("banded", "thomas"):
        raise InputError("linear_solver must be 'banded' or 'thomas'")
    for name, v in (("n_space", n_space), ("n_time", n_time)):
        if int(v) != v or v < 3:
            raise InputError(f"{name} must be an integer >= 3, got {v!r}")
    n_space, n_time = int(n_space), int(n_time)
    if n_sd <= 0:
        raise InputError("n_sd must be positive")
    if rannacher_steps < 0 or rannacher_steps > n_time:
        raise InputError("rannacher_steps must lie in [0, n_time]")
    if opt.T == 0 or mkt.sigma == 0 or mkt.S == 0:
        return _deterministic(opt, mkt, n_time, scheme)

    t_start = time.perf_counter()
    S0, K, T, r, q, sig = mkt.S, opt.K, opt.T, mkt.r, mkt.q, mkt.sigma
    theta_s = SCHEMES[scheme]
    nu = r - q - 0.5 * sig * sig
    x0, lk = math.log(S0), math.log(K)
    half = n_sd * sig * math.sqrt(T) + abs(nu) * T
    lo, hi = min(x0, lk) - half, max(x0, lk) + half
    dx = (hi - lo) / n_space
    i0 = int(round((x0 - lo) / dx))
    i0 = min(max(i0, 1), n_space - 1)
    lo = x0 - i0 * dx
    x = lo + dx * np.arange(n_space + 1)
    S = np.exp(x)
    h = T / n_time

    a = sig * sig / (2 * dx * dx) - nu / (2 * dx)
    b = -sig * sig / (dx * dx) - r
    c = sig * sig / (2 * dx * dx) + nu / (2 * dx)

    diag: dict = {
        "scheme": scheme, "theta": theta_s, "n_space": n_space, "n_time": n_time, "dx": dx, "dtau": h,
        "S_min": float(S[0]), "S_max": float(S[-1]), "n_sd": n_sd, "spot_node": i0,
        "smooth_payoff": smooth_payoff, "linear_solver": linear_solver,
        "rannacher_steps": rannacher_steps if scheme == "cn" else 0,
        "implicit_part_M_matrix": bool(a >= 0 and c >= 0),
    }
    stab = explicit_stability(a, b, c, h, r)
    if scheme == "explicit":
        diag.update(stab)
        if not stab["von_neumann_stable"] and not allow_unstable:
            raise UnstableSchemeError(
                f"explicit scheme unstable: max |g| = {stab['von_neumann_max_amplification']:.4f} > "
                f"{stab['von_neumann_allowed']:.4f} (dtau={h:.3g}, dx={dx:.3g}, sigma^2 dtau/dx^2="
                f"{sig * sig * h / dx / dx:.3f}). Increase n_time to at least about "
                f"{int(math.ceil(T * (sig * sig / dx / dx + abs(r)) * 1.01)) + 1}, use 'implicit'/'cn', "
                "or pass allow_unstable=True for an intentional demonstration.")

    is_call, american = opt.is_call, opt.is_american
    g = opt.payoff(S)
    V = _cell_average_payoff(x, dx, K, is_call) if smooth_payoff else g.copy()
    if american:
        V = np.maximum(V, g)
    history = [V.copy()] if return_grid else None

    # step plan: list of (theta, step size)
    plan: list[tuple[float, float]] = []
    if scheme == "cn" and rannacher_steps > 0:
        plan += [(1.0, h / 2)] * (2 * rannacher_steps)
        plan += [(0.5, h)] * (n_time - rannacher_steps)
    else:
        plan += [(theta_s, h)] * n_time

    tau = 0.0
    V_before = V
    psor_iters: list[int] = []
    psor_ok = True
    max_obstacle_violation = 0.0
    max_comp = 0.0
    b_t: list[float] = []
    b_S: list[float] = []
    errstate = np.errstate(over="ignore", invalid="ignore") if allow_unstable else np.errstate()
    with errstate:
        for th, step in plan:
            tau_new = tau + step
            lo_old, hi_old = _boundaries(opt, S[0], S[-1], r, q, tau)
            lo_new, hi_new = _boundaries(opt, S[0], S[-1], r, q, tau_new)
            Vi = V[1:-1]
            LV = a * V[:-2] + b * Vi + c * V[2:]
            rhs = Vi + (1.0 - th) * step * LV
            newV = np.empty_like(V)
            newV[0], newV[-1] = lo_new, hi_new
            if th == 0.0:
                newV[1:-1] = rhs
                if american:
                    newV = np.maximum(newV, g)
            else:
                lw, dg, up = -th * step * a, 1.0 - th * step * b, -th * step * c
                if american:
                    sol, its, ok = psor(lw, dg, up, rhs, g[1:-1], Vi, lo_new, hi_new, omega, tol, max_iter)
                    psor_iters.append(its)
                    if not ok:
                        psor_ok = False
                        if not allow_nonconvergence:
                            raise ConvergenceError(
                                f"PSOR did not converge at tau={tau_new:.6g} after {its} iterations "
                                f"(omega={omega}, tol={tol}). Increase max_iter, adjust omega, or refine dtau.")
                    newV[1:-1] = sol
                    Ax = lw * newV[:-2] + dg * sol + up * newV[2:]
                    resid = Ax - rhs
                    gap = sol - g[1:-1]
                    max_obstacle_violation = max(max_obstacle_violation, float(max(0.0, -gap.min())))
                    max_comp = max(max_comp, float(np.max(np.abs(np.minimum(gap, resid)))))
                else:
                    rr = rhs.copy()
                    rr[0] -= lw * lo_new
                    rr[-1] -= up * hi_new
                    if linear_solver == "thomas":
                        m = len(rr)
                        newV[1:-1] = thomas(np.full(m, lw), np.full(m, dg), np.full(m, up), rr)
                    else:
                        newV[1:-1] = _banded(lw, dg, up, rr)
            if american:
                ex = (newV - g <= 1e-9 * K) & (g > 0)
                ex[0] = ex[-1] = False
                if ex.any():
                    b_t.append(T - tau_new)
                    b_S.append(float(S[ex].max() if not is_call else S[ex].min()))
            V_before = V
            V = newV
            tau = tau_new
            if history is not None:
                history.append(V.copy())
    V_last_step_start = V_before  # value one (last) step before tau = T
    price = float(V[i0])
    res = PDEResult(price, scheme, n_space, n_time)
    res.delta = float((V[i0 + 1] - V[i0 - 1]) / (2 * dx) / S0)
    vxx = (V[i0 + 1] - 2 * V[i0] + V[i0 - 1]) / (dx * dx)
    vx = (V[i0 + 1] - V[i0 - 1]) / (2 * dx)
    res.gamma = float((vxx - vx) / (S0 * S0))
    last_step = plan[-1][1]
    res.theta = float(-(V[i0] - V_last_step_start[i0]) / last_step)
    diag["nonfinite_values"] = bool(not np.all(np.isfinite(V)))
    if scheme != "explicit":
        diag["explicit_reference_stability"] = stab["von_neumann_stable"]
    if american and scheme == "explicit":
        diag["american_method"] = "explicit step followed by projection max(V, g) (exact LCP for explicit)"
    elif american:
        diag.update({
            "american_method": "PSOR (red-black) on each implicit/CN step",
            "psor_omega": omega, "psor_tol": tol, "psor_max_iter": max_iter,
            "psor_converged": psor_ok,
            "psor_total_iterations": int(sum(psor_iters)),
            "psor_max_iterations_per_step": int(max(psor_iters)) if psor_iters else 0,
            "psor_mean_iterations_per_step": float(np.mean(psor_iters)) if psor_iters else 0.0,
            "obstacle_violation_max": max_obstacle_violation,
            "complementarity_residual_max": max_comp,
        })
    res.diagnostics = diag
    res.boundary_t = np.asarray(b_t[::-1])
    res.boundary_S = np.asarray(b_S[::-1])
    if return_grid:
        res.S_grid = S
        res.V_final = V
        taus = np.concatenate([[0.0], np.cumsum([p[1] for p in plan])])
        res.tau_grid = taus
        res.V_history = np.asarray(history)
    res.runtime = time.perf_counter() - t_start
    return res
