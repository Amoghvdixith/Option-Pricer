"""Monte Carlo pricing of European options under risk-neutral GBM.

Terminal prices are sampled exactly,

    S_T = S exp((r - q - sigma^2/2) T + sigma sqrt(T) Z),   Z ~ N(0, 1),

so the estimator has **no time-discretisation bias**; its only error is sampling error.
That is quantified by the standard error of the mean of independent estimation units:

* plain sampling: each discounted payoff is one unit;
* antithetic sampling: each pair average ``(f(Z) + f(-Z)) / 2`` is one unit. The SE is
  computed from the pair averages, never from the 2n correlated payoffs.

Large runs are processed in chunks, and the moments are combined with Chan's parallel
update, so memory stays bounded. A Monte Carlo confidence interval describes sampling
uncertainty only. It says nothing about model error and cannot be used to justify the
deterministic bias of a tree or PDE.

Only European (terminal-payoff) contracts are supported here. American exercise
needs a regression method such as Longstaff-Schwartz, which is **not** implemented.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from .errors import InputError, UnsupportedMethodError
from .instruments import MarketData, Option


@dataclass
class MCResult:
    """Monte Carlo estimate with its sampling uncertainty."""

    price: float
    std_error: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_units: int
    n_normals: int
    antithetic: bool
    seed: int | None
    runtime: float
    method: str = "mc_terminal_gbm"

    def to_dict(self) -> dict:
        return asdict(self)


class _Moments:
    """Streaming count / mean / M2 (Chan et al. parallel combination)."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, x: np.ndarray) -> None:
        nb = x.size
        if nb == 0:
            return
        mb = float(x.mean())
        m2b = float(((x - mb) ** 2).sum())
        n = self.n + nb
        delta = mb - self.mean
        self.mean += delta * nb / n
        self.m2 += m2b + delta * delta * self.n * nb / n
        self.n = n

    @property
    def var(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else math.nan


def mc_european(opt: Option, mkt: MarketData, n_paths: int = 100_000, seed: int | np.random.Generator | None = 0,
                antithetic: bool = False, chunk_size: int = 250_000, ci_level: float = 0.95) -> MCResult:
    """Price a European option by exact terminal GBM sampling.

    Parameters
    ----------
    n_paths : number of normal draws. With ``antithetic=True`` it must be even, and the
        number of independent units is ``n_paths // 2``.
    seed : int seed or an existing ``np.random.Generator`` (for common random numbers).
    chunk_size : normals generated per chunk (memory bound).
    """
    if opt.is_american:
        raise UnsupportedMethodError(
            "terminal-payoff Monte Carlo prices European options only; American exercise needs "
            "an early-exercise algorithm (e.g. Longstaff-Schwartz), which is not implemented")
    if int(n_paths) != n_paths or n_paths < 2:
        raise InputError("n_paths must be an integer >= 2")
    if antithetic and n_paths % 2:
        raise InputError("n_paths must be even with antithetic sampling (units are pairs)")
    if not 0 < ci_level < 1:
        raise InputError("ci_level must lie in (0, 1)")
    if chunk_size < 2:
        raise InputError("chunk_size must be >= 2")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    t0 = time.perf_counter()
    S, r, q, sig, T, K = mkt.S, mkt.r, mkt.q, mkt.sigma, opt.T, opt.K
    drift = (r - q - 0.5 * sig * sig) * T
    vol = sig * math.sqrt(T)
    disc = math.exp(-r * T)
    is_call = opt.is_call

    def pay(z: np.ndarray) -> np.ndarray:
        ST = S * np.exp(drift + vol * z)
        return disc * (np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0))

    mom = _Moments()
    n_units = n_paths // 2 if antithetic else n_paths
    remaining = n_units
    per_chunk = max(1, chunk_size // (2 if antithetic else 1))
    while remaining > 0:
        m = min(per_chunk, remaining)
        z = rng.standard_normal(m)
        x = 0.5 * (pay(z) + pay(-z)) if antithetic else pay(z)
        mom.add(x)
        remaining -= m
    se = math.sqrt(mom.var / mom.n)
    zq = float(norm.ppf(0.5 + ci_level / 2))
    seed_val = seed if isinstance(seed, int) else None
    return MCResult(mom.mean, se, mom.mean - zq * se, mom.mean + zq * se, ci_level, n_units,
                    n_paths, antithetic, seed_val, time.perf_counter() - t0)


def mcsim_available() -> bool:
    """True if the optional sibling package ``mcsim`` (monte-carlo-sim) is importable."""
    try:
        import mcsim.models.gbm  # noqa: F401
        import mcsim.pricing.engine  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def mcsim_crosscheck(opt: Option, mkt: MarketData, n_paths: int = 200_000, seed: int = 0) -> dict:
    """Price a European option with the external ``mcsim`` engine (if installed).

    Uses ``mcsim.models.gbm.GBM(sigma, measure="Q", r, q)`` with one time step (exact
    terminal sampling) and ``mcsim.pricing.engine.price_mc``. Returns a dict with price, SE
    and CI. Raises ``ImportError`` if ``mcsim`` is not installed. This is an independent
    implementation used only as a cross-check.
    """
    if opt.is_american:
        raise UnsupportedMethodError("mcsim cross-check supports European options only")
    from mcsim.models.gbm import GBM
    from mcsim.pricing.engine import price_mc
    from mcsim.pricing.payoffs import european_call, european_put

    model = GBM(sigma=mkt.sigma, measure="Q", r=mkt.r, q=mkt.q)
    payoff = european_call(opt.K) if opt.is_call else european_put(opt.K)
    res = price_mc(model, payoff, s0=mkt.S, T=opt.T, n_steps=1, n_paths=n_paths, seed=seed)
    return {"price": res.price, "std_error": res.std_error, "ci_low": res.ci_low, "ci_high": res.ci_high,
            "n_paths": res.n_paths, "runtime": res.seconds, "engine": "mcsim.price_mc"}
