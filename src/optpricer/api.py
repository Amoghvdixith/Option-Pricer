"""Unified pricing entry point.

>>> from optpricer import Option, MarketData, price
>>> res = price(Option("put", 100, 1.0, "american"), MarketData(100, 0.05, 0.0, 0.2), method="lr", n=501)
>>> round(res.price, 2)
6.09

Supported methods by exercise style:

============  ========  ========
method        European  American
============  ========  ========
bs            yes       no (raises UnsupportedMethodError)
mc            yes       no (raises; Longstaff-Schwartz not implemented)
crr, crr_avg  yes       yes
lr            yes       yes (odd n only)
explicit      yes       yes (stability-checked; projection)
implicit, cn  yes       yes (PSOR)
============  ========  ========
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import binomial, bsm, mc, pde
from .errors import OptPricerError, UnsupportedMethodError
from .instruments import MarketData, Option

METHODS = ("bs", "crr", "crr_avg", "lr", "explicit", "implicit", "cn", "mc")
EUROPEAN_ONLY = ("bs", "mc")


@dataclass
class PricingResult:
    """Price plus method metadata. ``std_error``/CI are set only for Monte Carlo."""

    price: float
    method: str
    kind: str
    style: str
    std_error: float = math.nan
    ci_low: float = math.nan
    ci_high: float = math.nan
    delta: float = math.nan
    gamma: float = math.nan
    theta: float = math.nan
    runtime: float = 0.0
    settings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def supported(method: str, style: str) -> bool:
    """Whether ``method`` can price an option of exercise ``style``."""
    return method in METHODS and not (style == "american" and method in EUROPEAN_ONLY)


def price(option: Option, market: MarketData, method: str = "bs", **settings) -> PricingResult:
    """Price ``option`` under ``market`` with ``method`` (see module table).

    Settings: trees ``n``; PDE ``n_space, n_time, rannacher_steps, n_sd, omega, tol, ...``
    (see :func:`optpricer.pde.fd_price`); MC ``n_paths, seed, antithetic, chunk_size``.
    """
    method = method.lower()
    if method not in METHODS:
        raise UnsupportedMethodError(f"unknown method {method!r}; choose from {METHODS}")
    if not supported(method, option.style):
        raise UnsupportedMethodError(
            f"method {method!r} prices European options only; {option.style} options need a tree "
            "(crr, crr_avg, lr) or PDE (explicit, implicit, cn) method")
    t0 = time.perf_counter()
    base: dict = dict(kind=option.kind, style=option.style)
    if method == "bs":
        g = bsm.greeks(market.S, option.K, option.T, market.r, market.q, market.sigma, option.kind)
        return PricingResult(float(g["price"]), "bs", **base, delta=float(g["delta"]), gamma=float(g["gamma"]),
                             theta=float(g["theta"]), runtime=time.perf_counter() - t0)
    if method in ("crr", "crr_avg", "lr"):
        n = int(settings.get("n", 201 if method == "lr" else 1000))
        if method == "crr":
            tr = binomial.crr(option, market, n)
        elif method == "crr_avg":
            tr = binomial.crr_averaged(option, market, n)
        else:
            tr = binomial.leisen_reimer(option, market, n)
        return PricingResult(tr.price, method, **base, delta=tr.delta, gamma=tr.gamma, theta=tr.theta,
                             runtime=time.perf_counter() - t0, settings={"n": n},
                             diagnostics={"u": tr.u, "d": tr.d, "p": tr.p, "notes": tr.notes})
    if method in ("explicit", "implicit", "cn"):
        pr = pde.fd_price(option, market, scheme=method, **settings)
        return PricingResult(pr.price, method, **base, delta=pr.delta, gamma=pr.gamma, theta=pr.theta,
                             runtime=time.perf_counter() - t0,
                             settings={"n_space": pr.n_space, "n_time": pr.n_time}, diagnostics=pr.diagnostics)
    res = mc.mc_european(option, market, **settings)
    return PricingResult(res.price, "mc", **base, std_error=res.std_error, ci_low=res.ci_low, ci_high=res.ci_high,
                         runtime=time.perf_counter() - t0,
                         settings={"n_paths": res.n_normals, "antithetic": res.antithetic, "seed": res.seed},
                         diagnostics={"n_units": res.n_units, "ci_level": res.ci_level})


def price_all(option: Option, market: MarketData, methods=None, settings: dict | None = None) -> pd.DataFrame:
    """Price with several methods.

    Unsupported combinations and failed methods (e.g. an explicit grid rejected as unstable) are
    listed with a status message instead of aborting the whole comparison.
    """
    methods = methods or METHODS
    settings = settings or {}
    rows = []
    for m in methods:
        if not supported(m, option.style):
            rows.append({"method": m, "price": math.nan, "status": "unsupported for " + option.style})
            continue
        try:
            r = price(option, market, m, **settings.get(m, {}))
        except OptPricerError as exc:  # e.g. an unstable explicit grid: report, do not abort the others
            rows.append({"method": m, "price": math.nan, "status": f"error: {type(exc).__name__}: {exc}"})
            continue
        rows.append({"method": m, "price": r.price, "std_error": r.std_error, "ci_low": r.ci_low,
                     "ci_high": r.ci_high, "delta": r.delta, "gamma": r.gamma, "theta": r.theta,
                     "runtime_s": r.runtime, "settings": r.settings, "status": "ok"})
    return pd.DataFrame(rows)
