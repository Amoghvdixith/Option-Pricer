"""Numerical-validation helpers: tolerances, convergence orders, bump sweeps, run metadata."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import bsm


def close(actual, expected, abs_tol: float, rel_tol: float) -> np.ndarray:
    """Element-wise ``|a - e| <= abs_tol + rel_tol * |e|`` (combined tolerance near zero)."""
    a, e = np.asarray(actual, float), np.asarray(expected, float)
    return np.abs(a - e) <= abs_tol + rel_tol * np.abs(e)


def observed_order(h, err) -> float:
    """Least-squares slope of ``log|err|`` on ``log h`` (observed convergence order)."""
    h, err = np.asarray(h, float), np.abs(np.asarray(err, float))
    ok = err > 0
    if ok.sum() < 2:
        return math.nan
    return float(np.polyfit(np.log(h[ok]), np.log(err[ok]), 1)[0])


def pairwise_orders(h, err) -> np.ndarray:
    """Local orders ``log(e_i/e_{i+1}) / log(h_i/h_{i+1})``."""
    h, err = np.asarray(h, float), np.abs(np.asarray(err, float))
    return np.log(err[:-1] / err[1:]) / np.log(h[:-1] / h[1:])


_BUMP_PARAM = {"delta": "S", "vega": "sigma", "rho": "r", "psi": "q"}


def bump_greeks(S, K, T, r, q, sigma, kind="call", h_rel: float = 1e-4) -> dict[str, float]:
    """Central finite-difference Greeks of the BSM price (raw units).

    Bumps: ``S`` by ``h_rel*S``; ``sigma``, ``r`` and ``q`` by ``h_rel`` (absolute); ``T`` by
    ``h_rel`` years. Theta = ``-dV/dT``. Second derivatives (gamma, volga) use the
    3-point formula, and vanna uses a 4-point cross difference.
    """
    P = lambda **kw: float(bsm.price(**{**dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, kind=kind), **kw}))  # noqa: E731
    hS, h = h_rel * S, h_rel
    out = {
        "delta": (P(S=S + hS) - P(S=S - hS)) / (2 * hS),
        "gamma": (P(S=S + hS) - 2 * P() + P(S=S - hS)) / hS**2,
        "vega": (P(sigma=sigma + h) - P(sigma=sigma - h)) / (2 * h),
        "rho": (P(r=r + h) - P(r=r - h)) / (2 * h),
        "psi": (P(q=q + h) - P(q=q - h)) / (2 * h),
        "theta": -(P(T=T + h) - P(T=T - h)) / (2 * h),
        "volga": (P(sigma=sigma + h) - 2 * P() + P(sigma=sigma - h)) / h**2,
        "vanna": (P(S=S + hS, sigma=sigma + h) - P(S=S + hS, sigma=sigma - h)
                  - P(S=S - hS, sigma=sigma + h) + P(S=S - hS, sigma=sigma - h)) / (4 * hS * h),
    }
    return out


def pde_identity_residual(S, K, T, r, q, sigma, kind="call") -> np.ndarray:
    """``theta + 1/2 sigma^2 S^2 gamma + (r - q) S delta - r V`` using raw annual Greeks."""
    g = bsm.greeks(S, K, T, r, q, sigma, kind)
    S = np.asarray(S, float)
    return g["theta"] + 0.5 * np.asarray(sigma) ** 2 * S**2 * g["gamma"] + (np.asarray(r) - np.asarray(q)) * S * g[
        "delta"] - np.asarray(r) * g["price"]


def run_metadata(config: dict, seed: int | None, runtime_s: float, provenance: dict) -> dict:
    """Versions, platform, configuration, seed, runtime and data provenance for a run."""
    import matplotlib
    import scipy

    from . import __version__

    return {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "optpricer_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "seed": seed, "runtime_seconds": runtime_s, "config": config, "data_provenance": provenance,
        "cpu_count": os.cpu_count(),
    }


def write_json(obj: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p
