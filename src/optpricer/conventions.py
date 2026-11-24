"""Market and unit conventions.

Package-wide conventions:

* **Time** is a year fraction on the ACT/365F basis (calendar days / 365). Volatility is
  annualised on the same calendar basis. Trading-day (/252) clocks are never mixed in
  silently. A function that uses 252 says so in its name and argument.
* **Rates and dividend yields** are continuously compounded per annum, so the
  discount factor is ``exp(-r T)``.
* **Greeks** are computed internally in raw mathematical units:
  - ``delta = dV/dS``, ``gamma = d2V/dS2`` (per unit of underlying)
  - ``vega = dV/dsigma`` (per 1.00 = 100 vol points)
  - ``rho = dV/dr`` (per 1.00 = 10,000 bp)
  - ``psi = dV/dq`` (per 1.00 change in dividend yield)
  - ``theta = dV/dt`` in *calendar time* with expiry date fixed, so theta = -dV/dT
    where T is time to expiry. Unit: per year.
  - ``vanna = d2V/(dS dsigma)``, ``volga = d2V/dsigma2`` (raw)
* **Display units** (as on trading screens) are produced only by :func:`to_display_units`:
  vega and rho per 1 percentage point, theta per calendar day, vanna per 1 vol point,
  volga per (1 vol point)^2.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping

import numpy as np

DAYS_PER_YEAR = 365.0
"""Calendar days per year for the ACT/365F year fraction and per-day theta."""

TRADING_DAYS_PER_YEAR = 252
"""Used only where explicitly requested (e.g. hedging rebalance frequency labels)."""


def year_fraction(start: _dt.date | _dt.datetime, end: _dt.date | _dt.datetime) -> float:
    """ACT/365F year fraction between two dates or datetimes.

    Datetimes contribute fractional days, so an option expiring at 16:00 measured from
    10:00 the same day has a positive year fraction. Raises if ``end < start``.
    """
    if isinstance(start, _dt.datetime) != isinstance(end, _dt.datetime):
        # promote dates to midnight datetimes so they can be subtracted consistently
        if not isinstance(start, _dt.datetime):
            start = _dt.datetime.combine(start, _dt.time())
        if not isinstance(end, _dt.datetime):
            end = _dt.datetime.combine(end, _dt.time())
    delta = end - start  # type: ignore[operator]
    days = delta.total_seconds() / 86400.0
    if days < 0:
        raise ValueError(f"end ({end}) is before start ({start})")
    return days / DAYS_PER_YEAR


def discount_factor(r: float | np.ndarray, T: float | np.ndarray) -> np.ndarray:
    """Continuously compounded discount factor ``exp(-r T)``."""
    return np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))


def forward_price(S, T, r, q) -> np.ndarray:
    """Forward price ``S exp((r - q) T)`` under continuous rate and yield."""
    return np.asarray(S, dtype=float) * np.exp((np.asarray(r, float) - np.asarray(q, float)) * np.asarray(T, float))


# Scaling of each raw Greek for screen display, and the unit label it then carries.
DISPLAY_SCALING: dict[str, tuple[float, str]] = {
    "price": (1.0, "currency"),
    "delta": (1.0, "per 1 unit of underlying"),
    "gamma": (1.0, "delta per 1 unit of underlying"),
    "vega": (0.01, "per 1 vol point (sigma +0.01)"),
    "theta": (1.0 / DAYS_PER_YEAR, "per calendar day (expiry fixed)"),
    "rho": (0.01, "per 1 rate percentage point (r +0.01)"),
    "psi": (0.01, "per 1 dividend-yield percentage point (q +0.01)"),
    "vanna": (0.01, "delta change per 1 vol point"),
    "volga": (1e-4, "vega(raw) change x 0.01 per vol point, i.e. per (1 vol point)^2"),
}


def to_display_units(raw: Mapping[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    """Convert raw Greeks (see module docstring) to screen units.

    Keys without a known scaling are passed through unchanged.
    """
    out: dict[str, np.ndarray | float] = {}
    for key, val in raw.items():
        scale = DISPLAY_SCALING.get(key, (1.0, ""))[0]
        out[key] = np.asarray(val) * scale if isinstance(val, np.ndarray) else float(val) * scale
    return out
