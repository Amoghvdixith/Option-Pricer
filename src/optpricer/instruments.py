"""Immutable contract and market-data objects.

``Option`` describes the contract: kind, exercise style, strike, time to expiry.
``MarketData`` describes the state used to value it: spot, rate, dividend yield and
volatility. Both validate their inputs when constructed. Time is an ACT/365F year
fraction; ``r`` and ``q`` are continuously compounded; ``sigma`` is annualised
(see :mod:`optpricer.conventions`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .errors import InputError

Kind = Literal["call", "put"]
Style = Literal["european", "american"]

_KINDS = ("call", "put")
_STYLES = ("european", "american")


def _finite(name: str, x: float) -> float:
    try:
        xf = float(x)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be a real number, got {x!r}") from exc
    if not math.isfinite(xf):
        raise InputError(f"{name} must be finite, got {x!r}")
    return xf


def normalise_kind(kind: str) -> Kind:
    """Lower-case and validate an option kind ('call' or 'put'; 'c'/'p' accepted)."""
    k = str(kind).strip().lower()
    k = {"c": "call", "p": "put"}.get(k, k)
    if k not in _KINDS:
        raise InputError(f"option kind must be 'call' or 'put', got {kind!r}")
    return k  # type: ignore[return-value]


def normalise_style(style: str) -> Style:
    """Lower-case and validate an exercise style ('european' or 'american')."""
    s = str(style).strip().lower()
    if s not in _STYLES:
        raise InputError(f"exercise style must be 'european' or 'american', got {style!r}")
    return s  # type: ignore[return-value]


@dataclass(frozen=True)
class Option:
    """A vanilla option contract.

    Parameters
    ----------
    kind : 'call' or 'put'.
    K : strike, > 0.
    T : time to expiry in years (ACT/365F), >= 0. ``T == 0`` means at expiry; the value
        is then intrinsic.
    style : 'european' (default) or 'american'.
    """

    kind: Kind
    K: float
    T: float
    style: Style = "european"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", normalise_kind(self.kind))
        object.__setattr__(self, "style", normalise_style(self.style))
        K = _finite("strike K", self.K)
        T = _finite("time to expiry T", self.T)
        if K <= 0:
            raise InputError(f"strike K must be positive, got {K}")
        if T < 0:
            raise InputError(f"time to expiry T must be non-negative, got {T}")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "T", T)

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    @property
    def is_american(self) -> bool:
        return self.style == "american"

    def payoff(self, S):
        """Intrinsic value ``max(S-K, 0)`` or ``max(K-S, 0)`` (array-friendly)."""
        import numpy as np

        S = np.asarray(S, dtype=float)
        return np.maximum(S - self.K, 0.0) if self.is_call else np.maximum(self.K - S, 0.0)

    def with_(self, **changes) -> Option:
        """Return a copy with some fields changed (validated again)."""
        return replace(self, **changes)


@dataclass(frozen=True)
class MarketData:
    """Market state for a single underlying.

    Parameters
    ----------
    S : spot, >= 0.
    r : continuously compounded risk-free rate. Negative values are allowed.
    q : continuous dividend yield. Negative values are allowed and represent net
        carry benefit; the trees and PDEs accept them as long as their probabilities
        and boundaries stay valid.
    sigma : annualised volatility, >= 0. ``sigma == 0`` is the deterministic limit.
    """

    S: float
    r: float = 0.0
    q: float = 0.0
    sigma: float = 0.2

    def __post_init__(self) -> None:
        S = _finite("spot S", self.S)
        r = _finite("rate r", self.r)
        q = _finite("dividend yield q", self.q)
        sigma = _finite("volatility sigma", self.sigma)
        if S < 0:
            raise InputError(f"spot S must be non-negative, got {S}")
        if sigma < 0:
            raise InputError(f"volatility sigma must be non-negative, got {sigma}")
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "sigma", sigma)

    def with_(self, **changes) -> MarketData:
        """Return a copy with some fields changed (validated again)."""
        return replace(self, **changes)
