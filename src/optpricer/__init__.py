"""optpricer: an educational options pricing and research library.

Not production software. See README.md for scope and limitations.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import binomial, bsm, conventions, hedging, implied_vol, market, mc, pde, surface, validation  # noqa: E402
from .api import METHODS, PricingResult, price, price_all, supported  # noqa: E402
from .errors import (  # noqa: E402
    ConvergenceError,
    InputError,
    OptPricerError,
    ProbabilityError,
    UnstableSchemeError,
    UnsupportedMethodError,
)
from .instruments import MarketData, Option  # noqa: E402

__all__ = [
    "Option", "MarketData", "price", "price_all", "supported", "PricingResult", "METHODS",
    "bsm", "binomial", "pde", "mc", "implied_vol", "market", "surface", "hedging", "conventions", "validation",
    "OptPricerError", "InputError", "UnsupportedMethodError", "UnstableSchemeError", "ConvergenceError",
    "ProbabilityError", "__version__",
]
