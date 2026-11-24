"""Exception types used across the package.

Each error says what went wrong and, where it can, what to change. Numerical failures
(non-convergence, unstable schemes) are raised and never hidden.
"""

from __future__ import annotations


class OptPricerError(Exception):
    """Base class for package errors."""


class InputError(OptPricerError, ValueError):
    """An input was invalid: non-finite, negative where it must not be, or of the wrong kind."""


class UnsupportedMethodError(OptPricerError, ValueError):
    """The pricing method does not support this exercise style or contract.

    Example: Black-Scholes closed form, or terminal-payoff Monte Carlo, for an American option.
    """


class UnstableSchemeError(OptPricerError, ValueError):
    """A finite-difference configuration fails its stability check.

    Explicit schemes are rejected by default. Pass ``allow_unstable=True`` only for an
    intentional instability experiment.
    """


class ConvergenceError(OptPricerError, RuntimeError):
    """An iterative solver (PSOR, the implied-vol fallback, ...) did not meet its tolerance."""


class ProbabilityError(OptPricerError, ValueError):
    """A lattice risk-neutral probability fell outside (0, 1).

    The tree is not arbitrage-free at this step size; increase n or check the inputs.
    """
