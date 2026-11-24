"""Analytical Greeks: bump-and-revalue checks, bump-size sensitivity, symmetries, the PDE identity."""

import numpy as np
import pytest

from optpricer import bsm
from optpricer.conventions import to_display_units
from optpricer.validation import bump_greeks, close, pde_identity_residual

CASES = [
    (100, 100, 1.0, 0.05, 0.00, 0.20),
    (100, 90, 0.25, 0.03, 0.02, 0.30),
    (100, 120, 2.0, 0.01, 0.03, 0.15),
    (50, 55, 0.5, -0.01, 0.00, 0.40),
]

# Central differences have truncation O(h^2) and round-off O(eps/h) (first derivatives) or
# O(eps/h^2) (second derivatives). h_rel = 1e-4 balances these for these moderate cases.
# Tolerances: first-order Greeks rel 1e-6, second-order (gamma, volga, vanna) rel 1e-4,
# with an absolute floor for Greeks near zero.
TOL = {"delta": (1e-7, 1e-6), "vega": (1e-6, 1e-6), "rho": (1e-6, 1e-6), "psi": (1e-6, 1e-6),
       "theta": (1e-6, 1e-6), "gamma": (1e-6, 1e-4), "volga": (1e-4, 1e-4), "vanna": (1e-5, 1e-4)}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", ["call", "put"])
def test_analytic_vs_bump(case, kind):
    g = bsm.greeks(*case, kind)
    b = bump_greeks(*case, kind, h_rel=1e-4)
    for name, (atol, rtol) in TOL.items():
        assert close(g[name], b[name], atol, rtol), (name, float(g[name]), b[name])


def test_bump_size_sensitivity_shows_error_tradeoff():
    """Too large a bump -> truncation error; too small -> round-off. Middle bumps are best."""
    case = (100, 100, 1.0, 0.05, 0.0, 0.2)
    exact = float(bsm.gamma(*case))
    errs = {h: abs(bump_greeks(*case, h_rel=h)["gamma"] - exact) for h in (1e-1, 1e-4, 1e-8)}
    assert errs[1e-4] < errs[1e-1]
    assert errs[1e-4] < errs[1e-8]


def test_call_put_relationships():
    rng = np.random.default_rng(4)
    S, K = rng.uniform(50, 150, 100), rng.uniform(50, 150, 100)
    T, r, q, s = rng.uniform(0.05, 2, 100), rng.uniform(-0.01, 0.08, 100), rng.uniform(0, 0.05, 100), \
        rng.uniform(0.1, 0.6, 100)
    c, p = bsm.greeks(S, K, T, r, q, s, "call"), bsm.greeks(S, K, T, r, q, s, "put")
    np.testing.assert_allclose(c["delta"] - p["delta"], np.exp(-q * T), atol=1e-14)
    for name in ("gamma", "vega", "vanna", "volga"):
        np.testing.assert_allclose(c[name], p[name], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(c["rho"] - p["rho"], K * T * np.exp(-r * T), rtol=1e-12)
    np.testing.assert_allclose(c["psi"] - p["psi"], -S * T * np.exp(-q * T), rtol=1e-12)
    # theta difference = d/dt of (S e^{-q tau} - K e^{-r tau}) = q S e^{-qT} - r K e^{-rT}
    np.testing.assert_allclose(c["theta"] - p["theta"], q * S * np.exp(-q * T) - r * K * np.exp(-r * T), atol=1e-10)


def test_pde_identity_raw_units():
    rng = np.random.default_rng(5)
    n = 2000
    S, K = rng.uniform(20, 300, n), rng.uniform(20, 300, n)
    T, r, q, s = rng.uniform(0.02, 3, n), rng.uniform(-0.02, 0.1, n), rng.uniform(0, 0.06, n), rng.uniform(0.05, 1.0, n)
    for kind in ("call", "put"):
        res = pde_identity_residual(S, K, T, r, q, s, kind)
        scale = np.maximum(1.0, np.maximum(S, K))
        assert np.max(np.abs(res) / scale) < 1e-10


def test_pde_identity_fails_with_screen_units():
    """Using per-day theta (display units) in the identity breaks it: units matter."""
    case = (100, 100, 1.0, 0.05, 0.0, 0.2)
    g = bsm.greeks(*case)
    disp = to_display_units(g)
    S, r, sigma = 100, 0.05, 0.2
    wrong = disp["theta"] + 0.5 * sigma**2 * S**2 * g["gamma"] + r * S * g["delta"] - r * g["price"]
    assert abs(wrong) > 1.0


def test_known_atm_values():
    g = bsm.greeks(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    # independent textbook values for S=K=100, T=1, r=5%, sigma=20%
    assert g["delta"] == pytest.approx(0.636831, abs=1e-6)
    assert g["gamma"] == pytest.approx(0.018762, abs=1e-6)
    assert g["vega"] == pytest.approx(37.5240, abs=1e-4)
    assert g["theta"] == pytest.approx(-6.4140, abs=1e-4)
    assert g["rho"] == pytest.approx(53.2325, abs=1e-4)


def test_greeks_at_kink_are_nan_and_regular_limits():
    g = bsm.greeks(np.array([90.0, 100.0, 110.0]), 100, 0.0, 0.05, 0.0, 0.2, "call")
    assert np.isnan(g["delta"][1]) and np.isnan(g["gamma"][1])
    assert g["delta"][0] == 0.0 and g["delta"][2] == 1.0
    assert g["gamma"][0] == 0.0 and g["vega"][2] == 0.0
