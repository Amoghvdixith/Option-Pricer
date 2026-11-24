"""Implied volatility: round trips, ill-conditioned regions, bounds, failures, style safeguards."""

import math

import numpy as np
import pytest

from optpricer import UnsupportedMethodError, bsm
from optpricer.implied_vol import implied_vol, implied_vol_array

SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.5]
MONEYNESS = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]


def test_round_trip_identifiable_region_recovers_sigma():
    """Where vega is not tiny, sigma is recovered to 1e-8."""
    checked = 0
    for T in (0.1, 0.25, 1.0):
        for s in SIGMAS:
            for m in MONEYNESS:
                K = 100 * m
                kind = "call" if K >= 100 else "put"  # OTM side
                p = bsm.price(100, K, T, 0.03, 0.01, s, kind)
                res = implied_vol(p, 100, K, T, 0.03, 0.01, kind)
                if not res.poorly_identified:
                    assert res.converged, res
                    assert res.iv == pytest.approx(s, abs=1e-8), (T, s, m, res)
                    checked += 1
    assert checked > 50


def test_ill_conditioned_region_reproduces_price():
    """Deep OTM / low vol: the price is reproduced even when sigma is not identifiable."""
    for T in (0.1, 1.0):
        for s in SIGMAS:
            for m in MONEYNESS:
                K = 100 * m
                for kind in ("call", "put"):
                    p = bsm.price(100, K, T, 0.03, 0.01, s, kind)
                    res = implied_vol(p, 100, K, T, 0.03, 0.01, kind)
                    if res.converged and math.isfinite(res.iv):
                        assert abs(bsm.price(100, K, T, 0.03, 0.01, res.iv, kind) - p) <= 1e-9 * max(1.0, p)
                    else:
                        # only boundary cases (time value numerically zero) may fail to give a finite solution
                        assert res.status in ("zero_time_value", "converged_bracket"), res
    # an example of saturation: deep OTM short-dated low vol has no time value at double precision
    p = bsm.price(100, 200, 0.1, 0.03, 0.01, 0.05, "call")
    res = implied_vol(p, 100, 200, 0.1, 0.03, 0.01, "call")
    assert res.status == "zero_time_value" and res.poorly_identified


def test_low_vega_flagged():
    p = bsm.price(100, 60, 0.25, 0.03, 0.0, 0.15, "put")
    res = implied_vol(p, 100, 60, 0.25, 0.03, 0.0, "put")
    assert res.poorly_identified
    assert res.vol_uncertainty > 1e-8


def test_bounds_and_failures():
    lo, hi = bsm.european_bounds(100, 90, 1.0, 0.05, 0.0, "call")
    assert implied_vol(float(lo) - 0.5, 100, 90, 1.0, 0.05, 0.0, "call").status == "below_lower_bound"
    assert implied_vol(float(hi) + 0.5, 100, 90, 1.0, 0.05, 0.0, "call").status == "at_or_above_upper_bound"
    z = implied_vol(float(lo), 100, 90, 1.0, 0.05, 0.0, "call")
    assert z.status == "zero_time_value" and z.iv == 0.0
    assert implied_vol(5.0, 100, 100, 0.0, 0.05, 0.0, "call").status == "expired"
    # a price needing sigma > sigma_max is reported, not silently clipped
    p = bsm.price(100, 100, 1.0, 0.05, 0.0, 8.0, "call")
    r = implied_vol(p, 100, 100, 1.0, 0.05, 0.0, "call", sigma_max=4.0)
    assert r.status == "exceeds_sigma_max" and not r.converged
    # bracket expansion finds large finite vols beyond the initial bracket
    r2 = implied_vol(p, 100, 100, 1.0, 0.05, 0.0, "call", initial_upper=0.5, sigma_max=20.0)
    assert r2.converged and r2.iv == pytest.approx(8.0, rel=1e-6)


def test_american_quote_safeguard():
    with pytest.raises(UnsupportedMethodError):
        implied_vol(6.0, 100, 100, 1.0, 0.05, 0.0, "put", "american")
    r = implied_vol(6.0, 100, 100, 1.0, 0.05, 0.0, "put", "american", approximate_american=True)
    assert r.label == "european_bsm_approximation_of_american_quote"


def test_newton_iterations_small_near_the_money():
    p = bsm.price(100, 100, 0.5, 0.02, 0.0, 0.3, "call")
    r = implied_vol(p, 100, 100, 0.5, 0.02, 0.0, "call")
    assert r.converged and r.iterations <= 6 and r.method.startswith("newton")


def test_array_wrapper_broadcasts():
    K = np.array([80.0, 100.0, 120.0])
    prices = bsm.price(100, K, 1.0, 0.03, 0.0, 0.25, np.array(["put", "call", "call"]))
    df = implied_vol_array(prices, 100, K, 1.0, 0.03, 0.0, np.array(["put", "call", "call"]))
    assert len(df) == 3
    np.testing.assert_allclose(df["iv"], 0.25, atol=1e-9)
