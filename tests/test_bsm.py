"""Black-Scholes-Merton prices: benchmarks, parity, bounds, broadcasting and edge cases."""

import math

import numpy as np
import pytest

from optpricer import InputError, bsm


def _ref_price(S, K, T, r, q, sig, kind):
    """Independent scalar implementation using math.erf (shares no code with the package)."""
    def N(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if kind == "call":
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)


def test_hull_example_unrounded_and_rounded():
    # Hull: S=42, K=40, r=10%, sigma=20%, T=0.5 -> C = 4.76, P = 0.81 (published to 2 dp)
    c = bsm.price(42, 40, 0.5, 0.10, 0.0, 0.20, "call")
    p = bsm.price(42, 40, 0.5, 0.10, 0.0, 0.20, "put")
    assert round(c, 2) == 4.76 and round(p, 2) == 0.81
    assert c == pytest.approx(_ref_price(42, 40, 0.5, 0.10, 0.0, 0.20, "call"), abs=1e-12)
    assert p == pytest.approx(_ref_price(42, 40, 0.5, 0.10, 0.0, 0.20, "put"), abs=1e-12)
    assert c == pytest.approx(4.759422, abs=5e-7)
    assert p == pytest.approx(0.808599, abs=5e-7)


def test_guide_atm_benchmarks():
    assert bsm.price(100, 100, 1, 0.05, 0, 0.2, "call") == pytest.approx(10.4506, abs=5e-5)
    assert bsm.price(100, 100, 1, 0.05, 0, 0.2, "put") == pytest.approx(5.5735, abs=5e-5)


def test_haug_generalised_bsm_put_with_carry():
    # Haug (2007) generalised BSM example: S=75, X=70, T=0.5, r=10%, b=5% (q = r - b = 5%), sigma=35% -> put 4.0870
    assert bsm.price(75, 70, 0.5, 0.10, 0.05, 0.35, "put") == pytest.approx(4.0870, abs=5e-5)


def test_independent_reference_random():
    rng = np.random.default_rng(1)
    for _ in range(200):
        S, K = rng.uniform(20, 200), rng.uniform(20, 200)
        T, r, q, s = rng.uniform(0.01, 3), rng.uniform(-0.02, 0.1), rng.uniform(0, 0.06), rng.uniform(0.05, 1.0)
        for kind in ("call", "put"):
            ref = _ref_price(S, K, T, r, q, s, kind)
            assert bsm.price(S, K, T, r, q, s, kind) == pytest.approx(ref, abs=1e-10, rel=1e-10)


def test_put_call_parity_random():
    rng = np.random.default_rng(2)
    n = 5000
    S, K = rng.uniform(1, 500, n), rng.uniform(1, 500, n)
    T, r, q, s = rng.uniform(0, 5, n), rng.uniform(-0.03, 0.15, n), rng.uniform(-0.02, 0.1, n), rng.uniform(0, 2, n)
    c = bsm.price(S, K, T, r, q, s, "call")
    p = bsm.price(S, K, T, r, q, s, "put")
    resid = bsm.parity_residual(c, p, S, K, T, r, q)
    scale = np.maximum.reduce([np.ones(n), S, K])
    # absolute floor 1e-12 relative to the price scale: a few ulps of max(S, K)
    assert np.max(np.abs(resid) / scale) < 1e-12


def test_bounds_hold():
    rng = np.random.default_rng(3)
    n = 3000
    S, K = rng.uniform(1, 300, n), rng.uniform(1, 300, n)
    T, r, q, s = rng.uniform(0, 4, n), rng.uniform(-0.02, 0.1, n), rng.uniform(0, 0.08, n), rng.uniform(0, 1.5, n)
    for kind in ("call", "put"):
        v = bsm.price(S, K, T, r, q, s, kind)
        lo, hi = bsm.european_bounds(S, K, T, r, q, kind)
        assert np.all(v >= lo - 1e-10 * (1 + K)) and np.all(v <= hi + 1e-10 * (1 + K))


def test_broadcasting_and_mixed_kinds():
    K = np.array([90.0, 100.0, 110.0])
    T = np.array([[0.5], [1.0]])
    v = bsm.price(100, K, T, 0.05, 0.0, 0.2, "call")
    assert v.shape == (2, 3)
    mixed = bsm.price(100, K, 1.0, 0.05, 0.0, 0.2, np.array(["call", "put", "call"]))
    assert mixed[1] == pytest.approx(bsm.price(100, 100, 1.0, 0.05, 0.0, 0.2, "put"))
    assert isinstance(bsm.price(100, 100, 1, 0.05, 0, 0.2), float)


def test_expiry_zero_vol_zero_spot_limits():
    assert bsm.price(110, 100, 0.0, 0.05, 0.0, 0.2, "call") == pytest.approx(10.0)
    assert bsm.price(90, 100, 0.0, 0.05, 0.0, 0.2, "put") == pytest.approx(10.0)
    F = 100 * math.exp(0.05 - 0.01)
    assert bsm.price(100, 100, 1.0, 0.05, 0.01, 0.0, "call") == pytest.approx(math.exp(-0.05) * (F - 100))
    assert bsm.price(100, 100, 1.0, 0.05, 0.01, 0.0, "put") == 0.0
    assert bsm.price(0.0, 100, 1.0, 0.05, 0.0, 0.3, "call") == 0.0
    assert bsm.price(0.0, 100, 1.0, 0.05, 0.0, 0.3, "put") == pytest.approx(100 * math.exp(-0.05))
    # continuity: tiny vol / tiny T approach the limits, with floating-point errors raised (none occur)
    with np.errstate(all="raise"):
        v = bsm.price(100, 100, 1e-12, 0.05, 0.0, 0.2, "call")
        v2 = bsm.price(100, 100, 1.0, 0.05, 0.0, 1e-12, "call")
        bsm.greeks(np.array([0.0, 100.0, 120.0]), 100, np.array([0.0, 1.0, 0.0]), 0.05, 0.0, 0.2, "put")
    # ATM small-time asymptotic: C ~ S sigma sqrt(T) / sqrt(2 pi)
    assert v == pytest.approx(100 * 0.2 * 1e-6 / math.sqrt(2 * math.pi), rel=1e-4)
    assert v2 == pytest.approx(math.exp(-0.05) * (100 * math.exp(0.05) - 100), rel=1e-8)


def test_deep_itm_otm_stability():
    with np.errstate(all="raise"):
        deep_otm = bsm.price(100, 1e4, 0.1, 0.05, 0.0, 0.2, "call")
        deep_itm_put = bsm.price(100, 1e4, 0.1, 0.05, 0.0, 0.2, "put")
    assert 0.0 <= deep_otm < 1e-300
    assert deep_itm_put == pytest.approx(1e4 * math.exp(-0.005) - 100, rel=1e-12)


@pytest.mark.parametrize("bad", [dict(K=0), dict(S=-1), dict(T=-1), dict(sigma=-0.1), dict(S=np.nan)])
def test_invalid_inputs(bad):
    args = dict(S=100, K=100, T=1, r=0.05, q=0.0, sigma=0.2)
    args.update(bad)
    with pytest.raises(InputError):
        bsm.price(**args)


def test_invalid_kind():
    with pytest.raises(InputError):
        bsm.price(100, 100, 1, 0.05, 0, 0.2, "straddle")


def test_american_parity_bounds_require_nonnegative_rates():
    lo, hi = bsm.american_parity_bounds(100, 100, 1, 0.05, 0.02)
    assert lo < hi
    with pytest.raises(InputError):
        bsm.american_parity_bounds(100, 100, 1, -0.01, 0.0)
