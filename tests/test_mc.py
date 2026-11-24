"""Monte Carlo: unbiasedness within SE, uncertainty accounting, antithetic pairs, reproducibility."""

import math

import numpy as np
import pytest

from optpricer import InputError, MarketData, Option, UnsupportedMethodError, bsm
from optpricer.mc import mc_european, mcsim_available, mcsim_crosscheck


@pytest.mark.parametrize("kind", ["call", "put"])
def test_mc_within_four_se_of_bsm(kind):
    m = MarketData(100, 0.05, 0.02, 0.25)
    o = Option(kind, 105, 1.0)
    res = mc_european(o, m, 200_000, seed=123)
    ref = bsm.price(100, 105, 1.0, 0.05, 0.02, 0.25, kind)
    assert abs(res.price - ref) < 4 * res.std_error
    assert res.ci_low < res.price < res.ci_high
    assert res.n_units == 200_000


def test_terminal_moments_match_risk_neutral_forward():
    """E[S_T] = S e^{(r-q)T}: price a zero-strike call, then undiscount."""
    m = MarketData(100, 0.04, 0.01, 0.3)
    res = mc_european(Option("call", 1e-9, 2.0), m, 400_000, seed=5)
    fwd = 100 * math.exp(0.03 * 2)
    assert abs(res.price * math.exp(0.08) - fwd) < 4 * res.std_error * math.exp(0.08)


def test_antithetic_uses_pairs_and_reduces_se_for_monotone_payoff():
    m = MarketData(100, 0.05, 0.0, 0.2)
    o = Option("call", 100, 1.0)
    plain = mc_european(o, m, 200_000, seed=1)
    anti = mc_european(o, m, 200_000, seed=1, antithetic=True)
    assert anti.n_units == 100_000 and anti.n_normals == 200_000
    assert anti.std_error < plain.std_error
    assert abs(anti.price - bsm.price(100, 100, 1, 0.05, 0, 0.2, "call")) < 4 * anti.std_error
    with pytest.raises(InputError):
        mc_european(o, m, 1001, antithetic=True)


def test_se_scales_like_inverse_sqrt_n():
    m, o = MarketData(100, 0.05, 0.0, 0.2), Option("put", 100, 1.0)
    s1 = mc_european(o, m, 40_000, seed=2).std_error
    s2 = mc_european(o, m, 160_000, seed=3).std_error
    assert s1 / s2 == pytest.approx(2.0, rel=0.05)


def test_reproducible_and_chunk_invariant():
    m, o = MarketData(100, 0.05, 0.0, 0.2), Option("call", 110, 0.5)
    a = mc_european(o, m, 100_000, seed=9, chunk_size=100_000)
    b = mc_european(o, m, 100_000, seed=9, chunk_size=100_000)
    c = mc_european(o, m, 100_000, seed=9, chunk_size=7_919)
    assert a.price == b.price
    assert c.price == pytest.approx(a.price, abs=1e-12)  # same normal stream, chunked moments
    assert c.std_error == pytest.approx(a.std_error, rel=1e-9)


def test_american_rejected():
    with pytest.raises(UnsupportedMethodError):
        mc_european(Option("put", 100, 1.0, "american"), MarketData(100, 0.05, 0, 0.2), 1000)


@pytest.mark.skipif(not mcsim_available(), reason="optional sibling package mcsim not installed")
def test_optional_mcsim_crosscheck():
    m, o = MarketData(100, 0.05, 0.01, 0.2), Option("call", 100, 1.0)
    ref = bsm.price(100, 100, 1.0, 0.05, 0.01, 0.2, "call")
    ext = mcsim_crosscheck(o, m, 200_000, seed=777)
    ours = mc_european(o, m, 200_000, seed=778)
    se = math.hypot(ext["std_error"], ours.std_error)
    assert abs(ext["price"] - ours.price) < 4 * se
    assert abs(ext["price"] - ref) < 4 * ext["std_error"]
    # identical seed -> identical draws -> identical estimates (same exact terminal transform)
    same = mcsim_crosscheck(o, m, 50_000, seed=11)
    assert same["price"] == pytest.approx(mc_european(o, m, 50_000, seed=11).price, abs=1e-10)


def test_random_generator_argument_supports_common_numbers():
    m, o = MarketData(100, 0.05, 0.0, 0.2), Option("call", 100, 1.0)
    g1, g2 = np.random.default_rng(4), np.random.default_rng(4)
    assert mc_european(o, m, 10_000, seed=g1).price == mc_european(o, m, 10_000, seed=g2).price
