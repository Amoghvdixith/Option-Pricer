"""Binomial trees: hand-computed small trees, ordering, probabilities, convergence, American checks."""

import math

import numpy as np
import pytest

from optpricer import InputError, MarketData, Option, ProbabilityError, bsm
from optpricer.binomial import crr, crr_averaged, crr_parameters, leisen_reimer, lr_parameters, peizer_pratt


def _hand_two_step(S, K, T, r, q, sig, kind, american):
    """Explicit node-by-node 2-step CRR tree (no vectorisation, written out by hand)."""
    dt = T / 2
    u = math.exp(sig * math.sqrt(dt))
    d = 1 / u
    p = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    def pay(s):
        return max(s - K, 0.0) if kind == "call" else max(K - s, 0.0)

    Vuu, Vud, Vdd = pay(S * u * u), pay(S), pay(S * d * d)
    Vu = disc * (p * Vuu + (1 - p) * Vud)
    Vd = disc * (p * Vud + (1 - p) * Vdd)
    if american:
        Vu, Vd = max(Vu, pay(S * u)), max(Vd, pay(S * d))
    V0 = disc * (p * Vu + (1 - p) * Vd)
    if american:
        V0 = max(V0, pay(S))
    delta = (Vu - Vd) / (S * u - S * d)
    return V0, delta


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("style", ["european", "american"])
def test_two_step_tree_matches_hand_computation(kind, style):
    m = MarketData(100, 0.05, 0.01, 0.3)
    opt = Option(kind, 105, 1.0, style)
    res = crr(opt, m, n=2)
    V0, delta = _hand_two_step(100, 105, 1.0, 0.05, 0.01, 0.3, kind, style == "american")
    assert res.price == pytest.approx(V0, abs=1e-13)
    assert res.delta == pytest.approx(delta, abs=1e-13)


def test_one_step_textbook_numbers():
    # Hand calculation: S=100, K=100, T=1, r=5%, sigma=20%, n=1:
    # u = e^0.2 = 1.221402758, d = 0.818730753, p = (e^0.05 - d)/(u - d) = 0.579236
    u, d, p = crr_parameters(1.0, 1, 0.05, 0.0, 0.2)
    assert u == pytest.approx(1.221402758, abs=1e-9)
    assert p == pytest.approx((math.exp(0.05) - d) / (u - d))
    call = crr(Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2), n=1).price
    assert call == pytest.approx(math.exp(-0.05) * p * (100 * u - 100), abs=1e-12)


def test_node_order_and_probability_admissibility():
    with pytest.raises(ProbabilityError):
        # huge carry relative to vol at a coarse step: p > 1
        crr(Option("call", 100, 1.0), MarketData(100, 0.50, 0.0, 0.05), n=2)
    res = crr(Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2), n=50)
    assert 0 < res.p < 1 and res.u > 1 > res.d


def test_lr_requires_odd_steps_no_silent_rounding():
    with pytest.raises(InputError, match="odd"):
        leisen_reimer(Option("call", 100, 1.0), MarketData(100, 0.05, 0, 0.2), n=200)
    with pytest.raises(InputError):
        lr_parameters(100, 100, 1.0, 100, 0.05, 0.0, 0.2)


def test_peizer_pratt_properties():
    assert peizer_pratt(0.0, 101) == pytest.approx(0.5)
    assert 0 < peizer_pratt(-3.0, 101) < 0.5 < peizer_pratt(3.0, 101) < 1
    assert peizer_pratt(1.0, 101) + peizer_pratt(-1.0, 101) == pytest.approx(1.0)


def test_lr_martingale_condition():
    S, K, T, n, r, q, s = 100, 95, 0.75, 101, 0.04, 0.01, 0.25
    u, d, p = lr_parameters(S, K, T, n, r, q, s)
    assert p * u + (1 - p) * d == pytest.approx(math.exp((r - q) * T / n), rel=1e-14)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_european_convergence_to_bsm(kind):
    m = MarketData(100, 0.05, 0.02, 0.25)
    opt = Option(kind, 105, 1.0)
    ref = bsm.price(100, 105, 1.0, 0.05, 0.02, 0.25, kind)
    assert abs(leisen_reimer(opt, m, 201).price - ref) < 2e-4
    assert abs(crr(opt, m, 1000).price - ref) < 5e-3
    assert abs(crr_averaged(opt, m, 1000).price - ref) < 1e-3
    errs = [abs(leisen_reimer(opt, m, n).price - ref) for n in (51, 101, 201, 401)]
    assert errs[-1] < errs[0] / 20  # LR error falls quickly for this smooth European case


def test_tree_greeks_close_to_bsm():
    m = MarketData(100, 0.05, 0.0, 0.2)
    opt = Option("call", 100, 1.0)
    g = bsm.greeks(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    r = crr(opt, m, 2000)
    assert r.delta == pytest.approx(g["delta"], abs=2e-3)
    assert r.gamma == pytest.approx(g["gamma"], abs=5e-4)
    assert r.theta == pytest.approx(g["theta"], abs=2e-2)
    lr = leisen_reimer(opt, m, 501)
    assert lr.delta == pytest.approx(g["delta"], abs=1e-4)
    assert math.isnan(lr.theta)


def test_american_put_obstacle_and_premium():
    m = MarketData(100, 0.05, 0.0, 0.2)
    am = crr(Option("put", 100, 1.0, "american"), m, 1000, record_boundary=True)
    eu = crr(Option("put", 100, 1.0), m, 1000)
    assert am.price > eu.price + 0.4
    assert am.price >= 0.0  # intrinsic at S=K is 0
    deep = crr(Option("put", 100, 1.0, "american"), m.with_(S=60), 500)
    assert deep.price == pytest.approx(40.0, abs=1e-10)  # deep ITM: immediate exercise, V = K - S
    assert len(am.boundary_S) > 0 and np.all(am.boundary_S < 100)
    # boundary rises towards K as expiry approaches
    assert am.boundary_S[-1] > am.boundary_S[0]


def test_american_call_equals_european_without_dividends_nonnegative_rate():
    """Holds for q = 0 and r >= 0 (early exercise of a call is then never optimal)."""
    for r in (0.0, 0.03, 0.08):
        m = MarketData(100, r, 0.0, 0.3)
        a = crr(Option("call", 95, 1.0, "american"), m, 801).price
        e = crr(Option("call", 95, 1.0, "european"), m, 801).price
        assert a == pytest.approx(e, abs=1e-12)


def test_american_call_negative_rate_can_exceed_european():
    """With r < 0 (q = 0) early exercise of a deep ITM call can be optimal."""
    m = MarketData(150, -0.03, 0.0, 0.15)
    a = crr(Option("call", 100, 2.0, "american"), m, 801).price
    e = crr(Option("call", 100, 2.0, "european"), m, 801).price
    assert a > e + 1.0


def test_american_call_with_dividends_exceeds_european():
    m = MarketData(120, 0.03, 0.06, 0.2)
    a = leisen_reimer(Option("call", 100, 1.0, "american"), m, 401).price
    e = leisen_reimer(Option("call", 100, 1.0), m, 401).price
    assert a > e + 0.1


def test_degenerate_cases():
    m0 = MarketData(100, 0.05, 0.0, 0.0)
    assert crr(Option("call", 100, 1.0), m0, 10).price == pytest.approx(
        bsm.price(100, 100, 1.0, 0.05, 0.0, 0.0, "call"))
    assert crr(Option("put", 110, 0.0, "american"), MarketData(100, 0.05, 0, 0.3), 10).price == pytest.approx(10.0)
    # zero vol American put with r > 0: exercise now dominates later
    assert crr(Option("put", 110, 1.0, "american"), m0, 10).price == pytest.approx(10.0)
