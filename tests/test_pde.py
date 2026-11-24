"""Finite differences: tridiagonal solver, stability, boundaries/domain, Rannacher, PSOR/LCP."""

import numpy as np
import pytest

from optpricer import ConvergenceError, InputError, MarketData, Option, UnstableSchemeError, bsm
from optpricer.binomial import leisen_reimer
from optpricer.pde import explicit_stability, fd_price, psor, thomas


def test_thomas_matches_dense_solver():
    rng = np.random.default_rng(10)
    n = 60
    lower, upper = rng.uniform(-1, 0, n), rng.uniform(-1, 0, n)
    diag = 2.5 + rng.uniform(0, 1, n)
    rhs = rng.normal(size=n)
    A = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    np.testing.assert_allclose(thomas(lower, diag, upper, rhs), np.linalg.solve(A, rhs), rtol=1e-12, atol=1e-12)


def test_thomas_and_banded_give_same_price():
    o, m = Option("put", 100, 1.0), MarketData(100, 0.05, 0.01, 0.25)
    a = fd_price(o, m, "cn", 120, 60, linear_solver="thomas").price
    b = fd_price(o, m, "cn", 120, 60, linear_solver="banded").price
    assert a == pytest.approx(b, abs=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_cn_matches_bsm_on_400x400(kind, q):
    m = MarketData(100, 0.05, q, 0.2)
    res = fd_price(Option(kind, 100, 1.0), m, "cn", 400, 400)
    ref = bsm.price(100, 100, 1.0, 0.05, q, 0.2, kind)
    assert abs(res.price - ref) < 5e-4
    g = bsm.greeks(100, 100, 1.0, 0.05, q, 0.2, kind)
    assert res.delta == pytest.approx(g["delta"], abs=1e-4)
    assert res.gamma == pytest.approx(g["gamma"], abs=1e-4)
    assert res.theta == pytest.approx(g["theta"], abs=5e-3)


def test_implicit_first_order_in_time_cn_faster():
    o, m = Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2)
    ref = bsm.price(100, 100, 1.0, 0.05, 0.0, 0.2, "call")
    e_imp = [abs(fd_price(o, m, "implicit", 800, N).price - ref) for N in (20, 40, 80)]
    e_cn = [abs(fd_price(o, m, "cn", 800, N).price - ref) for N in (20, 40, 80)]
    # implicit: halving dt roughly halves the error; CN errors are much smaller
    assert 1.6 < e_imp[0] / e_imp[1] < 2.4 and 1.6 < e_imp[1] / e_imp[2] < 2.4
    assert all(c < i for c, i in zip(e_cn, e_imp))


def test_explicit_unstable_rejected_with_explanation():
    o, m = Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2)
    with pytest.raises(UnstableSchemeError, match="unstable"):
        fd_price(o, m, "explicit", 200, 50)
    res = fd_price(o, m, "explicit", 200, 50, allow_unstable=True)
    assert res.diagnostics["von_neumann_stable"] is False
    assert res.diagnostics["nonfinite_values"] or abs(res.price) > 1e3


def test_explicit_stable_configuration_accurate():
    o, m = Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2)
    res = fd_price(o, m, "explicit", 200, 2000)
    assert res.diagnostics["von_neumann_stable"] and res.diagnostics["monotone"]
    assert abs(res.price - bsm.price(100, 100, 1, 0.05, 0, 0.2, "call")) < 2e-3


def test_explicit_stability_helper_matches_analysis():
    # pure diffusion (nu = 0, r = 0): stable iff lambda = sigma^2 h / dx^2 <= 1
    sig, dx = 0.2, 0.01
    for lam, ok in ((0.9, True), (1.1, False)):
        h = lam * dx * dx / sig**2
        a = c = sig**2 / (2 * dx * dx)
        b = -sig**2 / dx**2
        assert explicit_stability(a, b, c, h, 0.0)["von_neumann_stable"] is ok


def test_domain_truncation_sensitivity():
    o, m = Option("put", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2)
    ref = bsm.price(100, 100, 1.0, 0.05, 0.0, 0.2, "put")
    # fixed dx roughly: scale n_space with the width
    e15 = abs(fd_price(o, m, "cn", 200, 400, n_sd=1.5).price - ref)
    e3 = abs(fd_price(o, m, "cn", 400, 400, n_sd=3.0).price - ref)
    e6 = abs(fd_price(o, m, "cn", 800, 400, n_sd=6.0).price - ref)
    assert e15 > 100 * e6  # truncation dominates on a narrow domain
    assert abs(e3 - e6) < 1e-6  # beyond ~3 sd the truncation error is negligible here


def test_rannacher_damps_gamma_oscillation():
    o, m = Option("call", 100, 0.25), MarketData(100, 0.05, 0.0, 0.3)
    exact = bsm.greeks(100, 100, 0.25, 0.05, 0.0, 0.3, "call")["gamma"]
    kw = dict(n_space=800, n_time=20, smooth_payoff=False)
    no = fd_price(o, m, "cn", rannacher_steps=0, **kw)
    yes = fd_price(o, m, "cn", rannacher_steps=2, **kw)
    assert abs(yes.gamma - exact) < abs(no.gamma - exact)
    assert yes.diagnostics["rannacher_steps"] == 2


def test_input_validation():
    o, m = Option("call", 100, 1.0), MarketData(100, 0.05, 0.0, 0.2)
    with pytest.raises(InputError):
        fd_price(o, m, "leapfrog")
    with pytest.raises(InputError):
        fd_price(o, m, "cn", 2, 10)


# --------------------------------------------------------------------------- American / PSOR
def test_psor_solves_small_lcp():
    rng = np.random.default_rng(11)
    m = 40
    lw, dg, up = -0.4, 2.0, -0.5
    rhs = rng.normal(size=m)
    g = rng.normal(size=m)
    x, its, ok = psor(lw, dg, up, rhs, g, np.zeros(m), 0.0, 0.0, omega=1.3, tol=1e-13)
    assert ok
    A = np.diag(np.full(m, dg)) + np.diag(np.full(m - 1, lw), -1) + np.diag(np.full(m - 1, up), 1)
    resid = A @ x - rhs
    assert np.all(x >= g - 1e-12)
    assert np.all(resid >= -1e-10)
    assert np.max(np.abs(np.minimum(x - g, resid))) < 1e-10


def test_american_put_pde_agrees_with_tree_and_reports_diagnostics():
    m = MarketData(100, 0.05, 0.0, 0.2)
    o = Option("put", 100, 1.0, "american")
    res = fd_price(o, m, "cn", 800, 800)
    ref = leisen_reimer(o, m, 4001).price
    assert abs(res.price - ref) < 2e-3
    d = res.diagnostics
    assert d["psor_converged"] and d["psor_total_iterations"] > 0
    assert d["obstacle_violation_max"] == 0.0
    assert d["complementarity_residual_max"] < 1e-8
    assert len(res.boundary_S) > 0 and np.all(res.boundary_S <= 100)


def test_american_put_at_least_european_and_intrinsic():
    m = MarketData(100, 0.05, 0.0, 0.3)
    for S in (70.0, 90.0, 100.0, 120.0):
        mm = m.with_(S=S)
        a = fd_price(Option("put", 100, 1.0, "american"), mm, "cn", 400, 200).price
        e = fd_price(Option("put", 100, 1.0), mm, "cn", 400, 200).price
        assert a >= e - 1e-12
        assert a >= max(100 - S, 0) - 1e-9


def test_psor_nonconvergence_is_raised_not_hidden():
    m = MarketData(100, 0.05, 0.0, 0.2)
    o = Option("put", 100, 1.0, "american")
    with pytest.raises(ConvergenceError):
        fd_price(o, m, "cn", 200, 50, max_iter=2)
    r = fd_price(o, m, "cn", 200, 50, max_iter=2, allow_nonconvergence=True)
    assert r.diagnostics["psor_converged"] is False


def test_american_call_with_dividend_boundary():
    m = MarketData(100, 0.03, 0.07, 0.25)
    o = Option("call", 100, 1.0, "american")
    pde_px = fd_price(o, m, "cn", 800, 800).price
    tree_px = leisen_reimer(o, m, 2001).price
    assert abs(pde_px - tree_px) < 3e-3
    assert pde_px > bsm.price(100, 100, 1.0, 0.03, 0.07, 0.25, "call") + 0.05
