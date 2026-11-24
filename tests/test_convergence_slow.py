"""Slower convergence experiments (run with: pytest -m slow).

These measure observed orders instead of assuming them. Bands are wide on purpose: the
claim is "close to the theoretical order for this smooth case", not exact slopes.
"""

import numpy as np
import pytest

from optpricer import MarketData, Option, bsm
from optpricer.binomial import crr, leisen_reimer
from optpricer.pde import fd_price
from optpricer.validation import observed_order

pytestmark = pytest.mark.slow

M = MarketData(100, 0.05, 0.0, 0.2)
REF_CALL = bsm.price(100, 100, 1.0, 0.05, 0.0, 0.2, "call")


def test_crr_first_order_lr_second_order_european():
    ns = np.array([101, 201, 401, 801, 1601])
    o = Option("call", 105, 1.0)
    ref = bsm.price(100, 105, 1.0, 0.05, 0.0, 0.2, "call")
    e_crr = [crr(o, M, n).price - ref for n in ns]
    e_lr = [leisen_reimer(o, M, n).price - ref for n in ns]
    assert 0.6 < -observed_order(ns, e_crr) < 1.5
    assert 1.7 < -observed_order(ns, e_lr) < 2.3


def test_cn_second_order_in_space_and_time():
    o = Option("call", 100, 1.0)
    Ms = np.array([100, 200, 400, 800])
    errs = []
    dxs = []
    for m in Ms:
        r = fd_price(o, M, "cn", int(m), 1600)
        errs.append(r.price - REF_CALL)
        dxs.append(r.diagnostics["dx"])
    assert 1.8 < observed_order(dxs, errs) < 2.2
    Ns = np.array([20, 40, 80, 160])
    fine = fd_price(o, M, "cn", 800, 2560).price
    et = [fd_price(o, M, "cn", 800, int(n)).price - fine for n in Ns]
    assert 1.7 < observed_order(1 / Ns, et) < 2.3


def test_american_put_pde_and_tree_converge_together():
    o = Option("put", 100, 1.0, "american")
    ref = leisen_reimer(o, M, 8001).price
    assert ref == pytest.approx(6.0903, abs=2e-4)  # widely quoted value for this benchmark
    errs = [abs(fd_price(o, M, "cn", n, n).price - ref) for n in (200, 400, 800, 1600)]
    assert errs[-1] < 2e-4 and errs[-1] < errs[0]


def test_large_mc_run_chunked():
    from optpricer.mc import mc_european

    r = mc_european(Option("put", 100, 1.0), M, 2_000_000, seed=31, chunk_size=300_000)
    assert abs(r.price - bsm.price(100, 100, 1.0, 0.05, 0.0, 0.2, "put")) < 4 * r.std_error
    assert r.std_error < 0.01
