"""Delta-hedging accounting: ledger reconciliation, cost accounting, identities, frequency effects."""

import math

import numpy as np
import pytest

from optpricer import bsm
from optpricer.hedging import (
    HedgeSetup,
    frequency_study,
    gamma_pnl_approximation,
    hedge_paths,
    rebalance_grid,
    reconcile_ledger,
    simulate_paths,
    union_grid,
)


def _paths(setup, f=52, n=200, seed=0):
    t = rebalance_grid(setup.T, f)
    S = simulate_paths(setup, t, n, np.random.default_rng(seed))
    return t, S


def test_ledger_reconciles_with_costs_and_dividends():
    s = HedgeSetup(q=0.02, cost_rate=0.001)
    t, S = _paths(s)
    out = hedge_paths(s, t, S, ledger_path=3)
    rec = reconcile_ledger(out["ledger"], s.r, s.T)
    assert abs(rec["difference"]) < 1e-9
    assert out["ledger"]["cash"].iloc[-1] == pytest.approx(out["pnl"][3], abs=1e-12)


def test_discounted_gains_identity_no_costs_no_dividends():
    """e^{-rT} PnL = V0 - e^{-rT} payoff + sum_k Delta_k (e^{-r t_{k+1}} S_{k+1} - e^{-r t_k} S_k)."""
    s = HedgeSetup(q=0.0, cost_rate=0.0)
    t, S = _paths(s, f=21, n=50)
    out = hedge_paths(s, t, S)
    V0 = bsm.price(s.S0, s.K, s.T, s.r, s.q, s.sigma_implied, "call")
    disc = np.exp(-s.r * t)
    gains = np.zeros(S.shape[0])
    for k in range(len(t) - 1):
        d = np.asarray(bsm.delta(S[:, k], s.K, s.T - t[k], s.r, s.q, s.hedge_vol, "call"))
        gains += d * (disc[k + 1] * S[:, k + 1] - disc[k] * S[:, k])
    pay = np.maximum(S[:, -1] - s.K, 0)
    np.testing.assert_allclose(math.exp(-s.r * s.T) * out["pnl"], V0 - disc[-1] * pay + gains, atol=1e-10)


def test_initial_and_final_costs_and_cost_drag():
    s0 = HedgeSetup(cost_rate=0.0)
    sc = HedgeSetup(cost_rate=0.002)
    t, S = _paths(s0, f=12, n=20)
    led = hedge_paths(sc, t, S, ledger_path=0)["ledger"]
    d0 = bsm.delta(100, 100, 1.0, 0.03, 0.0, 0.2, "call")
    assert led["cost"].iloc[0] == pytest.approx(-0.002 * d0 * 100)
    last_delta = led["delta_held"].iloc[-2]
    assert led["cost"].iloc[-1] == pytest.approx(-0.002 * abs(last_delta) * S[0, -1])
    p0 = hedge_paths(s0, t, S)["pnl"]
    pc = hedge_paths(sc, t, S, ledger_path=0)
    fv_costs = float((-led["cost"] * np.exp(s0.r * (s0.T - led["t"]))).sum())
    assert p0[0] - pc["pnl"][0] == pytest.approx(fv_costs, rel=1e-10)


def test_union_grid_nests_frequencies():
    allt, idx = union_grid(1.0, (4, 21, 252, 1000))
    for f, ii in idx.items():
        np.testing.assert_allclose(allt[ii], rebalance_grid(1.0, f), atol=1e-12)
    assert allt[0] == 0 and allt[-1] == 1.0


def test_hedging_error_falls_with_frequency_on_common_paths():
    summ, _ = frequency_study(HedgeSetup(), freqs=(4, 21, 252), n_paths=3000, seed=1)
    sd = summ.set_index("freq_per_year")["std_pnl"]
    assert sd[4] > sd[21] > sd[252]
    # matched implied = realised: the mean P&L is small relative to the premium
    prem = summ["premium"].iloc[0]
    assert abs(summ.set_index("freq_per_year").loc[252, "mean_pnl"]) < 0.05 * prem


def test_vol_mismatch_sign():
    hi = frequency_study(HedgeSetup(sigma_realised=0.30), freqs=(252,), n_paths=2000, seed=2)[0]
    lo = frequency_study(HedgeSetup(sigma_realised=0.12), freqs=(252,), n_paths=2000, seed=2)[0]
    assert hi["mean_pnl"].iloc[0] < 0 < lo["mean_pnl"].iloc[0]  # short option loses when realised > implied


def test_gamma_approximation_correlates_with_actual():
    s = HedgeSetup(sigma_realised=0.25)
    t, S = _paths(s, f=252, n=500, seed=4)
    pnl = hedge_paths(s, t, S)["pnl"]
    approx = gamma_pnl_approximation(s, t, S)
    assert np.corrcoef(pnl, approx)[0, 1] > 0.9


def test_reproducible_by_seed():
    a, pa = frequency_study(HedgeSetup(), freqs=(21,), n_paths=500, seed=9)
    b, pb = frequency_study(HedgeSetup(), freqs=(21,), n_paths=500, seed=9)
    np.testing.assert_array_equal(pa[21], pb[21])


def test_invalid_setup():
    with pytest.raises(ValueError):
        HedgeSetup(cost_rate=-0.1)
    with pytest.raises(ValueError):
        HedgeSetup(sigma_implied=0.0)
