"""Synthetic chain, cleaning, forward estimation, smiles and arbitrage diagnostics."""

import numpy as np
import pandas as pd
import pytest

from optpricer import UnsupportedMethodError
from optpricer.market import (
    CleaningPolicy,
    SSVIParams,
    clean_chain,
    cleaning_summary,
    load_snapshot,
    save_snapshot,
    synthetic_european_chain,
)
from optpricer.surface import (
    TotalVarianceSurface,
    arbitrage_diagnostics,
    build_smiles,
    implied_forwards,
    risk_neutral_density,
)


@pytest.fixture(scope="module")
def chain():
    df, truth = synthetic_european_chain(seed=7)
    return df, truth


def _true_chain(truth_seed=7):
    """Noise-free chain: bid = ask = true BSM price on the SSVI surface."""
    df, truth = synthetic_european_chain(seed=truth_seed, inject_defects=False)
    df["bid"] = df["ask"] = df["mid"] = df["true_price"]
    df["accepted"] = True
    return df, truth


def test_synthetic_chain_labelled_and_seeded(chain):
    df, _ = chain
    assert set(df["data_label"]) == {"SYNTHETIC"}
    assert set(df["exercise_style"]) == {"european"}
    df2, _ = synthetic_european_chain(seed=7)
    pd.testing.assert_frame_equal(df, df2)


def test_ssvi_generator_enforces_condition():
    with pytest.raises(ValueError):
        SSVIParams(rho=-0.8, eta=1.5)


def test_cleaning_catches_every_injected_defect(chain):
    df, truth = chain
    cl = clean_chain(df, CleaningPolicy(r=truth["r"], q=truth["q"]))
    inj = cl[cl["injected_defect"] != ""]
    assert len(inj) == 12
    for _, row in inj.iterrows():
        assert not row["accepted"], row
        assert row["injected_defect"] in row["reasons"], (row["injected_defect"], row["reasons"])
    assert len(cl) == len(df)  # rejected rows are kept with reasons
    summ = cleaning_summary(cl)
    assert int(summ.loc[summ.reason == "TOTAL_REJECTED_ROWS", "count"].iloc[0]) >= 12


def test_zero_bid_policy_flag_keeps_row():
    df = pd.DataFrame([dict(T=0.5, strike=150.0, kind="call", bid=0.0, ask=0.04, underlying_price=100.0,
                            exercise_style="european")])
    assert not clean_chain(df)["accepted"].iloc[0]
    kept = clean_chain(df, CleaningPolicy(zero_bid="flag"))
    assert kept["accepted"].iloc[0] and "zero_bid" in kept["flags"].iloc[0]


def test_short_expiry_rule():
    df = pd.DataFrame([dict(T=3 / 365, strike=100.0, kind="call", bid=1.0, ask=1.1, underlying_price=100.0,
                            exercise_style="european")])
    assert "short_expiry" in clean_chain(df)["reasons"].iloc[0]


def test_forward_estimation_recovers_truth(chain):
    df, truth = chain
    cl = clean_chain(df, CleaningPolicy(r=truth["r"], q=truth["q"]))
    fw = implied_forwards(cl)
    for (_, row), t in zip(fw.iterrows(), truth["expiries"]):
        # truth within 4 reported standard errors (quote noise makes DF the harder parameter)
        assert abs(row["F"] - t["F"]) < 4 * row["F_se"] + 1e-9 and row["F_se"] < 0.05
        assert abs(row["DF"] - t["DF"]) < 4 * row["DF_se"] + 1e-9 and row["DF_se"] < 0.01


def test_american_quotes_refused_for_parity_tools(chain):
    df, _ = chain
    am = df.copy()
    am["exercise_style"] = "american"
    am["accepted"] = True
    with pytest.raises(UnsupportedMethodError):
        implied_forwards(am)
    fw = implied_forwards(am, approximate=True)
    assert fw["label"].str.contains("APPROXIMATE").all()


def test_true_surface_passes_finite_checks_and_recovers_iv():
    df, truth = _true_chain()
    fw = pd.DataFrame([dict(expiry=e["expiry"], T=e["T"], F=e["F"], DF=e["DF"], implied_rate=truth["r"],
                            implied_yield=truth["q"]) for e in truth["expiries"]])
    sm = build_smiles(df, fw)
    np.testing.assert_allclose(sm["iv"], sm["true_iv"], atol=1e-7)
    viol, summ = arbitrage_diagnostics(df, fw, sm, tol=1e-9)
    assert summ["violations_mid"].sum() == 0, viol


def _single_expiry(df, fw):
    e = fw.iloc[0]["expiry"]
    return df[df["expiry"] == e].copy(), fw.iloc[[0]]


def test_injected_convexity_violation_detected_nonuniform_strikes():
    df, truth = _true_chain()
    fw = pd.DataFrame([dict(expiry=e["expiry"], T=e["T"], F=e["F"], DF=e["DF"]) for e in truth["expiries"]])
    one, fw1 = _single_expiry(df, fw)
    # bump one call near the 2.5/1.0 spacing change (non-uniform neighbours)
    idx = one[(one.kind == "call") & (one.strike == 90.0)].index[0]
    one.loc[idx, ["mid", "bid", "ask"]] += 0.5
    one = one[one.kind == "call"]
    viol, summ = arbitrage_diagnostics(one, fw1)
    conv = viol[viol.check == "convexity"]
    assert len(conv) >= 1 and conv["executable"].any()
    assert any("90" in s for s in conv["strikes"])


def test_injected_monotonicity_and_bound_violations():
    df, truth = _true_chain()
    fw = pd.DataFrame([dict(expiry=e["expiry"], T=e["T"], F=e["F"], DF=e["DF"]) for e in truth["expiries"]])
    one, fw1 = _single_expiry(df, fw)
    one = one[one.kind == "call"].copy()
    ks = sorted(one.strike)
    k_hi = ks[ks.index(100.0) + 1]
    one.loc[one.strike == k_hi, ["mid", "bid", "ask"]] = one.loc[one.strike == 100.0, "mid"].iloc[0] + 0.3
    viol, _ = arbitrage_diagnostics(one, fw1)
    assert (viol.check == "monotonicity").any()
    one2 = df[(df.expiry == fw1.iloc[0]["expiry"]) & (df.kind == "call")].copy()
    one2.loc[one2.strike == 70.0, ["mid", "bid", "ask"]] = 0.1  # below DF*(F-K)
    viol2, _ = arbitrage_diagnostics(one2, fw1)
    assert (viol2.check == "bounds").any()


def test_calendar_violation_detected_in_total_variance():
    df, truth = _true_chain()
    fw = pd.DataFrame([dict(expiry=e["expiry"], T=e["T"], F=e["F"], DF=e["DF"], implied_rate=truth["r"],
                            implied_yield=truth["q"]) for e in truth["expiries"]])
    sm = build_smiles(df, fw)
    t1 = sorted(sm["T"].unique())[1]
    sm.loc[sm["T"] == t1, "w"] *= 0.3  # make the second expiry's total variance too small
    viol, summ = arbitrage_diagnostics(df, fw, sm)
    assert int(summ.loc[summ.check == "calendar", "violations_mid"].iloc[0]) > 0


def test_total_variance_surface_interpolation():
    df, truth = _true_chain()
    fw = pd.DataFrame([dict(expiry=e["expiry"], T=e["T"], F=e["F"], DF=e["DF"], implied_rate=truth["r"],
                            implied_yield=truth["q"]) for e in truth["expiries"]])
    sm = build_smiles(df, fw)
    surf = TotalVarianceSurface(sm)
    T0 = surf.T[2]
    k = np.array([-0.1, 0.0, 0.1])
    np.testing.assert_allclose(surf.w(k, T0), surf._slice_w(2, k))
    mid_T = 0.5 * (surf.T[2] + surf.T[3])
    w = surf.w(k, mid_T)
    assert np.all(w >= surf.w(k, surf.T[2]) - 1e-12) and np.all(w <= surf.w(k, surf.T[3]) + 1e-12)


def test_density_integrates_near_one():
    df, truth = _true_chain()
    e = truth["expiries"][3]
    calls = df[(df.expiry == e["expiry"]) & (df.kind == "call")].sort_values("strike")
    dens = risk_neutral_density(calls.strike, calls.mid, e["DF"])
    assert (dens.density >= -1e-10).all()
    mass = np.trapezoid(dens.density, dens.strike)
    assert 0.8 < mass <= 1.0 + 1e-6  # truncated strike range


def test_snapshot_roundtrip(tmp_path, chain):
    df, truth = chain
    save_snapshot(df, {"source": "synthetic", "label": "SYNTHETIC"}, tmp_path, "snap")
    df2, meta = load_snapshot(tmp_path / "snap.csv")
    assert meta["label"] == "SYNTHETIC" and len(df2) == len(df)
