"""Local Streamlit interface for optpricer (educational; not deployed).

Run from the project root:  streamlit run app/streamlit_app.py

All financial calculations come from the ``optpricer`` package; this file only collects
inputs and displays results. Everything works offline by default. Market data is
downloaded only when the user presses the explicit fetch button.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from optpricer import MarketData, Option, bsm, price, supported
from optpricer.binomial import crr, crr_averaged, leisen_reimer
from optpricer.conventions import DISPLAY_SCALING, to_display_units
from optpricer.errors import OptPricerError
from optpricer.hedging import HedgeSetup, frequency_study
from optpricer.implied_vol import implied_vol
from optpricer.market import CleaningPolicy, clean_chain, synthetic_european_chain
from optpricer.surface import arbitrage_diagnostics, build_smiles, implied_forwards

st.set_page_config(page_title="optpricer - local research app", layout="wide")
st.title("optpricer: options pricing research library (local, educational)")
st.caption("Not production software and not investment advice. Conventions: T in ACT/365F years; r, q "
           "continuously compounded; sigma annualised. Default data are model inputs or SYNTHETIC.")

# ------------------------------------------------------------------ sidebar inputs
with st.sidebar:
    st.header("Contract and market")
    kind = st.radio("Option kind", ["call", "put"], horizontal=True)
    style = st.radio("Exercise style", ["european", "american"], horizontal=True)
    S = st.number_input("Spot S", min_value=0.0, value=100.0, step=1.0)
    K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
    T = st.number_input("Time to expiry T (years, ACT/365F)", min_value=0.0, value=1.0, step=0.05)
    r = st.number_input("Rate r (cont. comp., decimal)", value=0.05, step=0.005, format="%.4f")
    q = st.number_input("Dividend yield q (cont., decimal)", value=0.0, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility sigma (decimal)", min_value=0.0, value=0.20, step=0.01, format="%.4f")
    st.header("Numerical settings")
    crr_n = st.number_input("CRR steps", 10, 5000, 1000, step=50)
    lr_n = st.number_input("LR steps (odd only)", 11, 5001, 201, step=2)
    pde_m = st.number_input("PDE space intervals M", 50, 3200, 400, step=50)
    pde_n = st.number_input("PDE time steps N", 10, 3200, 400, step=50)
    mc_paths = st.number_input("MC paths", 1000, 2_000_000, 200_000, step=10_000)
    mc_seed = st.number_input("MC seed", 0, 10**9, 2026)

try:
    opt = Option(kind, K, T, style)
    mkt = MarketData(S, r, q, sigma)
except OptPricerError as exc:
    st.error(f"Invalid input: {exc}")
    st.stop()

tabs = st.tabs(["1 Pricing comparison", "2 Greeks", "3 Implied vol and smiles", "4 Convergence", "5 Delta hedging"])

# ------------------------------------------------------------------ 1 pricing
with tabs[0]:
    st.subheader(f"{style.title()} {kind}: method comparison")
    methods = ["bs", "crr", "crr_avg", "lr", "implicit", "cn", "mc"]
    chosen = st.multiselect("Methods", methods, default=[m for m in methods if supported(m, style)],
                            help="BS and terminal-payoff MC price European options only; they are disabled "
                                 "for American options.")
    blocked = [m for m in chosen if not supported(m, style)]
    if blocked:
        st.warning(f"Not available for {style} options (disabled): {', '.join(blocked)}")
    if lr_n % 2 == 0:
        st.error("Leisen-Reimer needs an odd number of steps; it is not rounded for you.")
    if st.button("Price", type="primary"):
        rows = []
        settings = {"crr": {"n": crr_n}, "crr_avg": {"n": crr_n}, "lr": {"n": lr_n},
                    "implicit": {"n_space": pde_m, "n_time": pde_n}, "cn": {"n_space": pde_m, "n_time": pde_n},
                    "mc": {"n_paths": int(mc_paths) // 2 * 2, "seed": int(mc_seed)}}
        for m in chosen:
            if not supported(m, style):
                continue
            try:
                res = price(opt, mkt, m, **settings.get(m, {}))
                d = res.diagnostics
                note = ""
                if m == "mc":
                    note = f"95% CI [{res.ci_low:.4f}, {res.ci_high:.4f}] (sampling error only)"
                if m in ("implicit", "cn") and style == "american":
                    note = (f"PSOR iters {d['psor_total_iterations']}, converged {d['psor_converged']}, "
                            f"complementarity {d['complementarity_residual_max']:.1e}")
                rows.append({"method": m, "price": res.price,
                             "std error": res.std_error if m == "mc" else np.nan,
                             "runtime s": res.runtime, "settings": str(res.settings), "notes": note})
            except OptPricerError as exc:
                rows.append({"method": m, "price": np.nan, "notes": f"ERROR: {exc}"})
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch")
        if style == "european":
            st.info("Trees and PDE differ from BSM by discretisation error; MC differs by sampling error (see its CI).")
        else:
            st.info("No closed form for American options; compare trees and PDE, and refine grids to judge accuracy.")

# ------------------------------------------------------------------ 2 Greeks
with tabs[1]:
    st.subheader("Analytical BSM Greeks (European)")
    g = bsm.greeks(S, K, T, r, q, sigma, kind)
    disp = to_display_units(g)
    st.dataframe(pd.DataFrame([{"greek": k, "raw": float(g[k]), "display": float(disp[k]),
                                "display unit": DISPLAY_SCALING[k][1]} for k in g]), width="stretch")
    if style == "american":
        st.warning("These are European BSM Greeks; for American options use tree/PDE Greeks below.")
        tg = crr(opt, mkt, int(crr_n))
        st.write({"CRR tree delta": tg.delta, "CRR tree gamma": tg.gamma, "CRR tree theta (per year)": tg.theta})
    Sg = np.linspace(max(0.2 * K, 1e-6), 1.8 * K, 161)
    gg = to_display_units(bsm.greeks(Sg, K, max(T, 1e-8), r, q, sigma, kind))
    pick = st.selectbox("Plot against spot", ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "price"])
    st.line_chart(pd.DataFrame({"S": Sg, pick: gg[pick]}).set_index("S"))
    st.caption(f"Units: {DISPLAY_SCALING[pick][1]}. Theta is calendar-time decay per day with expiry fixed.")

# ------------------------------------------------------------------ 3 IV
with tabs[2]:
    st.subheader("Implied volatility solver")
    c1, c2 = st.columns(2)
    with c1:
        p_in = st.number_input("Observed option price", min_value=0.0,
                               value=float(round(bsm.price(S, K, max(T, 1e-8), r, q, sigma, kind), 4)))
        approx = st.checkbox("Quote is American: compute a labelled European-BSM approximation", value=False)
        if st.button("Solve IV"):
            try:
                res = implied_vol(p_in, S, K, T, r, q, kind, "american" if approx else "european",
                                  approximate_american=approx)
                st.json(res.to_dict())
            except OptPricerError as exc:
                st.error(str(exc))
    with c2:
        st.markdown("**SYNTHETIC European chain** (offline, seeded) - smiles and arbitrage diagnostics")
        if st.button("Build synthetic smiles"):
            chain, truth = synthetic_european_chain(seed=7)
            cl = clean_chain(chain, CleaningPolicy(r=truth["r"], q=truth["q"]))
            fw = implied_forwards(cl, n_near=25)
            sm = build_smiles(cl, fw)
            piv = sm.pivot_table(index="k", columns="expiry", values="iv")
            st.line_chart(piv)
            _, summ = arbitrage_diagnostics(cl, fw, sm)
            st.dataframe(summ, width="stretch")
            st.caption("SYNTHETIC data. Finite-grid checks; passing them does not prove a surface arbitrage-free.")
    st.divider()
    st.markdown("**Optional delayed market chain (explicit action only)**")
    ticker = st.text_input("Ticker (yfinance; US options are American-style)", value="SPY")

    @st.cache_data(ttl=900, show_spinner="Downloading delayed chain...")
    def _fetch(t: str):
        from optpricer.market import fetch_yfinance_chain
        return fetch_yfinance_chain(t, 4)

    if st.button("Fetch delayed chain now"):
        try:
            df, meta = _fetch(ticker)
            st.warning(f"{meta['label']}. Retrieved {meta['retrieved_utc']} UTC; spot time "
                       f"{meta['underlying_spot_time_utc']}. Exercise style: AMERICAN. IVs below are descriptive "
                       "European-BSM approximations with r as set in the sidebar and q=0.")
            cl = clean_chain(df, CleaningPolicy(r=r, q=0.0))
            ok = cl[cl.accepted]
            Sx = meta["underlying_spot"]
            ok = ok[((ok.kind == "call") & (ok.strike >= Sx)) | ((ok.kind == "put") & (ok.strike < Sx))]
            if len(ok) < 10:
                st.error(f"Only {len(ok)} usable two-sided quotes (zero bids: {(df.bid == 0).mean():.0%}). The market "
                         "may be closed; no smile is shown.")
            else:
                ivs = [implied_vol(x.mid, Sx, x.strike, x.T, r, 0.0, x.kind, "american", approximate_american=True).iv
                       for x in ok.itertuples()]
                ok = ok.assign(iv_descriptive=ivs, k_spot=np.log(ok.strike / Sx))
                st.line_chart(ok.pivot_table(index="k_spot", columns="expiry", values="iv_descriptive"))
        except Exception as exc:  # network/provider errors are shown, never hidden
            st.error(f"Download failed: {type(exc).__name__}: {exc}")

# ------------------------------------------------------------------ 4 convergence
with tabs[3]:
    st.subheader("Tree convergence for the current contract")
    if T == 0 or sigma == 0 or S == 0:
        st.info("Degenerate inputs (T, sigma or S equal to 0): no lattice to converge.")
    elif st.button("Run tree convergence"):
        ns = [25, 51, 101, 201, 401, 801]
        rows = []
        for n in ns:
            rows.append({"n": n, "CRR": crr(opt, mkt, n).price, "CRR avg": crr_averaged(opt, mkt, n).price,
                         "LR": leisen_reimer(opt, mkt, n).price})
        df = pd.DataFrame(rows).set_index("n")
        ref = bsm.price(S, K, T, r, q, sigma, kind) if style == "european" else leisen_reimer(opt, mkt, 4001).price
        ref_name = "BSM closed form" if style == "european" else "LR n=4001 (itself approximate)"
        st.write(f"Reference: {ref_name} = {ref:.6f}")
        st.line_chart((df - ref).abs().apply(np.log10).rename(columns=lambda c: f"log10|{c} error|"))
        st.dataframe(df, width="stretch")
        st.caption("Observed convergence, not an assumed rate. LR uses odd n.")

# ------------------------------------------------------------------ 5 hedging
with tabs[4]:
    st.subheader("Self-financing delta hedge of a short European call (simulated)")
    c1, c2, c3 = st.columns(3)
    s_imp = c1.number_input("Implied/pricing vol", 0.01, 2.0, 0.20)
    s_real = c2.number_input("Realised (simulation) vol", 0.01, 2.0, 0.20)
    mu = c3.number_input("Physical drift mu (total return)", -0.5, 0.5, 0.07)
    cost = c1.number_input("Proportional cost (fraction of notional)", 0.0, 0.05, 0.0, format="%.4f")
    npaths = c2.number_input("Paths", 500, 50_000, 5_000, step=500)
    seed = c3.number_input("Seed", 0, 10**9, 42)
    if st.button("Run hedging simulation"):
        if T <= 0 or S <= 0:
            st.error("Need T > 0 and S > 0.")
        else:
            setup = HedgeSetup(S0=S, K=K, T=T, r=r, q=q, sigma_implied=s_imp, sigma_realised=s_real, mu=mu,
                               cost_rate=cost)
            summ, pnls = frequency_study(setup, (4, 21, 252), int(npaths), int(seed))
            st.dataframe(summ, width="stretch")
            hist = {f: np.histogram(v, bins=60) for f, v in pnls.items()}
            st.bar_chart(pd.DataFrame({f"{f}/yr": h[0] for f, h in hist.items()},
                                      index=np.round(hist[252][1][:-1], 2)))
            st.caption("SIMULATED GBM paths, same paths for every frequency. P&L in time-T currency. "
                       "Histogram bins are those of the 252/yr case; the gamma-weighted volatility explanation "
                       "is an approximation, not a per-path guarantee.")

st.sidebar.caption(f"BSM price for current inputs: {bsm.price(S, K, T, r, q, sigma, kind):.4f}"
                   if style == "european" else "American: see the Pricing tab")
_ = math
