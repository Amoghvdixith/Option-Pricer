"""Build and execute the four notebooks in notebooks/ (they call package functions only).

Usage: python scripts/build_notebooks.py [--no-execute]

The notebooks use small budgets so they run in about a minute. They do not write to
results/; the full-size outputs come from scripts/run_all.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

SETUP = """%matplotlib inline
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from optpricer import Option, MarketData, price, price_all, bsm
pd.set_option("display.width", 160); pd.set_option("display.precision", 6)"""

NOTEBOOKS = {
    "01_method_agreement.ipynb": [
        ("md", "# 01 - Method agreement\nEuropean prices from BSM, CRR, Leisen-Reimer, Crank-Nicolson and Monte Carlo, "
               "plus American options (trees and PDE only). **Inputs are model parameters, not market data.**"),
        ("code", SETUP),
        ("code", """m = MarketData(S=100, r=0.05, q=0.0, sigma=0.20)
for kind in ("call", "put"):
    df = price_all(Option(kind, 100, 1.0), m, methods=["bs", "crr", "lr", "cn", "mc"],
                   settings={"crr": {"n": 1000}, "lr": {"n": 201}, "mc": {"n_paths": 200_000, "seed": 1}})
    df["diff_vs_bs"] = df["price"] - df.loc[df.method == "bs", "price"].iloc[0]
    display(df[["method", "price", "std_error", "diff_vs_bs", "runtime_s", "settings"]])"""),
        ("md", "Deterministic methods differ from BSM by discretisation error. Monte Carlo differs by sampling error "
               "and should be judged against its standard error, never used to excuse tree/PDE bias."),
        ("code", """am = Option("put", 100, 1.0, "american")
display(price_all(am, m, methods=["bs", "mc", "crr", "lr", "cn"], settings={"lr": {"n": 1001}})[["method", "price", "status"]])
res = price(am, m, "cn", n_space=800, n_time=800)
{k: v for k, v in res.diagnostics.items() if k.startswith(("psor", "complementarity", "obstacle"))}"""),
        ("md", "BSM and terminal-payoff MC are refused for American options. The PSOR diagnostics report "
               "iterations, convergence and the complementarity residual of the linear complementarity problem."),
        ("code", """from optpricer.binomial import leisen_reimer
for r in (0.05, 0.0, -0.02):
    mm = MarketData(130, r, 0.0, 0.2)
    a = leisen_reimer(Option("call", 100, 2.0, "american"), mm, 1001).price
    e = leisen_reimer(Option("call", 100, 2.0), mm, 1001).price
    print(f"q=0, r={r:+.2f}: American - European call = {a - e:.3e}")"""),
        ("md", "With q = 0 the American call equals the European call when r >= 0. At a negative rate early "
               "exercise of a deep in-the-money call can be optimal, so the equality fails."),
    ],
    "02_convergence.ipynb": [
        ("md", "# 02 - Numerical convergence\nObserved (not assumed) convergence of trees and finite differences."),
        ("code", SETUP),
        ("code", """from optpricer.binomial import crr, crr_averaged, leisen_reimer
from optpricer.validation import observed_order
m, o = MarketData(100, 0.05, 0.0, 0.2), Option("call", 105, 1.0)
ref = bsm.price(100, 105, 1.0, 0.05, 0.0, 0.2, "call")
ns = np.array([51, 101, 201, 401, 801])
tab = pd.DataFrame({"n": ns,
    "CRR": [crr(o, m, n).price - ref for n in ns],
    "CRR avg": [crr_averaged(o, m, n).price - ref for n in ns],
    "LR": [leisen_reimer(o, m, n).price - ref for n in ns]})
display(tab)
print({c: round(-observed_order(ns, tab[c]), 2) for c in ["CRR", "CRR avg", "LR"]})
ax = tab.set_index("n").abs().plot(logy=True, logx=True, marker="o"); ax.set_ylabel("|error|");"""),
        ("code", """from optpricer.pde import fd_price
o = Option("call", 100, 1.0)
rows = []
for M in (100, 200, 400, 800):
    r = fd_price(o, m, "cn", M, 1600)
    rows.append({"M": M, "dx": r.diagnostics["dx"], "error": r.price - bsm.price(100, 100, 1, 0.05, 0, 0.2, "call")})
sp = pd.DataFrame(rows); display(sp)
print("observed spatial order:", round(observed_order(sp.dx, sp.error), 2))"""),
        ("code", """from optpricer import UnstableSchemeError
try:
    fd_price(o, m, "explicit", 200, 50)
except UnstableSchemeError as exc:
    print("Rejected by default:", str(exc)[:160], "...")
bad = fd_price(o, m, "explicit", 100, 60, allow_unstable=True)
print("intentional unstable run: price =", round(bad.price, 2), "max|g| =", round(bad.diagnostics["von_neumann_max_amplification"], 3))"""),
    ],
    "03_vol_surface.ipynb": [
        ("md", "# 03 - Volatility smiles and surface (SYNTHETIC European data)\nThe chain is generated offline from a "
               "known SSVI surface with seeded noise and deliberately injected bad quotes. **Not market data.**"),
        ("code", SETUP),
        ("code", """from optpricer.market import synthetic_european_chain, clean_chain, cleaning_summary, CleaningPolicy
from optpricer.surface import implied_forwards, build_smiles, arbitrage_diagnostics, TotalVarianceSurface
chain, truth = synthetic_european_chain(seed=7)
cl = clean_chain(chain, CleaningPolicy(r=truth["r"], q=truth["q"]))
display(cleaning_summary(cl))
display(cl.loc[cl.injected_defect != "", ["kind", "strike", "T", "bid", "ask", "injected_defect", "reasons"]])"""),
        ("code", """fw = implied_forwards(cl, n_near=25)
fw["F_true"] = [e["F"] for e in truth["expiries"]]; fw["DF_true"] = [e["DF"] for e in truth["expiries"]]
display(fw[["expiry", "T", "F", "F_se", "F_true", "DF", "DF_se", "DF_true", "n_pairs"]])"""),
        ("code", """sm = build_smiles(cl, fw)
fig, ax = plt.subplots(figsize=(8, 4.5))
for exp, g in sm.groupby("expiry"):
    g = g.sort_values("k"); ax.plot(g.k, g.iv, marker=".", label=exp)
ax.set_xlabel("k = ln(K/F)"); ax.set_ylabel("implied vol"); ax.legend(fontsize=7); ax.set_title("SYNTHETIC smiles")
print("max |IV - true IV| on cleaned OTM quotes:", sm.iv_error_vs_true.abs().max() if "iv_error_vs_true" in sm else (sm.iv - sm.true_iv).abs().max())"""),
        ("code", """viol, summ = arbitrage_diagnostics(cl, fw, sm)
display(summ)
display(viol.head(10))"""),
        ("md", "Mid-price inconsistencies appear because of quote noise and tick rounding, but none is executable "
               "at the quoted bid/ask. Passing finite-grid checks would still not prove that a continuous surface "
               "is arbitrage-free. American-style equity quotes (e.g. SPY) are refused by these European tools "
               "unless an explicit, labelled approximation is requested."),
    ],
    "04_delta_hedging.ipynb": [
        ("md", "# 04 - Delta hedging (SIMULATED paths)\nSelf-financing hedge of a short European call; the same "
               "simulated paths are used for every rebalancing frequency."),
        ("code", SETUP),
        ("code", """from optpricer.hedging import HedgeSetup, frequency_study, hedge_paths, rebalance_grid, simulate_paths, reconcile_ledger, gamma_pnl_approximation
s = HedgeSetup(S0=100, K=100, T=1.0, r=0.03, q=0.0, sigma_implied=0.2, sigma_realised=0.2, mu=0.07)
summ, pnls = frequency_study(s, (4, 21, 252, 1000), n_paths=4000, seed=42)
display(summ[["freq_per_year", "mean_pnl", "se_mean_pnl", "std_pnl", "q05", "q95", "mean_turnover_notional"]])
fig, ax = plt.subplots(figsize=(8, 4))
for f, v in pnls.items(): ax.hist(v, bins=80, histtype="step", density=True, label=f"{f}/yr")
ax.legend(); ax.set_xlabel("terminal P&L");"""),
        ("code", """rows = []
for sr in (0.12, 0.16, 0.20, 0.24, 0.28):
    sm, _ = frequency_study(HedgeSetup(sigma_realised=sr), (252,), 2000, 7)
    rows.append({"sigma_realised": sr, "mean_pnl": sm.mean_pnl[0], "se": sm.se_mean_pnl[0]})
pd.DataFrame(rows)"""),
        ("md", "A short option hedged at implied volatility loses on average when realised volatility exceeds "
               "implied volatility. The gamma-weighted variance formula explains this *approximately*; it is not "
               "an exact identity for every path."),
        ("code", """t = rebalance_grid(1.0, 21)
sq = HedgeSetup(q=0.02, cost_rate=0.001)
S = simulate_paths(sq, t, 1, np.random.default_rng(1))
out = hedge_paths(sq, t, S, ledger_path=0)
display(out["ledger"].round(5).head(8))
reconcile_ledger(out["ledger"], sq.r, sq.T)"""),
        ("code", """t = rebalance_grid(1.0, 252); s2 = HedgeSetup(sigma_realised=0.25)
S = simulate_paths(s2, t, 2000, np.random.default_rng(3))
act, apx = hedge_paths(s2, t, S)["pnl"], gamma_pnl_approximation(s2, t, S)
print("corr(actual, gamma approximation) =", round(np.corrcoef(act, apx)[0, 1], 4),
      "; RMS difference =", round(float(np.sqrt(np.mean((act - apx) ** 2))), 4))"""),
    ],
}


def build(execute: bool = True) -> None:
    NB.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
                    for kind, src in cells]
        if execute:
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbformat.write(nb, NB / name)
        print(f"{'executed' if execute else 'built'}: notebooks/{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true")
    build(not ap.parse_args().no_execute)
    sys.exit(0)
