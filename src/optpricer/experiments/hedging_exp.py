"""Delta-hedging experiments (simulated GBM paths; no market data)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from ..hedging import (
    HedgeSetup,
    frequency_study,
    gamma_pnl_approximation,
    hedge_paths,
    rebalance_grid,
    reconcile_ledger,
    simulate_paths,
)
from .common import PALETTE, plt, save_fig, save_table

LABEL = "SIMULATED GBM paths (physical measure), not market data"


def _setup(cfg: dict, **over) -> HedgeSetup:
    keys = ("S0", "K", "T", "r", "q", "sigma_implied", "sigma_realised", "mu", "cost_rate")
    base = {k: cfg[k] for k in keys}
    base.update(over)
    return HedgeSetup(**base)


def _foot(s: HedgeSetup, extra: str = "", cost_note: str | None = None) -> str:
    return (f"{LABEL}. Short 1 European call K={s.K:g}, T={s.T:g}y, S0={s.S0:g}; r={s.r:.1%}, q={s.q:.1%}; "
            f"mu(P)={s.mu:.0%}; sigma_impl={s.sigma_implied:.0%} (premium & hedge), sigma_real={s.sigma_realised:.0%}; "
            f"{cost_note or f'cost={s.cost_rate * 1e4:.0f} bp of traded notional'}; ACT/365F, "
            f"rebalance grid k/f. {extra}")


def hedging_experiments(cfg: dict) -> dict:
    seed = cfg["seed"]
    s0 = _setup(cfg)
    freqs = tuple(cfg["frequencies"])
    summ, pnls = frequency_study(s0, freqs, cfg["n_paths"], seed)
    summ["std_x_sqrt_n"] = summ["std_pnl"] * np.sqrt(summ["n_rebalances"])
    save_table(summ, "hedging_frequency_summary")
    pnl_df = pd.DataFrame({f"freq_{f}": v for f, v in pnls.items()})
    save_table(pnl_df, "hedging_pnl_paths")

    fig, ax = plt.subplots(figsize=(8, 4.4))
    lo, hi = np.quantile(pnls[freqs[0]], [0.005, 0.995])
    bins = np.linspace(lo, hi, 90).tolist()
    for i, f in enumerate(freqs):
        ax.hist(pnls[f], bins=bins, histtype="step", lw=1.8, color=PALETTE[i], density=True,
                label=f"{f}/yr: sd {np.std(pnls[f], ddof=1):.3f}")
    ax.set_xlabel("terminal net P&L (time-T currency)")
    ax.set_ylabel("density")
    ax.set_title("Delta-hedged short call: P&L distribution by rebalancing frequency (same paths)")
    ax.legend(fontsize=8)
    save_fig(fig, "14_hedging_pnl_distributions",
             _foot(s0, f"{cfg['n_paths']:,} common paths; premium {summ['premium'].iloc[0]:.4f}."))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(summ["n_rebalances"], summ["std_pnl"], yerr=summ["std_pnl"] / np.sqrt(2 * (summ["n_paths"] - 1)),
                marker="o", color=PALETTE[0], capsize=3, label="simulated SD of P&L (+-1 approx. SE)")
    n = summ["n_rebalances"].to_numpy(float)
    ref = summ["std_pnl"].iloc[-1] * np.sqrt(n[-1] / n)
    ax.plot(n, ref, ls=":", color="#52514e", lw=1.2, label="1/sqrt(n) reference through last point")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rebalances over the option's life")
    ax.set_ylabel("SD of terminal P&L")
    ax.set_title("Hedging error vs rebalancing frequency")
    ax.legend(fontsize=8)
    save_fig(fig, "15_hedging_error_vs_frequency", _foot(s0, "Reference line is a guide, not a fitted formula."))

    # vol mismatch
    mm_rows: list[dict] = []
    for sr in cfg["mismatch_realised_vols"]:
        s = _setup(cfg, sigma_realised=sr)
        sm, pn = frequency_study(s, (cfg["mismatch_frequency"],), cfg["n_paths_mismatch"], seed)
        row = sm.iloc[0].to_dict()
        row.update(sigma_realised=sr, mismatch=sr - s.sigma_implied)
        mm_rows.append(row)
    mm = pd.DataFrame(mm_rows)
    save_table(mm, "hedging_vol_mismatch")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(mm["mismatch"], mm["mean_pnl"], yerr=2 * mm["se_mean_pnl"], marker="o", color=PALETTE[0], capsize=3,
                label="mean P&L +- 2 SE")
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axvline(0, color="#52514e", lw=0.8, ls=":")
    ax.set_xlabel("sigma_realised - sigma_implied")
    ax.set_ylabel("mean terminal P&L of hedged short call")
    ax.set_title("Mean hedged P&L vs volatility mismatch")
    ax.legend(fontsize=8)
    save_fig(fig, "16_mean_pnl_vs_vol_mismatch",
             _foot(_setup(cfg), f"Realised vol varied; {cfg['mismatch_frequency']} rebalances/yr; "
                                f"{cfg['n_paths_mismatch']:,} paths per point, same seed."))

    # costs vs frequency
    cr_parts: list[pd.DataFrame] = []
    for c in cfg["cost_rates"]:
        s = _setup(cfg, cost_rate=c)
        sm, _ = frequency_study(s, tuple(cfg["cost_frequencies"]), cfg["n_paths_cost"], seed, keep_pnl=False)
        sm["cost_rate"] = c
        cr_parts.append(sm)
    cr = pd.concat(cr_parts, ignore_index=True)
    cr["rmse_pnl"] = np.sqrt(cr["mean_pnl"] ** 2 + cr["std_pnl"] ** 2)
    save_table(cr, "hedging_cost_frequency_tradeoff")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, c in enumerate(cfg["cost_rates"]):
        g = cr[cr["cost_rate"] == c]
        axes[0].plot(g["freq_per_year"], g["mean_pnl"], marker="o", ms=3.5, color=PALETTE[i], label=f"{c * 1e4:.0f} bp")
        axes[1].plot(g["freq_per_year"], g["rmse_pnl"], marker="o", ms=3.5, color=PALETTE[i], label=f"{c * 1e4:.0f} bp")
    for ax, lab in zip(axes, ("mean terminal P&L", "RMS P&L = sqrt(mean^2 + SD^2)")):
        ax.set_xscale("log")
        ax.set_xlabel("rebalances per year")
        ax.set_ylabel(lab)
        ax.legend(title="cost", fontsize=8)
    axes[0].set_title("Costs drag the mean as frequency rises")
    axes[1].set_title("Risk/cost trade-off: RMS P&L has an interior minimum with costs")
    save_fig(fig, "17_cost_frequency_tradeoff",
             _foot(_setup(cfg), f"{cfg['n_paths_cost']:,} common paths per cost level; proportional costs on "
                                "initial trade, every rebalance and final liquidation.",
                   cost_note="cost varies by line (0-20 bp of traded notional)"))

    # accounting audit for one path, plus a dividend case
    sq = _setup(cfg, q=cfg["dividend_case_q"], cost_rate=0.001)
    t = rebalance_grid(sq.T, 21)
    S = simulate_paths(sq, t, 1, np.random.default_rng(seed))
    out = hedge_paths(sq, t, S, ledger_path=0)
    led = out["ledger"]
    save_table(led, "hedging_accounting_audit_example")
    rec = reconcile_ledger(led, sq.r, sq.T)
    s0_q = _setup(cfg, q=cfg["dividend_case_q"])
    sq_sum, _ = frequency_study(s0_q, freqs, cfg["n_paths_mismatch"], seed, keep_pnl=False)
    sq_sum["q"] = cfg["dividend_case_q"]
    save_table(sq_sum, "hedging_frequency_summary_with_dividends")
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), sharex=True)
    axes[0].plot(led["t"], led["S"], color=PALETTE[0], label="spot S")
    axes[0].axhline(sq.K, color="#52514e", ls=":", lw=1)
    axes[0].set_ylabel("spot")
    ax2 = axes[1]
    ax2.step(led["t"], led["cash"], where="post", color=PALETTE[1], label="cash account")
    ax2.plot(led["t"], led["portfolio_value"], color=PALETTE[2], marker="o", ms=3,
             label="marked value (cash + hedge - option)")
    ax2.set_ylabel("currency")
    ax2.set_xlabel("t (years)")
    ax2.legend(fontsize=8)
    axes[0].set_title(f"Accounting audit, one path, 21 rebalances/yr: final P&L {rec['final_cash']:.4f}; "
                      f"FV of flows {rec['future_value_of_flows']:.4f} (diff {rec['difference']:.1e})")
    save_fig(fig, "18_hedging_accounting_audit",
             _foot(sq, "Dividends credited as q*Delta*dt*(S_k+S_{k+1})/2 (trapezoid). Full ledger in "
                       "tables/hedging_accounting_audit_example.csv."))

    # gamma approximation vs actual (diagnostic)
    sg = _setup(cfg, sigma_realised=0.25)
    t = rebalance_grid(sg.T, 252)
    S = simulate_paths(sg, t, 3000, np.random.default_rng(seed + 1))
    actual = hedge_paths(sg, t, S)["pnl"]
    approx = gamma_pnl_approximation(sg, t, S)
    gp = pd.DataFrame({"actual_pnl": actual, "gamma_approx": approx})
    save_table(gp, "hedging_gamma_approximation")
    return {"frequency": summ, "mismatch": mm, "costs": cr, "audit": rec, "dividend_case": sq_sum,
            "gamma_corr": float(np.corrcoef(actual, approx)[0, 1]),
            "gamma_rmse": float(np.sqrt(np.mean((actual - approx) ** 2))),
            "setup": dataclasses.asdict(s0)}
