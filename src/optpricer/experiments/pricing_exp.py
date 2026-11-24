"""Pricing experiments: method agreement, American options, convergence, Greeks, implied vol.

All inputs are synthetic model parameters (no market data). Every figure footnote states
the settings, and every figure has a CSV with its numbers in ``results/tables``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from .. import bsm
from ..binomial import crr, crr_averaged, leisen_reimer
from ..conventions import to_display_units
from ..implied_vol import implied_vol
from ..instruments import MarketData, Option
from ..mc import mc_european, mcsim_available, mcsim_crosscheck
from ..pde import fd_price
from ..validation import bump_greeks, observed_order, pde_identity_residual
from .common import DIVERGING, PALETTE, SEQ_BLUE, plt, save_fig, save_table

LABEL = "MODEL PARAMETERS ONLY (no market data)"


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


# =========================================================================== 1. method agreement
def method_comparison(cfg: dict) -> dict:
    S = cfg["spot"]
    rows = []
    seed = cfg["seed"]
    for sc in cfg["scenarios"]:
        m = MarketData(S, sc["r"], sc["q"], sc["sigma"])
        for T in cfg["maturities"]:
            for K in cfg["strikes"]:
                for kind in ("call", "put"):
                    o = Option(kind, K, T)
                    ref = float(bsm.price(S, K, T, m.r, m.q, m.sigma, kind))
                    row = dict(scenario=sc["name"], r=m.r, q=m.q, sigma=m.sigma, K=K, T=T, kind=kind, bs=ref)
                    t0 = time.perf_counter()
                    row["crr"] = crr(o, m, cfg["crr_n"]).price
                    row["crr_s"] = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    row["lr"] = leisen_reimer(o, m, cfg["lr_n"]).price
                    row["lr_s"] = time.perf_counter() - t0
                    pr = fd_price(o, m, "cn", cfg["pde_n_space"], cfg["pde_n_time"])
                    row["cn"], row["cn_s"] = pr.price, pr.runtime
                    seed += 1
                    mres = mc_european(o, m, cfg["mc_paths"], seed=seed, antithetic=False)
                    row.update(mc=mres.price, mc_se=mres.std_error, mc_s=mres.runtime, mc_seed=seed)
                    for meth in ("crr", "lr", "cn", "mc"):
                        row[f"{meth}_err"] = row[meth] - ref
                    row["mc_z"] = row["mc_err"] / row["mc_se"]
                    rows.append(row)
    df = pd.DataFrame(rows)
    df["budget"] = (f"CRR n={cfg['crr_n']}; LR n={cfg['lr_n']}; CN {cfg['pde_n_space']}x{cfg['pde_n_time']} "
                    f"(Rannacher 2, n_sd=6); MC {cfg['mc_paths']} paths plain")
    save_table(df, "method_agreement_european")
    summ = []
    for meth in ("crr", "lr", "cn"):
        e = df[f"{meth}_err"].abs()
        summ.append(dict(method=meth, max_abs_error=e.max(), median_abs_error=e.median(),
                         n_cases=len(e), n_above_1e_3=int((e > 1e-3).sum()),
                         worst_case={c: df[c].iloc[int(np.argmax(e.to_numpy()))]
                                     for c in ("scenario", "K", "T", "kind")},
                         mean_runtime_s=df[f"{meth}_s"].mean()))
    z = df["mc_z"]
    summ.append(dict(method="mc", max_abs_error=df["mc_err"].abs().max(), median_abs_error=df["mc_err"].abs().median(),
                     n_cases=len(z), n_above_1e_3=int((df["mc_err"].abs() > 1e-3).sum()),
                     worst_case={"max_abs_z": float(z.abs().max()), "share_abs_z_gt_2": float((z.abs() > 2).mean())},
                     mean_runtime_s=df["mc_s"].mean()))
    summary = pd.DataFrame(summ)
    save_table(summary, "method_agreement_summary")

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (meth, lab) in enumerate([("crr", f"CRR n={cfg['crr_n']}"), ("lr", f"LR n={cfg['lr_n']}"),
                                     ("cn", f"CN {cfg['pde_n_space']}x{cfg['pde_n_time']}"),
                                     ("mc", f"MC {cfg['mc_paths']:,} (|err|)")]):
        e = df[f"{meth}_err"].abs().clip(lower=1e-12)
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(e))
        ax.scatter(np.full(len(e), i) + jitter, e, s=14, color=PALETTE[i], alpha=0.75, edgecolor="none", label=lab)
    ax.scatter(np.full(len(df), 3) + 0.32, 2 * df["mc_se"], marker="_", s=60, color=MUTED_GREY, label="MC 2 x SE")
    ax.axhline(1e-3, color=PALETTE[7], lw=1, ls="--")
    ax.text(3.45, 1.1e-3, "1e-3", color=PALETTE[7], fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(range(4), ["CRR", "LR", "Crank-Nicolson", "Monte Carlo"])
    ax.set_ylabel("|price - BSM closed form|")
    ax.set_title(f"European method agreement vs BSM: {len(df)} cases (K 80-120, T 1m-2y, 3 (r,q,sigma) scenarios)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    save_fig(fig, "01_method_agreement_errors",
             f"{LABEL}. European calls/puts, S=100. Deterministic methods: discretisation error. MC: sampling error "
             "(compare with 2 SE). MC error is not a measure of tree/PDE bias.")

    # headline table (guide's example) with American row
    m = MarketData(**{k: cfg["headline"][k] for k in ("S", "r", "q", "sigma")})
    K, T = cfg["headline"]["K"], cfg["headline"]["T"]
    head_rows: list[dict] = []
    for kind, style in (("call", "european"), ("put", "european"), ("put", "american")):
        o = Option(kind, K, T, style)  # type: ignore[arg-type]
        r: dict = {"option": f"{style} {kind} K={K:g} T={T:g}"}
        r["bs"] = float(bsm.price(m.S, K, T, m.r, m.q, m.sigma, kind)) if style == "european" else math.nan
        r[f"crr_n{cfg['crr_n']}"] = crr(o, m, cfg["crr_n"]).price
        r[f"lr_n{cfg['lr_n']}"] = leisen_reimer(o, m, cfg["lr_n"]).price
        pr = fd_price(o, m, "cn", cfg["pde_n_space"], cfg["pde_n_time"])
        r[f"cn_{cfg['pde_n_space']}x{cfg['pde_n_time']}"] = pr.price
        if style == "european":
            mres = mc_european(o, m, cfg["mc_paths"], seed=cfg["seed"])
            r["mc"], r["mc_se"] = mres.price, mres.std_error
            r["mc_paths"] = cfg["mc_paths"]
        else:
            r["mc"] = r["mc_se"] = math.nan
            r["mc_paths"] = "n/a (Longstaff-Schwartz not implemented)"
            r["psor_iterations"] = pr.diagnostics["psor_total_iterations"]
            r["complementarity_residual"] = pr.diagnostics["complementarity_residual_max"]
        head_rows.append(r)
    head = pd.DataFrame(head_rows)
    save_table(head, "method_agreement_headline")
    out = {"n_cases": len(df), "summary": summary, "headline": head}
    if mcsim_available():
        o = Option("call", K, T)
        ext = mcsim_crosscheck(o, m, cfg["mc_paths"], seed=cfg["seed"] + 999)
        ours = mc_european(o, m, cfg["mc_paths"], seed=cfg["seed"] + 1000)
        xc = pd.DataFrame([dict(engine="optpricer.mc", price=ours.price, se=ours.std_error),
                           dict(engine="mcsim.price_mc (sibling project, optional)", price=ext["price"],
                                se=ext["std_error"]),
                           dict(engine="bsm closed form", price=float(bsm.price(m.S, K, T, m.r, m.q, m.sigma, "call")),
                                se=0.0)])
        save_table(xc, "mc_crosscheck_mcsim")
        out["mcsim"] = xc
    return out


MUTED_GREY = "#52514e"


# =========================================================================== 2. American options
def american_reference(m: MarketData, o: Option, lr_ns=(4001, 8001), pde_grid=(3200, 3200)) -> dict:
    a = leisen_reimer(o, m, lr_ns[0]).price
    b = leisen_reimer(o, m, lr_ns[1]).price
    p = fd_price(o, m, "cn", pde_grid[0], pde_grid[1])
    ref = b
    unc = max(abs(b - a), abs(b - p.price))
    return dict(reference=ref, lr_coarse=a, lr_fine=b, pde_fine=p.price, uncertainty=unc,
                pde_psor_iters=p.diagnostics["psor_total_iterations"],
                pde_complementarity=p.diagnostics["complementarity_residual_max"])


def american_examples(cfg: dict) -> dict:
    hd = cfg["headline"]
    m = MarketData(hd["S"], hd["r"], hd["q"], hd["sigma"])
    o = Option("put", hd["K"], hd["T"], "american")
    ar = cfg["american_reference"]
    ref = american_reference(m, o, tuple(ar["lr_n"]), tuple(ar["pde_grid"]))
    ref_df = pd.DataFrame([dict(option="american put S=K=100 T=1 r=5% q=0 sigma=20%", **ref,
                                reference_method=f"LR n={ar['lr_n'][1]}; uncertainty = max gap to LR n="
                                                 f"{ar['lr_n'][0]} and CN {ar['pde_grid'][0]}x{ar['pde_grid'][1]}")])
    save_table(ref_df, "american_put_reference")

    rows = []
    for sc in cfg["scenarios"][:2]:
        mm = MarketData(cfg["spot"], sc["r"], sc["q"], sc["sigma"])
        for T in (0.25, 1.0):
            for K in cfg["strikes"]:
                for kind in ("put", "call"):
                    oo = Option(kind, K, T, "american")
                    rf = leisen_reimer(oo, mm, 4001).price
                    c = crr(oo, mm, cfg["crr_n"]).price
                    lr = leisen_reimer(oo, mm, cfg["lr_n"]).price
                    pr = fd_price(oo, mm, "cn", cfg["pde_n_space"], cfg["pde_n_time"])
                    eu = float(bsm.price(cfg["spot"], K, T, mm.r, mm.q, mm.sigma, kind))
                    rows.append(dict(scenario=sc["name"], r=mm.r, q=mm.q, sigma=mm.sigma, K=K, T=T, kind=kind,
                                     ref_lr4001=rf, crr=c, lr=lr, cn=pr.price, crr_err=c - rf, lr_err=lr - rf,
                                     cn_err=pr.price - rf, european_bsm=eu, early_ex_premium=rf - eu,
                                     psor_iters=pr.diagnostics["psor_total_iterations"],
                                     psor_max_iter_step=pr.diagnostics["psor_max_iterations_per_step"],
                                     complementarity=pr.diagnostics["complementarity_residual_max"],
                                     psor_converged=pr.diagnostics["psor_converged"]))
    am = pd.DataFrame(rows)
    save_table(am, "american_method_comparison")

    # American call checks with the rate regime stated explicitly
    chk = []
    for label, r, q in (("q=0, r=5% (>=0): equality expected", 0.05, 0.0), ("q=0, r=0: equality expected", 0.0, 0.0),
                        ("q=0, r=-2% (<0): early exercise can pay", -0.02, 0.0),
                        ("q=4%, r=3%: early exercise can pay", 0.03, 0.04)):
        mm = MarketData(130, r, q, 0.2)
        a = leisen_reimer(Option("call", 100, 2.0, "american"), mm, 2001).price
        e = leisen_reimer(Option("call", 100, 2.0), mm, 2001).price
        chk.append(dict(case=label, S=130, K=100, T=2.0, sigma=0.2, american_lr2001=a, european_lr2001=e,
                        difference=a - e))
    save_table(pd.DataFrame(chk), "american_call_rate_regimes")

    # Early-exercise premium vs moneyness for several rates
    S_grid = np.linspace(70, 130, 31)
    prem_rows = []
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    for i, r in enumerate((0.0, 0.02, 0.05, 0.08)):
        prem = []
        for S in S_grid:
            mm = MarketData(S, r, 0.0, 0.2)
            a = leisen_reimer(Option("put", 100, 1.0, "american"), mm, 801).price
            e = leisen_reimer(Option("put", 100, 1.0), mm, 801).price
            prem.append(a - e)
            prem_rows.append(dict(S=S, K=100, T=1.0, r=r, q=0.0, sigma=0.2, american=a, european=e, premium=a - e))
        ax.plot(S_grid / 100, prem, color=PALETTE[i], label=f"r = {r:.0%}")
    ax.set_xlabel("moneyness S/K")
    ax.set_ylabel("American - European put (LR n=801, same tree)")
    ax.set_title("Early-exercise premium of an American put")
    ax.legend()
    save_fig(fig, "05_american_early_exercise_premium",
             f"{LABEL}. K=100, T=1, q=0, sigma=20%. Premium = American - European on the same LR tree (bias largely "
             "cancels). At r=0 (q=0) early exercise of a put has no value.")
    save_table(pd.DataFrame(prem_rows), "american_early_exercise_premium")

    # Exercise boundary
    pde_res = fd_price(o, m, "cn", 1600, 1600)
    tree_res = crr(o, m, 2000, record_boundary=True)
    bd = pd.concat([pd.DataFrame(dict(method="CN 1600x1600 + PSOR", t=pde_res.boundary_t, S_star=pde_res.boundary_S)),
                    pd.DataFrame(dict(method="CRR n=2000", t=tree_res.boundary_t, S_star=tree_res.boundary_S))])
    save_table(bd, "american_put_exercise_boundary")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(tree_res.boundary_t, tree_res.boundary_S, color=PALETTE[1], lw=1.2, label="CRR n=2000 (node-level)")
    ax.plot(pde_res.boundary_t, pde_res.boundary_S, color=PALETTE[0], lw=2, label="CN 1600x1600 + PSOR (grid-level)")
    ax.axhline(100, color=MUTED_GREY, lw=1, ls=":")
    ax.text(0.02, 100.4, "strike K = 100", fontsize=8, color=MUTED_GREY)
    ax.set_xlabel("calendar time t (years); expiry at t = 1")
    ax.set_ylabel("critical spot S*(t): exercise if S <= S*")
    ax.set_title("American put early-exercise boundary")
    ax.legend(loc="lower right")
    save_fig(fig, "06_american_put_exercise_boundary",
             f"{LABEL}. K=100, T=1, r=5%, q=0, sigma=20%. Boundary is resolved only to grid/node spacing; "
             f"near expiry it tends to K (r >= q). PSOR: omega 1.2, tol 1e-10, total iterations "
             f"{pde_res.diagnostics['psor_total_iterations']}.")
    return {"reference": ref_df, "american": am, "calls": pd.DataFrame(chk)}


# =========================================================================== 3. convergence
def tree_convergence(cfg: dict) -> pd.DataFrame:
    rows = []
    for case_name in ("case", "off_atm_case"):
        c = cfg[case_name]
        m = MarketData(c["S"], c["r"], c["q"], c["sigma"])
        o = Option("call", c["K"], c["T"])
        ref = float(bsm.price(c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], "call"))
        for n in cfg["tree_steps"]:
            rows.append(dict(case=case_name, style="european", n=n, method="CRR", price=crr(o, m, n).price, ref=ref))
            rows.append(dict(case=case_name, style="european", n=n, method="CRR avg(n,n+1)",
                             price=crr_averaged(o, m, n).price, ref=ref))
            rows.append(dict(case=case_name, style="european", n=_odd(n), method="LR",
                             price=leisen_reimer(o, m, _odd(n)).price, ref=ref))
    # American put: reference from a fine LR tree
    c = cfg["case"]
    m = MarketData(c["S"], c["r"], c["q"], c["sigma"])
    oa = Option("put", c["K"], c["T"], "american")
    ref_am = leisen_reimer(oa, m, 8001).price
    for n in cfg["tree_steps"]:
        rows.append(dict(case="case", style="american_put", n=n, method="CRR", price=crr(oa, m, n).price, ref=ref_am))
        rows.append(dict(case="case", style="american_put", n=_odd(n), method="LR",
                         price=leisen_reimer(oa, m, _odd(n)).price, ref=ref_am))
    df = pd.DataFrame(rows)
    df["error"] = df["price"] - df["ref"]
    orders = []
    for (case, style, meth), g in df.groupby(["case", "style", "method"]):
        orders.append(dict(case=case, style=style, method=meth,
                           observed_order_fit=-observed_order(g["n"], g["error"]),
                           max_abs_error_n_ge_400=g.loc[g.n >= 400, "error"].abs().max()))
    save_table(df, "tree_convergence")
    save_table(pd.DataFrame(orders), "tree_convergence_orders")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    panels = [("case", "european", "ATM European call"), ("off_atm_case", "european", "K=105, q=2% European call"),
              ("case", "american_put", "ATM American put (ref LR n=8001)")]
    for ax, (case, style, title) in zip(axes, panels):
        g = df[(df.case == case) & (df["style"] == style)]
        for i, meth in enumerate(["CRR", "CRR avg(n,n+1)", "LR"]):
            gg = g[g.method == meth]
            if len(gg):
                ax.loglog(gg["n"], gg["error"].abs().clip(lower=1e-14), marker="o", ms=3.5, color=PALETTE[i],
                          label=meth)
        ax.set_title(title)
        ax.set_xlabel("steps n")
    axes[0].set_ylabel("|tree price - reference|")
    axes[0].legend()
    save_fig(fig, "02_tree_convergence",
             f"{LABEL}. ATM: S=K=100, T=1, r=5%, q=0, sigma=20%. Observed orders (log-log fit) in "
             "tables/tree_convergence_orders.csv. LR uses odd n (n+1 where n is even). American LR reference "
             "itself carries ~1e-5 error.")
    return df


def pde_convergence(cfg: dict) -> dict:
    c = cfg["case"]
    m = MarketData(c["S"], c["r"], c["q"], c["sigma"])
    o = Option("call", c["K"], c["T"])
    ref = float(bsm.price(c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], "call"))
    rows = []
    Nfix = cfg["pde_time_fixed_space"]
    for M in cfg["pde_space"]:
        for smooth in (True, False):
            r = fd_price(o, m, "cn", M, Nfix, smooth_payoff=smooth, rannacher_steps=cfg["rannacher_steps"])
            rows.append(dict(study="space", scheme="cn", M=M, N=Nfix, smooth_payoff=smooth, price=r.price,
                             error=r.price - ref, dx=r.diagnostics["dx"], runtime=r.runtime))
    Mfix = cfg["pde_time_fixed_space"]
    for scheme, rs in (("cn", cfg["rannacher_steps"]), ("cn", 0), ("implicit", 0)):
        fine = fd_price(o, m, scheme, Mfix, 4 * cfg["pde_time"][-1], rannacher_steps=rs).price
        for N in cfg["pde_time"]:
            r = fd_price(o, m, scheme, Mfix, N, rannacher_steps=rs)
            rows.append(dict(study="time", scheme=f"{scheme}{' + Rannacher' if rs else ''}", M=Mfix, N=N,
                             smooth_payoff=True, price=r.price, error=r.price - fine, dx=r.diagnostics["dx"],
                             dtau=r.diagnostics["dtau"], runtime=r.runtime))
    df = pd.DataFrame(rows)
    save_table(df, "pde_convergence")
    orders = []
    for keys, g in df.groupby(["study", "scheme", "smooth_payoff"]):
        study_k, scheme_k, smooth_k = keys  # type: ignore[misc]
        h = g["dx"] if study_k == "space" else 1.0 / g["N"]
        orders.append(dict(study=study_k, scheme=scheme_k, smooth_payoff=smooth_k,
                           observed_order_fit=observed_order(h, g["error"])))
    ords = pd.DataFrame(orders)
    save_table(ords, "pde_convergence_orders")

    # domain truncation (boundary error)
    dom_rows: list[dict] = []
    for nsd in cfg["domain_n_sd"]:
        M = int(round(400 * nsd / 6.0))
        for kind in ("call", "put"):
            r = fd_price(Option(kind, c["K"], c["T"]), m, "cn", max(M, 20), 400, n_sd=nsd)
            refk = float(bsm.price(c["S"], c["K"], c["T"], c["r"], c["q"], c["sigma"], kind))
            dom_rows.append(dict(n_sd=nsd, kind=kind, M=max(M, 20), N=400, S_min=r.diagnostics["S_min"],
                            S_max=r.diagnostics["S_max"], price=r.price, error=r.price - refk))
    dom = pd.DataFrame(dom_rows)
    save_table(dom, "pde_domain_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    g = df[df.study == "space"]
    for i, smooth in enumerate((True, False)):
        gg = g[g.smooth_payoff == smooth]
        ax = axes[0]
        ax.loglog(gg["M"], gg["error"].abs(), marker="o", ms=3.5, color=PALETTE[i],
                  label="cell-averaged payoff" if smooth else "point payoff")
    xs = np.array(cfg["pde_space"], float)
    axes[0].loglog(xs, g["error"].abs().iloc[0] * (xs[0] / xs) ** 2, ls=":", color=MUTED_GREY, lw=1,
                   label="slope -2 guide")
    axes[0].set_title(f"Space: CN, N={Nfix} fixed, error vs BSM")
    axes[0].set_xticks(cfg["pde_space"], [str(v) for v in cfg["pde_space"]])
    axes[0].minorticks_off()
    axes[0].set_xlabel("space intervals M")
    axes[0].set_ylabel("|error|")
    axes[0].legend(fontsize=8)
    g = df[df.study == "time"]
    for i, sch in enumerate(g.scheme.unique()):
        gg = g[g.scheme == sch]
        axes[1].loglog(gg["N"], gg["error"].abs().clip(lower=1e-14), marker="o", ms=3.5, color=PALETTE[i], label=sch)
    axes[1].set_title(f"Time: M={Mfix} fixed, error vs same-grid N={4 * cfg['pde_time'][-1]}")
    axes[1].set_xlabel("time steps N")
    axes[1].set_xticks(cfg["pde_time"], [str(v) for v in cfg["pde_time"]])
    axes[1].minorticks_off()
    axes[1].legend(fontsize=8)
    for i, kind in enumerate(("call", "put")):
        gg = dom[dom.kind == kind]
        axes[2].semilogy(gg["n_sd"], gg["error"].abs().clip(lower=1e-14), marker="o", ms=3.5, color=PALETTE[i],
                         label=kind)
    axes[2].set_title("Domain truncation: half-width in sigma*sqrt(T)")
    axes[2].set_xlabel("n_sd (dx held ~constant)")
    axes[2].legend(fontsize=8)
    save_fig(fig, "03_pde_convergence",
             f"{LABEL}. European call S=K=100, T=1, r=5%, q=0, sigma=20%, log-spot grid with S on a node (no "
             "interpolation error). Spatial errors vs closed form; temporal errors vs a same-grid fine-time solution, "
             "so they exclude spatial error. Orders in tables/pde_convergence_orders.csv.")
    return {"table": df, "orders": ords, "domain": dom}


def explicit_instability(cfg: dict) -> pd.DataFrame:
    c = cfg["case"]
    d = cfg["instability_demo"]
    m = MarketData(c["S"], c["r"], c["q"], c["sigma"])
    o = Option("call", c["K"], d["T"])
    stable = fd_price(o, m, "explicit", d["n_space"], d["n_time_stable"], return_grid=True)
    unstable = fd_price(o, m, "explicit", d["n_space"], d["n_time_unstable"], return_grid=True, allow_unstable=True)
    rows = []
    for lab, r in (("stable", stable), ("unstable (allow_unstable=True)", unstable)):
        dd = r.diagnostics
        rows.append(dict(config=lab, M=r.n_space, N=r.n_time, dx=dd["dx"], dtau=dd["dtau"],
                         lambda_sigma2_dt_over_dx2=c["sigma"] ** 2 * dd["dtau"] / dd["dx"] ** 2,
                         von_neumann_max_amp=dd["von_neumann_max_amplification"],
                         von_neumann_stable=dd["von_neumann_stable"], monotone=dd["monotone"], price=r.price,
                         bsm=float(bsm.price(c["S"], c["K"], d["T"], c["r"], c["q"], c["sigma"], "call")),
                         max_abs_value=float(np.nanmax(np.abs(r.V_history)))))
    tab = pd.DataFrame(rows)
    save_table(tab, "explicit_stability_demo")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    assert unstable.V_history is not None and unstable.S_grid is not None and unstable.V_final is not None
    assert stable.S_grid is not None and stable.V_final is not None
    H = unstable.V_history
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.log10(np.abs(H) + 1.0)
    S = unstable.S_grid
    keep = (S > 30) & (S < 300)
    im = axes[0].pcolormesh(S[keep], unstable.tau_grid, Z[:, keep], cmap=SEQ_BLUE, shading="auto")
    fig.colorbar(im, ax=axes[0], label="log10(|V| + 1)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("spot S (log scale)")
    axes[0].set_ylabel("time to expiry tau")
    axes[0].set_title(f"Unstable explicit scheme: M={d['n_space']}, N={d['n_time_unstable']}")
    axes[1].plot(stable.S_grid, stable.V_final, color=PALETTE[0], label=f"explicit stable (N={d['n_time_stable']})")
    axes[1].plot(S, np.clip(unstable.V_final, -60, 160), color=PALETTE[1], lw=1.2,
                 label=f"explicit unstable (N={d['n_time_unstable']}), clipped to [-60,160]")
    axes[1].plot(S, bsm.price(S, c["K"], d["T"], c["r"], c["q"], c["sigma"], "call"), color=MUTED_GREY, ls=":",
                 lw=1.5, label="BSM closed form")
    axes[1].set_xlim(40, 220)
    axes[1].set_xlabel("spot S")
    axes[1].set_ylabel("value at tau = T")
    axes[1].set_title("Final-time profile")
    axes[1].legend(fontsize=8)
    save_fig(fig, "04_explicit_instability",
             f"{LABEL}. Isolated, intentional demonstration (allow_unstable=True). Call K=100, T=1, r=5%, sigma=20%, "
             f"log grid n_sd=6. Von Neumann max|g| = {rows[1]['von_neumann_max_amp']:.3f} (unstable) vs "
             f"{rows[0]['von_neumann_max_amp']:.3f} (stable). Default behaviour rejects the unstable grid.")
    return tab


# =========================================================================== 4. Greeks
def greeks_experiments(cfg: dict) -> dict:
    S = np.linspace(50, 150, 201)
    K, r, q, sig = 100.0, 0.05, 0.0, 0.2
    rows = []
    for T in (0.1, 0.5, 1.0):
        for kind in ("call", "put"):
            g = to_display_units(bsm.greeks(S, K, T, r, q, sig, kind))
            arrs = {k: np.asarray(v) for k, v in g.items()}
            for i, s in enumerate(S):
                rows.append(dict(S=s, T=T, kind=kind, **{k: float(v[i]) for k, v in arrs.items()}))
    gdf = pd.DataFrame(rows)
    save_table(gdf, "greeks_vs_spot")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6))
    spec = [("delta", "delta (per 1 unit S)"), ("gamma", "gamma (per 1 unit S)"), ("vega", "vega per 1 vol point"),
            ("theta", "theta per calendar day"), ("rho", "rho per 1 rate point"), ("vanna", "vanna per 1 vol point")]
    for ax, (key, lab) in zip(axes.ravel(), spec):
        for i, T in enumerate((0.1, 0.5, 1.0)):
            for kind, ls in (("call", "-"), ("put", "--")):
                gg = gdf[(gdf["T"] == T) & (gdf.kind == kind)]
                if key in ("gamma", "vega", "vanna") and kind == "put":
                    continue  # identical to call
                ax.plot(gg["S"], gg[key], color=PALETTE[i], ls=ls, lw=1.6,
                        label=f"T={T} {kind}" if key == "delta" else None)
        ax.set_title(lab)
        ax.set_xlabel("spot S")
    axes[0, 0].legend(fontsize=7, ncol=2)
    save_fig(fig, "07_greeks_vs_spot",
             f"{LABEL}. K=100, r=5%, q=0, sigma=20%. Solid = call, dashed = put (gamma, vega, vanna are equal for "
             "calls and puts). Display units: vega/100, rho/100, theta/365 (calendar day), vanna/100.")

    # validation table vs central bumps
    cases = [(100, 100, 1.0, 0.05, 0.0, 0.2), (100, 90, 0.25, 0.03, 0.02, 0.3), (100, 120, 2.0, 0.01, 0.03, 0.15),
             (50, 55, 0.5, -0.01, 0.0, 0.4), (100, 150, 0.05, 0.05, 0.0, 0.2), (100, 100, 0.002, 0.05, 0.0, 0.2)]
    val = []
    for case in cases:
        for kind in ("call", "put"):
            a = bsm.greeks(*case, kind)
            b = bump_greeks(*case, kind, h_rel=1e-4)
            for k in b:
                av = float(a[k])
                val.append(dict(S=case[0], K=case[1], T=case[2], r=case[3], q=case[4], sigma=case[5], kind=kind,
                                greek=k, analytic=av, bump=b[k], abs_err=abs(av - b[k]),
                                rel_err=abs(av - b[k]) / max(abs(av), 1e-300)))
    vdf = pd.DataFrame(val)
    save_table(vdf, "greek_validation_bump")
    # bump-size sweep at ATM
    sweep = []
    case = (100, 100, 1.0, 0.05, 0.0, 0.2)
    a = bsm.greeks(*case)
    for h in cfg["greek_bumps"]:
        b = bump_greeks(*case, h_rel=h)
        for k in b:
            sweep.append(dict(h=h, greek=k, rel_err=abs(b[k] - float(a[k])) / abs(float(a[k]))))
    sw = pd.DataFrame(sweep)
    save_table(sw, "greek_bump_size_sweep")
    rng = np.random.default_rng(5)
    n = 2000
    Sx, Kx = rng.uniform(20, 300, n), rng.uniform(20, 300, n)
    Tx, rx = rng.uniform(0.02, 3, n), rng.uniform(-0.02, 0.1, n)
    qx, sx = rng.uniform(0, 0.06, n), rng.uniform(0.05, 1, n)
    pde_res = []
    for kind in ("call", "put"):
        res = pde_identity_residual(Sx, Kx, Tx, rx, qx, sx, kind)
        pde_res.append(dict(kind=kind, n=n, max_abs_residual=float(np.max(np.abs(res))),
                            max_scaled_residual=float(np.max(np.abs(res) / np.maximum(Sx, Kx)))))
    save_table(pd.DataFrame(pde_res), "greek_pde_identity_residuals")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, k in enumerate(["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]):
        gg = sw[sw.greek == k]
        axes[0].loglog(gg["h"], gg["rel_err"].clip(lower=1e-16), marker="o", ms=3.5, color=PALETTE[i], label=k)
    axes[0].set_xlabel("bump size h (relative for S, absolute for sigma, r, T)")
    axes[0].set_ylabel("|bump - analytic| / |analytic|")
    axes[0].set_title("Bump-size sensitivity (ATM, central differences)")
    axes[0].legend(fontsize=8, ncol=2)
    piv = vdf.groupby(["greek"])["rel_err"].max().sort_values()
    axes[1].barh(piv.index, piv.clip(lower=1e-16).values, color=PALETTE[0], height=0.6)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("max relative error over 12 cases (h=1e-4)")
    axes[1].set_title("Analytic vs bump: worst case per Greek")
    save_fig(fig, "08_greek_validation_errors",
             f"{LABEL}. Truncation error dominates for large h, round-off for small h. Worst cases include near-expiry "
             "(T=0.002) and deep-OTM short-dated options, where bump estimates are least accurate.")
    return {"greeks": gdf, "validation": vdf, "sweep": sw, "pde_identity": pd.DataFrame(pde_res)}


# =========================================================================== 5. implied vol
def iv_roundtrip(cfg: dict) -> pd.DataFrame:
    g = cfg["iv_grid"]
    rows = []
    for T in g["maturities"]:
        for s in g["sigmas"]:
            for mny in g["moneyness"]:
                K = 100 * mny
                for kind in ("call", "put"):
                    p = float(bsm.price(100, K, T, 0.03, 0.01, s, kind))
                    res = implied_vol(p, 100, K, T, 0.03, 0.01, kind)
                    rp = float(bsm.price(100, K, T, 0.03, 0.01, res.iv, kind)) if np.isfinite(res.iv) else np.nan
                    rows.append(dict(T=T, sigma_true=s, moneyness_K_over_S=mny, kind=kind, price=p, iv=res.iv,
                                     vol_error=res.iv - s, price_residual=rp - p, status=res.status,
                                     iterations=res.iterations, method=res.method, vega=res.vega,
                                     vol_uncertainty=res.vol_uncertainty, poorly_identified=res.poorly_identified))
    df = pd.DataFrame(rows)
    save_table(df, "iv_roundtrip")
    fig, axes = plt.subplots(1, len(g["maturities"]), figsize=(6 * len(g["maturities"]), 4.4), squeeze=False)
    for ax, T in zip(axes[0], g["maturities"]):
        d = df[(df["T"] == T) & (((df.kind == "call") & (df.moneyness_K_over_S >= 1))
                                 | ((df.kind == "put") & (df.moneyness_K_over_S < 1)))]
        piv = d.pivot(index="sigma_true", columns="moneyness_K_over_S", values="vol_error").abs()
        Z = np.log10(piv.fillna(1.0).clip(lower=1e-16))
        im = ax.imshow(Z.values, cmap=DIVERGING, vmin=-16, vmax=0, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
        ax.set_xlabel("K / S (OTM side: puts < 1 <= calls)")
        ax.set_ylabel("true sigma")
        ax.set_title(f"log10 |IV - true sigma|, T = {T}")
        poor = d.pivot(index="sigma_true", columns="moneyness_K_over_S", values="poorly_identified")
        for (i, j), v in np.ndenumerate(poor.values):
            st = d[(d.sigma_true == poor.index[i]) & (d.moneyness_K_over_S == poor.columns[j])]["status"].iloc[0]
            if st != "converged":
                ax.text(j, i, "B" if st == "zero_time_value" else "x", ha="center", va="center", fontsize=8)
            elif v:
                ax.text(j, i, "*", ha="center", va="center", fontsize=10)
        fig.colorbar(im, ax=ax)
    save_fig(fig, "09_iv_roundtrip",
             f"{LABEL}. S=100, r=3%, q=1%. '*' = flagged poorly identified (price resolution / vega > 1e-8), "
             "'B' = no time value at double precision (status zero_time_value, IV=0 boundary, error = -sigma). "
             "Prices reproduced to <=1e-9 wherever a finite IV is returned; see tables/iv_roundtrip.csv.")
    return df


def run_all_pricing(pricing_cfg: dict, numerical_cfg: dict) -> dict:
    out: dict = {}
    out["agreement"] = method_comparison(pricing_cfg)
    out["american"] = american_examples(pricing_cfg)
    out["tree_conv"] = tree_convergence(numerical_cfg)
    out["pde_conv"] = pde_convergence(numerical_cfg)
    out["instability"] = explicit_instability(numerical_cfg)
    out["greeks"] = greeks_experiments(numerical_cfg)
    out["iv"] = iv_roundtrip(numerical_cfg)
    return out
