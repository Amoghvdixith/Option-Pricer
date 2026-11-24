"""Run every experiment, write figures/tables, logs and results/run_metadata.json."""

from __future__ import annotations

import json
import logging
import sys
import time

from ..paths import load_config, results_dir
from ..validation import run_metadata, write_json

STAGES = ("pricing", "american", "convergence", "greeks", "iv", "surface", "hedging")


def _logger(mode: str) -> logging.Logger:
    log = logging.getLogger("optpricer.run_all")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(results_dir("logs") / f"run_all_{mode}.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


def run(mode: str = "fast", stages=STAGES) -> dict:
    from . import hedging_exp, pricing_exp, surface_exp

    log = _logger(mode)
    cfgs = {n: load_config(n, mode) for n in ("pricing", "numerical", "market_data", "hedging")}
    times: dict[str, float] = {}
    out: dict = {}
    t_all = time.perf_counter()
    plan = {
        "pricing": lambda: pricing_exp.method_comparison(cfgs["pricing"]),
        "american": lambda: pricing_exp.american_examples(cfgs["pricing"]),
        "convergence": lambda: (pricing_exp.tree_convergence(cfgs["numerical"]),
                                pricing_exp.pde_convergence(cfgs["numerical"]),
                                pricing_exp.explicit_instability(cfgs["numerical"])),
        "greeks": lambda: pricing_exp.greeks_experiments(cfgs["numerical"]),
        "iv": lambda: pricing_exp.iv_roundtrip(cfgs["numerical"]),
        "surface": lambda: (surface_exp.surface_experiments(cfgs["market_data"]),
                            surface_exp.market_snapshot_summary()),
        "hedging": lambda: hedging_exp.hedging_experiments(cfgs["hedging"]),
    }
    for st in stages:
        log.info("stage %s (mode=%s) started", st, mode)
        t0 = time.perf_counter()
        out[st] = plan[st]()
        times[st] = round(time.perf_counter() - t0, 2)
        log.info("stage %s finished in %.2f s", st, times[st])
    total = time.perf_counter() - t_all
    snap = out.get("surface", (None, None))[1] if "surface" in out else None
    provenance = {
        "pricing/convergence/greeks/iv": "model parameters only; no market data",
        "surface": "SYNTHETIC European chain from optpricer.market.synthetic_european_chain (seeded, SSVI truth); "
                   "saved to data/synthetic/synthetic_european_chain.csv",
        "hedging": "simulated GBM paths under P with seeds from configs/hedging.yaml",
        "market_snapshot": "none used" if snap is None else
        "delayed yfinance snapshot in data/snapshots (AMERICAN; descriptive IV only)",
    }
    meta = run_metadata({k: v for k, v in cfgs.items()}, seed=None, runtime_s=round(total, 2), provenance=provenance)
    meta["mode"] = mode
    meta["stage_runtimes_s"] = times
    meta["seeds"] = {"pricing": cfgs["pricing"]["seed"], "synthetic_chain": cfgs["market_data"]["synthetic"]["seed"],
                     "hedging": cfgs["hedging"]["seed"]}
    full = tuple(stages) == STAGES
    target = results_dir() / "run_metadata.json" if full else (
        results_dir("logs") / f"run_metadata_{mode}_{'_'.join(stages)}.json")
    write_json(meta, target)
    log.info("stages done in %.1f s; metadata written to %s", total, target.relative_to(results_dir().parent))
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Run all optpricer experiments")
    ap.add_argument("--mode", choices=("fast", "research"), default="fast")
    ap.add_argument("--stages", nargs="*", default=list(STAGES), choices=list(STAGES))
    a = ap.parse_args(argv)
    run(a.mode, a.stages)
    print(json.dumps({"status": "ok", "mode": a.mode}))
    return 0
