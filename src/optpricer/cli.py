"""Command-line interface.

Examples::

    optpricer price --kind put --style american --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --method lr --n 501
    optpricer compare --kind call --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2
    optpricer run-all --mode fast
"""

from __future__ import annotations

import argparse
import json
import sys

from .api import METHODS, price, price_all
from .errors import OptPricerError
from .instruments import MarketData, Option


def _contract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kind", choices=("call", "put"), default="call")
    p.add_argument("--style", choices=("european", "american"), default="european")
    p.add_argument("--S", type=float, default=100.0, help="spot")
    p.add_argument("--K", type=float, default=100.0, help="strike")
    p.add_argument("--T", type=float, default=1.0, help="years to expiry (ACT/365F)")
    p.add_argument("--r", type=float, default=0.05, help="continuously compounded rate")
    p.add_argument("--q", type=float, default=0.0, help="continuous dividend yield")
    p.add_argument("--sigma", type=float, default=0.2, help="annualised volatility")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="optpricer", description="Educational options pricing library")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("price", help="price one option with one method")
    _contract_args(p)
    p.add_argument("--method", choices=METHODS, default="bs")
    p.add_argument("--n", type=int, help="tree steps")
    p.add_argument("--n-space", type=int, help="PDE space intervals")
    p.add_argument("--n-time", type=int, help="PDE time steps")
    p.add_argument("--n-paths", type=int, help="Monte Carlo paths")
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", help="price with every supported method")
    _contract_args(c)
    r = sub.add_parser("run-all", help="regenerate results/ (figures, tables, metadata)")
    r.add_argument("--mode", choices=("fast", "research"), default="fast")
    a = ap.parse_args(argv)

    if a.cmd == "run-all":
        from .experiments.run_all import main as run_main

        return run_main(["--mode", a.mode])
    try:
        opt = Option(a.kind, a.K, a.T, a.style)
        mkt = MarketData(a.S, a.r, a.q, a.sigma)
        if a.cmd == "compare":
            df = price_all(opt, mkt, settings={"mc": {"n_paths": 200_000, "seed": 0},
                                                     "explicit": {"n_space": 200, "n_time": 2000}})
            print(df[["method", "price", "std_error", "runtime_s", "status"]].to_string(index=False))
            return 0
        kw: dict = {}
        if a.method in ("crr", "crr_avg", "lr") and a.n:
            kw["n"] = a.n
        if a.method in ("explicit", "implicit", "cn"):
            if a.n_space:
                kw["n_space"] = a.n_space
            if a.n_time:
                kw["n_time"] = a.n_time
        if a.method == "mc":
            kw.update(n_paths=a.n_paths or 200_000, seed=a.seed)
        res = price(opt, mkt, a.method, **kw)
        print(json.dumps(res.to_dict(), indent=2, default=str))
        return 0
    except OptPricerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
