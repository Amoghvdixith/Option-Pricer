"""OPTIONAL: download a delayed public option chain with yfinance and save it to data/snapshots.

Needs network access and the [market] extra. Run only on purpose:

    python scripts/fetch_snapshot.py --ticker SPY

The snapshot is labelled MARKET_DELAYED and exercise_style=american (US-listed equity/ETF
options). Downstream code treats its BSM implied vols as *descriptive approximations*
and never uses it to infer an exact forward or to certify European no-arbitrage.
"""

import argparse
import datetime as dt
import sys

from optpricer.market import fetch_yfinance_chain, save_snapshot
from optpricer.paths import data_dir, load_config


def main() -> int:
    cfg = load_config("market_data")["optional_download"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=cfg["ticker"])
    ap.add_argument("--max-expiries", type=int, default=cfg["max_expiries"])
    a = ap.parse_args()
    try:
        df, meta = fetch_yfinance_chain(a.ticker, a.max_expiries)
    except Exception as exc:  # report the exact blocker; nothing is fabricated
        print(f"DOWNLOAD FAILED: {type(exc).__name__}: {exc}")
        return 1
    if df.empty:
        print("DOWNLOAD RETURNED NO ROWS; nothing saved")
        return 1
    meta["rate_assumption_for_descriptive_iv"] = cfg["rate_assumption_for_descriptive_iv"]
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    csv, js = save_snapshot(df, meta, data_dir("snapshots"), f"{a.ticker}_{stamp}")
    zb = float((df["bid"].fillna(0) == 0).mean())
    if zb > 0.5:
        print(f"WARNING: {zb:.0%} of contracts have zero bid (market closed or stale feed?); "
              "quotes are unlikely to be usable for implied volatility")
    print(f"saved {len(df)} rows to {csv.name} (+ {js.name}); label MARKET_DELAYED, exercise style american")
    return 0


if __name__ == "__main__":
    sys.exit(main())
