"""Option-chain data: synthetic European chains, snapshots, cleaning, optional download.

Data sources
------------
A. :func:`synthetic_european_chain` builds an offline, seeded, **synthetic** European chain.
   Prices come from BSM on a known SSVI total-variance surface (Gatheral & Jacquier 2014),
   with power-law ``phi(theta) = eta * theta^-gamma``, ``gamma = 1/2`` and
   ``eta (1 + |rho|) <= 2``. This satisfies their sufficient conditions for no static
   arbitrage in the *continuous* surface. Mids then get seeded noise and tick rounding,
   and a few deliberately bad quotes are injected (labelled ``injected_defect``). Every
   row carries ``data_label = "SYNTHETIC"``.
B. :func:`load_snapshot` / :func:`save_snapshot` read and write CSV + JSON metadata for
   user-provided or downloaded chains.
C. :func:`fetch_yfinance_chain` (optional, network) downloads a **delayed** public chain.
   It runs only when called explicitly. US equity/ETF options such as SPY are
   **American-style**; the snapshot is labelled as such, and downstream European tools
   refuse it unless the caller explicitly asks for a labelled approximation.

Cleaning policy (:func:`clean_chain`)
-------------------------------------
Every rule tags rows with a reason; nothing is silently dropped. The result has
``accepted`` (bool) and ``reasons`` (``;``-separated) columns. Rules:
``missing_quote``, ``negative_quote``, ``crossed_market`` (bid > ask), ``zero_bid`` (see
policy), ``wide_spread`` (ask - bid > max_rel_spread * mid and > min_abs_spread),
``stale_quote`` (quote older than ``stale_seconds`` relative to the spot timestamp, when
timestamps exist), ``short_expiry`` (< ``min_days`` calendar days), ``bound_violation``
(mid outside the model-free bounds for the row's exercise style), ``low_liquidity``
(volume or open interest below thresholds, when present).
Zero-bid policy: ``"reject"`` (default) excludes zero-bid quotes from IV work because
their mid is not a reliable price. ``"flag"`` keeps them but tags the reason.
Zero-bid contracts are **not** assumed to be worthless.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bsm
from .conventions import DAYS_PER_YEAR
from .errors import InputError

CHAIN_COLUMNS = ["expiry", "T", "strike", "kind", "bid", "ask", "mid", "volume", "open_interest",
                 "quote_time", "underlying_price", "underlying_time", "exercise_style", "source",
                 "data_label"]


# --------------------------------------------------------------------------- synthetic
@dataclass(frozen=True)
class SSVIParams:
    """SSVI surface ``w(k, theta) = theta/2 (1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2))``."""

    rho: float = -0.6
    eta: float = 1.1
    gamma: float = 0.5
    atm_vols: tuple[float, ...] = (0.22, 0.21, 0.20, 0.195, 0.19, 0.185)

    def __post_init__(self) -> None:
        if not -1 < self.rho < 1:
            raise InputError("SSVI rho must lie in (-1, 1)")
        if self.gamma != 0.5 or self.eta * (1 + abs(self.rho)) > 2 + 1e-12:
            raise InputError("this generator requires gamma = 0.5 and eta (1 + |rho|) <= 2 "
                             "(Gatheral-Jacquier sufficient no-butterfly condition)")


def ssvi_total_variance(k, theta, p: SSVIParams) -> np.ndarray:
    """SSVI total implied variance at log-forward moneyness ``k`` and ATM total variance ``theta``."""
    k = np.asarray(k, float)
    theta = np.asarray(theta, float)
    phi = p.eta / np.power(theta, p.gamma)
    return 0.5 * theta * (1 + p.rho * phi * k + np.sqrt((phi * k + p.rho) ** 2 + 1 - p.rho**2))


def synthetic_european_chain(S: float = 100.0, r: float = 0.03, q: float = 0.015,
                             expiry_days: tuple[int, ...] = (30, 60, 91, 182, 365, 730),
                             params: SSVIParams | None = None, seed: int = 7,
                             valuation_time: dt.datetime | None = None,
                             inject_defects: bool = True, tick: float = 0.01) -> tuple[pd.DataFrame, dict]:
    """Seeded synthetic European chain with a known SSVI smile.

    Strikes are non-uniform: 1-point spacing within +-10% of spot, 2.5 out to +-25%, and 5
    beyond. Mids carry +-25% of half-spread noise and are rounded to the tick; quotes are
    widened to cover the true price. ATM total variance increases with maturity, so the
    continuous surface has no calendar arbitrage.

    Returns ``(chain, truth)``. ``truth`` holds the generating parameters and the true
    forwards/discount factors used for validation.
    """
    p = params or SSVIParams()
    if len(p.atm_vols) < len(expiry_days):
        raise InputError("need one ATM vol per expiry")
    rng = np.random.default_rng(seed)
    vt = valuation_time or dt.datetime(2026, 1, 15, 15, 30, 0)
    rows = []
    expiries: list[dict] = []
    truth = {"S": S, "r": r, "q": q, "valuation_time": vt.isoformat(), "ssvi": dict(p.__dict__), "expiries": expiries}
    theta_prev = 0.0
    for i, days in enumerate(expiry_days):
        T = days / DAYS_PER_YEAR
        theta = p.atm_vols[i] ** 2 * T
        if theta <= theta_prev:
            raise InputError("ATM total variance must increase with maturity")
        theta_prev = theta
        F = S * math.exp((r - q) * T)
        DF = math.exp(-r * T)
        strikes = np.unique(np.round(np.concatenate([
            np.arange(0.60 * S, 0.75 * S, 5.0), np.arange(0.75 * S, 0.90 * S, 2.5),
            np.arange(0.90 * S, 1.10 * S + 1e-9, 1.0), np.arange(1.125 * S, 1.25 * S + 1e-9, 2.5),
            np.arange(1.30 * S, 1.50 * S + 1e-9, 5.0)]), 6))
        k = np.log(strikes / F)
        w = ssvi_total_variance(k, theta, p)
        iv = np.sqrt(w / T)
        expiry = (vt + dt.timedelta(days=days)).date().isoformat()
        expiries.append({"expiry": expiry, "T": T, "F": F, "DF": DF, "theta_atm": theta})
        for kind in ("call", "put"):
            true_px = np.asarray(bsm.price(S, strikes, T, r, q, iv, kind))
            half = np.maximum(0.5 * tick, 0.01 + 0.015 * true_px)
            mid = true_px + rng.uniform(-0.25, 0.25, size=strikes.size) * half
            bid = np.maximum(np.floor((mid - half) / tick) * tick, 0.0)
            ask = np.ceil((mid + half) / tick) * tick
            bid = np.minimum(bid, np.floor(true_px / tick) * tick)  # quotes bracket the true price
            ask = np.maximum(ask, np.ceil(true_px / tick) * tick)
            vol = rng.integers(0, 2000, size=strikes.size)
            oi = rng.integers(10, 20000, size=strikes.size)
            age = rng.uniform(0, 120, size=strikes.size)  # seconds before the spot timestamp
            for j, K in enumerate(strikes):
                rows.append({
                    "expiry": expiry, "T": T, "strike": float(K), "kind": kind,
                    "bid": round(float(bid[j]), 10), "ask": round(float(ask[j]), 10),
                    "volume": int(vol[j]), "open_interest": int(oi[j]),
                    "quote_time": (vt - dt.timedelta(seconds=float(age[j]))).isoformat(),
                    "underlying_price": S, "underlying_time": vt.isoformat(),
                    "exercise_style": "european", "source": "synthetic_ssvi_generator",
                    "data_label": "SYNTHETIC", "true_iv": float(iv[j]), "true_price": float(true_px[j]),
                    "injected_defect": "",
                })
    df = pd.DataFrame(rows)
    if inject_defects:
        df = _inject_defects(df, rng, vt)
    df["mid"] = np.where(df["bid"].notna() & df["ask"].notna(), 0.5 * (df["bid"] + df["ask"]), np.nan)
    return df.reset_index(drop=True), truth


def _inject_defects(df: pd.DataFrame, rng: np.random.Generator, vt: dt.datetime) -> pd.DataFrame:
    """Inject a documented handful of bad quotes so the cleaning rules are exercised."""
    df = df.copy()
    idx = rng.choice(df.index.to_numpy(), size=12, replace=False)
    kinds = ["crossed_market", "crossed_market", "negative_quote", "missing_quote", "wide_spread",
             "wide_spread", "stale_quote", "stale_quote", "zero_bid", "zero_bid", "bound_violation",
             "bound_violation"]
    for i, kind in zip(idx, kinds):
        b, a = float(df.at[i, "bid"]), float(df.at[i, "ask"])  # type: ignore[arg-type]
        if kind == "crossed_market":
            df.at[i, "bid"], df.at[i, "ask"] = a + 0.05, b
        elif kind == "negative_quote":
            df.at[i, "bid"] = -0.05
        elif kind == "missing_quote":
            df.at[i, "ask"] = np.nan
        elif kind == "wide_spread":
            df.at[i, "bid"], df.at[i, "ask"] = max(0.0, round(0.3 * b, 2)), round(2.5 * a + 0.5, 2)
        elif kind == "stale_quote":
            df.at[i, "quote_time"] = (vt - dt.timedelta(hours=3)).isoformat()
        elif kind == "zero_bid":
            df.at[i, "bid"] = 0.0
        elif kind == "bound_violation":
            S, K = float(df.at[i, "underlying_price"]), float(df.at[i, "strike"])  # type: ignore[arg-type]
            ub = S if df.at[i, "kind"] == "call" else K
            df.at[i, "bid"], df.at[i, "ask"] = round(ub + 1.0, 2), round(ub + 1.2, 2)
        df.at[i, "injected_defect"] = kind
    return df


# --------------------------------------------------------------------------- cleaning
@dataclass
class CleaningPolicy:
    """Thresholds for :func:`clean_chain` (see module docstring)."""

    max_rel_spread: float = 0.5
    min_abs_spread: float = 0.05
    min_days: float = 7.0
    stale_seconds: float = 900.0
    zero_bid: str = "reject"  # or "flag"
    min_volume: int | None = None
    min_open_interest: int | None = None
    r: float | None = None  # needed for the bound check; None skips it
    q: float | None = None
    extra: dict = field(default_factory=dict)


def clean_chain(df: pd.DataFrame, policy: CleaningPolicy | None = None) -> pd.DataFrame:
    """Tag each quote with rejection reasons; returns a copy with ``accepted`` and ``reasons``.

    Rows are never dropped; filter on ``accepted`` to get the clean subset.
    """
    pol = policy or CleaningPolicy()
    if pol.zero_bid not in ("reject", "flag"):
        raise InputError("zero_bid policy must be 'reject' or 'flag'")
    out = df.copy()
    n = len(out)
    reasons: list[list[str]] = [[] for _ in range(n)]
    flags: list[list[str]] = [[] for _ in range(n)]
    bid = out["bid"].to_numpy(float)
    ask = out["ask"].to_numpy(float)
    mid = 0.5 * (bid + ask)
    S = out["underlying_price"].to_numpy(float)
    K = out["strike"].to_numpy(float)
    T = out["T"].to_numpy(float)
    kinds = out["kind"].astype(str).str.lower().to_numpy()
    styles = out.get("exercise_style", pd.Series(["european"] * n)).astype(str).str.lower().to_numpy()
    qt = pd.to_datetime(out["quote_time"], errors="coerce", utc=True, format="ISO8601") if "quote_time" in out else None
    ut = (pd.to_datetime(out["underlying_time"], errors="coerce", utc=True, format="ISO8601")
          if "underlying_time" in out else None)

    for i in range(n):
        if not (np.isfinite(bid[i]) and np.isfinite(ask[i])):
            reasons[i].append("missing_quote")
            continue
        if bid[i] < 0 or ask[i] < 0:
            reasons[i].append("negative_quote")
            continue
        if bid[i] > ask[i]:
            reasons[i].append("crossed_market")
            continue
        if bid[i] == 0:
            (reasons if pol.zero_bid == "reject" else flags)[i].append("zero_bid")
        spread = ask[i] - bid[i]
        if mid[i] > 0 and spread > pol.max_rel_spread * mid[i] and spread > pol.min_abs_spread:
            reasons[i].append("wide_spread")
        if T[i] * DAYS_PER_YEAR < pol.min_days:
            reasons[i].append("short_expiry")
        if qt is not None and ut is not None and pd.notna(qt.iloc[i]) and pd.notna(ut.iloc[i]):
            age = (ut.iloc[i] - qt.iloc[i]).total_seconds()
            if age > pol.stale_seconds:
                reasons[i].append("stale_quote")
        if pol.r is not None and pol.q is not None:
            if styles[i] == "american":
                lo, hi = bsm.american_bounds(S[i], K[i], T[i], pol.r, pol.q, kinds[i])
            else:
                lo, hi = bsm.european_bounds(S[i], K[i], T[i], pol.r, pol.q, kinds[i])
            # executable-side test: ask below the lower bound or bid above the upper bound
            if ask[i] < float(lo) - 1e-9 or bid[i] > float(hi) + 1e-9:
                reasons[i].append("bound_violation")
        if pol.min_volume is not None and "volume" in out and not (out["volume"].iloc[i] >= pol.min_volume):
            reasons[i].append("low_liquidity")
        if (pol.min_open_interest is not None and "open_interest" in out
                and not (out["open_interest"].iloc[i] >= pol.min_open_interest)):
            reasons[i].append("low_liquidity")
    out["mid"] = mid
    out["accepted"] = [len(r) == 0 for r in reasons]
    out["reasons"] = [";".join(r) for r in reasons]
    out["flags"] = [";".join(f) for f in flags]
    return out


def cleaning_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Counts of each rejection reason (a row may have several)."""
    rs = cleaned.loc[~cleaned["accepted"], "reasons"].str.split(";").explode()
    counts = rs.value_counts().rename_axis("reason").reset_index(name="count")
    total = pd.DataFrame([{"reason": "TOTAL_REJECTED_ROWS", "count": int((~cleaned["accepted"]).sum())},
                          {"reason": "TOTAL_ACCEPTED_ROWS", "count": int(cleaned["accepted"].sum())}])
    return pd.concat([counts, total], ignore_index=True)


# --------------------------------------------------------------------------- snapshots
def save_snapshot(df: pd.DataFrame, meta: dict, directory: str | Path, name: str) -> tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.meta.json`` (metadata: source, times, style, ...)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = d / f"{name}.csv", d / f"{name}.meta.json"
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return csv_path, meta_path


def load_snapshot(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load a snapshot CSV and its metadata JSON (if present next to it)."""
    p = Path(csv_path)
    df = pd.read_csv(p)
    meta_path = p.with_name(p.stem + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    missing = [c for c in ("T", "strike", "kind", "bid", "ask", "underlying_price") if c not in df]
    if missing:
        raise InputError(f"snapshot is missing required columns: {missing}")
    if "exercise_style" not in df:
        raise InputError("snapshot must state exercise_style per row (european/american)")
    return df, meta


def fetch_yfinance_chain(ticker: str = "SPY", max_expiries: int = 6, min_days: int = 7) -> tuple[pd.DataFrame, dict]:
    """Download a **delayed** public option chain with yfinance (network access needed).

    Rows are labelled ``exercise_style = "american"`` (US-listed equity/ETF options) and
    ``data_label = "MARKET_DELAYED"``. The metadata records the retrieval time and the
    caveat that quotes and the spot price are not synchronised.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is not installed; install the [market] extra") from exc
    retrieved = dt.datetime.now(dt.UTC)
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d", interval="1m")
    if hist.empty:
        raise RuntimeError(f"no recent price history returned for {ticker}")
    spot = float(hist["Close"].iloc[-1])
    spot_time = hist.index[-1].to_pydatetime().astimezone(dt.UTC)
    rows = []
    used = []
    for exp in tk.options:
        exp_dt = dt.datetime.fromisoformat(exp).replace(hour=20, tzinfo=dt.UTC)  # 16:00 New York ~ 20/21 UTC
        T = (exp_dt - retrieved).total_seconds() / 86400 / DAYS_PER_YEAR
        if T * DAYS_PER_YEAR < min_days:
            continue
        ch = tk.option_chain(exp)
        for kind, frame in (("call", ch.calls), ("put", ch.puts)):
            for _, rw in frame.iterrows():
                lt = rw.get("lastTradeDate")
                rows.append({
                    "expiry": exp, "T": T, "strike": float(rw["strike"]), "kind": kind,
                    "bid": float(rw["bid"]) if pd.notna(rw["bid"]) else np.nan,
                    "ask": float(rw["ask"]) if pd.notna(rw["ask"]) else np.nan,
                    "volume": float(rw["volume"]) if pd.notna(rw.get("volume")) else np.nan,
                    "open_interest": float(rw["openInterest"]) if pd.notna(rw.get("openInterest")) else np.nan,
                    "quote_time": None,  # yfinance exposes no per-quote timestamp
                    "last_trade_time": pd.Timestamp(lt).isoformat() if pd.notna(lt) else None,
                    "underlying_price": spot, "underlying_time": spot_time.isoformat(),
                    "exercise_style": "american", "source": f"yfinance:{ticker}",
                    "data_label": "MARKET_DELAYED",
                })
        used.append(exp)
        if len(used) >= max_expiries:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["mid"] = 0.5 * (df["bid"] + df["ask"])
    meta = {
        "source": f"yfinance public option chain for {ticker}",
        "retrieved_utc": retrieved.isoformat(),
        "underlying_spot": spot, "underlying_spot_time_utc": spot_time.isoformat(),
        "expiries": used, "currency": "USD", "exercise_style": "american",
        "settlement": "physical delivery of shares (US equity/ETF options)",
        "expiry_time_assumption": "20:00 UTC on expiry date (approx. 16:00 New York; ignores DST)",
        "quote_timestamp_note": "yfinance gives lastTradeDate per contract, not a quote timestamp; bid/ask "
                                "and spot are delayed and not synchronised. 'quote_time' is left empty; "
                                "staleness can only be judged from last_trade_time.",
        "label": "MARKET_DELAYED - not live, not synchronised",
        "rate_assumption": "none stored; supply r explicitly downstream",
    }
    return df, meta
