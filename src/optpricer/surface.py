"""Implied forwards, smiles, a total-variance surface and static-arbitrage diagnostics.

Scope and limits
----------------
* Forward and discount factor are estimated per expiry from **matched European**
  call/put mids via put-call parity ``C - P = DF (F - K)``. This is a weighted least-squares
  fit of ``C - P`` on ``K`` (slope ``-DF``, intercept ``DF F``) over the most liquid
  near-the-money pairs. The result is an *implied carry*, not an observed dividend
  forecast. For American quotes parity is only an inequality, so the estimator refuses
  them unless ``approximate=True``, and the output is then labelled.
* Smiles use log-forward moneyness ``k = ln(K / F)`` and out-of-the-money quotes (puts
  for ``K < F``, calls for ``K >= F``).
* :class:`TotalVarianceSurface` interpolates total variance ``w = sigma^2 T`` linearly in
  ``k`` within each slice and linearly in ``T`` at fixed ``k``. **Interpolation alone does
  not guarantee absence of arbitrage**; run :func:`arbitrage_diagnostics` on the quotes and
  check the interpolated surface separately if you use it.
* :func:`arbitrage_diagnostics` checks a *finite grid* of quotes. Passing all checks does
  not make a continuous surface arbitrage-free. Each violation is labelled ``mid`` (a
  mid-price inconsistency) or ``executable`` (tradeable at the quoted bid/ask, before
  fees, size and timing constraints).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .errors import InputError, UnsupportedMethodError
from .implied_vol import implied_vol


@dataclass
class ForwardEstimate:
    """Per-expiry parity-implied forward and discount factor."""

    expiry: str
    T: float
    F: float
    DF: float
    implied_rate: float
    implied_carry: float  # r - q implied by F/S
    implied_yield: float  # q = r - carry
    F_se: float           # WLS standard error (delta method), quote-noise only
    DF_se: float
    n_pairs: int
    residual_std: float
    max_abs_residual: float
    approximate: bool
    label: str

    def to_dict(self) -> dict:
        return asdict(self)


def _require_european(df: pd.DataFrame, approximate: bool, what: str) -> bool:
    styles = set(df.get("exercise_style", pd.Series(["european"])).astype(str).str.lower())
    if styles - {"european"}:
        if not approximate:
            raise UnsupportedMethodError(
                f"{what} requires European quotes; got exercise styles {sorted(styles)}. American prices "
                "violate exact parity (early exercise, discrete dividends). Pass approximate=True for a "
                "labelled descriptive approximation.")
        return True
    return False


def implied_forward(expiry_quotes: pd.DataFrame, S: float, n_near: int = 25,
                    approximate: bool = False) -> ForwardEstimate:
    """Estimate ``F`` and ``DF`` for one expiry from matched accepted call/put mids."""
    is_approx = _require_european(expiry_quotes, approximate, "implied_forward")
    q = expiry_quotes
    if "accepted" in q:
        q = q[q["accepted"]]
    finite = np.isfinite(q[["bid", "ask", "mid"]].to_numpy(float)).all(axis=1) & (q["bid"] <= q["ask"]).to_numpy()
    q = q[finite]
    calls = q[q["kind"] == "call"].set_index("strike")
    puts = q[q["kind"] == "put"].set_index("strike")
    common = calls.index.intersection(puts.index)
    if len(common) < 3:
        raise InputError(f"need >= 3 matched call/put strikes, got {len(common)}")
    near = np.asarray(sorted(common, key=lambda k: abs(math.log(k / S))))[:n_near]
    c, p = calls.loc[near], puts.loc[near]
    y = (c["mid"] - p["mid"]).to_numpy(float)
    K = np.asarray(near, float)
    spread = (c["ask"] - c["bid"]).to_numpy(float) + (p["ask"] - p["bid"]).to_numpy(float)
    w = 1.0 / np.maximum(spread, 1e-4) ** 2
    X = np.column_stack([np.ones_like(K), K])
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(X * W, y * W[:, 0], rcond=None)
    alpha, beta = coef
    DF = -beta
    if not (DF > 0):
        raise InputError(f"parity regression gave non-positive discount factor {DF:.4g}")
    F = alpha / DF
    T = float(q["T"].iloc[0])
    resid = y - X @ coef
    dof = max(len(K) - 2, 1)
    s2 = float((w * resid**2).sum() / dof)
    cov = s2 * np.linalg.inv((X * w[:, None]).T @ X)
    DF_se = math.sqrt(cov[1, 1])
    # F = alpha / DF with DF = -beta: gradient wrt (alpha, beta) = (1/DF, alpha/DF^2)
    grad = np.array([1.0 / DF, alpha / DF**2])
    F_se = math.sqrt(float(grad @ cov @ grad))
    r_imp = -math.log(DF) / T
    carry = math.log(F / S) / T
    label = "parity_implied" if not is_approx else "APPROXIMATE_from_american_quotes"
    return ForwardEstimate(str(q["expiry"].iloc[0]), T, float(F), float(DF), r_imp, carry, r_imp - carry,
                           F_se, DF_se, len(K), float(resid.std(ddof=2)) if len(K) > 2 else math.nan,
                           float(np.abs(resid).max()), is_approx, label)


def implied_forwards(chain: pd.DataFrame, S: float | None = None, approximate: bool = False,
                     n_near: int = 25) -> pd.DataFrame:
    """:func:`implied_forward` for every expiry. Returns one row per expiry."""
    rows = []
    for _exp, grp in chain.groupby("expiry", sort=True):
        S_ = float(grp["underlying_price"].iloc[0]) if S is None else S
        rows.append(implied_forward(grp, S_, n_near=n_near, approximate=approximate).to_dict())
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def build_smiles(chain: pd.DataFrame, forwards: pd.DataFrame, approximate: bool = False,
                 otm_only: bool = True) -> pd.DataFrame:
    """Implied vols (mid, bid, ask) of accepted quotes in ``k = ln(K/F)``.

    The IV inputs use the parity-implied rate and carry of each expiry, so ``S e^{(r-q)T} = F``.
    """
    is_approx = _require_european(chain, approximate, "build_smiles")
    style = "american" if is_approx else "european"
    q = chain[chain["accepted"]] if "accepted" in chain else chain
    out = []
    for _, fw in forwards.iterrows():
        g = q[q["expiry"] == fw["expiry"]]
        S = float(g["underlying_price"].iloc[0])
        T, F, r_imp, q_imp = fw["T"], fw["F"], fw["implied_rate"], fw["implied_yield"]
        for _, row in g.iterrows():
            K, kind = float(row["strike"]), row["kind"]
            if otm_only and ((kind == "call" and K < F) or (kind == "put" and K >= F)):
                continue
            kw = dict(approximate_american=is_approx)
            res = implied_vol(row["mid"], S, K, T, r_imp, q_imp, kind, style, **kw)
            ivb = implied_vol(row["bid"], S, K, T, r_imp, q_imp, kind, style, **kw) if row["bid"] > 0 else None
            iva = implied_vol(row["ask"], S, K, T, r_imp, q_imp, kind, style, **kw)
            out.append({
                "expiry": fw["expiry"], "T": T, "F": F, "DF": fw["DF"], "strike": K, "kind": kind,
                "k": math.log(K / F), "mid": row["mid"], "bid": row["bid"], "ask": row["ask"],
                "iv": res.iv, "iv_status": res.status, "iv_poorly_identified": res.poorly_identified,
                "iv_bid": ivb.iv if ivb is not None and ivb.converged else np.nan,
                "iv_ask": iva.iv if iva.converged else np.nan,
                "w": res.iv**2 * T if res.converged else np.nan,
                "true_iv": row.get("true_iv", np.nan),
                "label": res.label, "data_label": row.get("data_label", ""),
            })
    return pd.DataFrame(out)


class TotalVarianceSurface:
    """Piecewise-linear total-variance surface ``w(k, T)`` built from smile slices.

    Within a slice, ``w`` is interpolated linearly in ``k`` and held flat beyond the quoted
    range (a documented, crude extrapolation). Across maturities, ``w`` is linear in ``T`` at
    fixed ``k``, and before the first slice ``w`` is scaled proportionally to ``T``. If each
    slice is butterfly-free and ``w`` increases with ``T``, linear-in-T interpolation keeps
    calendar monotonicity, but slice-wise linear interpolation in ``k`` can still create
    butterfly issues. This is a diagnostic/visual tool, not an arbitrage-free fit.
    """

    def __init__(self, smiles: pd.DataFrame):
        s = smiles.dropna(subset=["w"]).sort_values(["T", "k"])
        self.slices = []
        for T, grp in s.groupby("T", sort=True):
            g = grp.drop_duplicates("k")
            if len(g) >= 2:
                self.slices.append((float(T), g["k"].to_numpy(float), g["w"].to_numpy(float)))  # type: ignore[arg-type]
        if not self.slices:
            raise InputError("no usable smile slices")
        self.T = np.array([t for t, _, _ in self.slices])

    def _slice_w(self, i: int, k: np.ndarray) -> np.ndarray:
        _, ks, ws = self.slices[i]
        return np.interp(k, ks, ws)

    def w(self, k, T: float) -> np.ndarray:
        """Total variance at log-moneyness ``k`` (array) and maturity ``T``."""
        k = np.asarray(k, float)
        if T <= self.T[0]:
            return self._slice_w(0, k) * (T / self.T[0])
        if T >= self.T[-1]:
            return self._slice_w(len(self.T) - 1, k)
        j = int(np.searchsorted(self.T, T))
        t0, t1 = self.T[j - 1], self.T[j]
        a = (T - t0) / (t1 - t0)
        return (1 - a) * self._slice_w(j - 1, k) + a * self._slice_w(j, k)

    def iv(self, k, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.w(k, T), 0.0) / T)


def _call_equivalent(chain: pd.DataFrame, fw: pd.Series) -> pd.DataFrame:
    """Call-equivalent mid/bid/ask per strike, using parity with the estimated F and DF.

    Puts map to calls via ``C = P + DF (F - K)``. When both exist, the tighter one is used.
    """
    g = chain[(chain["expiry"] == fw["expiry"])]
    if "accepted" in g:
        g = g[g["accepted"]]
    rows = []
    for K, grp in g.groupby("strike"):
        best: tuple | None = None
        for _, r_ in grp.iterrows():
            adj = 0.0 if r_["kind"] == "call" else fw["DF"] * (fw["F"] - K)
            cand = (r_["ask"] - r_["bid"], r_["mid"] + adj, r_["bid"] + adj, r_["ask"] + adj, r_["kind"])
            if best is None or cand[0] < best[0]:
                best = cand
        assert best is not None
        rows.append({"strike": float(K), "C_mid": best[1], "C_bid": best[2], "C_ask": best[3], "source": best[4]})
    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)


def arbitrage_diagnostics(chain: pd.DataFrame, forwards: pd.DataFrame, smiles: pd.DataFrame | None = None,
                          tol: float = 1e-10, approximate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Static no-arbitrage diagnostics on a finite quote grid.

    Per expiry, on call-equivalent prices ``C(K)`` (non-uniform strikes allowed):

    * ``bounds``: ``DF max(F - K, 0) <= C <= DF F``.
    * ``monotonicity``: ``C(K_{i+1}) <= C(K_i)``.
    * ``vertical_spread``: slope ``s_i = (C_{i+1} - C_i) / (K_{i+1} - K_i)`` within ``[-DF, 0]``.
    * ``convexity``: slopes non-decreasing (``s_i >= s_{i-1}``), i.e. non-negative butterflies with
      non-uniform weights ``w1 = (K3-K2)/(K3-K1)``, ``w3 = (K2-K1)/(K3-K1)``.

    Across expiries (when ``smiles`` is given): ``calendar``. At each ``k`` on a common grid
    inside both slices' quoted ranges, total variance must not fall from ``T_i`` to ``T_{i+1}``.
    This forward-normalised check is valid under deterministic proportional carry. Raw call
    prices are not compared across maturities.

    Executable labels: a monotonicity violation is executable if ``bid(K_{i+1}) > ask(K_i)``;
    a butterfly if ``w1 ask(K1) + w3 ask(K3) - bid(K2) < 0``; a vertical-slope violation
    beyond ``-DF`` if ``bid(K_i) - ask(K_{i+1}) > DF (K_{i+1} - K_i)``; a bound violation if the
    bid/ask quote lies entirely outside the bound; a calendar violation if the later expiry's
    ask-IV total variance is below the earlier bid-IV total variance (an IV-space proxy, not a
    priced trade).

    Returns ``(violations, summary)``.
    """
    _require_european(chain, approximate, "arbitrage_diagnostics")
    viol = []
    for _, fw in forwards.iterrows():
        ce = _call_equivalent(chain, fw)
        K = ce["strike"].to_numpy()
        Cm, Cb, Ca = ce["C_mid"].to_numpy(), ce["C_bid"].to_numpy(), ce["C_ask"].to_numpy()
        DF, F = fw["DF"], fw["F"]
        lo, hi = DF * np.maximum(F - K, 0.0), DF * F
        for i in range(len(K)):
            if Cm[i] < lo[i] - tol or Cm[i] > hi + tol:
                execu = Ca[i] < lo[i] - tol or Cb[i] > hi + tol
                viol.append(dict(expiry=fw["expiry"], check="bounds", strikes=f"{K[i]:g}",
                                 magnitude=float(max(lo[i] - Cm[i], Cm[i] - hi)), basis="mid",
                                 executable=bool(execu)))
        dK = np.diff(K)
        s = np.diff(Cm) / dK
        for i in range(len(s)):
            if Cm[i + 1] > Cm[i] + tol:
                viol.append(dict(expiry=fw["expiry"], check="monotonicity", strikes=f"{K[i]:g}/{K[i + 1]:g}",
                                 magnitude=float(Cm[i + 1] - Cm[i]), basis="mid",
                                 executable=bool(Cb[i + 1] > Ca[i] + tol)))
            if s[i] < -DF - tol:
                viol.append(dict(expiry=fw["expiry"], check="vertical_spread", strikes=f"{K[i]:g}/{K[i + 1]:g}",
                                 magnitude=float(-DF - s[i]), basis="mid",
                                 executable=bool(Cb[i] - Ca[i + 1] > DF * dK[i] + tol)))
        for i in range(1, len(s)):
            if s[i] < s[i - 1] - tol:
                K1, K2, K3 = K[i - 1], K[i], K[i + 1]
                w1, w3 = (K3 - K2) / (K3 - K1), (K2 - K1) / (K3 - K1)
                fly_mid = w1 * Cm[i - 1] + w3 * Cm[i + 1] - Cm[i]
                fly_exec = w1 * Ca[i - 1] + w3 * Ca[i + 1] - Cb[i]
                viol.append(dict(expiry=fw["expiry"], check="convexity", strikes=f"{K1:g}/{K2:g}/{K3:g}",
                                 magnitude=float(-fly_mid), basis="mid", executable=bool(fly_exec < -tol)))
    if smiles is not None and len(smiles):
        sm = smiles.dropna(subset=["w"]).sort_values(["T", "k"])
        Ts = sorted(sm["T"].unique())
        for t0, t1 in zip(Ts[:-1], Ts[1:]):
            a, b = sm[sm["T"] == t0], sm[sm["T"] == t1]
            kgrid = np.linspace(max(a["k"].min(), b["k"].min()), min(a["k"].max(), b["k"].max()), 41)
            if kgrid[0] >= kgrid[-1]:
                continue
            w0 = np.interp(kgrid, a["k"], a["w"])
            w1_ = np.interp(kgrid, b["k"], b["w"])
            wb0 = np.interp(kgrid, a["k"], (a["iv_bid"].fillna(0.0) ** 2) * t0)
            wa1 = np.interp(kgrid, b["k"], (b["iv_ask"].fillna(np.inf) ** 2) * t1)
            bad = w1_ < w0 - tol
            for kk, d0, d1, e0, e1 in zip(kgrid[bad], w0[bad], w1_[bad], wb0[bad], wa1[bad]):
                viol.append(dict(expiry=f"{a['expiry'].iloc[0]}->{b['expiry'].iloc[0]}", check="calendar",
                                 strikes=f"k={kk:.4f}", magnitude=float(d0 - d1), basis="mid_iv",
                                 executable=bool(e1 < e0 - tol)))
    vdf = pd.DataFrame(viol, columns=["expiry", "check", "strikes", "magnitude", "basis", "executable"])
    checks = ["bounds", "monotonicity", "vertical_spread", "convexity", "calendar"]
    summary = pd.DataFrame([{
        "check": c, "violations_mid": int((vdf["check"] == c).sum()),
        "violations_executable": int(((vdf["check"] == c) & vdf["executable"]).sum()),
        "max_magnitude": float(vdf.loc[vdf["check"] == c, "magnitude"].max()) if (vdf["check"] == c).any() else 0.0,
    } for c in checks])
    return vdf, summary


def risk_neutral_density(strikes, calls, DF: float) -> pd.DataFrame:
    """Breeden-Litzenberger density ``q(K) = (1/DF) d2C/dK2`` on a non-uniform strike grid.

    Uses the second divided difference ``2 (s_i - s_{i-1}) / (K_{i+1} - K_{i-1})``. Finite
    differences amplify quote noise and depend strongly on interpolation. Treat the output
    as a diagnostic: negative values signal convexity violations or noise.
    """
    K = np.asarray(strikes, float)
    C = np.asarray(calls, float)
    order = np.argsort(K)
    K, C = K[order], C[order]
    s = np.diff(C) / np.diff(K)
    d2 = 2.0 * np.diff(s) / (K[2:] - K[:-2])
    return pd.DataFrame({"strike": K[1:-1], "density": d2 / DF})
