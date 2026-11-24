r"""Self-financing discrete delta hedging of a short European option.

Accounting (all amounts in currency; time in ACT/365F years)
------------------------------------------------------------
At ``t_0 = 0``: sell the option for the premium ``V_0 = BSM(sigma_implied)``, buy ``Delta_0`` shares
(computed with ``sigma_hedge``), pay the proportional cost ``c |Delta_0| S_0``, and put the
residual ``V_0 - Delta_0 S_0 - cost`` in the cash account (negative = borrowing).

Over each interval ``[t_k, t_{k+1}]``:
1. cash accrues interest: ``cash *= exp(r dt)``;
2. the shares held earn the dividend yield ``q``. The continuous dividend
   ``q Delta_k \int S dt`` is approximated by the trapezoid ``q Delta_k dt (S_k + S_{k+1})/2``.
   This is the only approximation in the ledger, and it is zero when ``q = 0`` (the default);
3. at ``t_{k+1} < T`` the hedge is reset to ``Delta(S_{k+1}, T - t_{k+1}; sigma_hedge)``. Trades are
   paid from cash, plus the cost ``c |trade| S``.
At ``T``: pay the option payoff, sell the remaining shares (with cost), and the remaining cash
is the terminal net P&L (in time-T currency). No external capital is injected after t_0,
so the strategy is self-financing.

Simulation (physical measure)
-----------------------------
``dS/S = (mu - q) dt + sigma_realised dW``, where ``mu`` is the total expected return
including dividends. Paths are sampled **exactly** at the union of all rebalance grids
(e.g. 4, 21, 252 and 1000 per year). Every frequency therefore hedges the *same* paths
(common random numbers), and differences between frequencies are not sampling noise
from different paths.

Interpretation
--------------
For small intervals the hedged short-option P&L is approximately
``sum_k 1/2 Gamma_k S_k^2 (sigma_implied^2 dt - (dS/S)^2)`` (gamma-weighted variance
difference), when the hedge uses the implied volatility. This is an approximation used for
explanation (see :func:`gamma_pnl_approximation`). It is not an exact pathwise identity and
does not guarantee any single path's outcome. No closed-form hedging-error formula is
hard-coded; error scaling with rebalance frequency is measured empirically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import bsm
from .errors import InputError


@dataclass(frozen=True)
class HedgeSetup:
    """Parameters of a delta-hedging experiment (short one European option)."""

    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0
    r: float = 0.03
    q: float = 0.0
    sigma_implied: float = 0.20   # premium / marking volatility
    sigma_hedge: float | None = None  # volatility used for delta (defaults to sigma_implied)
    sigma_realised: float = 0.20  # simulation volatility
    mu: float = 0.07              # physical total expected return (incl. dividends)
    cost_rate: float = 0.0        # proportional cost per unit of traded notional
    kind: str = "call"

    def __post_init__(self) -> None:
        for name in ("S0", "K", "T", "sigma_implied", "sigma_realised"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0):
                raise InputError(f"{name} must be positive and finite")
        if self.sigma_hedge is not None and not self.sigma_hedge > 0:
            raise InputError("sigma_hedge must be positive")
        if self.cost_rate < 0:
            raise InputError("cost_rate must be non-negative")
        if self.kind not in ("call", "put"):
            raise InputError("kind must be 'call' or 'put'")

    @property
    def hedge_vol(self) -> float:
        return self.sigma_hedge if self.sigma_hedge is not None else self.sigma_implied

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hedge_vol_used"] = self.hedge_vol
        return d


def rebalance_grid(T: float, per_year: float) -> np.ndarray:
    """Rebalance dates ``0, 1/f, 2/f, ...`` capped at ``T`` (last interval may be shorter)."""
    if per_year <= 0:
        raise InputError("rebalance frequency must be positive")
    n = int(math.ceil(per_year * T - 1e-9))
    return np.minimum(np.arange(n + 1) / per_year, T)


def union_grid(T: float, freqs) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Union of rebalance grids and, per frequency, indices into the union."""
    grids = {f: rebalance_grid(T, f) for f in freqs}
    allt = np.unique(np.round(np.concatenate(list(grids.values())), 12))
    idx = {f: np.searchsorted(allt, np.round(g, 12)) for f, g in grids.items()}
    return allt, idx


def simulate_paths(setup: HedgeSetup, times: np.ndarray, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Exact GBM under P at the given times; shape ``(n_paths, len(times))``."""
    dt = np.diff(times)
    z = rng.standard_normal((n_paths, dt.size))
    sig = setup.sigma_realised
    inc = (setup.mu - setup.q - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * z
    logS = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1)
    return setup.S0 * np.exp(logS)


def _delta(setup: HedgeSetup, S, tau):
    return np.asarray(bsm.delta(S, setup.K, tau, setup.r, setup.q, setup.hedge_vol, setup.kind))


def hedge_paths(setup: HedgeSetup, times: np.ndarray, S: np.ndarray, ledger_path: int | None = None) -> dict:
    """Run the self-financing hedge on given paths (``S`` shape ``(n_paths, len(times))``).

    Returns per-path arrays ``pnl, costs, turnover_shares, turnover_notional`` and, if
    ``ledger_path`` is given, a step-by-step audit ``ledger`` DataFrame for that path.
    """
    times = np.asarray(times, float)
    if times[0] != 0 or abs(times[-1] - setup.T) > 1e-12 or np.any(np.diff(times) <= 0):
        raise InputError("times must start at 0, end at T and increase strictly")
    K, T, r, q, c = setup.K, setup.T, setup.r, setup.q, setup.cost_rate
    is_call = setup.kind == "call"
    n_paths, m1 = S.shape
    V0 = float(bsm.price(setup.S0, K, T, r, q, setup.sigma_implied, setup.kind))
    S0 = S[:, 0]
    pos = _delta(setup, S0, T)
    cost = c * np.abs(pos) * S0
    cash = V0 - pos * S0 - cost
    costs = cost.copy()
    turn_sh = np.abs(pos).copy()
    turn_not = np.abs(pos) * S0
    ledger = []
    if ledger_path is not None:
        p = ledger_path
        mark = V0
        ledger.append(dict(t=0.0, S=S0[p], premium=V0, shares_traded=pos[p], trade_cash=-pos[p] * S0[p],
                           cost=-cost[p], interest=0.0, dividend=0.0, payoff=0.0, liquidation=0.0,
                           delta_held=pos[p], cash=cash[p], option_mark=mark,
                           portfolio_value=cash[p] + pos[p] * S0[p] - mark))
    for k in range(m1 - 1):
        dt = times[k + 1] - times[k]
        Sk, Sn = S[:, k], S[:, k + 1]
        interest = cash * math.expm1(r * dt)
        cash = cash + interest
        div = pos * q * dt * 0.5 * (Sk + Sn)
        cash = cash + div
        last = k + 1 == m1 - 1
        if not last:
            new = _delta(setup, Sn, T - times[k + 1])
            trade = new - pos
            tc = c * np.abs(trade) * Sn
            cash = cash - trade * Sn - tc
            costs += tc
            turn_sh += np.abs(trade)
            turn_not += np.abs(trade) * Sn
            pos = new
            payoff = np.zeros_like(Sn)
            liq = np.zeros_like(Sn)
            mark_arr = None
        else:
            payoff = np.maximum(Sn - K, 0.0) if is_call else np.maximum(K - Sn, 0.0)
            cash = cash - payoff
            tc = c * np.abs(pos) * Sn
            liq = pos * Sn
            cash = cash + liq - tc
            costs += tc
            turn_sh += np.abs(pos)
            turn_not += np.abs(pos) * Sn
            trade = -pos
            pos = np.zeros_like(pos)
            mark_arr = payoff
        if ledger_path is not None:
            p = ledger_path
            if last:
                mark = 0.0  # settled
                ledger.append(dict(t=times[k + 1], S=Sn[p], premium=0.0, shares_traded=trade[p], trade_cash=0.0,
                                   cost=-tc[p], interest=interest[p], dividend=div[p], payoff=-payoff[p],
                                   liquidation=liq[p], delta_held=0.0, cash=cash[p], option_mark=mark,
                                   portfolio_value=cash[p]))
            else:
                mark = float(bsm.price(Sn[p], K, T - times[k + 1], r, q, setup.sigma_implied, setup.kind))
                ledger.append(dict(t=times[k + 1], S=Sn[p], premium=0.0, shares_traded=trade[p],
                                   trade_cash=-trade[p] * Sn[p], cost=-tc[p], interest=interest[p],
                                   dividend=div[p], payoff=0.0, liquidation=0.0, delta_held=pos[p], cash=cash[p],
                                   option_mark=mark, portfolio_value=cash[p] + pos[p] * Sn[p] - mark))
        del mark_arr
    out = {"pnl": cash, "costs": costs, "turnover_shares": turn_sh, "turnover_notional": turn_not,
           "premium": V0}
    if ledger_path is not None:
        out["ledger"] = pd.DataFrame(ledger)
    return out


def reconcile_ledger(ledger: pd.DataFrame, r: float, T: float) -> dict:
    """Check that the external cash flows, carried forward to T at rate r, equal final cash.

    External flows are premium, trade cash, costs, dividends, payoff and liquidation.
    Interest is the carry, not a flow. Returns the two totals and their difference.
    """
    flows = ledger[["premium", "trade_cash", "cost", "dividend", "payoff", "liquidation"]].sum(axis=1)
    fv = float((flows * np.exp(r * (T - ledger["t"]))).sum())
    final = float(ledger["cash"].iloc[-1])
    return {"future_value_of_flows": fv, "final_cash": final, "difference": final - fv}


def gamma_pnl_approximation(setup: HedgeSetup, times: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Per-path ``sum_k 1/2 Gamma_k S_k^2 (sigma_implied^2 dt_k - (ln S_{k+1}/S_k)^2) e^{r(T - t_{k+1})}``.

    This is the textbook gamma-weighted variance approximation for a short option hedged at the
    implied volatility, carried to T. It is a diagnostic approximation, not an identity.
    """
    out = np.zeros(S.shape[0])
    for k in range(S.shape[1] - 1):
        dt = times[k + 1] - times[k]
        tau = setup.T - times[k]
        g = np.asarray(bsm.gamma(S[:, k], setup.K, tau, setup.r, setup.q, setup.sigma_implied, setup.kind))
        ret2 = np.log(S[:, k + 1] / S[:, k]) ** 2
        carry = math.exp(setup.r * (setup.T - times[k + 1]))
        out += 0.5 * g * S[:, k] ** 2 * (setup.sigma_implied**2 * dt - ret2) * carry
    return out


def summarise(pnl: np.ndarray, costs: np.ndarray, turnover: np.ndarray) -> dict:
    """Mean, SD, SE of the mean, quantiles, mean cost and turnover across paths."""
    n = pnl.size
    q = np.quantile(pnl, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {"n_paths": n, "mean_pnl": float(pnl.mean()), "std_pnl": float(pnl.std(ddof=1)),
            "se_mean_pnl": float(pnl.std(ddof=1) / math.sqrt(n)),
            "q01": q[0], "q05": q[1], "median": q[2], "q95": q[3], "q99": q[4],
            "mean_cost": float(costs.mean()), "mean_turnover_notional": float(turnover.mean())}


def frequency_study(setup: HedgeSetup, freqs=(4, 21, 252, 1000), n_paths: int = 20_000, seed: int = 42,
                    keep_pnl: bool = True) -> tuple[pd.DataFrame, dict]:
    """Hedge the same simulated paths at several rebalance frequencies (per year)."""
    rng = np.random.default_rng(seed)
    allt, idx = union_grid(setup.T, freqs)
    S = simulate_paths(setup, allt, n_paths, rng)
    rows, pnls = [], {}
    for f in freqs:
        ii = idx[f]
        res = hedge_paths(setup, allt[ii], S[:, ii])
        row = {"freq_per_year": f, "n_rebalances": len(ii) - 1, "premium": res["premium"]}
        row.update(summarise(res["pnl"], res["costs"], res["turnover_notional"]))
        rows.append(row)
        if keep_pnl:
            pnls[f] = res["pnl"]
    return pd.DataFrame(rows), pnls
