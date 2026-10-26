"""Tables of the report."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import CRITICAL_T
from hedging import ESTIMATORS, RULES

BASE = "Equal"


def nw_t(x) -> float:
    """Newey-West t of mean(x), Bartlett lag floor(4 (T/100)^(2/9)), no small-sample correction."""
    x = np.asarray(x, dtype=float)
    lags = int(4 * (len(x) / 100) ** (2 / 9))
    fit = sm.OLS(x, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return fit.tvalues[0]


def volatility(X: pd.DataFrame) -> pd.DataFrame:
    """PC1 daily standard deviation (bp) and variance share by half-year."""
    rows = {}
    for (year, half), g in X.groupby([X.index.year, (X.index.month > 6) + 1]):
        ev = np.linalg.eigvalsh(np.cov(g.T, bias=True))
        rows[f"{year} H{half}"] = {"PC1 sd": np.sqrt(ev[-1]),
                                   "PC1 share %": 100 * ev[-1] / ev.sum()}
    return pd.DataFrame(rows).T


def _squares(oos: pd.DataFrame) -> pd.DataFrame:
    """Squared P&L of every hedged position, one column per (rule, k, est, hedge)."""
    h = oos[oos.rule != "None"]
    return h.assign(sq=h.pnl ** 2).pivot_table(index="date", columns=["rule", "k", "est", "hedge"],
                                               values="sq")


def residual_risk(oos: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample residual risk (%), median over the hedge set."""
    unhedged = (oos[(oos.rule == "None") & (oos.est == BASE)].pnl ** 2).sum()
    rr = 100 * _squares(oos).sum() / unhedged
    table = rr.groupby(["rule", "k", "est"]).median().unstack("est")
    return table.reindex(index=RULES, level="rule")[list(ESTIMATORS)]


def dm(oos: pd.DataFrame) -> pd.DataFrame:
    """Per-hedge Diebold-Mariano tests of squared P&L against Equal, counts at the 5% level.
    Estimators whose hedges equal Equal's (PCA under LW and OAS) are left out."""
    sq = _squares(oos)
    rows = {}
    for rule, k in sq.columns.droplevel(["est", "hedge"]).unique():
        base = sq[(rule, k, BASE)]
        for est in sq[(rule, k)].columns.get_level_values("est").unique().drop(BASE):
            d = sq[(rule, k, est)] - base
            if d.abs().to_numpy().max() <= 1e-9 * base.to_numpy().max():
                continue
            ts = np.array([nw_t(d[c]) for c in d])
            rows[(rule, k, est)] = {"better": int((ts < -CRITICAL_T).sum()),
                                    "worse": int((ts > CRITICAL_T).sum()), "hedges": d.shape[1]}
    return pd.DataFrame(rows).T.reindex(index=RULES, level=0)


def calibration(oos: pd.DataFrame) -> pd.DataFrame:
    """Mean z^2 and NW t of H0: E[z^2] = 1, unhedged and for two-swap hedges."""
    z2 = oos[oos.k.isin([0, 2])].assign(z2=lambda d: d.pnl ** 2 / d["var"])
    daily = z2.groupby(["rule", "est", "date"]).z2.mean()
    rows = {}
    for (rule, est), s in daily.groupby(["rule", "est"]):
        name = "Unhedged" if rule == "None" else rule
        rows.setdefault(est, {})[(name, "E[z2]")] = s.mean()
        rows[est][(name, "t")] = nw_t(s - 1)
    return pd.DataFrame(rows).T.loc[list(ESTIMATORS), ["Unhedged", *RULES]]


def inflation(oos: pd.DataFrame) -> pd.DataFrame:
    """Medians of rho(q) = q'Sq / q'q and of q' S_e q / q'Sq, S Equal, unhedged and PCA pairs."""
    sel = oos[oos.rule.isin(["None", "PCA"]) & oos.k.isin([0, 2])]
    var = sel.pivot_table(index=["rule", "date", "hedge"], columns="est", values="var")
    qq = sel[sel.est == BASE].set_index(["rule", "date", "hedge"]).qq
    out = pd.DataFrame({"rho": var[BASE] / qq, "LW": var["LW"] / var[BASE],
                        "OAS": var["OAS"] / var[BASE]})
    names = {"None": "Unhedged", "PCA": "PCA pairs"}
    return out.groupby("rule").median().rename(index=names).rename_axis(None)
