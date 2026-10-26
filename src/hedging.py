"""Covariance estimators, hedge rules and the out-of-sample loop."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf

from config import DATA, DV01, EWMA, MAX_COND, TENORS, WINDOW

R = np.array(DV01, dtype=float)
RULES = ("PCA", "MinVar")
ESTIMATORS = ("Equal", "LW", "OAS") + tuple(f"EWMA {lam}" for lam in EWMA)


def changes() -> pd.DataFrame:
    """Daily swap-rate changes (bp); holiday rows with partial quotes are dropped first."""
    rates = pd.read_csv(DATA, index_col=0, parse_dates=True, date_format="%m/%d/%Y")
    x = rates.dropna().diff().dropna() * 100
    x.columns = list(TENORS)
    return x


def estimators(x: np.ndarray, i: int) -> dict:
    """Covariance estimates from the changes before day i."""
    win = x[i - WINDOW:i]
    est = {"Equal": np.cov(win.T, bias=True),
           "LW": LedoitWolf().fit(win).covariance_,
           "OAS": OAS().fit(win).covariance_}
    for lam in EWMA:
        w = lam ** np.arange(i)[::-1]
        est[f"EWMA {lam}"] = (x[:i] * (w / w.sum())[:, None]).T @ x[:i]
    return est


def top_pcs(S: np.ndarray, k: int) -> np.ndarray:
    """Eigenvectors of the k largest eigenvalues."""
    return np.linalg.eigh(S)[1][:, ::-1][:, :k]


def hedged(rule: str, S: np.ndarray, cols: list) -> np.ndarray:
    """DV01 vector q = R + B h after hedging with swaps at the tenors in cols."""
    B = np.eye(len(R))[:, cols]
    if rule == "PCA":
        V = top_pcs(S, len(cols))
        h = np.linalg.solve(V.T @ B, -V.T @ R)
    else:
        h = np.linalg.solve(B.T @ S @ B, -B.T @ S @ R)
    return R + B @ h


def hedge_sets(x: np.ndarray) -> list:
    """Every single tenor, and the pairs well conditioned in the first window."""
    V = top_pcs(np.cov(x[:WINDOW].T, bias=True), 2)
    pairs = [list(p) for p in combinations(range(len(R)), 2)
             if np.linalg.cond(V[list(p)]) < MAX_COND]
    return [[j] for j in range(len(R))] + pairs


def out_of_sample(X: pd.DataFrame) -> pd.DataFrame:
    """P&L, forecast variance and q'q of every position on every test day."""
    x = X.to_numpy()
    sets = hedge_sets(x)
    rows = []
    for i in range(WINDOW, len(x)):
        for est, S in estimators(x, i).items():
            rows.append((X.index[i], est, "None", 0, "-", -R @ x[i], R @ S @ R, R @ R))
            for rule in RULES:
                for cols in sets:
                    q = hedged(rule, S, cols)
                    rows.append((X.index[i], est, rule, len(cols),
                                 "-".join(TENORS[j] for j in cols), -q @ x[i], q @ S @ q, q @ q))
    return pd.DataFrame(rows, columns=["date", "est", "rule", "k", "hedge", "pnl", "var", "qq"])
