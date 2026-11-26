"""Spearman correlation of each per-session predictor with decoding accuracy, with session-bootstrap 95 % CIs.

Across the 80 healthy sessions, decoding accuracy r (session_master.csv, column r_v10) is correlated (Spearman) with:
- k1 / k2 / k3 rest-corrected gait-phase locking (excess_k1..k3; drawn rows, BH-FDR across the three harmonics),
- mu/beta ERD (drawn row; a control on a different measure, tested with its uncorrected p),
- k1 locking partialled on k2 and k3 (rank-residual partial correlation, t-test p with n - 2 - 2 df),
- k1 / k2 / k3 locking of the rest windows alone (rest_k1..k3).
CI = 2.5th / 97.5th percentile of 1000 session-bootstrap resamples, all drawn from one numpy default_rng(0) stream in
the order: k1-k3 rest-corrected, k1-k3 rest, ERD, partial. `sig` = q < 0.05 for k1-k3, p < 0.05 for ERD, and CI
excluding 0 for the rows not drawn (in_panel = 0).

Named input: session_master.csv.
Named output: derived/locking_forest_stats.csv.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as tdist

from analysis_common import inp, write_table

rng = np.random.default_rng(0)


def boot_rho(x, y, n=1000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y)
    i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    return r0, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), p, len(x)


def partial_rho(x, y, covs, n=1000):
    d = pd.DataFrame({"x": x, "y": y})
    for i, c in enumerate(covs):
        d[f"c{i}"] = c
    d = d.dropna()
    R = d.apply(rankdata)
    Z = np.column_stack([np.ones(len(R))] + [R[f"c{i}"] for i in range(len(covs))])

    def res(v):
        return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]

    rx, ry = res(R.x.values), res(R.y.values)
    r0 = np.corrcoef(rx, ry)[0, 1]
    i = np.arange(len(rx))
    rs = [np.corrcoef(rx[s], ry[s])[0, 1] for s in (rng.choice(i, len(i), True) for _ in range(n))]
    df = len(rx) - 2 - len(covs)
    tt = r0 * np.sqrt(df / (1 - r0 ** 2))
    p = 2 * tdist.sf(abs(tt), df)
    return r0, np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5), p, len(rx)


def bh(ps):
    """Benjamini-Hochberg q = p * m / rank (capped at 1)."""
    ps = np.array(ps)
    o = np.argsort(ps)
    r = np.empty(len(ps), int)
    r[o] = np.arange(1, len(ps) + 1)
    return np.minimum(1.0, ps * len(ps) / r)


def build():
    m = pd.read_csv(inp("session_master.csv"))

    zr = {k: boot_rho(m[f"excess_k{k}"], m.r_v10) for k in (1, 2, 3)}
    zc = {k: boot_rho(m[f"rest_k{k}"], m.r_v10) for k in (1, 2, 3)}
    er = boot_rho(m.mubeta_erd_mean, m.r_v10)
    fam = [(f"k{k}_taskevoked", *zr[k]) for k in (1, 2, 3)] + [("mubeta_erd", *er)]
    q3 = bh([f[4] for f in fam[:3]])
    q4 = list(q3) + [np.nan]
    tv = list(q3) + [fam[3][4]]

    stat = []
    for i, (key, r0, lo, hi, p, n) in enumerate(fam):
        stat.append([key, r0, lo, hi, p, q4[i], n, int(tv[i] < 0.05), 1])
    pr = partial_rho(m.excess_k1, m.r_v10, [m.excess_k2, m.excess_k3])
    stat.append(["k1_partial_given_k2k3", *pr[:4], np.nan, pr[4], int(pr[1] > 0 or pr[2] < 0), 0])
    for k in (1, 2, 3):
        r0, lo, hi, p, n = zc[k]
        stat.append([f"k{k}_restcontrol", r0, lo, hi, p, np.nan, n, int(lo > 0 or hi < 0), 0])

    t = pd.DataFrame(stat, columns=["predictor", "rho", "ci_lo", "ci_hi", "p_spearman", "q_bh_fdr",
                                    "n", "sig", "in_panel"])
    p = write_table(t, "locking_forest_stats.csv", index=False)
    print(t.to_string(index=False))
    print("wrote", p)


if __name__ == "__main__":
    build()
