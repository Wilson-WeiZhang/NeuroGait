"""Session-level statistics linking decoded phase agreement and cortical locking to decoding accuracy (healthy, 80 sessions).

agreement_real_tests.csv: phase agreement of the decoded output with the reference (session mean of cos of the phase difference,
  harmonics k1 and k2) in the 55 responder sessions against real walking in the same sessions; paired Wilcoxon, BH over the two
  measures.
partial_forest.csv: partial Spearman of session accuracy on the rest-corrected locking of harmonic k given the other two harmonics
  (ranks residualised on the other two by least squares; p from the Pearson r of the residuals with n - 2 - 2 df); BH over the
  three; 95 % percentile interval from 5000 bootstraps over sessions, default_rng(20261105).
replication_bootstrap.csv: in the 38 participants with two sessions, per session order, Spearman of session accuracy on the
  rest-corrected locking of k1, k2 (5000 bootstraps, default_rng(20261102), drawn in the order session 1 k1, k2, session 2 k1,
  k2) and of k3 (5000 bootstraps, default_rng(20261105), session 1 then session 2).

Named inputs: decoder_agreement_imagined.csv, decoder_agreement_real.csv (per-session decoder measures with the responder flag),
session_locking_table.csv (per-session accuracy r_v10 and rest-corrected locking excess_k1..3).
Named outputs: agreement_real_tests.csv, partial_forest.csv, replication_bootstrap.csv.
Usage: python locking_accuracy_stats.py
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import inp, write_table

SEED = 20261105
SEED_K12 = 20261102


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(q, 1)
    return out


def write(df, name):
    write_table(df, name, index=False, float_format="%.6g", lineterminator="\n")


def agreement_real():
    iw = pd.read_csv(inp("decoder_agreement_imagined.csv")).set_index("session")
    rw = pd.read_csv(inp("decoder_agreement_real.csv")).set_index("session")
    Rs = iw[iw.resp == True]    # noqa: E712
    M = ("cos_d1", "cos_d2")
    p = [stats.wilcoxon(rw[m].reindex(Rs.index), Rs[m]).pvalue for m in M]
    q = bh(p)
    rows = []
    for i, m in enumerate(M):
        rows.append(dict(measure=m, median_resp=Rs[m].median(), median_real_same_sessions=rw[m].reindex(Rs.index).median(),
                         real_above=int((rw[m].reindex(Rs.index) > Rs[m]).sum()), n=len(Rs), p=p[i], pFDR=q[i],
                         test="paired Wilcoxon, the 55 responder sessions, BH over the two measures"))
    write(pd.DataFrame(rows), "agreement_real_tests.csv")


def partial_multi(y, x, zs):
    ry, rx = stats.rankdata(y), stats.rankdata(x)
    X = np.c_[np.ones(len(ry)), *[stats.rankdata(z) for z in zs]]
    ey = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    ex = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    r = np.corrcoef(ey, ex)[0, 1]
    df = len(ry) - 2 - len(zs)
    t = r * np.sqrt(df / (1 - r ** 2))
    return float(r), float(2 * stats.t.sf(abs(t), df))


def partial_forest():
    m = pd.read_csv(inp("session_locking_table.csv"))
    y = m.r_v10.to_numpy()
    E = {k: m[f"excess_k{k}"].to_numpy() for k in (1, 2, 3)}
    res = {k: partial_multi(y, E[k], [E[j] for j in (1, 2, 3) if j != k]) for k in (1, 2, 3)}
    q = bh([res[k][1] for k in (1, 2, 3)])
    rng = np.random.default_rng(SEED)
    B = np.empty((5000, 3))
    for b in range(5000):
        i = rng.integers(0, len(y), len(y))
        for c, k in enumerate((1, 2, 3)):
            B[b, c] = partial_multi(y[i], E[k][i], [E[j][i] for j in (1, 2, 3) if j != k])[0]
    rows = []
    for c, k in enumerate((1, 2, 3)):
        lo, hi = np.percentile(B[:, c], [2.5, 97.5])
        rows.append(dict(harmonic_k=k, rho=res[k][0], ci_lo=lo, ci_hi=hi, p=res[k][1], pFDR=q[c], n=len(y)))
    write(pd.DataFrame(rows), "partial_forest.csv")


def boot_spearman(y, v, rng):
    rho = stats.spearmanr(y, v)[0]
    b = np.empty(5000)
    for j in range(5000):
        i = rng.integers(0, len(y), len(y))
        b[j] = stats.spearmanr(y[i], v[i])[0]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return rho, lo, hi


def replication():
    m = pd.read_csv(inp("session_locking_table.csv"))
    m["subj"] = m.session.str.split("_sess").str[0]
    m["sess"] = m.session.str.extract(r"_sess(\d+)")[0].astype(int)
    d = m[m.groupby("subj").session.transform("count") == 2].copy()
    d["order"] = d.groupby("subj").sess.rank().astype(int)
    rows = []
    for seed, ks in ((SEED_K12, (1, 2)), (SEED, (3,))):
        rng = np.random.default_rng(seed)
        for o in (1, 2):
            s = d[d.order == o]
            for k in ks:
                rho, lo, hi = boot_spearman(s.r_v10.to_numpy(), s[f"excess_k{k}"].to_numpy(), rng)
                rows.append(dict(panel="fig5_session_replication", session_order=o, harmonic_k=k, rho=rho, ci_lo=lo, ci_hi=hi, n=len(s)))
    write(pd.DataFrame(rows), "replication_bootstrap.csv")


def main():
    agreement_real()
    partial_forest()
    replication()


if __name__ == "__main__":
    main()
