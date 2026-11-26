"""Statistical mediation over healthy sessions: volitional imagery control (X, questionnaire item Q6_avoid) -> k1 gait-phase
locking above rest (M = mi_k1 - rest_k1) -> session decoding accuracy (Y = r_v10).

Sessions with the questionnaire (76 of 80); X, M, Y z-scored (pandas std, ddof = 1). OLS paths: a (M on X), b and the direct
effect c' (Y on X and M), total c (Y on X); indirect = a x b. Session-level percentile bootstrap, 1000 resamples of the sessions
with replacement, numpy default_rng(0): 95 % intervals of a x b and c' from the same resamples; two-sided bootstrap p of the
indirect effect = 2 min(P(a x b <= 0), P(a x b >= 0)). The other paths keep their OLS t-test p.

Named input: session_locking_table.csv (per-session table with Q6_avoid, mi_k1, rest_k1, r_v10).
Named output: mediation_paths.csv.
Usage: python mediation.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis_common import inp, write_table


def main():
    m = pd.read_csv(inp("session_locking_table.csv"))
    m["lock"] = m.mi_k1 - m.rest_k1
    d = m[["Q6_avoid", "lock", "r_v10"]].dropna().rename(columns={"Q6_avoid": "X", "lock": "M", "r_v10": "Y"})
    for c in ["X", "M", "Y"]:
        d[c] = (d[c] - d[c].mean()) / d[c].std()
    a = sm.OLS(d.M, sm.add_constant(d.X)).fit()
    bf = sm.OLS(d.Y, sm.add_constant(d[["X", "M"]])).fit()
    tot = sm.OLS(d.Y, sm.add_constant(d.X)).fit()
    aA, pa = a.params["X"], a.pvalues["X"]
    bB, pb = bf.params["M"], bf.pvalues["M"]
    cp, pcp = bf.params["X"], bf.pvalues["X"]
    cT, pcT = tot.params["X"], tot.pvalues["X"]
    rng = np.random.default_rng(0)
    n_s = len(d)
    inds, cps = [], []
    for _ in range(1000):
        dd = d.iloc[rng.integers(0, n_s, n_s)]
        a_b = sm.OLS(dd.M, sm.add_constant(dd.X)).fit().params["X"]
        b_f = sm.OLS(dd.Y, sm.add_constant(dd[["X", "M"]])).fit()
        inds.append(a_b * b_f.params["M"])
        cps.append(b_f.params["X"])
    lo, hi = np.percentile(inds, [2.5, 97.5])
    cp_lo, cp_hi = np.percentile(cps, [2.5, 97.5])
    inds = np.asarray(inds)
    p_boot = 2 * min((inds <= 0).mean(), (inds >= 0).mean())
    rows = [("a_X_to_M", aA, pa), ("b_M_to_Y_given_X", bB, pb), ("c_total_X_to_Y", cT, pcT),
            ("cprime_direct_X_to_Y_given_M", cp, pcp), ("indirect_axb", aA * bB, p_boot),
            ("indirect_CI_lo", lo, np.nan), ("indirect_CI_hi", hi, np.nan),
            ("indirect_boot_p_two_sided", p_boot, np.nan), ("indirect_n_boot", len(inds), np.nan),
            ("cprime_CI_lo", cp_lo, np.nan), ("cprime_CI_hi", cp_hi, np.nan)]
    out = pd.DataFrame([dict(panel="mediation_paths", row_type="value", path=k, beta=b, p=p) for k, b, p in rows])
    write_table(out, "mediation_paths.csv", index=False, float_format="%.6g", lineterminator="\n")


if __name__ == "__main__":
    main()
