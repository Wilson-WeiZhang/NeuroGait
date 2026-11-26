"""Self-report items against k1 gait-phase locking: Spearman rho of each of the six questionnaire items with the rest-corrected
k1 locking (excess_k1) over the healthy sessions with a questionnaire (76).

Per item: Spearman rho and its p; 95 % percentile interval from 1000 bootstrap resamples of the sessions (one numpy
default_rng(0) shared by the six items, drawn in the order Q1 ... Q6); Benjamini-Hochberg across the six items
(scipy false_discovery_control). Rows are ordered by rho, largest first.

Named input: session_locking_table.csv (per-session table with Q1_intent ... Q6_avoid and excess_k1).
Named output: selfreport_forest.csv.
Usage: python selfreport_forest.py
"""
import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, spearmanr

from analysis_common import inp, write_table

ITEMS = [("Q1_intent", "intent to move"), ("Q2_sync", "sync with walker"), ("Q3_still", "stayed still (0-1)"),
         ("Q4_pace", "matched pace"), ("Q5_relax", "felt relaxed"), ("Q6_avoid", "volitional control")]


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def boot_rho(x, y, rng, n=1000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0, p = spearmanr(x, y)
    i = np.arange(len(x))
    rs = np.array([spearmanr(x[s], y[s])[0] for s in (rng.choice(i, len(i), True) for _ in range(n))])
    rs = rs[np.isfinite(rs)]
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return r0, lo, hi, p, len(x)


def main():
    m = pd.read_csv(inp("session_locking_table.csv"))
    rng = np.random.default_rng(0)
    rows = [(lab, *boot_rho(m[col], m.excess_k1, rng)) for col, lab in ITEMS]
    rows.sort(key=lambda r: r[1])
    q = false_discovery_control([r[4] for r in rows], method="bh")
    d = pd.DataFrame([dict(label=lab, rho=r0, ci_lo=lo, ci_hi=hi, q=qq, n=n) for (lab, r0, lo, hi, p, n), qq in zip(rows, q)])
    d = d.sort_values("rho", ascending=False).reset_index(drop=True)
    out = [dict(panel="selfreport_forest", row_type="stat", group=r.label, p=float(r.q), mark=star(r.q),
                test="Spearman, BH across the six items", rho=r.rho, ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=r.n) for r in d.itertuples()]
    write_table(pd.DataFrame(out), "selfreport_forest.csv", index=False, float_format="%.6g", lineterminator="\n")


if __name__ == "__main__":
    main()
