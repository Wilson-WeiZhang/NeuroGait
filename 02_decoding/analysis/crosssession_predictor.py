"""Cross-session predictor: does one session's gait-phase locking predict the decoding accuracy of the other session?

For every participant with two recorded sessions (38 of 42), the first session (A) and the second session (B) give
the imagined-walking locking |c| at k1 (miA, miB), the rest-corrected locking (exA, exB) and the decoding accuracy
(rA, rB). The panel statistic is the Spearman correlation of exA with rB, across participants: locking and decoding
come from different recordings and different trials. Printed alongside: miA vs rB, the pooled two-direction
correlation (exA -> rB and exB -> rA, 2n points) and the within-session correlation of excess_k1 with r.

Named input: session_master.csv.
Named output: derived/crosssession_predictor.csv (one row per two-session participant).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis_common import inp, write_table


def build():
    M = pd.read_csv(inp("session_master.csv")).dropna(subset=["mi_k1", "excess_k1", "r_v10"]).copy()
    M["subj"] = M.session.str.extract(r"(subj\d+)")
    M["sess"] = M.session.str.extract(r"sess0*(\d+)").astype(int)
    rows = []
    for s, d in M.groupby("subj"):
        if d.sess.nunique() < 2:
            continue
        d = d.sort_values("sess")
        a, b = d.iloc[0], d.iloc[1]
        rows.append((s, a.mi_k1, a.excess_k1, a.r_v10, b.mi_k1, b.excess_k1, b.r_v10))
    P = pd.DataFrame(rows, columns=["subj", "miA", "exA", "rA", "miB", "exB", "rB"])

    rho_ex, p_ex = spearmanr(P.exA, P.rB)
    rho_mi, _ = spearmanr(P.miA, P.rB)
    rho_pool, p_pool = spearmanr(np.r_[P.exA, P.exB], np.r_[P.rB, P.rA])
    rho_within = spearmanr(M.excess_k1, M.r_v10)[0]
    print(f"n = {len(P)} two-session participants")
    print(f"excess A -> r B: rho = {rho_ex:+.4f}, p = {p_ex:.2g} | |c| A -> r B: rho = {rho_mi:+.4f}")
    print(f"pooled 2n: rho = {rho_pool:+.4f}, p = {p_pool:.2g} | within-session: rho = {rho_within:+.4f}")
    p = write_table(P, "crosssession_predictor.csv", index=False)
    print("wrote", p)


if __name__ == "__main__":
    build()
