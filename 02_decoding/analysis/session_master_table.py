"""Per-session master table of the healthy cohort: decoding accuracy, gait-phase locking, ERD and questionnaire scores.

One row per healthy session (80 sessions, 42 participants):
- r_v10: the session's decoding accuracy (mean of the finite calibrated per-fold cells, 10 folds x 6 joints).
- mi_k1..k3 / rest_k1..k3: gait-phase locking |c| at the harmonics k1-k3 during imagined walking and during the
  trial-matched rest windows; |c| is the modulus of the complex harmonic phase value, averaged over channels and
  gait-cycle points of the session.
- excess_k = mi_k - rest_k (rest-corrected locking).
- mu_erd_mean, beta_erd_mean, mubeta_erd_mean: per-session mu / beta event-related desynchronization (imagined
  walking vs the following rest).
- Q1_intent..Q6_avoid: the session's self-report scores (mean over that participant's rows of the session; Q3 yes = 1).

Named inputs: session_decoding_r.csv, iw_phase_maps.csv, rest_phase_maps.csv, session_erd.csv,
questionnaire_scores.xlsx.
Named output: derived/session_master.csv.
"""
import numpy as np
import pandas as pd

from analysis_common import inp, write_table


def cabs_by_k(path):
    """Session x harmonic mean of |c| over channels and gait-cycle points."""
    d = pd.read_csv(path, usecols=["session", "harmonic_k", "real", "imag"])
    d["cabs"] = np.hypot(d.real, d.imag)
    t = d.groupby(["session", "harmonic_k"])["cabs"].mean().unstack("harmonic_k")
    t.columns = [f"k{int(c)}" for c in t.columns]
    return t.reset_index()


def build():
    r = pd.read_csv(inp("session_decoding_r.csv")).rename(columns={"r": "r_v10"})

    mi = cabs_by_k(inp("iw_phase_maps.csv"))
    mi = mi.rename(columns={c: f"mi_{c}" for c in mi.columns if c.startswith("k")})
    rest = cabs_by_k(inp("rest_phase_maps.csv"))
    rest = rest.rename(columns={c: f"rest_{c}" for c in rest.columns if c.startswith("k")})

    m = r.merge(mi, on="session").merge(rest, on="session")
    for k in ("k1", "k2", "k3"):
        m[f"excess_{k}"] = m[f"mi_{k}"] - m[f"rest_{k}"]

    erd = pd.read_csv(inp("session_erd.csv"),
                      usecols=["session", "mu_erd_mean", "beta_erd_mean", "mubeta_erd_mean"])
    m = m.merge(erd, on="session", how="left")

    q = pd.read_excel(inp("questionnaire_scores.xlsx"), sheet_name="scores")
    cc = list(q.columns)
    ren = {cc[3]: "Q1_intent", cc[4]: "Q2_sync", cc[5]: "Q3_still",
           cc[6]: "Q4_pace", cc[7]: "Q5_relax", cc[8]: "Q6_avoid"}
    q = q.rename(columns=ren)
    q["Q3_still"] = q["Q3_still"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    q["subj_int"] = q["subj_number"].str.extract(r"SUB_0*(\d+)").astype(int)
    q["sess_int"] = q["session_number"].astype(int)
    qa = q.groupby(["subj_int", "sess_int"])[
        ["Q1_intent", "Q2_sync", "Q3_still", "Q4_pace", "Q5_relax", "Q6_avoid"]].mean().reset_index()
    m["subj_int"] = m.session.str.extract(r"subj0*(\d+)").astype(int)
    m["sess_int"] = m.session.str.extract(r"sess0*(\d+)").astype(int)
    m = m.merge(qa, on=["subj_int", "sess_int"], how="left")

    p = write_table(m, "session_master.csv", index=False)
    print(f"wrote {p}  ({len(m)} sessions, {m.subj_int.nunique()} participants, mean r = {m.r_v10.mean():.4f})")
    return m


if __name__ == "__main__":
    build()
