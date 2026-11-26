"""Figure 7 — physics simulation and walk-intent gate

The walk-intent gate on the causal preprocessing: gate_scores_within_session.py (within-session 10-fold by
event, converged Karcher mean, all-rest pool, earliest-rest reference, fused band-power and tangent-space
scores) run on the gate session set of build_gate_session_set_causal.py, decision windows of 3 s, with the
scorer's own leakage and stream self-checks.
Writes gate_within_session-style tables (scores, thresholds, folds, events, AUC, self-checks) to result/main/.

Reads:
  decoder output in data/decoder_output/ (not distributed)
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import sys
import time

PW = os.environ.get("GAIT_PROJECT_ROOT", "") + "/02_decoding/simulation"
sys.path.insert(0, PW)
import gate_scores_within_session as G

MAIN_ROOT = os.environ.get("GAIT_PROJECT_ROOT", "") + "/data/decoder_output/main/triples/pat_gate"
OUT = os.path.join(PW, "result", "main")
T = 3.0
TAG = "within10"


def main():
    t0 = time.time()
    assert os.path.exists(os.path.join(MAIN_ROOT, "READY")), "causal gate root not built (build_gate_session_set_causal.py)"
    if os.path.exists(os.path.join(OUT, "gate_within_session_scores_T3.csv")):
        sys.exit(f"{OUT} already holds a stream; refusing to overwrite")
    chk, cache, fold_rows = [], {}, []
    os.makedirs(OUT, exist_ok=True)
    per, mean_, lo, hi, stream, thr, rows, D, fmap, inv = G.run_rung(
        TAG, G.SESS11, MAIN_ROOT, T, "checked", 0.0, "earliest", True, chk, cache, fold_rows=fold_rows)
    G.check_no_leakage(f"{TAG}_T{T:.0f}", thr, D, G.SESS11, fmap, chk)
    G.check_stream(f"{TAG}_T{T:.0f}", stream, inv, fmap, chk)
    anchors = cache[(tuple(G.SESS11), MAIN_ROOT, T, 0.0)][1]
    base = dict(rung=TAG, T=T, sessions=11, root="pat_gate_causal", solver="checked", pool="allrest", ref_scheme="earliest")
    auc_rows = [dict(base, session="MEAN11", auc_tiled=mean_, ci_lo=lo, ci_hi=hi)]
    for r in rows:
        auc_rows.append(dict(base, session=r["session"], auc_tiled=r["auc_tiled"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                             n_win=r["n_win"], auc_trial=r.get("auc_trial")))
    for d in inv:
        chk.append(dict(check="loader_inventory", **d))
    for r in thr:
        r["rung"] = TAG
    G.M.write(os.path.join(OUT, f"gate_within_session_scores_T{T:.0f}.csv"), stream)
    G.M.write(os.path.join(OUT, f"gate_within_session_thresholds_T{T:.0f}.csv"), thr,
              head=["rung", "scheme", "T", "session", "fold"])
    G.M.write(os.path.join(OUT, "gate_within_session_folds.csv"), fold_rows)
    G.M.write(os.path.join(OUT, "gate_within_session_events.csv"), anchors)
    G.M.write(os.path.join(OUT, "gate_within_session_auc.csv"), auc_rows,
              head=["rung", "T", "sessions", "root", "solver", "pool", "ref_scheme", "session"])
    G.M.write(os.path.join(OUT, "gate_within_session_selfcheck.csv"), chk, head=["check", "key"])
    G.log(f"\nWITHIN-SESSION 10-FOLD / 3 s: {mean_:.6f} [{lo:.4f}, {hi:.4f}] (11 sessions)")
    G.log(f"DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
