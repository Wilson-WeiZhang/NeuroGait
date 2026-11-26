"""Per-trial decoding accuracy by joint group, healthy imagined walking.

Reads the per-trial, per-joint correlation between decoded and reference trajectory of the calibrated decoder
(model "calib", column rpt_r; scored trials only) and averages it over the joints of each group, NaN joints skipped:
Hip = left and right hip, Knee = left and right knee, Ankle = left and right ankle, All = the six joints. Trials without
any finite joint are dropped.

Named input: healthy_imagined_trial_r.csv (session, fold, trial_idx, test_ordinal, joint, model, rpt_r).
Named output (derived/): joint_trial_r.csv (one row per trial).
"""
import csv
import os
from collections import defaultdict
from statistics import mean

from analysis_common import inp, out_path

GROUPS = {"Hip": ["LHipW X", "RHipW X"], "Knee": ["LKneW X", "RKneW X"], "Ankle": ["LAnkW X", "RAnkW X"],
          "All": ["LHipW X", "RHipW X", "LKneW X", "RKneW X", "LAnkW X", "RAnkW X"]}


def nanmean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def main():
    per, meta = defaultdict(dict), {}
    with open(inp("healthy_imagined_trial_r.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] != "calib":
                continue
            key = (row["session"], int(row["trial_idx"]))
            try:
                v = float(row["rpt_r"])
                v = None if v != v else v
            except ValueError:
                v = None
            per[key][row["joint"]] = v
            meta[key] = (int(row["test_ordinal"]), int(row["fold"]))
    trials = []
    for (sess, tidx), jr in per.items():
        rec = {"subject": sess.split("_sess")[0], "session": sess, "trial_idx": tidx,
               "test_ordinal": meta[(sess, tidx)][0], "fold": meta[(sess, tidx)][1]}
        for g, js in GROUPS.items():
            rec[g] = nanmean([jr.get(j) for j in js])
        if rec["All"] is not None:
            trials.append(rec)
    trials.sort(key=lambda r: (r["session"], r["trial_idx"]))
    p = out_path("joint_trial_r.csv")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "session", "trial_idx", "test_ordinal", "fold", "Hip", "Knee", "Ankle", "All"])
        for r in trials:
            w.writerow([r["subject"], r["session"], r["trial_idx"], r["test_ordinal"], r["fold"]]
                       + [("" if r[g] is None else f"{r[g]:.6f}") for g in ("Hip", "Knee", "Ankle", "All")])
    os.replace(tmp, p)
    print("wrote", p, len(trials), "trials")


if __name__ == "__main__":
    main()
