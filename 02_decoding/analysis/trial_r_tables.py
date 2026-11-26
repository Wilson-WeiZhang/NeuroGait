"""Trial-level decoding tables: one row per (session, test trial, joint), from the per-session decoder outputs.

Each decoding run writes one `<session>_trial_r.csv` per session (columns session, fold, trial_idx, test_ordinal, joint,
model, r, rpt_r, n_samples, first_finite_time_sec). This script concatenates them in session-name order (header once,
non-finite values written as "NaN", every other field unchanged), checks that every trial has exactly six joint rows,
and optionally keeps only the scored trials, i.e. the trials whose onset follows the session's calibration segment
(table `scoring_mask.csv`, column `keep`; a trial absent from the mask stops the run).

Named inputs: `<set>_trial_r_dir` (folder of the per-session files; sets: healthy_imagined, healthy_rest, patient_postop,
patient_preop) and, with --scored, `scoring_mask.csv`.
Named outputs (under $NEUROGAIT_OUT/derived/): healthy_imagined_trial_r.csv, healthy_rest_trial_r.csv,
patient_trial_r.csv (post-operative and pre-operative sessions, post-operative first).

usage:
    python trial_r_tables.py healthy_imagined [--scored]
    python trial_r_tables.py healthy_rest [--scored]
    python trial_r_tables.py patient [--scored]
"""
import argparse
import csv
import math
import os
from collections import defaultdict

from analysis_common import inp, out_path

HEAD = ["session", "fold", "trial_idx", "test_ordinal", "joint", "model", "r", "rpt_r", "n_samples",
        "first_finite_time_sec"]
JOINTS = ["LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"]

# output name, folders (in order), expected number of sessions, condition of the test trials in the scoring mask
SETS = {
    "healthy_imagined": ("healthy_imagined_trial_r.csv", ["healthy_imagined_trial_r_dir"], 80, "IW"),
    "healthy_rest": ("healthy_rest_trial_r.csv", ["healthy_rest_trial_r_dir"], 80, "REST"),
    "patient": ("patient_trial_r.csv", ["patient_postop_trial_r_dir", "patient_preop_trial_r_dir"], 15, "IW"),
}


def read_mask(path):
    keep, known = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r["session"], r["condition"])
            known.setdefault(k, set()).add(int(r["trial_idx"]))
            if r["keep"] == "1":
                keep.setdefault(k, set()).add(int(r["trial_idx"]))
    return keep, known


def norm_nan(s):
    return "NaN" if s.strip().lower() == "nan" else s


def build(name, scored):
    out_name, dirs, n_sessions, condition = SETS[name]
    files = []
    for d in dirs:
        files += sorted(inp(d).glob("*_trial_r.csv"))
    sessions = [f.name[:-len("_trial_r.csv")] for f in files]
    if len(set(sessions)) != n_sessions or len(sessions) != n_sessions:
        raise SystemExit(f"{name}: {len(sessions)} per-session files, expected {n_sessions}")
    keep = known = None
    if scored:
        keep, known = read_mask(inp("scoring_mask.csv"))
    rows = []
    for f, s in zip(files, sessions):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            if next(rd) != HEAD:
                raise SystemExit(f"{f.name}: unexpected header")
            n = 0
            for row in rd:
                if row[0] != s or row[5] != "calib" or row[4] not in JOINTS:
                    raise SystemExit(f"{f.name}: unexpected row {row[:6]}")
                if scored:
                    t = int(float(row[2]))
                    if t not in known.get((s, condition), ()):
                        raise SystemExit(f"{s} trial {t} is not in the scoring mask")
                    if t not in keep.get((s, condition), ()):
                        continue
                rows.append([norm_nan(x) for x in row])
                n += 1
        if n == 0:
            raise SystemExit(f"{f.name}: no rows")
    per_trial = defaultdict(int)
    for row in rows:
        per_trial[(row[0], int(row[2]))] += 1
    if any(v != 6 for v in per_trial.values()):
        raise SystemExit(f"{name}: a trial without exactly six joint rows")
    p = out_path(out_name)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEAD)
        w.writerows(rows)
    os.replace(tmp, p)
    print(f"{p}: {len(rows)} rows, {len(per_trial)} trials, {len(set(sessions))} sessions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--scored", action="store_true", help="keep only scored trials")
    a = ap.parse_args()
    build(a.set, a.scored)
