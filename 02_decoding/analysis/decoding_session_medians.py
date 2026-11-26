"""Per-session decoding accuracy of imagined and real walking as the median over trials, with the paired comparison.

Per task (imagined walking, real walking), from the long table of per-trial, per-joint correlations of the scored
trials: keep model == 'calib'; per (session, trial) the trial value is the mean of rpt_r over the six joints that are
finite (a trial with no finite joint is dropped); per session the value is the median (and the mean) over trials.
The paired comparison real vs imagined walking is printed at two levels: across the 80 sessions (two-sided Wilcoxon
signed-rank and paired t on the session medians) and across the 42 participants (mean of each participant's session
medians).

Named inputs: iw_trial_r_long.csv, rw_trial_r_long.csv (session, trial_idx, joint, model, rpt_r).
Named output: derived/decoding_session_medians.csv.
"""
import csv
from collections import defaultdict
from statistics import mean, median

import numpy as np
from scipy import stats

from analysis_common import inp, out_path


def nanmean(vals):
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else None


def per_session_trial_values(src):
    """-> dict session -> list of per-trial values (mean over the finite joints)."""
    per = defaultdict(dict)
    for row in csv.DictReader(src.open(newline="")):
        if row["model"] != "calib":
            continue
        try:
            v = float(row["rpt_r"])
            if v != v:
                v = None
        except ValueError:
            v = None
        per[(row["session"], int(row["trial_idx"]))][row["joint"]] = v
    joints = sorted({j for jr in per.values() for j in jr})
    assert len(joints) == 6, joints
    bysess = defaultdict(list)
    for (sess, _tidx), jr in per.items():
        allv = nanmean([jr.get(j) for j in joints])
        if allv is not None:
            bysess[sess].append(allv)
    return bysess


def paired_tests(a, b):
    """b vs a, two-sided Wilcoxon signed-rank and paired t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = stats.wilcoxon(b, a, alternative="two-sided")
    t = stats.ttest_rel(b, a)
    return dict(W=float(w.statistic), p=float(w.pvalue), t=float(t.statistic), t_p=float(t.pvalue),
                n=int(len(a)), n_b_gt_a=int((b > a).sum()))


def build():
    iw = per_session_trial_values(inp("iw_trial_r_long.csv"))
    rw = per_session_trial_values(inp("rw_trial_r_long.csv"))
    assert set(iw) == set(rw)
    sessions = sorted(iw)
    rows = []
    for s in sessions:
        iv, rv = iw[s], rw[s]
        rows.append(dict(session=s, subject=s.split("_sess")[0], n_trials_iw=len(iv), n_trials_rw=len(rv),
                         iw_median=median(iv), rw_median=median(rv), iw_mean=mean(iv), rw_mean=mean(rv),
                         rw_minus_iw_median=median(rv) - median(iv)))

    sess = paired_tests([r["iw_median"] for r in rows], [r["rw_median"] for r in rows])
    bysub = defaultdict(lambda: {"iw": [], "rw": []})
    for r in rows:
        bysub[r["subject"]]["iw"].append(r["iw_median"])
        bysub[r["subject"]]["rw"].append(r["rw_median"])
    subs = sorted(bysub)
    part = paired_tests([mean(bysub[s]["iw"]) for s in subs], [mean(bysub[s]["rw"]) for s in subs])
    print(f"median of session medians: IW {median(r['iw_median'] for r in rows):.6f}, "
          f"RW {median(r['rw_median'] for r in rows):.6f}")
    for lab, d in (("sessions", sess), ("participants", part)):
        print(f"{lab} (n = {d['n']}): RW > IW {d['n_b_gt_a']}/{d['n']}, Wilcoxon W = {d['W']:.1f}, "
              f"p = {d['p']:.3e}; paired t = {d['t']:.3f}, p = {d['t_p']:.3e}")

    p = out_path("decoding_session_medians.csv")
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "subject", "n_trials_iw", "n_trials_rw",
                    "iw_median", "rw_median", "iw_mean", "rw_mean", "rw_minus_iw_median"])
        for r in rows:
            w.writerow([r["session"], r["subject"], r["n_trials_iw"], r["n_trials_rw"],
                        *(format(r[k], ".6f") for k in ("iw_median", "rw_median", "iw_mean", "rw_mean",
                                                        "rw_minus_iw_median"))])
    tmp.replace(p)
    print("wrote", p)


if __name__ == "__main__":
    build()
