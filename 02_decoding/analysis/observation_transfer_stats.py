"""Own-condition single-harmonic decoders of the six observation-experiment participants: imagined walking with observation (MI+AO)
against observation only (AO), for the decoders built on harmonic k1 and on harmonic k3.

From the per-fold decoder outputs (one CSV per participant x condition x fold x harmonic; rows of kind "within" = trials decoded by the
decoder trained on the same condition; columns renamed on reading, train_on / test_on = training and test condition):
  trial r      = mean over the six joints of the calibrated per-trial r (rpt_r);
  scored       = trial_idx > 10 (20 trials per condition);
  participant  = median of the 20 scored trials; cohort value = median of the six participant medians;
  statistics   = trial-level REML linear mixed model r ~ c + (1 + c | participant), c = +0.5 MI+AO / -0.5 AO; the best REML
                 likelihood among the admissible fits of lbfgs, powell and cg (converged, finite s.e., fixed effect equal to the
                 mean difference of the balanced design); random intercept only when no slope fit is admissible.

Named input: observation_folds (folder of <participant>__<condition>__f<fold>__<harmonic>.csv).
Named label: own_decoder (the decoder_trained_on value of the contrast rows).
Named output: observation_transfer_k3.csv (k1 and k3 rows).
Usage: python observation_transfer_stats.py
"""
import glob
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_common import inp, label, write_table

COND = {"miao": "MI+AO", "ao": "AO"}
FOLD_COLS = ["key", "train_on", "test_on", "feat", "fold", "kind", "trial_idx", "joint", "rpt_r", "n_samples"]
COLS = ["row_type", "features", "test_condition", "decoder_trained_on", "contrast", "participant", "n_trials", "median_r",
        "reference_median_of_participant_medians", "compared_median_of_participant_medians", "n_participants_compared_lower",
        "lme_estimate", "lme_se", "lme_p|random_effects"]


def lme(df):
    diff = df[df.c > 0].r.mean() - df[df.c < 0].r.mean()
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in ("lbfgs", "powell", "cg"):
            try:
                f = smf.mixedlm("r ~ c", df, groups=df["participant"], re_formula="~c").fit(reml=True, method=m)
            except Exception:
                continue
            if f.converged and np.isfinite(f.bse["c"]) and abs(f.params["c"] - diff) < 1e-6 \
                    and (best is None or f.llf > best.llf):
                best = f
        if best is not None:
            return best, "intercept and slope"
        return smf.mixedlm("r ~ c", df, groups=df["participant"]).fit(reml=True), "intercept only (no admissible slope fit)"


def load(feat):
    fs = sorted(glob.glob(str(inp("observation_folds") / f"*__{feat}.csv")))
    assert len(fs) == 120, len(fs)
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    assert d.shape[1] == len(FOLD_COLS)
    d = d.set_axis(FOLD_COLS, axis=1)
    d = d[d.kind == "within"]
    assert (d.groupby(["key", "train_on", "test_on", "feat", "trial_idx"]).joint.nunique() == 6).all()
    t = d.groupby(["key", "train_on", "test_on", "feat", "kind", "trial_idx"]).rpt_r.mean().rename("r").reset_index()
    t = t[t.trial_idx > 10].copy()
    assert len(t) == 240
    t["participant"] = t.key.str.replace("sub", "SUB_", regex=False)
    return t


def rows_for(t, feat):
    rows = []
    for test in ("miao", "ao"):
        med = t[t.test_on == test].groupby("participant").r.median()
        for p, v in med.items():
            rows.append(["participant", feat, COND[test], COND[test], "", p, 20, round(v, 6), "", "", "", "", "", ""])
    sub = t.copy()
    sub["c"] = np.where(sub.test_on == "miao", 0.5, -0.5)
    med = sub.groupby(["participant", "test_on"]).r.median().unstack()
    f, spec = lme(sub)
    b, se, pv = float(f.params["c"]), float(f.bse["c"]), float(f.pvalues["c"])
    rows.append(["contrast", feat, "MI+AO vs AO", label("own_decoder"), "MI+AO - AO test trials, own decoders", "all6", len(sub), "",
                 round(med["ao"].median(), 6), round(med["miao"].median(), 6), int((med["miao"] < med["ao"]).sum()), round(b, 6),
                 round(se, 6), f"{pv:.3g}|{spec}"])
    return rows


def main():
    d = pd.DataFrame(rows_for(load("k1"), "k1") + rows_for(load("k3"), "k3"), columns=COLS)
    write_table(d, "observation_transfer_k3.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
