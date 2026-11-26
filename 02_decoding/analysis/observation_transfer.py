"""Decoder transfer between imagined walking (MI+AO) and observation only (AO), six supplementary participants.

For each condition and outer fold k the fold model of that condition (27 training trials) was applied to its own fold-k
trials (own decoder) and to the other condition's trials whose own fold is k (transferred decoder), so both decoders of a
test trial were trained on 27 trials. Feature sets: all 360 features, or the 120 features of k1 or k2 (PCA 30 of the
participant's leave-subject-out basis restricted to them).
  trial r     : mean over the six walker joints of the calibrated per-trial Pearson r
  scored      : trial index > 10 (cycles 11-30, after the calibration segment; 20 trials per condition)
  participant : median of the 20 scored trials; cohort value = median of the six participant medians
  statistics  : trial-level REML linear mixed model r ~ c + (1 + c | participant), c = +0.5 / -0.5; best REML fit among
                lbfgs / powell / cg that converged, has a finite standard error and an estimate equal to the mean
                difference (balanced design); random-intercept model when no slope fit qualifies (labelled)

Named inputs: transfer_folds (folder of per-fold, per-joint tables <key>__<condition>__f<k>__<features>.csv);
labels fold_train_column, fold_test_column (column names of the fold tables), own_decoder, other_decoder.
Named output: derived/observation_transfer.csv.
"""
import argparse
import csv
import io
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_common import inp, label, out_path

NAME = {"miao": "MI+AO", "ao": "AO"}


def trial_table():
    d = pd.concat([pd.read_csv(f) for f in sorted(inp("transfer_folds").glob("*.csv"))], ignore_index=True)
    d = d.rename(columns={label("fold_train_column"): "train", label("fold_test_column"): "test"})
    assert (d.groupby(["key", "train", "test", "feat", "trial_idx"]).joint.nunique() == 6).all()
    t = d.groupby(["key", "train", "test", "feat", "kind", "trial_idx"]).rpt_r.mean().rename("r").reset_index()
    t = t[t.trial_idx > 10].copy()
    t["participant"] = t.key.str.replace("sub", "SUB_", regex=False)
    return t


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


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    t = trial_table()
    rows = []
    for feat in ("all", "k1", "k2"):
        for test in ("miao", "ao"):
            other = "ao" if test == "miao" else "miao"
            sub = t[(t.feat == feat) & (t.test == test)].copy()
            sub["c"] = np.where(sub.kind == "cross", 0.5, -0.5)
            med = sub.groupby(["participant", "kind"]).r.median().unstack()
            for p, v in med.iterrows():
                for kind, trained in (("within", test), ("cross", other)):
                    rows.append(["participant", feat, NAME[test], NAME[trained], "", p, 20, round(v[kind], 6),
                                 "", "", "", "", "", ""])
            f, spec = lme(sub)
            b, se, pv = float(f.params["c"]), float(f.bse["c"]), float(f.pvalues["c"])
            rows.append(["contrast", feat, NAME[test], NAME[other], f"transferred - own decoder on {NAME[test]} trials",
                         "all6", len(sub), "", round(med["within"].median(), 6), round(med["cross"].median(), 6),
                         int((med["cross"] < med["within"]).sum()), round(b, 6), round(se, 6), f"{pv:.3g}|{spec}"])
        for kind in ("within", "cross"):
            sub = t[(t.feat == feat) & (t.kind == kind)].copy()
            sub["c"] = np.where(sub.test == "miao", 0.5, -0.5)
            med = sub.groupby(["participant", "test"]).r.median().unstack()
            f, spec = lme(sub)
            b, se, pv = float(f.params["c"]), float(f.bse["c"]), float(f.pvalues["c"])
            own = kind == "within"
            rows.append(["contrast", feat, "MI+AO vs AO", label("own_decoder") if own else label("other_decoder"),
                         f"MI+AO - AO test trials, {'own' if own else 'transferred'} decoders", "all6",
                         len(sub), "", round(med["ao"].median(), 6), round(med["miao"].median(), 6),
                         int((med["miao"] < med["ao"]).sum()), round(b, 6), round(se, 6), f"{pv:.3g}|{spec}"])

    cols = ["row_type", "features", "test_condition", "decoder_trained_on", "contrast", "participant", "n_trials",
            "median_r", "reference_median_of_participant_medians", "compared_median_of_participant_medians",
            "n_participants_compared_lower", "lme_estimate", "lme_se", "lme_p|random_effects"]
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    w.writerows(rows)
    p = out_path("observation_transfer.csv")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8", newline="\n")
    tmp.replace(p)
    print("wrote", p, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
