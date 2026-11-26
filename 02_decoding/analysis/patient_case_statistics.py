"""Fig. 6g,h. Patient-adjusted locking–accuracy association and single-case timing.

patient_locking_excess.csv and patient_accuracy_locking.csv provide the session-level
locking, surrogate mean and decoding accuracy. phase_offsets.csv has harmonic_k,
cohort (Healthy or SCI), unit (session label) and offset_pct (gait-cycle percentage).
"""
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, write_table
from patient_spatial_rsa import SESSIONS


def circular_mean(values):
    return np.angle(np.exp(2j * np.pi * np.asarray(values) / 50).mean()) * 50 / (2 * np.pi)


def build():
    locking = pd.read_csv(inp("patient_locking_excess.csv"))
    scores = pd.read_csv(inp("patient_accuracy_locking.csv")).set_index("session").decode_trial_median_r
    visits = [f"{case}_sess{visit:02d}" for case, numbers in SESSIONS.items() for visit in numbers]
    locking = locking[locking.session.isin(visits)].copy()
    locking["excess"] = locking.abs_c - locking.surrogate_mean
    locking["accuracy"] = locking.session.map(scores)
    locking["patient"] = locking.session.str.split("_").str[0]
    rows = []
    for k, group in locking.groupby("harmonic_k"):
        if group.accuracy.isna().any() or set(group.session) != set(visits):
            raise ValueError(f"Incomplete post-operative session pairs at harmonic {k}")
        fit = smf.ols("accuracy ~ excess + C(patient)", group).fit()
        t = fit.tvalues["excess"]
        rows.append(dict(harmonic_k=k, partial_r=t / np.sqrt(t*t + fit.df_resid),
                         df=fit.df_resid, p=fit.pvalues["excess"]))
    write_table(pd.DataFrame(rows), "patient_locking_accuracy.csv", index=False)
    offsets = pd.read_csv(inp("phase_offsets.csv"))
    offsets = offsets[offsets.harmonic_k == 2].copy()
    offsets["participant"] = offsets.unit.str.split("_").str[0]
    groups = offsets.groupby(["cohort", "participant"]).offset_pct.apply(circular_mean)
    reference = groups["Healthy"].to_numpy()
    centre = circular_mean(reference)
    def wrap(value):
        return (value - centre + 25) % 50 - 25
    reference = wrap(reference)
    n = len(reference)
    rows = []
    for case, value in groups["SCI"].items():
        t = (wrap(value) - reference.mean()) / (reference.std(ddof=1) * np.sqrt((n + 1) / n))
        rows.append(dict(patient=case, offset_pct=value, t=t, df=n-1, p=2*stats.t.sf(abs(t), n-1)))
    result = pd.DataFrame(rows)
    result["pFDR"] = multipletests(result.p, method="fdr_bh")[1]
    write_table(result, "patient_phase_offsets.csv", index=False)


if __name__ == "__main__":
    build()
