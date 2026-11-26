"""Fig. 7. Forward displacement, cadence and step imbalance after calibration.

Each test fits a session random intercept to paired trial differences. BH families
contain seven contrasts for displacement and three each for cadence and imbalance.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, label, write_table


def scored_trials(df, col="trial_idx"):
    """Rows of a simulation table whose trial is scored (after the calibration segment); rows with a negative trial index are kept."""
    m = pd.read_csv(inp("scoring_mask.csv"), usecols=["session", "condition", "trial_idx", "keep"])
    m = m[m.condition == "IW"]
    keep = set(zip(m.session[m.keep == 1], m.trial_idx[m.keep == 1]))
    known = set(zip(m.session, m.trial_idx))
    out = []
    for s, t in zip(df["session"], df[col]):
        t = int(float(t))
        if t >= 0 and (s, t) not in known:
            raise SystemExit(f"trial {s} {t} is not in the scoring mask")
        out.append(t < 0 or (s, t) in keep)
    return df.loc[out]


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def lme_paired_test(a, b, key):
    """Per-trial paired difference d = a - b matched on (session, trial_idx); b None = test of a's value itself against 0.
    LME d ~ 1 + (1 | session), REML, optimizers lbfgs / powell / cg; of the fits that converge with a finite intercept SE
    and log-likelihood the one with the largest log-likelihood is reported (no fit kept: the intercept-only fit when the
    session variance sits at the REML boundary 0 for every optimizer, else an error). Wald p of the intercept sets the star."""
    A = a.set_index(["session", "trial_idx"])[key].astype(float)
    dd = (A if b is None else A - b.set_index(["session", "trial_idx"])[key].astype(float)).dropna()
    sess = dd.index.get_level_values("session").astype(str)
    df = pd.DataFrame({"d": dd.to_numpy(), "subject": sess.str[:5], "session": sess})
    multi = df.session.nunique() >= 2
    fits = []
    for m in ("lbfgs", "powell", "cg") if multi else ():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm("d ~ 1", df, groups=df["session"], re_formula="~1").fit(reml=True, method=m, maxiter=2000)
        if f.converged and np.isfinite(f.bse_fe["Intercept"]) and np.isfinite(f.llf):
            fits.append((m, f))
    if fits:
        m, f = max(fits, key=lambda t: t[1].llf)
        beta, se, revar = float(f.fe_params["Intercept"]), float(f.bse_fe["Intercept"]), float(f.cov_re.iloc[0, 0])
    else:
        if multi and not _lme_at_boundary(df):
            raise RuntimeError(f"LME d ~ 1 + (1 | session) converged with no optimizer ({key}, n = {len(df)})")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.ols("d ~ 1", df).fit()
        m = "ols (session variance 0)" if multi else "ols (one session: intercept only)"
        beta, se, revar = float(f.params["Intercept"]), float(f.bse["Intercept"]), 0.0
    p = float(f.pvalues["Intercept"])
    return dict(n=len(df), n_sessions=df.session.nunique(), mean_difference=beta, se=se, session_variance=revar, optimizer=m,
                median_difference=float(np.median(df.d)), p=p, star=stars(p))


def _lme_at_boundary(df):
    """True when every optimizer converged with a finite log-likelihood and a session variance at the REML boundary."""
    res = []
    for m in ("lbfgs", "powell", "cg"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm("d ~ 1", df, groups=df["session"], re_formula="~1").fit(reml=True, method=m, maxiter=2000)
        res.append(bool(f.converged) and bool(np.isfinite(f.llf)) and float(f.cov_re.iloc[0, 0]) < 1e-4 * float(np.var(df.d)))
    return all(res)


def add_fdr(family):
    """Benjamini-Hochberg q and the star drawn from it, for a list of test results (p taken at 6 significant figures)."""
    q = multipletests(np.asarray([float("{:.6g}".format(r["p"])) for r in family], float), alpha=0.05, method="fdr_bh")[1]
    for r, v in zip(family, q):
        r["q"], r["star_drawn"] = float(v), stars(v)


def stat_rows(panel, contrasts):
    """Source Data rows of the tests behind the stars: contrast, n, mean difference, p and q."""
    return [dict(panel=panel, row_type="stat", contrast=c, n=r["n"], n_sessions=r["n_sessions"], mean_difference=r["mean_difference"],
                 median_difference=r["median_difference"], p=r["p"], q_bh=r["q"], mark=r["star_drawn"]) for c, r in contrasts]


def build():
    column = label("condition_column")
    if column == "condition_column":
        column = "condition"
    traces = scored_trials(pd.read_csv(inp("simulation_traces.csv.gz")))
    walker = scored_trials(pd.read_csv(inp("walker_window_traces.csv.gz")))
    traces = pd.concat([traces[traces[column] != "wphase"], walker], ignore_index=True)
    traces = traces[traces.t_s <= 3.0 + 1e-9].sort_values("t_s")
    end = traces.groupby([column, "session", "trial_idx"]).x_m.last().reset_index()
    rhythm = scored_trials(pd.read_csv(inp("rhythm_metrics.csv")))
    walker = scored_trials(pd.read_csv(inp("walker_window_rhythm.csv")))
    rhythm = pd.concat([rhythm[rhythm.stage == "confirm"], walker[walker.stage == "calib"]])
    rhythm = rhythm[(rhythm.phase_src == "k1") & (rhythm.lam.astype(float) == 0)]
    rhythm = rhythm.assign(cadence=rhythm.n_adv_events / rhythm.dur_s)
    conditions = {"Walker": "wphase_matched", "Decoded": "dphase", "Coordination null": "dphase_iscr", "In-phase null": "dphase_uscr"}
    rows = []
    for data, key in [(end, "x_m"), (rhythm, "cadence"), (rhythm, "r5_lr_asym")]:
        groups = {name: data[data[column] == tag] for name, tag in conditions.items()}
        pairs = [("Walker", "Decoded"), ("Decoded", "Coordination null"), ("Decoded", "In-phase null")]
        if key == "x_m":
            pairs.extend((label, None) for label in conditions)
        tests = [(f"{a} vs {b or 'zero'}", lme_paired_test(groups[a], groups.get(b), key)) for a, b in pairs]
        add_fdr([test for _, test in tests])
        rows.extend(stat_rows(key, tests))
    write_table(pd.DataFrame(rows), "rollout_statistics.csv", index=False)


if __name__ == "__main__":
    build()
