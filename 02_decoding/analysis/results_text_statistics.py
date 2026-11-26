"""Test statistics, effect sizes and 95% confidence intervals quoted in the Results text.

For the comparisons and associations reported in the Results, this script adds to the estimate the test statistic,
the p value, an effect size and a 95% percentile bootstrap interval (10,000 resamples of sessions or participants):

  decoding      imagined walking against resting input, healthy cohort and patients (paired Wilcoxon signed-rank,
                matched-pairs rank-biserial correlation, median paired difference)
  responders    decoding accuracy of responder against non-responder sessions (label permutation of the difference
                of medians, Mann-Whitney U, Cliff's delta)
  timing        absolute phase error of the decoded trajectories against resting input at the stride and step
                frequencies, both cohorts (paired Wilcoxon signed-rank)
  associations  bootstrap intervals of two Spearman correlations (stride-frequency share of decoded power against
                stride-frequency locking; first-session locking against second-session accuracy)
  equation      bootstrap interval of the r-squared between observed accuracy and the parameter-free equation
                (sessions resampled with all their joints)
  similarity    patient session maps against healthy resting sessions, similarity to the healthy group map
                (Mann-Whitney U, Benjamini-Hochberg over the two harmonics, Cliff's delta)

Every block first recomputes the value already printed in the text from the same table ("check" rows) and writes
its statistics only if that value is reproduced.

Named inputs: session_decoding_by_state.csv, responder_table.csv, timing_rows.csv, k1_share_by_session.csv,
crosssession_predictor.csv, equation_cells.csv, patient_session_similarity.csv, session_similarity_by_state.csv,
patient_trial_r.csv, patient_rest_dump, scoring_mask.csv.
Named output: derived/results_text_statistics.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import inp, write_table

N_BOOT = 10000
N_PERM = 100000
SEED = 20261117
rows = []


def add(block, quantity, value, lo=np.nan, hi=np.nan, n=np.nan, test=""):
    rows.append(dict(block=block, quantity=quantity, value=value, ci_lo=lo, ci_hi=hi, n=n, test=test))


def check(block, name, got, want, tol):
    ok = abs(got - want) <= tol
    add(block, "check: " + name, got, test="printed value %g reproduced: %s" % (want, "yes" if ok else "no"))
    print("%-12s %-58s %.6f (printed %g) %s" % (block, name, got, want, "ok" if ok else "MISMATCH"))
    return ok


def boot_ci(rng, fun, *arrays, paired=True):
    """95% percentile bootstrap interval; paired resamples rows jointly, otherwise each array on its own."""
    n = len(arrays[0])
    out = []
    for _ in range(N_BOOT):
        if paired:
            i = rng.integers(0, n, n)
            out.append(fun(*[a[i] for a in arrays]))
        else:
            out.append(fun(*[a[rng.integers(0, len(a), len(a))] for a in arrays]))
    return np.nanpercentile(out, [2.5, 97.5])


def rank_biserial(d):
    d = d[d != 0]
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def cliffs_delta(a, b):
    return np.sign(a[:, None] - b[None, :]).mean()


def spearman(x, y):
    return stats.spearmanr(x, y)[0]


def paired_block(rng, block, a, b, label):
    d = a - b
    w = stats.wilcoxon(a, b)
    add(block, label + ": Wilcoxon W", w.statistic, n=len(d), test="Wilcoxon signed-rank, two-sided")
    add(block, label + ": p", w.pvalue, n=len(d))
    add(block, label + ": pairs with a positive difference", int((d > 0).sum()), n=len(d))
    lo, hi = boot_ci(rng, rank_biserial, d)
    add(block, label + ": rank-biserial r", rank_biserial(d), lo, hi, len(d), "matched-pairs rank-biserial correlation")
    lo, hi = boot_ci(rng, np.median, d)
    add(block, label + ": median paired difference", np.median(d), lo, hi, len(d))


def unpaired_block(rng, block, a, b, label):
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    add(block, label + ": Mann-Whitney U", u.statistic, n=len(a) + len(b), test="Mann-Whitney, two-sided; n = %d and %d" % (len(a), len(b)))
    add(block, label + ": p", u.pvalue, n=len(a) + len(b))
    lo, hi = boot_ci(rng, cliffs_delta, a, b, paired=False)
    add(block, label + ": Cliff's delta", cliffs_delta(a, b), lo, hi, len(a) + len(b))
    return u.pvalue


def session_medians():
    sd = pd.read_csv(inp("session_decoding_by_state.csv"))
    sd = sd[sd.row_type == "value"]
    return sd.assign(state=sd.group.str.split(",").str[0]).pivot(index="unit", columns="state", values="value")


def decoding_healthy(rng, med):
    if check("decoding", "healthy imagined walking, median of session medians", med["Imag"].median(), 0.465586, 5e-5) and \
       check("decoding", "healthy resting input, median of session medians", med["Rest input"].median(), 0.014641, 5e-5):
        paired_block(rng, "decoding", med["Imag"].values, med["Rest input"].values, "healthy imagined walking minus resting input")


def decoding_patients(rng):
    """Per-session median of the per-trial mean over joints; resting-input trials kept by the scoring mask."""
    files = sorted(inp("patient_rest_dump").glob("*_trial_r.csv"))
    sessions = [f.name.removesuffix("_trial_r.csv") for f in files]
    p = pd.read_csv(inp("patient_trial_r.csv"))
    iw = p[p.session.isin(sessions)].groupby(["session", "trial_idx"]).rpt_r.mean().groupby("session").median()
    if not check("decoding", "patients imagined walking, median of session medians", iw.median(), 0.538915, 5e-5):
        return
    mask = pd.read_csv(inp("scoring_mask.csv"))
    mask = mask[(mask.condition == "REST") & (mask.keep == 1)]
    rest = {}
    for f in files:
        r = pd.read_csv(f)
        if not len(r):
            continue
        s = r.session.iloc[0]
        r = r[(r.model == "calib") & r.trial_idx.isin(mask[mask.session == s].trial_idx)]
        if len(r):
            rest[s] = r.groupby(["fold", "trial_idx"]).rpt_r.mean().median()
    rest = pd.Series(rest)
    add("decoding", "patient sessions with scored resting input", len(rest), n=len(sessions))
    if len(rest) and check("decoding", "patients resting input, median of session medians", rest.median(), -0.021402, 5e-5):
        j = pd.concat([iw, rest], axis=1, keys=["iw", "rest"]).dropna()
        paired_block(rng, "decoding", j.iw.values, j.rest.values, "patients imagined walking minus resting input")


def responders(rng, med):
    table = pd.read_csv(inp("responder_table.csv")).set_index("session")
    acc = med["Imag"].reindex(table.index)
    a = acc[table.resp.astype(str) == "True"].dropna().values
    b = acc[table.resp.astype(str) == "False"].dropna().values
    if not (check("responders", "responder sessions, median r", np.median(a), 0.57, 0.005)
            and check("responders", "non-responder sessions, median r", np.median(b), 0.22, 0.005)):
        return
    obs = np.median(a) - np.median(b)
    pool = np.concatenate([a, b])
    count = 0
    for _ in range(N_PERM):
        q = rng.permutation(pool)
        count += abs(np.median(q[:len(a)]) - np.median(q[len(a):])) >= abs(obs)
    add("responders", "permutation p, difference of medians", (count + 1) / (N_PERM + 1), n=len(pool),
        test="label permutation, two-sided, %d permutations; n = %d and %d" % (N_PERM, len(a), len(b)))
    lo, hi = boot_ci(rng, lambda x, y: np.median(x) - np.median(y), a, b, paired=False)
    add("responders", "difference of medians", obs, lo, hi, len(pool))
    unpaired_block(rng, "responders", a, b, "responder against non-responder accuracy")


def timing(rng):
    t = pd.read_csv(inp("timing_rows.csv"))
    printed = {"Healthy": {"IW": (11.9, 5.8), "REST": (24.5, 12.0)}, "SCI": {"IW": (8.9, 5.1), "REST": (23.6, 13.1)}}
    for cohort, want in printed.items():
        g = t[t.cohort == cohort].copy()
        g["k1"] = g.err_k1_pct.abs()
        g["k2"] = g.err_k2_pct.abs()
        ses = g.groupby(["condition", "session"])[["k1", "k2"]].median().reset_index()
        ok = True
        for cond in ("IW", "REST"):
            s = ses[ses.condition == cond]
            ok &= check("timing", "%s %s k1 median absolute phase error (%% cycle)" % (cohort, cond), s.k1.median(), want[cond][0], 0.05)
            ok &= check("timing", "%s %s k2 median absolute phase error (%% cycle)" % (cohort, cond), s.k2.median(), want[cond][1], 0.05)
        if ok:
            w = ses.pivot(index="session", columns="condition", values=["k1", "k2"]).dropna()
            for k in ("k1", "k2"):
                paired_block(rng, "timing", w[(k, "REST")].values, w[(k, "IW")].values,
                             "%s %s absolute phase error, resting input minus imagined walking (%% cycle)" % (cohort, k))


def associations(rng):
    d = pd.read_csv(inp("k1_share_by_session.csv"))
    d = d[d.row_type == "value"]
    x, y = d.rest_corrected_k1.values.astype(float), d.k1_share.values.astype(float)
    if check("associations", "rho, k1 share of decoded power against k1 locking", spearman(x, y), 0.680567, 5e-5):
        lo, hi = boot_ci(rng, spearman, x, y)
        add("associations", "Spearman rho, k1 share of decoded power against k1 locking", spearman(x, y), lo, hi, len(x), "bootstrap over sessions")
    d = pd.read_csv(inp("crosssession_predictor.csv"))
    x, y = d.exA.values, d.rB.values
    if check("associations", "rho, first-session k1 locking against second-session accuracy", spearman(x, y), 0.613962, 5e-5):
        lo, hi = boot_ci(rng, spearman, x, y)
        add("associations", "Spearman rho, first-session k1 locking against second-session accuracy", spearman(x, y), lo, hi, len(x), "bootstrap over participants")


def equation(rng):
    c = pd.read_csv(inp("equation_cells.csv"))
    for name, key, want in (("healthy", "IW", 0.884561), ("patients", "PAT", 0.719317)):
        g = c[c.set == key]
        r2 = np.corrcoef(g.r, g.P_shared)[0, 1] ** 2
        if not check("equation", "%s r-squared, observed against calculated accuracy" % name, r2, want, 5e-5):
            continue
        sessions = g.session.unique()
        idx = {s: g.index[g.session == s].values for s in sessions}
        out = []
        for _ in range(N_BOOT):
            ii = np.concatenate([idx[s] for s in rng.choice(sessions, len(sessions))])
            out.append(np.corrcoef(g.r.loc[ii], g.P_shared.loc[ii])[0, 1] ** 2)
        lo, hi = np.percentile(out, [2.5, 97.5])
        add("equation", "%s r-squared" % name, r2, lo, hi, len(g), "bootstrap over %d sessions, joints of a session kept together" % len(sessions))


def similarity(rng):
    q = pd.read_csv(inp("patient_session_similarity.csv"))
    q = q[q.row_type == "patient_session"]
    h = pd.read_csv(inp("session_similarity_by_state.csv"))
    h = h[(h.state == "REST") & (h.reference == "IW")]
    pvals = []
    rest_rng = np.random.default_rng(20261104)
    for k, want_patient, want_rest in ((1, 0.56, 0.26), (2, 0.69, 0.44)):
        a = q[q.harmonic_k == k].value.values.astype(float)
        b = h[h.harmonic_k == k].similarity.values
        if not (check("similarity", "k%d patient sessions, mean similarity" % k, a.mean(), want_patient, 0.005)
                and check("similarity", "k%d healthy resting sessions, mean similarity" % k, b.mean(), want_rest, 0.005)):
            return
        unpaired_block(rng, "similarity", a, b, "k%d patient against healthy resting sessions" % k)
        # Compare the 11-session patient mean with equally sized resting-session sets.
        null_means = np.array([rest_rng.choice(b, len(a), replace=False).mean() for _ in range(100000)])
        pvals.append((1 + np.count_nonzero(null_means >= a.mean())) / (1 + len(null_means)))
        add("similarity", "k%d patient sessions above the resting 95th percentile" % k, int((a > np.percentile(b, 95)).sum()), n=len(a))
    for k, p in zip((1, 2), stats.false_discovery_control(pvals, method="bh")):
        add("similarity", "k%d resting-set resampling p, Benjamini-Hochberg over the two harmonics" % k, p)


def build():
    med = session_medians()
    blocks = (lambda r: decoding_healthy(r, med), decoding_patients, lambda r: responders(r, med), timing, associations, equation, similarity)
    for i, block in enumerate(blocks):
        block(np.random.default_rng([SEED, i]))
    p = write_table(pd.DataFrame(rows), "results_text_statistics.csv", index=False)
    print("wrote", p, len(rows), "rows")


if __name__ == "__main__":
    build()
