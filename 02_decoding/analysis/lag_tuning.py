"""Lag tuning and timing error of the decoded joint trajectories against the reference trajectories.

For every scored trial and joint, the decoded trajectory is shifted by a lag given in % of that trial's gait cycle (positive =
decoded later than the reference) and correlated with the reference (Pearson r, lags -50 to +50 % in steps of 5 %). The gait
cycle is the stride frequency of the reference hip of that trial (peak of the Hann-windowed spectrum within 0.45-1.15 Hz).
  session lag curve   median of r over the trial x joint rows of the session, per lag.
  timing error        phase of the decoded minus the reference trajectory at k1 and k2 (z-scored traces projected on Hann-weighted
                      sinusoids at k x stride frequency), in % of the gait cycle; positive = decoded later.
  trial-level test    one value per trial and lag = median over its joints; per cohort and lag a linear mixed model with the trial
                      as the sample and a random intercept for the session, r ~ condition + (1 | session) (condition = imagined
                      walking against resting input), REML, Wald p; Benjamini-Hochberg over the 21 lags of that cohort.
Conditions: healthy imagined walking and resting input; patient imagined walking and resting input (scored trials only).

Named inputs: scoring_mask.csv; healthy_imagined_dump, healthy_rest_dump, patient_imagined_dump, patient_rest_dump (folders of
per-session decoder output, *_dump.h5); labels healthy, patients (cohort codes of the scoring mask).
Named outputs: lag_by_session.csv, timing_rows.csv, lag_trial_tests.csv.
Usage: python lag_tuning.py [--workers N]
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy import optimize, stats

from analysis_common import inp, label, write_table

FS = 250.0
JOINTS = ["LHip", "LKnee", "LAnk", "RHip", "RKnee", "RAnk"]
LAGS = np.arange(-50, 51, 5)
# decoder output of the four scored sets: (cohort label, condition label, folder name)
SETS = {("Healthy", "IW"): ("healthy", "IW", "healthy_imagined_dump"), ("Healthy", "REST"): ("healthy", "REST", "healthy_rest_dump"),
        ("SCI", "IW"): ("patients", "IW", "patient_imagined_dump"), ("SCI", "REST"): ("patients", "REST", "patient_rest_dump")}


def stride_freq(x):
    x = x - x.mean()
    n = 1 << 15
    f = np.fft.rfftfreq(n, 1 / FS)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= 0.45) & (f <= 1.15)
    return f[band][np.argmax(p[band])]


def proj(x, f, t):
    w = np.hanning(len(x))
    return np.sum(w * x * np.exp(-1j * 2 * np.pi * f * t))


def z(x):
    return (x - x.mean()) / x.std()


def one_session(args):
    """Lag curves (trial x joint x lag) and timing errors of the scored trials of one session."""
    fp, keep, cohort, cond = args
    with h5py.File(fp, "r") as h:
        s = h.attrs["session"]
        s = s.decode() if isinstance(s, bytes) else str(s)
        C, Y, T = h["rows/C_pred"][:], h["rows/Y_true"][:], h["rows/time_sec"][:]
        tid, off, nr = h["trials/trial_idx"][:], h["trials/row_offset"][:], h["trials/n_rows"][:]
    trials, tim = [], []
    for ti, o, n in zip(tid, off, nr):
        if (s, int(ti)) not in keep:
            continue
        c0, y0, t0 = C[o:o + n], Y[o:o + n], T[o:o + n]
        hip = 0 if np.isfinite(y0[:, 0]).all() else 3
        per_joint = []
        for j in range(6):
            ok = np.isfinite(c0[:, j]) & np.isfinite(y0[:, j]) & np.isfinite(y0[:, hip])
            if ok.sum() < 2 * FS:
                continue
            c, y, t = c0[ok, j], y0[ok, j], t0[ok]
            f0 = stride_freq(y0[ok, hip])
            rs = []
            for lag in LAGS:
                sh = int(round(lag / 100 * FS / f0))
                a, b = (c[sh:], y[:len(y) - sh]) if sh >= 0 else (c[:sh], y[-sh:])
                rs.append(np.corrcoef(a, b)[0, 1] if len(a) > FS and a.std() > 0 and b.std() > 0 else np.nan)
            per_joint.append(rs)
            rec = dict(cohort=cohort, condition=cond, session=s, trial_idx=int(ti), joint=JOINTS[j])
            for k in (1, 2):
                ap, at = proj(z(c), k * f0, t), proj(z(y), k * f0, t)
                rec[f"err_k{k}_pct"] = float(np.angle(ap * np.conj(at)) / (2 * np.pi * k) * 100)     # positive = decoded later
            tim.append(rec)
        if per_joint:
            trials.append((int(ti), np.array(per_joint)))
    return cohort, cond, s, trials, tim


def lmm(y, x, g):
    """y = b0 + b1 x + u[g] + e, REML, profiled over lambda = var(u) / var(e); Wald z and p for b1."""
    labels, gi = np.unique(g, return_inverse=True)
    X = np.c_[np.ones(len(y)), x]
    n = np.bincount(gi).astype(float)
    sX = np.vstack([np.bincount(gi, X[:, j]) for j in range(2)]).T
    sy = np.bincount(gi, y)
    XX, Xy, yy = X.T @ X, X.T @ y, y @ y
    N, p = len(y), 2

    def parts(loglam):
        lam = np.exp(loglam)
        w = lam / (1 + lam * n)
        A = XX - (sX.T * w) @ sX
        B = Xy - (sX.T * w) @ sy
        b = np.linalg.solve(A, B)
        rss = yy - (sy * w * sy).sum() - b @ B
        return A, b, rss, lam

    def nll(loglam):
        A, _, rss, lam = parts(loglam)
        return (N - p) * np.log(rss / (N - p)) + np.log(1 + lam * n).sum() + np.linalg.slogdet(A)[1]
    grid = np.linspace(-12, 8, 41)
    best = grid[int(np.argmin([nll(v) for v in grid]))]
    res = optimize.minimize_scalar(nll, bounds=(best - 1, best + 1), method="bounded", options=dict(xatol=1e-6))
    A, b, rss, _ = parts(res.x)
    se = np.sqrt(rss / (N - p) * np.linalg.inv(A)[1, 1])
    return float(b[1]), float(se), float(2 * stats.norm.sf(abs(b[1] / se)))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    r = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.empty_like(p)
    q[o] = np.minimum(np.minimum.accumulate(r[::-1])[::-1], 1.0)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    mask = pd.read_csv(inp("scoring_mask.csv"))
    jobs = []
    for (cohort, cond), (mc, mcond, folder) in SETS.items():
        m = mask[(mask.cohort == label(mc)) & (mask.condition == mcond) & (mask.keep == 1)]
        keep = set(zip(m.session, m.trial_idx))
        jobs += [(fp, keep, cohort, cond) for fp in sorted(inp(folder).glob("*_dump.h5"))]
    with Pool(a.workers) as pool:
        res = pool.map(one_session, jobs, chunksize=1)

    # session lag curve: median over the trial x joint rows of the session
    lag_rows, trial_rows = [], []
    for cohort, cond, s, trials, _ in res:
        if not trials:                                      # no scored trial in this output (a patient rest session before the cut)
            continue
        M = np.nanmedian(np.vstack([A for _, A in trials]), axis=0)
        lag_rows += [dict(cohort=cohort, condition=cond, session=s, lag_pct=int(l), r=float(v)) for l, v in zip(LAGS, M)]
        for ti, A in trials:
            trial_rows += [dict(cohort=cohort, condition=cond, session=s, trial_idx=ti, lag_pct=int(l), r=float(v)) for l, v in zip(LAGS, np.nanmedian(A, axis=0))]
    write_table(pd.DataFrame(lag_rows), "lag_by_session.csv", index=False, float_format="%.6g", lineterminator="\n")
    write_table(pd.DataFrame([r for *_, tim in res for r in tim]), "timing_rows.csv", index=False, float_format="%.6g", lineterminator="\n")

    # trial-level test of imagined walking against resting input, per lag
    D = pd.DataFrame(trial_rows)
    rows = []
    for co in ("Healthy", "SCI"):
        for lag in LAGS:
            g = D[(D.cohort == co) & (D.lag_pct == lag)].dropna(subset=["r"])
            beta, se, p_ = lmm(g.r.to_numpy(), (g.condition == "IW").to_numpy(float), g.session.to_numpy())
            rows.append(dict(cohort=co, lag_pct=int(lag), beta=beta, se=se, p=p_, n_trials=len(g), n_sessions=g.session.nunique()))
    T = pd.DataFrame(rows)
    T["pFDR"] = np.nan
    for co in ("Healthy", "SCI"):
        m = T.cohort == co
        T.loc[m, "pFDR"] = bh(T[m].p.to_numpy())
    write_table(T, "lag_trial_tests.csv", index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
