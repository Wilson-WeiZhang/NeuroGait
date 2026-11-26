"""Lead-leg traces: hip angle and the k1, k2, k3 EEG components on a fixed walker clock, for trials in which the walker starts with
the usual leg (Typical) and trials in which the walker starts with the other leg (Other leg), healthy sessions.

Trials (125 Hz trial packets of imagined walking; EEG, walker goniometer and walker phase).  Walker motion of each trial from
its goniometers (smoothed absolute velocity above half its median over the middle half of the trial: first and last moving
sample).  Onset phase = walker phase 1 s after the first motion minus the session's mean rate times that time; a trial is an
Other-leg trial when its onset phase deviates from the session's circular mean by more than 120 degrees.  Of those, a trial
enters only if the deviation exceeds 120 degrees and the direction of the first hip movement differs from that of the typical
trials.  Sessions with fewer than 10 typical trials are left out.
Signals.  Per trial, five windows of three walker cycles each over the motion span; per window and channel the sin^2-weighted
projection of the EEG on exp(-i k phase), k = 1, 2, 3; unit-modulus per window, mean over windows = the trial's map.  The
session's typical-trial map (leave the trial itself out for typical trials) over the central cluster, normalized, projects the
trial's band-passed EEG (band (k - 0.5, k + 0.5) x the session's stride rate, 4th-order Butterworth, zero phase, Hilbert): real
part of the projection.  Hip = the walker's left hip angle (right hip when the left is not finite in 90 % of the trial) minus its
trial mean.
Clock.  psi = walker phase minus the trial's signed onset-phase deviation; grid psi = 2 pi (c0 + m / 25), m = 0 .. 99 (4 cycles
from the first whole cycle after the first motion); a trial too short for the grid is skipped.
Summary.  Drawn: mean over sessions of the per-session mean over trials, per group and grid point; band = 95 % CI over sessions
(mean +- t(n - 1) SE).  Pointwise test, per signal and grid point, in the sessions that have both groups: paired Wilcoxon
signed-rank, two-sided, Typical against Other leg, on the per-session means; BH over the 100 grid points of that signal.

Named inputs: responder_table.csv (the session list); trial_packets, trial_packets_more (per-session npz trial packets; a session
is read from the first folder that holds it); for `summary`: lead_leg_session_means.csv.
Named outputs: lead_leg_session_means.csv (stage `sessions`), lead_leg_traces.csv and lead_leg_pointwise.csv (stage `summary`).
Usage: python lead_leg_traces.py sessions [--workers N] | summary
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import signal, stats

from analysis_common import inp, write_table

FS = 125.0
K = 5                      # windows per trial
STR = 3                    # walker cycles per window
KS = (1, 2, 3)
TWOPI = 2 * np.pi
EPS = 1e-12
CENTRAL = sorted({"FCz", "FC1", "FC2", "Cz", "C1", "C2", "CPz", "CP1", "CP2", "FC3", "C3", "CP3", "FC4", "C4", "CP4"})
NCYC, PER = 4, 25
BANDS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5))
SIG = ("hip", "k1", "k2", "k3")


# ---- trials ----------------------------------------------------------------------------------------------------------
def load(path):
    """Imagined-walking trials of one packet: EEG x, core start c0 and length L, walker phase ph, goniometer y."""
    with np.load(path) as z:
        labels = [str(c) for c in z["labels"]]
        x, seg, ph, y = z["x_IW"], z["seg_IW"], z["ph_IW"], z["y_IW"]
        ends = np.r_[seg[1:, 0], len(x)]
        T = [dict(x=x[o: ends[j]].astype(float), c0=int(c0), L=int(L), ph=ph[o: ends[j]], y=y[o: ends[j]].astype(float)) for j, (o, c0, L) in enumerate(seg)]
    return T, labels


def motion(t):
    """on, off in core samples and the distance over the core, from this trial's goniometers."""
    s = t["y"][t["c0"]: t["c0"] + t["L"]]
    live = np.isfinite(s).mean(axis=0) > 0.95
    s = np.nan_to_num(s[:, live])
    n = int(0.5 * FS)
    v = np.convolve(np.abs(np.gradient(s, axis=0)).mean(axis=1), np.ones(n) / n, mode="same")
    L = t["L"]
    above = np.flatnonzero(v > 0.5 * np.median(v[L // 4: 3 * L // 4]))
    i0, i1 = int(above[0]), int(above[-1])
    c = np.cumsum(v)
    return i0, i1, 1.0 + 8.0 * np.clip((c - c[i0]) / (c[i1] - c[i0]), 0, 1)


def window_z(x, ph, i0, i1, d):
    p = np.maximum.accumulate(ph)
    if p[i1] - p[i0] < (STR + 0.5) * TWOPI:
        return None
    nch = x.shape[1]
    Z, D = np.empty((K, len(KS), nch), complex), np.empty(K)
    for q, lo in enumerate(np.linspace(p[i0], p[i1] - STR * TWOPI, K)):
        hi = lo + STR * TWOPI
        a, b = np.searchsorted(p, [lo, hi])
        w = np.sin(np.pi * (p[a: b] - lo) / (hi - lo)) ** 2
        sw = w.sum()
        seg = x[a: b] - (w[:, None] * x[a: b]).sum(axis=0) / sw
        for kk, k in enumerate(KS):
            Z[q, kk] = 2.0 * (w[:, None] * seg * np.exp(-1j * k * ph[a: b])[:, None]).sum(axis=0) / sw
        D[q] = (w * d[a: b]).sum() / sw
    return Z, D


def per_window(T, SP):
    Z = []
    for t, (i0, i1, d) in zip(T, SP):
        c0, L = t["c0"], t["L"]
        n = min(L, len(d))
        ii1 = min(i1, n - 1)
        r = None if ii1 <= i0 else window_z(t["x"][c0: c0 + n], t["ph"][c0: c0 + n], i0, ii1, d[:n])
        Z.append(np.full((K, 3, t["x"].shape[1]), np.nan + 0j) if r is None else r[0])
    return np.array(Z)


def wrap(x):
    return np.angle(np.exp(1j * x))


def mean_rate(T):
    return np.median([(t["ph"][t["c0"] + t["L"] - 1] - t["ph"][t["c0"]]) / t["L"] for t in T])


def onset_phases(T, SP):
    rate = mean_rate(T)
    ph = []
    for t, (i0, i1, d) in zip(T, SP):
        p = t["ph"][t["c0"]: t["c0"] + t["L"]]
        a = min(i0 + int(1.0 * FS), len(p) - 1)
        ph.append(p[a] - rate * a)
    ph = np.array(ph)
    dev = np.angle(np.exp(1j * (ph - np.angle(np.exp(1j * ph).mean()))))
    return dev, np.abs(dev) > np.radians(120.0)


def onset_deviation(T, SP):
    """Signed deviation of each trial's start phase from the session's circular mean."""
    rate = mean_rate(T)
    ph_on = []
    for t, (i0, i1, d) in zip(T, SP):
        p = t["ph"][t["c0"]: t["c0"] + t["L"]]
        a = min(i0 + int(1.0 * FS), len(p) - 1)
        ph_on.append(p[a] - rate * (a - i0))
    ph_on = np.array(ph_on)
    return wrap(ph_on - np.angle(np.exp(1j * ph_on).mean()))


def trial_labels(T, SP):
    """Other-leg flag, |onset-phase deviation| in degrees, and whether the first hip movement differs from the typical trials."""
    _, other = onset_phases(T, SP)
    dm = onset_deviation(T, SP)
    first = []
    for t, (i0, i1, d) in zip(T, SP):
        y = t["y"][t["c0"]: t["c0"] + t["L"]]
        j = 0 if np.isfinite(y[:, 0]).mean() > 0.5 else 3
        lo, hi = max(i0 - int(0.2 * FS), 0), min(i0 + int(0.6 * FS), y.shape[0] - 1)
        first.append(np.sign(np.nan_to_num(y[hi, j] - y[lo, j])))
    first = np.array(first)
    typ = np.sign(np.median(first[~other])) if (~other).any() else 0
    return other, np.degrees(np.abs(dm)), first != typ, dm


# ---- session stage ---------------------------------------------------------------------------------------------------
def one_session(a):
    sess, path = a
    T, lab = load(path)
    SP = [motion(t) for t in T]
    other, dm_abs, diff, dm = trial_labels(T, SP)
    both = other & (dm_abs > 120) & diff
    if (~other).sum() < 10:
        return None
    ch = [lab.index(c) for c in CENTRAL]
    rate = mean_rate(T) * FS / (2 * np.pi)
    Z = per_window(T, SP)[..., ch]
    ok = np.isfinite(Z).all(axis=(1, 2, 3))
    Uk = np.where(ok[:, None, None, None], Z / np.maximum(np.abs(Z), EPS), 0).mean(axis=1)
    ty = ok & ~other
    n = ty.sum()
    sos = [signal.butter(4, [lo * rate, hi * rate], "bandpass", fs=FS, output="sos") for lo, hi in BANDS]
    Sm = [Uk[:, kk][ty].sum(axis=0) for kk in range(3)]
    rows = {}
    for j, (t, (i0, i1, d)) in enumerate(zip(T, SP)):
        if not (ty[j] or (ok[j] and both[j])):
            continue
        c0, L = t["c0"], t["L"]
        psi = t["ph"][c0: c0 + L] - dm[j]
        start = int(np.ceil(psi[min(i0, L - 1)] / (2 * np.pi)))
        targets = 2 * np.pi * (start + np.arange(NCYC * PER) / PER)
        if targets[-1] > psi[-1]:
            continue
        idx = np.clip(np.searchsorted(psi, targets), 0, L - 1)
        v = {}
        for kk in range(3):
            U = Uk[:, kk]
            m = (Sm[kk] - U[j]) / (n - 1) if ty[j] else Sm[kk] / n
            m = m / np.maximum(np.linalg.norm(m), EPS)
            za = signal.hilbert(signal.sosfiltfilt(sos[kk], t["x"][:, ch], axis=0), axis=0)
            v[f"k{kk + 1}"] = np.real((np.conj(m)[None] * za).sum(axis=1))[c0 + idx]
        yv = np.nan_to_num(t["y"][:, 0] if np.isfinite(t["y"][:, 0]).mean() > 0.9 else t["y"][:, 3])
        v["hip"] = yv[c0 + idx] - yv[c0: c0 + L].mean()
        rows[j] = ("Typical" if ty[j] else "Other leg", v)
    return sess, rows


def sessions_stage(workers):
    jobs = []
    for s in sorted(pd.read_csv(inp("responder_table.csv")).session):
        p = inp("trial_packets") / f"{s}.npz"
        jobs.append((s, p if p.exists() else inp("trial_packets_more") / f"{s}.npz"))
    with Pool(workers) as p:
        R = [r for r in p.map(one_session, jobs, chunksize=1) if r is not None]
    sm = []
    for sess, rows in R:
        for tag in ("Typical", "Other leg"):
            tr = [v for t_, v in rows.values() if t_ == tag]
            if not tr:
                continue
            for sig in SIG:
                A = np.array([v[sig] for v in tr])
                sm += [dict(session=sess, trials=tag, signal=sig, cycle=q / PER, n_trials=len(tr), value=float(x)) for q, x in enumerate(A.mean(axis=0))]
    write_table(pd.DataFrame(sm), "lead_leg_session_means.csv", index=False, float_format="%.6g")


# ---- summary ---------------------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[o] = np.minimum(q, 1)
    return out


def summary():
    D = pd.read_csv(inp("lead_leg_session_means.csv"))
    tr_rows = []
    for (tag, sig, cyc), g in D.groupby(["trials", "signal", "cycle"]):
        v = g.value.to_numpy()
        n = len(v)
        h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
        tr_rows.append(dict(trials=tag, signal=sig, cycle=cyc, mean=v.mean(), ci_lo=v.mean() - h, ci_hi=v.mean() + h, n_sessions=n, n_trials=int(g.n_trials.sum())))
    both = sorted(D[D.trials == "Other leg"].session.unique())          # the sessions with both groups
    pw = []
    for sig in SIG:
        g = D[(D.signal == sig) & D.session.isin(both)].pivot_table(index=["session", "cycle"], columns="trials", values="value").reset_index()
        cyc = sorted(g.cycle.unique())
        ps = np.array([stats.wilcoxon(q["Typical"], q["Other leg"]).pvalue for _, q in g.groupby("cycle")])
        qs = bh(ps)
        pw += [dict(signal=sig, cycle=c, p=p_, pFDR=q_, n_sessions=g[g.cycle == c].session.nunique()) for c, p_, q_ in zip(cyc, ps, qs)]
    write_table(pd.DataFrame(tr_rows), "lead_leg_traces.csv", index=False, float_format="%.6g")
    write_table(pd.DataFrame(pw), "lead_leg_pointwise.csv", index=False, float_format="%.6g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sessions", "summary"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "sessions":
        sessions_stage(a.workers)
    else:
        summary()


if __name__ == "__main__":
    main()
