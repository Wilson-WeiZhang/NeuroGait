"""Accuracy equation: how a joint's decoding accuracy follows from its harmonic composition and the decoded phase agreement.

Per trial and joint, on the scored rows (every 5th sample), the reference angle Y and the decoded angle C are fitted by least
squares on 1, cos k phi, sin k phi (k = 1, 2, 3), phi = the unwrapped Hilbert phase of the left-hip angle band-passed around the
session's stride frequency (third-order Butterworth, 0.7-1.3 x f0, zero phase; right hip when the left is not recorded). The
trial's Pearson r then splits exactly into r = rho1 + rho2 + rho3 + rho_cross + rho_res, with sY_k, sC_k the share of variance of
Y and C in harmonic k and cos_k the cosine of the phase difference of C against Y at k. Session x joint cells are the means over
scored trials. Three predictions without fitted parameters are added per cell:
  P_own     sum_k sY_k cos_k                      (own composition, own phase agreement)
  P_shared  sum_k sY_k mean over live joints of cos_k (own composition, one phase agreement per harmonic)
  P_wrong   P_shared with the composition of the next joint on the same side (hip -> knee -> ankle -> hip)
Healthy imagined walking (IW, 80 sessions), healthy real walking (RW, same sessions) and post-operative patients (PAT, 11 sessions).

The plotted fit of observed r on P_shared (IW and PAT): least-squares line on 60 grid points, a 95 % band from 2000 bootstrap
resamples of cells (default_rng(101) for IW, 102 for PAT), r squared and slope.

Named inputs: iw_dumps, rw_dumps, patient_dumps (decoder output folders, one <session>_dump.h5 per session), scoring_mask.csv.
Named outputs (under $NEUROGAIT_OUT/derived): equation_cells.csv, equation_fit.csv, equation_stat.csv.
Usage: python accuracy_equation.py [--workers N]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from analysis_common import inp, write_table

FS = 250.0
STEP = 5
JN = ("LHip", "LKnee", "LAnk", "RHip", "RKnee", "RAnk")
NEXT = {0: 1, 1: 2, 2: 0, 3: 4, 4: 5, 5: 3}
SETS = (("IW", "iw_dumps", "healthy", "IW"), ("RW", "rw_dumps", "healthy", "RW"), ("PAT", "patient_dumps", "postop", "IW"))


def stride_freq(x):
    x = x - x.mean()
    n = 1 << 15
    f = np.fft.rfftfreq(n, 1 / FS)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= 0.45) & (f <= 1.15)
    return f[band][np.argmax(p[band])]


def one_trial(C, Y, phi, live):
    rows = []
    ok_r = np.isfinite(C[:, live]).all(axis=1) & np.isfinite(Y[:, live]).all(axis=1)
    idx = np.flatnonzero(ok_r)
    if len(idx) < 2 * FS:
        return rows
    idx = idx[::STEP]
    p = phi[idx]
    X = np.column_stack([np.ones(len(p))] + [f(k * p) for k in (1, 2, 3) for f in (np.cos, np.sin)])
    Pinv = np.linalg.pinv(X)
    for j in live:
        y, c = Y[idx, j], C[idx, j]
        by, bc = Pinv @ y, Pinv @ c
        fy, fc = X @ by, X @ bc
        ey, ec = y - fy, c - fc
        vy, vc = np.var(y), np.var(c)
        if vy <= 0 or vc <= 0:
            continue
        den = np.sqrt(vy * vc)
        r = np.mean((y - y.mean()) * (c - c.mean())) / den
        rec = dict(joint=j, r=r, rho_res=np.mean(ey * ec) / den)
        parts_y, parts_c = [], []
        tot = 0.0
        for q, k in enumerate((1, 2, 3)):
            a, b = 1 + 2 * q, 3 + 2 * q
            py, pc = X[:, a: b] @ by[a: b], X[:, a: b] @ bc[a: b]
            parts_y.append(py - py.mean())
            parts_c.append(pc - pc.mean())
            rec[f"rho{k}"] = np.mean(parts_y[q] * parts_c[q]) / den
            tot += rec[f"rho{k}"]
            rec[f"sY{k}"] = np.mean(parts_y[q] ** 2) / vy
            rec[f"sC{k}"] = np.mean(parts_c[q] ** 2) / vc
            zy, zc = complex(by[a], -by[a + 1]), complex(bc[a], -bc[a + 1])
            rec[f"cos{k}"] = np.cos(np.angle(zc * np.conj(zy)))
        fy0, fc0 = fy - fy.mean(), fc - fc.mean()
        rec["rho_cross"] = np.mean(fy0 * fc0) / den - tot
        rows.append(rec)
    return rows


def session(job):
    tag, fp, keep = job
    with h5py.File(fp, "r") as h:
        s = h.attrs["session"]
        s = s.decode() if isinstance(s, bytes) else str(s)
        Cc, Yy = h["rows/C_pred"][:], h["rows/Y_true"][:]
        tid, off, nr = h["trials/trial_idx"][:], h["trials/row_offset"][:], h["trials/n_rows"][:]
    Tr = [(int(t), Cc[o: o + n], Yy[o: o + n]) for t, o, n in zip(tid, off, nr) if (s, int(t)) in keep and n >= 3 * FS]
    if len(Tr) < 5:
        return []
    live = [j for j in range(6) if np.isfinite(Yy[:, j]).mean() > 0.5 and np.isfinite(Cc[:, j]).mean() > 0.3]
    hip = 0 if 0 in live else 3
    f0 = float(np.median([stride_freq(np.nan_to_num(t[2][:, hip])) for t in Tr]))
    sos = signal.butter(3, [0.7 * f0, 1.3 * f0], btype="band", fs=FS, output="sos")
    rows = []
    for ti, C, Y in Tr:
        y = Y[:, hip]
        ok = np.isfinite(y)
        yy = y.copy()
        yy[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), y[ok]) if (~ok).any() else 0
        phi = np.unwrap(np.angle(signal.hilbert(signal.sosfiltfilt(sos, yy - yy.mean()))))
        for r in one_trial(C, Y, phi, live):
            r.update(set=tag, session=s, trial=ti)
            rows.append(r)
    return rows


def predictions(C):
    """C: cells of one set, session x joint with the session means."""
    out = []
    for s, g in C.groupby("session"):
        g = g.set_index("joint")
        cs = {k: g[f"cos{k}"].mean() for k in (1, 2, 3)}
        for j, r in g.iterrows():
            w = NEXT[j] if NEXT[j] in g.index else (NEXT[NEXT[j]] if NEXT[NEXT[j]] in g.index else j)
            out.append(dict(session=s, joint=j, P_own=sum(r[f"sY{k}"] * r[f"cos{k}"] for k in (1, 2, 3)),
                            P_shared=sum(r[f"sY{k}"] * cs[k] for k in (1, 2, 3)),
                            P_wrong=sum(g.loc[w, f"sY{k}"] * cs[k] for k in (1, 2, 3))))
    return C.merge(pd.DataFrame(out), on=["session", "joint"])


def cells(workers):
    mask = pd.read_csv(inp("scoring_mask.csv"))
    jobs = []
    for tag, folder, cohort, cond in SETS:
        m = mask[(mask.cohort.str.startswith(cohort)) & (mask.condition == cond) & (mask.keep == 1)]
        keep = set(zip(m.session, m.trial_idx.astype(int)))
        for fp in sorted(inp(folder).glob("*_dump.h5")):
            jobs.append((tag, fp, keep))
    with Pool(workers) as p:
        res = p.map(session, jobs, chunksize=1)
    A = pd.DataFrame([r for x in res for r in x])
    cols = ["r", "rho1", "rho2", "rho3", "rho_cross", "rho_res", "sY1", "sY2", "sY3", "sC1", "sC2", "sC3", "cos1", "cos2", "cos3"]
    Ce = A.groupby(["set", "session", "joint"])[cols].mean().reset_index()
    D = []
    for tag in ("IW", "RW", "PAT"):
        C = predictions(Ce[Ce.set == tag])
        C["set"] = tag
        D.append(C)
    D = pd.concat(D)
    D["joint_name"] = D.joint.map(lambda j: JN[j])
    return write_table(D, "equation_cells.csv", index=False, float_format="%.6g")


def fit_tables(C):
    fit, st = [], []
    for q, tag in enumerate(("IW", "PAT")):
        g = C[C.set == tag]
        x, y = g.P_shared.to_numpy(), g.r.to_numpy()
        gx = np.linspace(x.min(), x.max(), 60)
        rng = np.random.default_rng(101 + q)
        B = np.array([np.polyval(np.polyfit(x[i], y[i], 1), gx) for i in rng.integers(0, len(x), (2000, len(x)))])
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        line = np.polyval(np.polyfit(x, y, 1), gx)
        fit += [dict(set=tag, x=a, line=b, lo=c, hi=d) for a, b, c, d in zip(gx, line, lo, hi)]
        st.append(dict(set=tag, n=len(g), r_squared=np.corrcoef(x, y)[0, 1] ** 2, slope=np.polyfit(x, y, 1)[0]))
    write_table(pd.DataFrame(fit), "equation_fit.csv", index=False, float_format="%.6g")
    write_table(pd.DataFrame(st), "equation_stat.csv", index=False, float_format="%.6g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    p = cells(a.workers)
    fit_tables(pd.read_csv(p))    # the fit reads the cells as printed (6 significant digits)


if __name__ == "__main__":
    main()
