"""Speed tuning of gait-phase locking |c| in the healthy sessions, with responder groups defined on one half of the trials.

Single-trial projection.  The EEG of a channel cluster (k1: left + right sensorimotor, k2: midline) is band-passed (4th-order
Butterworth, zero phase, k f0 x [0.80, 1.20] widened by 0.18 Hz, low edge at least 0.2 Hz), turned into a unit phasor (Hilbert)
and projected on exp(-i k (phi0 + s (phi - phi0))), phi = the unwrapped phase of the walker's left hip in its k1 band, phi0 = its
value at the trial onset, s = speed scale 0.80 to 1.20 in steps of 0.02.  A[s, trial, channel] = mean over the trial.
|c|(s; trials) = mean over cluster channels of |mean over the trials of A|.  Imagined-walking trials: the good trials of at least
2 s; rest trials borrow the walker phase of imagined trial (j mod n), cut to the shorter length, at least 2 s.
Step waveforms.  Mean time-normalized EEG between a left-hip and the next right-hip flexion peak (left start) and the reverse
(right start), steps of 0.3-0.9 s, 100 points, all channels; on the wide-band EEG and with the k1 band removed (band-stop).

Groups (stage `groups`).  Trials numbered by time within the session, odd = 1st, 3rd, ... and even = 2nd, 4th, ....  A session is
a responder in a half when |c|(1.0) at k1 on that half exceeds the 95th percentile (numpy linear) of 200 phase-shift surrogates
on the same trials (each trial's projection at s = 1.0 rotated by its own uniform angle, shared across channels; seed
[20261105, digits of the session name, 1 for odd or 2 for even]).  The curves of a group come from the other half (two
directions, odd -> even and even -> odd; drawn curve = mean of the two directions' group medians, its SE = half the root sum of
squares of the two SEs).  All-session curve = per session the mean of its odd and even half curves, median over sessions; rest
= the yoked rest curve.  SE of a median = 1.253 sd / sqrt(n).  Tests: held-out |c|(1.0) against |c|(0.8), |c|(1.2) and the
session's yoked rest |c|(1.0), paired Wilcoxon, two-sided; BH over the six responder k1 tests and separately over the other 18;
the same three contrasts on the all-session curves (BH over 6); responder against non-responder on held-out |c|(1.0),
Mann-Whitney two-sided (BH over 4).

Named inputs: healthy_prep_triples (per-session EEG, goniometer and event exports, zero-phase preprocessing), gait_peak_bands.csv
(per-session stride frequency and k bands of the walker's goniometer); for `stack` and `groups`: speed_tuning_parts.npz,
responder_table.csv, speed_tuning_trials.npz, speed_tuning_sessions.npz.
Named outputs: speed_tuning_trials.npz (keys <session>|A_<cond>_k<k>, c_<cond>_k<k>, rank_iw, f0) and speed_tuning_parts.npz
(stage `sessions`); speed_tuning_sessions.npz (stage `stack`: sessions, resp, scales, c_<cond>_k<k> and trial_amp_<cond>_k<k> as
sessions x scales, step_diff_<cond>_<wide|nok1>_<left|right|mid> = left-start minus right-start waveform, cluster mean, sessions x
100); speed_tuning_curves.csv (stage `groups`).
Usage: python speed_tuning.py sessions [--workers N] | stack | groups
"""
import argparse
import os
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal, stats

from analysis_common import OUT, inp, write_npz, write_table

SCALES = np.round(np.arange(0.80, 1.2001, 0.02), 2)
MID = ["FCz", "FC1", "FC2", "Cz", "C1", "C2", "CPz", "CP1", "CP2"]
LEFT = ["FC3", "FC1", "C3", "C1", "CP3", "CP1"]
RIGHT = ["FC4", "FC2", "C4", "C2", "CP4", "CP2"]
CLUS = {1: LEFT + RIGHT, 2: MID}
CL = {"left": LEFT, "right": RIGHT, "mid": MID}
NPTS = 100
I08, I10, I12 = 0, 10, 20
SEED, B_NULL = 20261105, 200
PARTS = OUT / "speed_tuning_parts"


# ---- session stage: loaders --------------------------------------------------------------------------------------------
def load_triple(sess, root):
    base = root / sess
    with h5py.File(base / f"{sess}_eeg.mat") as h:
        g = h["eeg"]
        x = np.asarray(g["x"])
        fs = float(np.asarray(g["s"]).squeeze())
        labels = ["".join(map(chr, np.asarray(h[r]).squeeze())) for r in np.asarray(g["c"]["labels"]).squeeze()]
    with h5py.File(base / f"{sess}_goni.mat") as h:
        g = h["goni"]
        gx = np.asarray(g["x"])
        gfs = float(np.asarray(g["s"]).squeeze())
        glabels = ["".join(map(chr, np.asarray(h[r]).squeeze())) for r in np.asarray(g["c"]).squeeze()]
    if gx.shape[1] != len(glabels):
        gx = gx.T
    ev = sio.loadmat(base / f"{sess}_event.mat", squeeze_me=True, struct_as_record=False)["event"]
    return x, fs, labels, gx, gfs, glabels, ev


def fill(g):
    g = g.astype(float)
    bad = ~np.isfinite(g)
    if bad.any():
        g[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), g[~bad])
    return g


def goni_phase(gx, glabels, name, lo, hi, gfs):
    sos = signal.butter(2, [lo, hi], "bandpass", fs=gfs, output="sos")
    return np.angle(signal.hilbert(signal.sosfiltfilt(sos, fill(gx[:, glabels.index(name)]))))


def trials_of(ev, code):
    return [e for e in ev if str(e.code) == code and str(e.qc) == "good"]


def span(e, n_eeg, n_goni):
    et0, et1, gt0, gt1 = int(e.etstart), int(e.etstop), int(e.gtstart), int(e.gtstop)
    L = min(et1 - et0 + 1, gt1 - gt0 + 1, n_eeg - et0 + 1, n_goni - gt0 + 1)
    return et0 - 1, gt0 - 1, L


def unit_of(z):
    return z / np.maximum(np.abs(z), np.finfo(float).eps)


def bp(x, fs, band):
    return signal.sosfiltfilt(signal.butter(4, band, "bandpass", fs=fs, output="sos"), x, axis=0)


# ---- session stage: measures ---------------------------------------------------------------------------------------------
def projections(x, fs, lab, f0, sets):
    """Single-trial projections A[scale, trial, channel] per condition and harmonic."""
    out = {}
    for k in (1, 2):
        ch = lab.get_indexer(CLUS[k])
        band = [max(k * f0 * SCALES[0] - 0.18, 0.2), k * f0 * SCALES[-1] + 0.18]
        u = unit_of(signal.hilbert(bp(x[:, ch], fs, band), axis=0))
        for cond, (sp, ph) in sets.items():
            A = np.zeros((len(SCALES), len(sp), len(ch)), complex)
            for j, ((e0, _, L), p) in enumerate(zip(sp, ph)):
                seg = u[e0: e0 + L]
                for i, s in enumerate(SCALES):
                    A[i, j] = (seg * np.exp(-1j * k * (p[0] + s * (p - p[0])))[:, None]).mean(axis=0)
            out[cond, k] = A
    return out


def peaks(p):
    """Samples where the unwrapped phase passes a multiple of 2 pi: the hip-flexion peak, phase 0 of its band-passed trajectory."""
    return np.flatnonzero(np.diff(np.floor(p / (2 * np.pi))) > 0) + 1


def step_waveforms(x, fs, band_k1, trials_lr):
    """Mean time-normalized waveform of left-start and right-start steps, all channels."""
    out = {}
    stop = signal.butter(4, band_k1, "bandstop", fs=fs, output="sos")
    ver = {"wide": x, "nok1": signal.sosfiltfilt(stop, x, axis=0)}
    grid = np.linspace(0, 1, NPTS)
    for cond, items in trials_lr.items():
        for v, data in ver.items():
            acc = {"L": [], "R": []}
            for e0, pl, pr in items:
                evs = sorted([(i, "L") for i in peaks(pl)] + [(i, "R") for i in peaks(pr)])
                for (a, lab_a), (b, lab_b) in zip(evs, evs[1:]):
                    if lab_a == lab_b or not (0.3 * fs <= b - a <= 0.9 * fs):
                        continue
                    seg = data[e0 + a: e0 + b]
                    seg = seg - seg.mean(axis=0)
                    t = np.linspace(0, 1, seg.shape[0])
                    acc[lab_a].append(np.stack([np.interp(grid, t, seg[:, c]) for c in range(seg.shape[1])], axis=1))
            out[f"step_{cond}_{v}_L"] = np.mean(acc["L"], axis=0) if acc["L"] else np.full((NPTS, x.shape[1]), np.nan)
            out[f"step_{cond}_{v}_R"] = np.mean(acc["R"], axis=0) if acc["R"] else np.full((NPTS, x.shape[1]), np.nan)
    return out


def peak_bands():
    return pd.read_csv(inp("gait_peak_bands.csv"))


def one_session(sess):
    part = PARTS / f"{sess}.npz"
    if part.exists():
        return sess, "skipped", 0.0
    t0 = time.time()
    x, fs, labels, gx, gfs, glabels, ev = load_triple(sess, inp("healthy_prep_triples"))
    x = x.astype(float)
    lab = pd.Index(labels)
    b = peak_bands()
    b = b[b.session == sess].set_index("harmonic_k")
    bb1 = [b.loc[1].band_lo, b.loc[1].band_hi]
    f0 = float(b.loc[1].peak_hz)
    phiL = goni_phase(gx, glabels, "LHipW X", *bb1, gfs)
    keep = [e for e in trials_of(ev, "gaitimagery") if span(e, x.shape[0], len(phiL))[2] >= 2 * fs]
    sp_iw = [span(e, x.shape[0], len(phiL)) for e in keep]
    ph_iw = [np.unwrap(phiL[g0: g0 + L]) for _, g0, L in sp_iw]
    phr_iw = [p - np.pi for p in ph_iw]                  # the right-hip channel runs in phase with the left (sign convention): the second half-cycle of the left hip marks the right step
    sp_rest, ph_rest, phr_rest = [], [], []
    for j, t in enumerate(trials_of(ev, "rest")):        # rest trials borrow the walker phase of an imagined-walking trial
        e0, _, L = span(t, x.shape[0], len(phiL))
        i = j % len(ph_iw)
        n = min(L, len(ph_iw[i]))
        if n >= 2 * fs:
            sp_rest.append((e0, 0, n))
            ph_rest.append(ph_iw[i][:n])
            phr_rest.append(phr_iw[i][:n])
    A = projections(x, fs, lab, f0, {"IW": (sp_iw, ph_iw), "REST": (sp_rest, ph_rest)})
    out = {"labels": np.array(labels), "f0": np.array(f0)}
    for (cond, k), a in A.items():
        out[f"A_{cond}_k{k}"] = a.astype(np.complex64)
        out[f"c_{cond}_k{k}"] = np.abs(a.mean(axis=1)).mean(axis=1)
        out[f"trial_amp_{cond}_k{k}"] = np.abs(a).mean(axis=(1, 2))       # mean single-trial projection amplitude
    out["rank_iw"] = np.argsort(np.argsort([int(e.etstart) for e in keep]))    # 0 = first in time; odd trials (1st, 3rd, ...) have even rank
    out.update(step_waveforms(x, fs, bb1, {"IW": [(e0, p, q) for (e0, _, _), p, q in zip(sp_iw, ph_iw, phr_iw)],
                                           "REST": [(e0, p, q) for (e0, _, _), p, q in zip(sp_rest, ph_rest, phr_rest)]}))
    tmp = PARTS / f"{sess}.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, part)
    return sess, "done", time.time() - t0


def sessions_stage(workers, only):
    PARTS.mkdir(parents=True, exist_ok=True)
    sessions = only or sorted(peak_bands().session.unique())
    with Pool(workers) as p:
        for i, (s, st, dt) in enumerate(p.imap_unordered(one_session, sessions), 1):
            print(f"{i}/{len(sessions)} {s} {st} {dt:.1f} s", flush=True)
    if only:
        return
    TRIAL_KEYS = ("rank_iw", "f0") + tuple(f"{p}_{c}_k{k}" for p in ("A", "c") for c in ("IW", "REST") for k in (1, 2))
    PART_ONLY = ("rank_iw",) + tuple(f"A_{c}_k{k}" for c in ("IW", "REST") for k in (1, 2))
    parts = {s: dict(np.load(PARTS / f"{s}.npz")) for s in sessions}
    write_npz("speed_tuning_trials.npz", **{f"{s}|{k}": v for s, d in parts.items() for k, v in d.items() if k in TRIAL_KEYS})
    write_npz("speed_tuning_parts.npz", **{f"{s}|{k}": v for s, d in parts.items() for k, v in d.items() if k not in PART_ONLY})


# ---- stack stage: per-session curves and step waveforms, one array per measure ------------------------------------------
def stack_stage():
    z = np.load(inp("speed_tuning_parts.npz"))
    D = {}
    for k in z.files:
        s, q = k.split("|", 1)
        D.setdefault(s, {})[q] = z[k]
    g = pd.read_csv(inp("responder_table.csv")).set_index("session").resp.astype(bool)
    ss = list(D)
    curves = {f"{kind}_{c}_k{k}": np.stack([D[s][f"{kind}_{c}_k{k}"] for s in ss]) for k in (1, 2) for kind in ("c", "trial_amp") for c in ("IW", "REST")}
    lab = list(D[ss[0]]["labels"])
    W = {}
    for cond in ("IW", "REST"):
        for v in ("nok1", "wide"):
            for cl, names in CL.items():
                ch = [lab.index(c) for c in names]
                W[f"step_diff_{cond}_{v}_{cl}"] = np.stack([(D[s][f"step_{cond}_{v}_L"][:, ch] - D[s][f"step_{cond}_{v}_R"][:, ch]).mean(1) for s in ss])
    resp = np.array([g[s] for s in ss])
    write_npz("speed_tuning_sessions.npz", resp=resp, sessions=np.array(ss), scales=SCALES, **curves, **W)


# ---- groups stage ----------------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r = np.empty(len(p))
    r[o] = np.minimum(q, 1)
    return r


def wil2(a, b):
    return float(stats.wilcoxon(a, b).pvalue)


def se_med(A):
    return 1.253 * A.std(axis=0, ddof=1) / np.sqrt(len(A))


def groups_stage():
    z = np.load(inp("speed_tuning_trials.npz"))
    sess = [str(s) for s in np.load(inp("speed_tuning_sessions.npz"))["sessions"]]
    n = len(sess)
    C = {(h, k): np.zeros((n, len(SCALES))) for h in ("odd", "even") for k in (1, 2)}
    Rest = {k: np.zeros((n, len(SCALES))) for k in (1, 2)}
    AllT = {k: np.zeros((n, len(SCALES))) for k in (1, 2)}
    lab = {h: np.zeros(n, bool) for h in ("odd", "even")}
    p95 = {h: np.zeros(n) for h in ("odd", "even")}
    n_half = {h: np.zeros(n, int) for h in ("odd", "even")}
    n_rest = np.zeros(n, int)
    for i, s in enumerate(sess):
        rank = z[f"{s}|rank_iw"]
        mask = {"odd": rank % 2 == 0, "even": rank % 2 == 1}
        for k in (1, 2):
            A = z[f"{s}|A_IW_k{k}"].astype(complex)
            AllT[k][i] = z[f"{s}|c_IW_k{k}"]
            Rest[k][i] = z[f"{s}|c_REST_k{k}"]
            for h, m in mask.items():
                C[h, k][i] = np.abs(A[:, m].mean(axis=1)).mean(axis=1)
            if k == 1:
                a10 = A[I10]
        n_rest[i] = z[f"{s}|A_REST_k1"].shape[1]
        dig = int("".join(c for c in s if c.isdigit()))
        for h, idx in (("odd", 1), ("even", 2)):
            m = mask[h]
            rng = np.random.default_rng([SEED, dig, idx])
            d = rng.uniform(0, 2 * np.pi, (B_NULL, int(m.sum())))
            nul = np.abs(np.exp(-1j * d) @ a10[m] / m.sum()).mean(axis=1)
            p95[h][i] = np.percentile(nul, 95)
            lab[h][i] = C[h, 1][i, I10] > p95[h][i]
            n_half[h][i] = m.sum()
    # per direction: groups from one half, curves from the other
    DIRS = (("O>E", "odd", "even"), ("E>O", "even", "odd"))
    GROUPS = (("Resp", True), ("Non-resp", False))
    val, tests = {}, []
    for dn, lh, hh in DIRS:
        for g, gv in GROUPS:
            sel = lab[lh] == gv
            for k in (1, 2):
                H_ = C[hh, k][sel]
                med, se = np.median(H_, axis=0), se_med(H_)
                val[dn, g, k] = (med, se, H_, sel)
                pk = SCALES[med.argmax()]
                for con, b in (("vs s = 0.8", H_[:, I08]), ("vs s = 1.2", H_[:, I12]), ("vs yoked rest at 1.0", Rest[k][sel, I10])):
                    tests.append(dict(direction=dn, group=g, k=k, contrast=con, n=int(sel.sum()), median_diff=float(np.median(H_[:, I10] - b)), p=wil2(H_[:, I10], b), peak_scale=float(pk)))
    T = pd.DataFrame(tests)
    prim = (T.group == "Resp") & (T.k == 1)
    T.loc[prim, "pFDR"] = bh(T[prim].p.to_numpy())
    T.loc[~prim, "pFDR"] = bh(T[~prim].p.to_numpy())
    # all sessions, half footing
    H80 = {k: 0.5 * (C["odd", k] + C["even", k]) for k in (1, 2)}
    t80 = []
    for k in (1, 2):
        for con, b in (("vs s = 0.8", H80[k][:, I08]), ("vs s = 1.2", H80[k][:, I12]), ("vs yoked rest at 1.0", Rest[k][:, I10])):
            t80.append(dict(k=k, contrast=con, median_diff=float(np.median(H80[k][:, I10] - b)), p=wil2(H80[k][:, I10], b)))
    t80 = pd.DataFrame(t80)
    t80["pFDR"] = bh(t80.p.to_numpy())
    # responder against non-responder on held-out trials
    tg = []
    for dn, lh, hh in DIRS:
        for k in (1, 2):
            a, b = C[hh, k][lab[lh], I10], C[hh, k][~lab[lh], I10]
            tg.append(dict(direction=dn, k=k, resp=float(np.median(a)), non=float(np.median(b)), p=float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)))
    tg = pd.DataFrame(tg)
    tg["pFDR"] = bh(tg.p.to_numpy())
    # drawn curves
    drawn = {}
    for k in (1, 2):
        for g, _ in GROUPS:
            m1, s1 = val["O>E", g, k][:2]
            m2, s2 = val["E>O", g, k][:2]
            drawn[g, k] = (0.5 * (m1 + m2), 0.5 * np.sqrt(s1 ** 2 + s2 ** 2))
        drawn["All 80", k] = (np.median(H80[k], axis=0), se_med(H80[k]))
        drawn["Rest", k] = (np.median(Rest[k], axis=0), se_med(Rest[k]))
    name = "speed_tuning"
    rows = []
    for k in (1, 2):
        for g in ("Rest", "Non-resp", "Resp", "All 80"):
            rows += [dict(panel=name, row_type="curve", harmonic_k=k, group=g, direction="drawn", scale=a, median_c=b, se=c) for a, b, c in zip(SCALES, *drawn[g, k])]
        for dn, _, _ in DIRS:
            for g, _ in GROUPS:
                rows += [dict(panel=name, row_type="curve", harmonic_k=k, group=g, direction=dn, scale=a, median_c=b, se=c) for a, b, c in zip(SCALES, val[dn, g, k][0], val[dn, g, k][1])]
        rows += [dict(panel=name, row_type="curve", harmonic_k=k, group="All 80, all trials", direction="none", scale=a, median_c=b) for a, b in zip(SCALES, np.median(AllT[k], axis=0))]
    rows += [dict(panel=name, row_type="stat", harmonic_k=r.k, group=r.group, direction=r.direction, contrast=r.contrast, n=r.n, median_diff=r.median_diff, p=r.p, pFDR=r.pFDR, test="paired Wilcoxon, two-sided, held-out trials") for r in T.itertuples()]
    rows += [dict(panel=name, row_type="stat", harmonic_k=r.k, group="All 80", direction="none", contrast=r.contrast, n=80, median_diff=r.median_diff, p=r.p, pFDR=r.pFDR, test="paired Wilcoxon, two-sided, mean of the odd and even half curves") for r in t80.itertuples()]
    rows += [dict(panel=name, row_type="stat", harmonic_k=r.k, group="Resp vs Non-resp", direction=r.direction, contrast="held-out |c| at 1.0", resp=r.resp, non=r.non, p=r.p, pFDR=r.pFDR, test="Mann-Whitney, two-sided") for r in tg.itertuples()]
    rows += [dict(panel=name, row_type="session", unit=s, responder_odd=bool(lab["odd"][i]), responder_even=bool(lab["even"][i]), c_odd_k1=C["odd", 1][i, I10], c_even_k1=C["even", 1][i, I10], p95_odd=p95["odd"][i], p95_even=p95["even"][i], n_odd=int(n_half["odd"][i]), n_even=int(n_half["even"][i]), n_rest=int(n_rest[i])) for i, s in enumerate(sess)]
    write_table(pd.DataFrame(rows), "speed_tuning_curves.csv", index=False, float_format="%.6g", lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sessions", "stack", "groups"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("session", nargs="*", help="sessions stage only: these sessions, no merge")
    a = ap.parse_args()
    if a.stage == "sessions":
        sessions_stage(a.workers, a.session)
    elif a.stage == "stack":
        stack_stage()
    else:
        groups_stage()


if __name__ == "__main__":
    main()
