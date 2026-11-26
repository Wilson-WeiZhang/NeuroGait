"""Cross-state phase read-out: the real-walking scalp pattern of each harmonic read out on single trials of imagined walking and rest.

Stage `export` (per healthy session, from the preprocessed EEG / goniometer / event triple and the per-session harmonic bands):
  For k = 1, 2, 3 the 60 channels are band-passed in the session's band of harmonic k (fourth-order Butterworth, zero phase),
  u_c(t) = analytic signal / its modulus, kept at 50 Hz (every 5th sample). Phase phi: the walker's left hip for imagined
  walking, the participant's own left hip for real walking (second-order Butterworth in the k = 1 band, Hilbert phase), rest
  yoked to imagined walking (rest trial j takes imagined trial j mod n). Good trials of at least 2 s.
  Weights (matched filter, no free parameter): W = conj(mean over real-walking trials of mean_t u_c(t) exp(-i k phi(t))).
  Read-out: zhat(t) = sum_c W_c u_c(t), z_j = mean_t exp(i (angle zhat(t) - k phi(t))). A 10-fold partition by trial
  (default_rng(20261110 + digits of the session name), drawn once before the harmonic loop) gives the constant lag of a held-out
  trial from the sum of z over the training folds. Real-to-imagined (RW2IW) and real-to-rest (RW2REST) use the weights of all
  real-walking trials; real-to-real (RW2RW) is cross-validated by trial (weights and lag from the training folds).
  H: 24 x 24 histogram of (true phase, read-out phase - lag) over the trial samples.
Stage `summary` (from the merged per-session arrays):
  density = mean over the 80 sessions of each session's histogram normalised to mean 1 (Rest, Imag, Real; k1-k3);
  delta = per session, median over trials of Re(z_j exp(-i lag_j)); Imag vs Rest by paired two-sided Wilcoxon over sessions,
  Benjamini-Hochberg across k1, k2, k3.

Named inputs: eeg_triples (folder <session>/<session>_{eeg,goni,event}.mat), harmonic_bands.csv (export);
readout_phases.npz (summary; keys "<session>|<array>").
Named outputs: readout_parts/<session>.npz and readout_phases.npz (export), readout_density.csv (summary).
Usage: python readout_phase.py export [--workers N] [session ...]  |  python readout_phase.py summary
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import re
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal, stats

from analysis_common import inp, out_path, write_npz, write_table

STEP = 5
RNG_BASE = 20261110
EPS = np.finfo(float).eps
KS = (1, 2, 3)
N_BIN = 24


# ------------------------------------------------------------------------------------------------ inputs
def load_triple(sess):
    base = inp("eeg_triples") / sess
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


# ------------------------------------------------------------------------------------------------ read-out
def analytic(x, fs, band):
    sos = signal.butter(4, band, "bandpass", fs=fs, output="sos")
    za = signal.hilbert(signal.sosfiltfilt(sos, x, axis=0), axis=0)
    return za / np.maximum(np.abs(za), EPS)


def trial_phasors(u, spans, phases, k):
    """Uf: n x channels, mean over the trial of u exp(-i k phi); U, P: per-trial u and k phi at 50 Hz."""
    Uf = np.empty((len(spans), u.shape[1]), np.complex64)
    U, P = [], []
    for j, ((e0, L), p) in enumerate(zip(spans, phases)):
        Uf[j] = (u[e0: e0 + L] * np.exp(-1j * k * p)[:, None]).mean(axis=0)
        U.append(u[e0: e0 + L][::STEP].copy())
        P.append(k * p[::STEP])
    return dict(Uf=Uf, U=U, P=P)


def read(U, P, W, idx=None):
    idx = range(len(U)) if idx is None else idx
    z = []
    for j in idx:
        v = U[j] @ W
        z.append((v / np.maximum(np.abs(v), EPS) * np.exp(-1j * P[j])).mean())
    return np.array(z)


def lag_folds(z, fold):
    tot = z.sum()
    lag = np.empty(len(z))
    for f in np.unique(fold):
        lag[fold == f] = np.angle(tot - z[fold == f].sum())
    return lag


def hist2d(U, P, W, lag):
    H = np.zeros((N_BIN, N_BIN))
    for j in range(len(U)):
        v = U[j] @ W
        act = np.mod(P[j], 2 * np.pi)
        est = np.mod(np.angle(v) - lag[j], 2 * np.pi)
        H += np.histogram2d(act, est, bins=N_BIN, range=[[0, 2 * np.pi], [0, 2 * np.pi]])[0]
    return H.astype(np.uint32)


def block(k, T, fold, out):
    RW = T["RW"]
    W_rw = np.conj(RW["Uf"].mean(axis=0))
    out[f"W_k{k}"] = W_rw.astype(np.complex64)
    for c in ("IW", "REST"):
        z = read(T[c]["U"], T[c]["P"], W_rw)
        lag = lag_folds(z, fold[c])
        out[f"z_RW2{c}_k{k}"], out[f"lag_RW2{c}_k{k}"] = z.astype(np.complex64), lag.astype(np.float32)
        out[f"H_RW2{c}_k{k}"] = hist2d(T[c]["U"], T[c]["P"], W_rw, lag)
    # real walking on itself, cross-validated by trial
    n = RW["Uf"].shape[0]
    z, lag = np.empty(n, complex), np.empty(n)
    H = np.zeros((N_BIN, N_BIN))
    for f in np.unique(fold["RW"]):
        tr, te = np.flatnonzero(fold["RW"] != f), np.flatnonzero(fold["RW"] == f)
        W = np.conj(RW["Uf"][tr].mean(axis=0))
        z[te] = read(RW["U"], RW["P"], W, idx=te)
        lag[te] = np.angle(read(RW["U"], RW["P"], W, idx=tr).sum())
    for f in np.unique(fold["RW"]):
        te = np.flatnonzero(fold["RW"] == f)
        W = np.conj(RW["Uf"][np.flatnonzero(fold["RW"] != f)].mean(axis=0))
        H += hist2d([RW["U"][j] for j in te], [RW["P"][j] for j in te], W, lag[te])
    out[f"z_RW2RW_k{k}"], out[f"lag_RW2RW_k{k}"] = z.astype(np.complex64), lag.astype(np.float32)
    out[f"H_RW2RW_k{k}"] = H.astype(np.uint32)


def one_session(sess):
    part = out_path("readout_parts") / f"{sess}.npz"
    if part.exists():
        return sess
    part.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_BASE + int(re.sub(r"\D", "", sess)))
    x, fs, labels, gx, gfs, gl, ev = load_triple(sess)
    assert x.shape[1] == len(labels) == 60 and fs == gfs == 250.0, (sess, x.shape, fs, gfs)
    x = x.astype(float)
    b = pd.read_csv(inp("harmonic_bands.csv"))
    b = b[b.session == sess].set_index("harmonic_k")
    bands = {k: [float(b.loc[k].band_lo), float(b.loc[k].band_hi)] for k in KS}
    phi_w = goni_phase(gx, gl, "LHipW X", *bands[1], gfs)
    phi_s = goni_phase(gx, gl, "LHipS X", *bands[1], gfs)
    sp_iw, up_iw = [], []
    for e in trials_of(ev, "gaitimagery"):
        s = span(e, x.shape[0], len(phi_w))
        if s[2] >= 2 * fs:
            sp_iw.append((s[0], s[2]))
            up_iw.append(np.unwrap(phi_w[s[1]: s[1] + s[2]]))
    sp_rw = [s for s in (span(e, x.shape[0], len(phi_s)) for e in trials_of(ev, "gaitexecution")) if s[2] >= 2 * fs]
    up_rw = [np.unwrap(phi_s[g0: g0 + L]) for _, g0, L in sp_rw]
    sp_re, up_re = [], []
    for j, e in enumerate(trials_of(ev, "rest")):
        e0, _, L = span(e, x.shape[0], len(phi_w))
        p = up_iw[j % len(up_iw)]
        n = min(L, len(p))
        if n >= 2 * fs:
            sp_re.append((e0, n))
            up_re.append(p[:n])
    sets = {"RW": ([(e0, L) for e0, _, L in sp_rw], up_rw), "IW": (sp_iw, up_iw), "REST": (sp_re, up_re)}
    out = {"labels": np.array(labels)}
    fold = {c: rng.permutation(len(sets[c][0])) % 10 for c in sets}
    for c in sets:
        out[f"fold_{c}"] = fold[c].astype(np.int8)
    for k in KS:
        u = analytic(x, fs, bands[k])
        T = {c: trial_phasors(u, spans, phases, k) for c, (spans, phases) in sets.items()}
        del u
        block(k, T, fold, out)
    tmp = part.with_name(part.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, part)
    return sess


def export(workers, names):
    all_ = sorted(pd.read_csv(inp("harmonic_bands.csv")).session.unique())
    with Pool(workers) as p:
        for s in p.imap_unordered(one_session, names or all_):
            print("done", s, flush=True)
    if not names:
        d = out_path("readout_parts")
        write_npz("readout_phases.npz", **{f"{s}|{k}": v for s in all_ for k, v in np.load(d / f"{s}.npz").items()})


# ------------------------------------------------------------------------------------------------ summary
def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r = np.empty(len(p))
    r[o] = np.minimum(q, 1)
    return r


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def summary():
    Z = np.load(inp("readout_phases.npz"))
    sess = sorted({k.split("|")[0] for k in Z.files})
    assert len(sess) == 80
    tags = (("Rest", "RW2REST"), ("Imag", "RW2IW"), ("Real", "RW2RW"))
    D = {(c, k): np.mean([Z[f"{s}|H_{t}_k{k}"] / Z[f"{s}|H_{t}_k{k}"].sum() * N_BIN ** 2 for s in sess], axis=0) for c, t in tags for k in KS}
    delta = {(t, k): pd.Series({s: np.median((Z[f"{s}|z_{t}_k{k}"].astype(complex) * np.exp(-1j * Z[f"{s}|lag_{t}_k{k}"].astype(float))).real)
                                for s in sess}) for _, t in tags for k in KS}
    q = bh([stats.wilcoxon(delta["RW2IW", k], delta["RW2REST", k]).pvalue for k in KS])
    name = "readout_density"
    rows = [dict(panel=name, row_type="density", group=c, harmonic_k=k, walker_bin=a, reconstructed_bin=b, density=D[c, k][a, b])
            for c, _ in tags for k in KS for a in range(N_BIN) for b in range(N_BIN)]
    for i, k in enumerate(KS):
        for g, t in (("Rest, RW weights", "RW2REST"), ("Imag, RW weights", "RW2IW")):
            v = delta[t, k]
            rows += [dict(panel=name, row_type="value", group=g, unit=s, value=float(v[s]), harmonic_k=k) for s in sess]
        rows.append(dict(panel=name, row_type="stat", group="Imag vs Rest", p=float(q[i]), mark=star(q[i]),
                         test="paired Wilcoxon over sessions, two-sided, BH across k1, k2, k3", harmonic_k=k))
    write_table(pd.DataFrame(rows), "readout_density.csv", index=False, float_format="%.6g", lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("export", "summary"))
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "export":
        export(a.workers, a.sessions)
    else:
        summary()


if __name__ == "__main__":
    main()
