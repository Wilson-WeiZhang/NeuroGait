"""Figure 7 — physics simulation and walk-intent gate

gate_features — session loader, decision windows, band power and calibration split of the walk-intent gate.

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import numpy as np
import scipy.io as sio
import h5py
from scipy.signal import welch

SESSIONS = ["P1_sess04", "P1_sess06", "P1_sess07", "P1_sess09",
            "P2_sess03", "P2_sess05", "P2_sess06"]
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "mu": (8.0, 13.0), "beta": (13.0, 30.0)}
WIN_S = 3.0
BASE_N = 4
FS_EXPECT = 250.0


def load_session(root, sess):
    d = os.path.join(root, sess)
    with h5py.File(os.path.join(d, f"{sess}_eeg.mat"), "r") as h:
        fs = float(np.array(h["eeg/s"]).ravel()[0])
        X = np.array(h["eeg/x"])
        lab = h["eeg/c/labels"]
        ch = ["".join(chr(c) for c in np.array(h[lab[i, 0]]).ravel())
              for i in range(lab.shape[0])]
    m = sio.loadmat(os.path.join(d, f"{sess}_event.mat"), squeeze_me=True,
                    struct_as_record=False)
    ev = m["event"] if "event" in m else m["events"]
    out = []
    for e in ev:
        if str(e.qc) != "good":
            continue
        out.append(dict(code=str(e.code), a=int(round(float(e.etstart))),
                        b=int(round(float(e.etstop)))))
    out.sort(key=lambda r: r["a"])
    return X, fs, ch, out


def windows(X, a, b, fs):
    """Non-overlapping WIN_S windows tiled from the event start."""
    n = int(round(WIN_S * fs))
    out = []
    s = a
    while s + n <= min(b, len(X)):
        out.append(X[s:s + n])
        s += n
    return out


def bandpower(W, fs):
    """[n_ch * n_band] Welch band power of one window, band order = BANDS order."""
    f, P = welch(W, fs=fs, nperseg=min(len(W), int(fs * 2)), axis=0)
    feats = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        feats.append(P[m].mean(axis=0))
    return np.concatenate(feats)


def calib_split(rest, fs, min_s=None):
    """(calibration events, evaluated events)."""
    if min_s is None:
        return rest[:BASE_N], rest[BASE_N:]
    cum, k = 0.0, 0
    while k < len(rest) and cum < min_s:
        cum += (rest[k]["b"] - rest[k]["a"]) / fs
        k += 1
    return rest[:k], rest[k:]
