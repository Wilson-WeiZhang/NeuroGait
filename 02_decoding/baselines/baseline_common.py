"""Shared pieces of the deep-baseline runs (Supplementary Table 2): named inputs, session readers, trial table, scoring.

The three networks (EEGNet-8,2, DeepConvNet, EEG2GAIT) are trained per session on the same trials, the same ten cross-validation
folds and the same goniometer targets as the decoder, and are scored with the decoder's definitions: the per-trial correlation of
prediction and target (`corr_1d`), the mean of the per-trial values over trials (`rpt`) and the correlation over all test samples
(`r`).

Inputs are named, not pathed: `inp(name)` resolves a name under $NEUROGAIT_DATA, through the optional name -> relative-path table
in $NEUROGAIT_INPUTS (two tab-separated columns).
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch
import torch.nn as nn

DATA = Path(os.environ.get("NEUROGAIT_DATA", "."))
_NAMES = {}
if os.environ.get("NEUROGAIT_INPUTS"):
    with open(os.environ["NEUROGAIT_INPUTS"], encoding="utf-8") as f:
        _NAMES = {r[0]: r[1] for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")}


def inp(name):
    """Path of a named input."""
    return DATA / _NAMES.get(name, name)


TARGET_LABELS = ["LHipW X", "LKneW X", "LAnkW X", "RHipW X", "RKneW X", "RAnkW X"]


# ----------------------------------------------------------------------------- norm-constrained layers
class Conv2dC(nn.Conv2d):
    """Norm-constrained conv (max_norm on the output-channel axis)."""

    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LinearC(nn.Linear):
    """Norm-constrained linear layer, the torch equivalent of Keras `Dense(..., kernel_constraint=max_norm(rate))`.

    Keras stores the Dense kernel as (in_features, out_features) and MaxNorm's default axis=0 reduces over axis 0, giving one norm
    per output unit. torch stores it transposed, (out_features, in_features), so renorm over dim=0 constrains the same vectors.
    """

    def __init__(self, *args, max_norm=0.25, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


# ------------------------------------------------------------------------------- io
@dataclass
class Trial:
    eeg: np.ndarray  # (T, C) at the model rate
    y: np.ndarray  # (T, 6) at the model rate, theta - theta0


def _mstr(ds) -> str:
    return "".join(chr(int(v)) for v in np.asarray(ds).flatten() if int(v) != 0)


def _refs(f, dset) -> list[str]:
    return [_mstr(f[r]) for r in np.asarray(dset).flatten()]


def read_h5_matrix(path: Path, root: str):
    """Signal matrix (samples x channels), sampling rate, channel labels, electrode positions and source tag of one recording."""
    with h5py.File(path, "r") as f:
        x = np.asarray(f[f"{root}/x"], dtype=np.float32)
        fs = float(np.asarray(f[f"{root}/s"]).squeeze())
        if root == "eeg":
            labels = _refs(f, f["eeg/c/labels"])
            xyz = np.stack(
                [np.array([float(np.asarray(f[r]).squeeze()) for r in np.asarray(f[f"eeg/c/{k}"]).flatten()])
                 for k in ("X", "Y", "Z")],
                axis=1,
            )
            src = _mstr(f["eeg/source_set"]) if "source_set" in f["eeg"] else ""
        else:
            labels, xyz, src = _refs(f, f["goni/c"]), None, ""
        if x.shape[1] != len(labels) and x.shape[0] == len(labels):
            x = x.T
        return x, fs, labels, xyz, src


def _sstr(x) -> str:
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, np.ndarray):
        return _sstr(x.item()) if x.shape == () else "".join(str(v) for v in x.tolist())
    return str(x)


def read_events(path: Path) -> list[dict]:
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    evs = np.atleast_1d(d["event"] if "event" in d else d["events"])
    return [
        {
            "code": _sstr(e.code),
            "qc": _sstr(e.qc),
            "etstart": int(round(float(e.etstart))) - 1,
            "etstop": int(round(float(e.etstop))) - 1,
            "gtstart": int(round(float(e.gtstart))) - 1,
            "gtstop": int(round(float(e.gtstop))) - 1,
        }
        for e in evs
    ]


def trial_filter(events: list[dict], condition: str) -> list[dict]:
    """The decoder's trial selection (imagined or executed walking, quality-control status good), order preserved."""
    target = "gaitimagery" if condition == "MI" else "gaitexecution"
    return [e for e in events if e["code"] == target and e["qc"] == "good"]


def read_foldmap(path: Path, sessions: list[str]):
    """{session: {fold: [0-based trial indices]}} of the decoder's random ten-fold split."""
    fm: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if sessions and row["session"] not in sessions:
                continue
            fm[row["session"]][int(row["fold"])].append(int(row["trial_idx"]) - 1)
    return fm


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Zero-phase polyphase resampling; used for the goniometer targets only, never for the EEG."""
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    g = math.gcd(int(round(fs_in)), int(round(fs_out)))
    return signal.resample_poly(x, int(round(fs_out)) // g, int(round(fs_in)) // g, axis=0).astype(np.float32)


# ------------------------------------------------------------------------- windowing and scoring
def fit_scalers(trials, idx, time_step, stride):
    e = np.vstack([trials[i].eeg for i in idx])
    emu, esd = np.nanmean(e, 0), np.nanstd(e, 0)
    esd[~np.isfinite(esd) | (esd <= 0)] = 1
    ys = [trials[i].y[np.arange(time_step - 1, len(trials[i].y), stride)] for i in idx]
    ys = [a for a in ys if len(a)]
    ya = np.vstack(ys)
    ymu, ysd = np.zeros(ya.shape[1], np.float32), np.ones(ya.shape[1], np.float32)
    for j in range(ya.shape[1]):
        ok = np.isfinite(ya[:, j])
        if ok.any():
            ymu[j] = float(np.mean(ya[ok, j]))
            s = float(np.std(ya[ok, j]))
            if np.isfinite(s) and s > 0:
                ysd[j] = s
    return emu.astype(np.float32), esd.astype(np.float32), ymu, ysd


def corr_1d(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return np.nan
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    d = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


def masked_mse(pred, tgt, mask):
    return (((pred - tgt) ** 2) * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def score(pred_z, y, mask, tid, ymu, ysd):
    """r = correlation over all concatenated test samples; rpt = mean over trials of the within-trial correlation."""
    pred = np.where(mask > 0, pred_z * ysd + ymu, np.nan)
    truth = np.where(mask > 0, y * ysd + ymu, np.nan)
    r, rpt = [], []
    for j in range(truth.shape[1]):
        r.append(corr_1d(truth[:, j], pred[:, j]))
        pt = [corr_1d(truth[tid == t, j], pred[tid == t, j]) for t in np.unique(tid)]
        rpt.append(float(np.nanmean(pt)) if np.any(np.isfinite(pt)) else np.nan)
    return r, rpt


def trial_scores(pred_z, y, mask, tid, ymu, ysd):
    """The per-trial within-trial correlation of `score` (same masking and `corr_1d`), as rows
    (trial list index, joint index, r, number of scored samples)."""
    pred = np.where(mask > 0, pred_z * ysd + ymu, np.nan)
    truth = np.where(mask > 0, y * ysd + ymu, np.nan)
    rows = []
    for t in np.unique(tid):
        sel = tid == t
        for j in range(truth.shape[1]):
            n = int((np.isfinite(truth[sel, j]) & np.isfinite(pred[sel, j])).sum())
            rows.append((int(t), j, corr_1d(truth[sel, j], pred[sel, j]), n))
    return rows
