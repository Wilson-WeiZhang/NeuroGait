"""Figure 7 — physics simulation and walk-intent gate

gate_fusion — Riemannian tools of the walk-intent gate: covariance bands, Karcher mean, tangent-space features and the logistic readout.

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gate_features import BANDS, calib_split, load_session


BASE_MIN_S = 30.0
BASE_N = 4
WIN_S, HOP_S = 3.0, 1.0
IQR_K, OFF_Q = 0.25, 90
LAM_RA = 1e-3
C_GRID = [0.01, 0.1, 1.0]


def _eig(C):
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    return np.maximum(w, 1e-20), V


def logm_sym(C):
    w, V = _eig(C)
    return (V * np.log(w)) @ V.T


def powm_sym(C, p):
    w, V = _eig(C)
    return (V * (w ** p)) @ V.T


def _expm_sym(S):
    w, V = np.linalg.eigh((S + S.T) / 2.0)
    return (V * np.exp(w)) @ V.T


def riemann_mean(Cs, tol=1e-10, itmax=60):
    """AIRM (Karcher) mean by the standard fixed-point iteration, init = arithmetic mean."""
    R = np.mean(Cs, axis=0)
    for _ in range(itmax):
        Rm, Rp = powm_sym(R, -0.5), powm_sym(R, 0.5)
        S = np.mean([logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
        R = Rp @ _expm_sym(S) @ Rp
        if np.linalg.norm(S, "fro") < tol:
            break
    return (R + R.T) / 2.0


def shrink(C, lam):
    n = C.shape[0]
    return (1.0 - lam) * C + lam * (np.trace(C) / n) * np.eye(n)


def vec(L):
    return L[np.triu_indices(L.shape[0])]


_SOS = {b: butter(4, (lo, hi), btype="band", fs=250.0) for b, (lo, hi) in BANDS.items()}


def band_covs(w):
    """The 4 band covariances of one window, band order = BANDS order."""
    out = []
    for bb, ab in _SOS.values():
        wf = filtfilt(bb, ab, w, axis=0)
        out.append((wf.T @ wf) / len(wf))
    return out


def ra_feat(covs, whit, lam=LAM_RA):
    """Riemannian-aligned log map: vec(log(W C W)) per band, W = R^-1/2 of the rest reference."""
    out = []
    for C, W in zip(covs, whit):
        Cs = shrink(C, lam)
        out.append(vec(logm_sym(W @ Cs @ W)))
    return np.concatenate(out)


def session_layout(root, sess, base_min_s):
    """(X, fs, calib_events, eval_events) with eval_events in build()'s exact order/naming."""
    X, fs, _ch, ev = load_session(root, sess)
    rest = [e for e in ev if e["code"] == "rest"]
    iw = [e for e in ev if e["code"] == "gaitimagery"]
    if base_min_s is None:
        cal, eval_rest = rest[:BASE_N], rest[BASE_N:]
    else:
        cal, eval_rest = calib_split(rest, fs, base_min_s)
    ordered_eval = [("iw", k, e) for k, e in enumerate(iw)] + \
                   [("rest", k, e) for k, e in enumerate(eval_rest)]
    return X, fs, cal, ordered_eval


def fit_ts(Xtr, ytr, gtr):
    gs = GridSearchCV(LogisticRegression(max_iter=5000), {"C": C_GRID}, scoring="roc_auc",
                      cv=GroupKFold(n_splits=3), n_jobs=8)
    gs.fit(Xtr, ytr, groups=gtr)
    return gs.best_estimator_, gs.best_params_["C"]
