"""Figure 7 — physics simulation and walk-intent gate

gate_window3_within_session — helpers of the within-session 10-fold gate: fold assignment by event, event table, bootstrap draws, decision windows.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import csv
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_features as pgf
import gate_fusion as PF

SESSIONS = list(pgf.SESSIONS)
LAM = PF.LAM_RA
BASE_MIN_S = PF.BASE_MIN_S
HOP_S = 1.0
NFOLD = 10
SEED = 0
B_BOOT = 1000


def auc_np(y, x):
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(x)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def dense_windows(X, a, b, fs, win_s, hop_s=HOP_S):
    n, hop = int(round(win_s * fs)), int(round(hop_s * fs))
    out, s, k = [], a, 0
    while s + n <= min(b, len(X)):
        out.append((k * hop_s + win_s, X[s:s + n]))
        s += hop
        k += 1
    return out


def event_table(D, s):
    """The session's evaluated events in canonical row order, with label and onset."""
    m = D["sess"] == s
    names, seen = [], set()
    for e in D["evn"][m]:
        if e not in seen:
            seen.add(e)
            names.append(e)
    lab = np.array([1 if n.startswith("iw") else 0 for n in names])
    ons = np.array([D["onset"][(s, n)] for n in names], float)
    return names, lab, ons


def folds_for(D, s, regime):
    names, lab, ons = event_table(D, s)
    n = len(names)
    if regime == "within_random":
        skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
        return names, [(tr, te) for tr, te in skf.split(np.zeros(n), lab)]
    if regime == "within_block":
        order = np.argsort(ons, kind="mergesort")
        blocks = np.array_split(order, NFOLD)
        out = []
        for b in blocks:
            te = np.asarray(b, int)
            tr = np.array([i for i in range(n) if i not in set(te.tolist())], int)
            out.append((tr, te))
        return names, out
    raise ValueError(regime)


def boot_index(D):
    """Shared bootstrap resampling indices (seed 0), identical across every cell."""
    rs = np.random.RandomState(SEED)
    counts = []
    for s in SESSIONS:
        m = D["sess"] == s
        ue = np.unique(D["evn"][m])
        counts.append((sum(1 for e in ue if e.startswith("iw")),
                       sum(1 for e in ue if e.startswith("rest"))))
    return [[(rs.randint(0, n1, n1), rs.randint(0, n0, n0)) for (n1, n0) in counts]
            for _ in range(B_BOOT)]


def write(path, rows, head=None):
    cols, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                cols.append(k)
    if head:
        cols = head + [c for c in cols if c not in head]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}: {len(rows)} rows", flush=True)
