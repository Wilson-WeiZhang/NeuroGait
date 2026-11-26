"""Fig. 6. Case-level electrode RSA with a rotation null and session-bootstrap intervals.

Input tables healthy_phase_imagined.csv, healthy_phase_rest.csv and patient_maps.csv
have session, harmonic_k, cycle_fraction, channel, real and imag columns.
Patient session labels follow the P1_sess04 convention. Estimates compare each
patient's cross-validated distance matrix with the healthy imagined-walking matrix.
"""
import itertools
import math

import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, write_table
from rsa_session_spin import rand_rot

SESSIONS = {"P1": (4, 5, 6, 7, 8, 9), "P2": (3, 5, 6, 8, 9)}


def features(raw, k, sessions, common):
    """Sessions by electrodes by complex gait-cycle features."""
    d = raw[raw.harmonic_k == k]
    if sessions is None:
        sessions = sorted(d.session.unique())
    d = d[d.session.isin(sessions)]
    cyc = sorted(d.cycle_fraction.unique())
    piv = d.set_index(["session", "cycle_fraction", "channel"]).sort_index()
    re, im = np.full((len(sessions), len(cyc), len(common)), np.nan), np.full((len(sessions), len(cyc), len(common)), np.nan)
    for si, s in enumerate(sessions):
        for ci, c in enumerate(cyc):
            sub = piv.loc[(s, c)]
            re[si, ci] = sub["real"].reindex(common).to_numpy()
            im[si, ci] = sub["imag"].reindex(common).to_numpy()
    return np.concatenate([re.transpose(0, 2, 1), im.transpose(0, 2, 1)], 2)


def crossnobis_w(P_, w):
    """Cross-validated squared distance for sessions counted w times: the feature scale over all copies; each held-out session against the mean of the other sessions, its own copies excluded.  With w = 1 this is the recipe, written with matrix products."""
    F = P_.shape[2]
    keep = w > 0
    Pk = np.repeat(P_[keep], w[keep], axis=0)
    sd = np.nanstd((Pk - np.nanmean(Pk, 0)[None]).reshape(-1, F), 0) + 1e-9
    Pw, wk = P_[keep] / sd, w[keep]
    n = wk.sum()
    total = (wk[:, None, None] * Pw).sum(0)
    C = P_.shape[1]
    rdm_ = np.zeros((C, C))
    for u in range(len(wk)):
        B = (total - wk[u] * Pw[u]) / (n - wk[u])
        G = Pw[u] @ B.T
        dg = np.diag(G)
        rdm_ += wk[u] * (dg[:, None] + dg[None, :] - G - G.T)
    return rdm_ / (n * F)


def weighted_q(v, w, q):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(c, q), len(v) - 1)])


def bootstrap(features, reference, upper):
    n = len(features)
    values, weights = [], []
    ranks = stats.rankdata(reference[upper])
    for sample in itertools.combinations_with_replacement(range(n), n):
        counts = np.bincount(sample, minlength=n)
        if np.count_nonzero(counts) < 2:
            continue
        distance = crossnobis_w(features, counts)[upper]
        values.append(np.corrcoef(stats.rankdata(distance), ranks)[0, 1])
        weights.append(math.factorial(n) / math.prod(math.factorial(int(x)) for x in counts))
    return [weighted_q(np.array(values), np.array(weights), q) for q in (0.025, 0.975)]


def build():
    patient, healthy, rest = [pd.read_csv(inp(name)) for name in
                              ("patient_maps.csv", "healthy_phase_imagined.csv", "healthy_phase_rest.csv")]
    channels = sorted(set(patient.channel) & set(healthy.channel) & set(rest.channel))
    positions = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    xyz = np.array([positions[c] for c in channels])
    xyz -= xyz.mean(axis=0)
    upper = np.triu_indices(len(channels), 1)
    rng = np.random.default_rng(0)
    rows = []
    for k in (1, 2, 3):
        group = features(healthy, k, None, channels)
        reference = crossnobis_w(group, np.ones(len(group), int))
        group = features(rest, k, None, channels)
        resting = crossnobis_w(group, np.ones(len(group), int))
        for case, visits in SESSIONS.items():
            sessions = [f"{case}_sess{n:02d}" for n in visits]
            case_features = features(patient, k, sessions, channels)
            distance = crossnobis_w(case_features, np.ones(len(visits), int))[upper]
            observed = stats.spearmanr(distance, reference[upper]).statistic
            null = []
            for _ in range(1000):
                cost = np.linalg.norm(xyz[:, None] - (xyz @ rand_rot(rng).T)[None], axis=2)
                _, perm = linear_sum_assignment(cost)
                null.append(stats.spearmanr(distance, reference[perm[upper[0]], perm[upper[1]]]).statistic)
            # The channel-shuffle control follows each spin test in the study RNG stream.
            shuffled = []
            for _ in range(5000):
                perm = rng.permutation(len(channels))
                shuffled.append(stats.spearmanr(distance, reference[perm[upper[0]], perm[upper[1]]]).statistic)
            lo, hi = bootstrap(case_features, reference, upper)
            rows.append(dict(harmonic_k=k, patient=case, rho=observed, ci_lo=lo, ci_hi=hi,
                             rest_rho=stats.spearmanr(distance, resting[upper]).statistic,
                             spin_p=round((1 + np.count_nonzero(np.asarray(null) >= observed)) / 1001, 4),
                             shuffle_p=(1 + np.count_nonzero(np.asarray(shuffled) >= observed)) / 5001))
    table = pd.DataFrame(rows)
    table["pFDR"] = multipletests(table.spin_p, method="fdr_bh")[1]
    write_table(table, "patient_spatial_rsa.csv", index=False)


if __name__ == "__main__":
    build()
