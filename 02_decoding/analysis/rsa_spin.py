"""Per-session spatial RSA between real and imagined walking maps, with a spin null, and its cohort statistics.

Per harmonic k (1-3) and healthy session: the 60 x 60 electrode RDM of each condition is the Euclidean distance between
the 18-value electrode feature vectors [re(phi_1..9), im(phi_1..9)] of the session's phase map; rho_s = Spearman
correlation of the imagined- and real-walking RDMs (upper triangle, 1,770 pairs). Spin null: one shared set of 1,000
random rotations of the standard_1005 electrode positions (seed 0; each rotation turned into an electrode permutation by
linear assignment on the centred positions), applied to the real-walking RDM of every session and reused for all k.
Cohort test: mean_s rho_s against the 1,000 null means, P = (1 + #{null >= observed}) / 1,001, BH across k. The CI of
the cohort mean is a percentile bootstrap over the 80 sessions (2,000 resamples, seed 20261102). Values are rounded to
six decimals before the cohort step, as in the per-session table.

Named inputs: healthy_phase_imagined.csv, healthy_phase_real.csv.
Named output (derived/): rsa_cohort_stats.csv (one row per harmonic: cohort mean rho, bootstrap CI, spin-null mean and
2.5-97.5th percentiles, BH-adjusted spin p and its mark).
"""
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, out_path

N_SPIN, SPIN_SEED, BOOT, BOOT_SEED = 1000, 0, 2000, 20261102
KS = (1, 2, 3)


def star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def load(path, k, common):
    """[session, cycle_point, channel] real / imaginary arrays of harmonic k."""
    d = pd.read_csv(path)
    d = d[d.harmonic_k == k]
    sess = sorted(d.session.unique())
    cyc = sorted(d.cycle_fraction.unique())
    re = np.full((len(sess), len(cyc), len(common)), np.nan)
    im = np.full_like(re, np.nan)
    piv = d.set_index(["session", "cycle_fraction", "channel"])
    for si, s in enumerate(sess):
        for ci, c in enumerate(cyc):
            sub = piv.loc[(s, c)]
            re[si, ci] = sub["real"].reindex(common).to_numpy()
            im[si, ci] = sub["imag"].reindex(common).to_numpy()
    return re, im, sess, cyc


def rand_rot(rng):
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rm)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def session_rdm(re_s, im_s):
    F = np.concatenate([re_s, im_s], axis=0).T            # (channel, 18)
    d2 = np.sum((F[:, None, :] - F[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def main():
    iwf, rwf = inp("healthy_phase_imagined.csv"), inp("healthy_phase_real.csv")
    common = sorted(set(pd.read_csv(iwf).channel) & set(pd.read_csv(rwf).channel))
    E = len(common)
    iuE = np.triu_indices(E, 1)
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in common])
    xyz0 = xyz - xyz.mean(0)
    rng = np.random.default_rng(SPIN_SEED)
    PM = np.empty((N_SPIN, E), int)
    for b in range(N_SPIN):
        Dm = np.linalg.norm(xyz0[:, None] - (xyz0 @ rand_rot(rng).T)[None], axis=2)
        _, PM[b] = linear_sum_assignment(Dm)
    FLAT = PM[:, iuE[0]] * E + PM[:, iuE[1]]               # flat indices realising M[pm][:, pm][iuE]

    stat = {}
    for K in KS:
        reI, imI, sessI, _ = load(iwf, K, common)
        reR, imR, sessR, _ = load(rwf, K, common)
        assert sessI == sessR
        S = len(sessI)
        rho_obs, null = np.empty(S), np.empty((S, N_SPIN))
        for si in range(S):
            A = session_rdm(reI[si], imI[si])
            B = session_rdm(reR[si], imR[si])
            ra = stats.rankdata(A[iuE])
            rb_mat = np.zeros((E, E))
            rb_mat[iuE] = stats.rankdata(B[iuE])
            rb_mat += rb_mat.T
            rho_obs[si] = stats.spearmanr(A[iuE], B[iuE])[0]
            Prb = rb_mat.ravel()[FLAT]                      # permuted real-walking ranks; Pearson on ranks = Spearman
            a0 = ra - ra.mean()
            P0 = Prb - Prb.mean(1, keepdims=True)
            null[si] = (P0 @ a0) / (np.sqrt((a0 ** 2).sum()) * np.sqrt((P0 ** 2).sum(1)))
        T_null = null.mean(0)
        stat[K] = dict(rho=np.round(rho_obs, 6), mean=round(float(rho_obs.mean()), 6),
                       null_mean=round(float(null.mean(1).mean()), 6),
                       p_spin=(np.sum(T_null >= rho_obs.mean()) + 1) / (N_SPIN + 1), T_null=np.round(T_null, 6))
    q = np.round(multipletests([stat[k]["p_spin"] for k in KS], method="fdr_bh")[1], 6)

    brng = np.random.default_rng(BOOT_SEED)
    rows = []
    for i, k in enumerate(KS):
        rho = stat[k]["rho"]
        lo, hi = np.percentile(rho[brng.integers(0, rho.size, (BOOT, rho.size))].mean(axis=1), [2.5, 97.5])
        nlo, nhi = np.percentile(stat[k]["T_null"], [2.5, 97.5])
        rows.append(dict(panel="states_rsa_real_imag_small", row_type="stat", group="cohort mean vs spin null",
                         p=float(q[i]), mark=star(q[i]), test="spin test, BH across k", harmonic_k=k,
                         mean_rho=stat[k]["mean"], ci_lo=lo, ci_hi=hi, null_mean=stat[k]["null_mean"],
                         null_p2_5=nlo, null_p97_5=nhi))
    p = out_path("rsa_cohort_stats.csv")
    tmp = p.with_name(p.name + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False, float_format="%.6g", lineterminator="\n")
    tmp.replace(p)


if __name__ == "__main__":
    main()
