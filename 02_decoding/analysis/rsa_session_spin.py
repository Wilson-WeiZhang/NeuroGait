"""Per-session spatial RSA between imagined and real walking, calibrated against a spin null, per harmonic k1-k3.

Per harmonic k and session s:
- Electrode RDM per condition: 60 x 60 Euclidean distances between the electrodes' 18-value feature vectors
  [re(phi_1..phi_9), im(phi_1..phi_9)] (complex harmonic phase value at nine gait-cycle points); upper triangle = 1770.
- rho_obs: Spearman correlation of the imagined-walking and real-walking RDMs.
- Spin null: one shared set of 1000 random-rotation electrode permutations (numpy default_rng(0); a random 3-D
  rotation of the standard_1005 electrode positions, matched back to electrodes by linear_sum_assignment), applied
  to the real-walking RDM of every session and reused for all k. Permuting electrodes only reorders the 1770
  distances, so the permuted Spearman is a Pearson correlation of the fixed imagined-walking ranks with a gather of
  the real-walking rank matrix.
- null_mean, null_sd (ddof 1), null_p95 of the session's 1000 null rho; z = (rho_obs - null_mean) / null_sd.
The cohort-level test printed per k is the cohort-mean spin test, P = (1 + #{mean_s null_s^(r) >= mean_s rho_s}) /
(1000 + 1), with BH-FDR across the three harmonics (statsmodels).

Named inputs: iw_phase_maps.csv, rw_phase_maps.csv (session, harmonic_k, cycle_fraction, channel, real, imag).
Named output: derived/rsa_session_rho.csv.
"""
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, write_table

N_SPIN = 1000
SEED = 0


def load(path, k, common):
    """[session, cycle_point, channel] real and imaginary arrays of harmonic k."""
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
    """60 x 60 Euclidean RDM between the 18-value electrode feature vectors of one session."""
    F = np.concatenate([re_s, im_s], axis=0).T
    d2 = np.sum((F[:, None, :] - F[None, :, :]) ** 2, axis=2)
    return np.sqrt(np.maximum(d2, 0.0))


def build():
    iwf, rwf = inp("iw_phase_maps.csv"), inp("rw_phase_maps.csv")
    common = sorted(set(pd.read_csv(iwf).channel) & set(pd.read_csv(rwf).channel))
    E = len(common)
    iuE = np.triu_indices(E, 1)

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in common])
    xyz0 = xyz - xyz.mean(0)
    rng = np.random.default_rng(SEED)
    PM = np.empty((N_SPIN, E), int)
    for b in range(N_SPIN):
        Dm = np.linalg.norm(xyz0[:, None] - (xyz0 @ rand_rot(rng).T)[None], axis=2)
        _, pm = linear_sum_assignment(Dm)
        PM[b] = pm
    FLAT = PM[:, iuE[0]] * E + PM[:, iuE[1]]                # flat indices realising M[pm][:, pm][iuE]

    rows, p_spin = [], []
    for K in (1, 2, 3):
        reI, imI, sessI, cyc = load(iwf, K, common)
        reR, imR, sessR, cycR = load(rwf, K, common)
        assert sessI == sessR and cyc == cycR
        S = len(sessI)
        rho_obs = np.empty(S)
        null = np.empty((S, N_SPIN))
        for si in range(S):
            A = session_rdm(reI[si], imI[si])
            B = session_rdm(reR[si], imR[si])
            ra = stats.rankdata(A[iuE])
            rb_mat = np.zeros((E, E))
            rb_mat[iuE] = stats.rankdata(B[iuE])
            rb_mat += rb_mat.T
            rho_obs[si] = stats.spearmanr(A[iuE], B[iuE])[0]
            Prb = rb_mat.ravel()[FLAT]
            a0 = ra - ra.mean()
            sa = np.sqrt((a0 ** 2).sum())
            P0 = Prb - Prb.mean(1, keepdims=True)
            sb = np.sqrt((P0 ** 2).sum(1))
            null[si] = (P0 @ a0) / (sa * sb)
        nm, ns = null.mean(1), null.std(1, ddof=1)
        p95 = np.percentile(null, 95, axis=1)
        z = (rho_obs - nm) / ns
        T_obs, T_null = float(rho_obs.mean()), null.mean(0)
        p_spin.append((np.sum(T_null >= T_obs) + 1) / (N_SPIN + 1))
        print(f"k{K}: mean rho = {T_obs:+.4f} | mean null = {nm.mean():+.4f} | P_spin = {p_spin[-1]:.4f}")
        for si, s in enumerate(sessI):
            rows.append(dict(session=s, participant=s.split("_")[0], harmonic_k=K,
                             rho_obs=round(float(rho_obs[si]), 6), null_mean=round(float(nm[si]), 6),
                             null_sd=round(float(ns[si]), 6), null_p95=round(float(p95[si]), 6),
                             z=round(float(z[si]), 6)))
    print("P_spin BH across k1-k3:", np.round(multipletests(p_spin, method="fdr_bh")[1], 6))
    p = write_table(pd.DataFrame(rows), "rsa_session_rho.csv", index=False)
    print("wrote", p)


if __name__ == "__main__":
    build()
