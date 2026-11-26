"""Reconstruction quality of the decoded trajectories in the post-operative patient sessions.

Every session in the inputs is a post-operative session (11 sessions, P1 and P2); missing joints are dropped per joint,
never interpolated (two P1 sessions have no left-knee goniometer signal and drop out of every left-knee row and pair).
Four parts, all over sessions:
- e, cycle curves: per left-limb joint, mean and 95 % half-width (1.96 SD / sqrt(n)) of the cycle-averaged reference and
  decoded trajectories (50 gait-cycle bins, standardized within each session and joint before averaging cycles), and the median and quartiles of the per-session
  per-cycle correlation.
- f, spectra: per session with at least 5 trials and a finite stride frequency f0, the reference and decoded power
  spectra are put on a cadence-relative axis (frequency / f0, 200 points over 0.2-4) and divided by their maximum; mean
  and 95 % half-width are taken in log10 (geometric mean; half-width in log10 units).
- g, cyclograms: per joint pair, sessions with complete curves for both joints in both series; session-mean curves
  z-scored over the cycle, 95 % half-width per axis in z units.
- h, correspondence: 6 x 6 matrices (reference joint x reference / decoded joint, correlation over the session's
  concatenated held-out trials), mean over the sessions with a complete matrix, display polarity applied (right hip,
  left knee and left ankle x -1). Same-joint diagonal per session (mean of the six diagonal cells) against the mean |r| of
  the 30 off-diagonal cells (Wilcoxon signed-rank, two-sided); the diagonal under imagined-walking input against
  resting input over the sessions present in both (paired Wilcoxon, two-sided). The resting-input run decodes one
  contiguous window of the session's resting EEG in place of each test trial with the same trained folds.

Named inputs: cycle_average_patient.npz (obs_curves, dec_curves, percycle_r); spectra_patient.npz (freq, psd_obs, psd_dec,
f0, n_trials); correspondence_patient_imagined.npz, correspondence_patient_rest_input.npz (sessions, obs_mat, cross_mat).
Named output (derived/): patient_reconstruction.csv (panel, quantity, key, x, value, n_sessions_contributing).
"""
import csv
import os

import numpy as np
from scipy import stats

from analysis_common import inp, out_path

NB = 50
KEY = {0: "LHip", 1: "LKne", 2: "LAnk"}
LAB6 = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]
POLARITY = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])   # LKne, LAnk, RHip flipped for display
S6 = np.outer(POLARITY, POLARITY)
ROWS = []


def rec(panel, quantity, key, x, value, n):
    ROWS.append([panel, quantity, key, "" if x is None else f"{float(x):.6g}",
                 "" if value is None or not np.isfinite(value) else f"{float(value):.6g}", int(n)])


def cycle_curves(obs, dec, percycle_r):
    x = np.linspace(0, 100, NB)
    for j in range(3):
        v = np.isfinite(obs[:, j, :]).all(1) & np.isfinite(dec[:, j, :]).all(1)
        A_o, A_d = obs[v, j, :], dec[v, j, :]
        n = int(v.sum())
        mo, cio = A_o.mean(0), 1.96 * A_o.std(0) / np.sqrt(n)
        mp, cip = A_d.mean(0), 1.96 * A_d.std(0) / np.sqrt(n)
        for i in range(NB):
            rec("e", "ref_mean", KEY[j], x[i], mo[i], n)
            rec("e", "ref_ci95", KEY[j], x[i], cio[i], n)
            rec("e", "dec_mean", KEY[j], x[i], mp[i], n)
            rec("e", "dec_ci95", KEY[j], x[i], cip[i], n)
        pr = percycle_r[v, j]
        pr = pr[np.isfinite(pr)]
        rec("e", "percycle_r_median", KEY[j], None, float(np.median(pr)), n)
        rec("e", "percycle_r_p25", KEY[j], None, float(np.percentile(pr, 25)), n)
        rec("e", "percycle_r_p75", KEY[j], None, float(np.percentile(pr, 75)), n)


def spectra(zs):
    keepi = [i for i in range(len(zs["sessions"])) if zs["n_trials"][i] >= 5 and np.isfinite(zs["f0"][i])]
    freq, f0 = zs["freq"], zs["f0"]
    relgrid = np.linspace(0.2, 4.0, 200)
    for j in range(3):
        Po, Pd = [], []
        for i in keepi:
            if not (np.isfinite(zs["psd_obs"][i, j]).all() and np.isfinite(zs["psd_dec"][i, j]).all()):
                continue
            rel = freq / f0[i]
            o = np.interp(relgrid, rel, zs["psd_obs"][i, j])
            d = np.interp(relgrid, rel, zs["psd_dec"][i, j])
            Po.append(o / o.max())
            Pd.append(d / d.max())
        Lo, Ld = np.log10(np.array(Po) + 1e-12), np.log10(np.array(Pd) + 1e-12)
        mo, md = 10 ** Lo.mean(0), 10 ** Ld.mean(0)
        co = 1.96 * Lo.std(0) / np.sqrt(len(Lo))
        cd = 1.96 * Ld.std(0) / np.sqrt(len(Ld))
        n = len(Lo)
        for i in range(len(relgrid)):
            rec("f", "psd_ref_mean", KEY[j], relgrid[i], mo[i], n)
            rec("f", "psd_ref_ci95_log10", KEY[j], relgrid[i], co[i], n)
            rec("f", "psd_dec_mean", KEY[j], relgrid[i], md[i], n)
            rec("f", "psd_dec_ci95_log10", KEY[j], relgrid[i], cd[i], n)


def zloop(A, B, n):
    ma, mb = A.mean(0), B.mean(0)
    sa, sb = ma.std() + 1e-9, mb.std() + 1e-9
    return ((ma - ma.mean()) / sa, (mb - mb.mean()) / sb,
            1.96 * A.std(0) / np.sqrt(n) / sa, 1.96 * B.std(0) / np.sqrt(n) / sb)


def cyclograms(obs, dec):
    phase = np.linspace(0, 100, NB)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        v = (np.isfinite(obs[:, a, :]).all(1) & np.isfinite(obs[:, b, :]).all(1)
             & np.isfinite(dec[:, a, :]).all(1) & np.isfinite(dec[:, b, :]).all(1))
        n = int(v.sum())
        pkey = f"{KEY[a]}-{KEY[b]}"
        mxo, myo, cxo, cyo = zloop(obs[v, a], obs[v, b], n)
        mxd, myd, cxd, cyd = zloop(dec[v, a], dec[v, b], n)
        for i in range(NB):
            for q, arr in (("ref_x", mxo), ("ref_y", myo), ("dec_x", mxd), ("dec_y", myd), ("ref_ci95_x", cxo),
                           ("ref_ci95_y", cyo), ("dec_ci95_x", cxd), ("dec_ci95_y", cyd)):
                rec("g", q, pkey, phase[i], arr[i], n)


def correspondence(zx, zr):
    compl = np.isfinite(zx["cross_mat"]).all(axis=(1, 2))
    complr = np.isfinite(zr["cross_mat"]).all(axis=(1, 2))
    cx, cr = zx["cross_mat"][compl], zr["cross_mat"][complr]
    Cobs = np.nanmean(zx["obs_mat"][compl], axis=0) * S6
    Ciw = np.nanmean(cx, axis=0) * S6
    Cre = np.nanmean(cr, axis=0) * S6
    diag_iw = np.array([np.mean(np.diag(m)) for m in cx])
    diag_re = np.array([np.mean(np.diag(m)) for m in cr])
    n_iw, n_re = int(compl.sum()), int(complr.sum())
    for M, q, nn in ((Cobs, "reference", n_iw), (Ciw, "decoded_IW", n_iw), (Cre, "decoded_REST", n_re)):
        for i in range(6):
            for jj in range(6):
                rec("h", q, f"{LAB6[i]}|{LAB6[jj]}", None, M[i, jj], nn)
    rec("h", "diag_median_IW", "all_joints", None, float(np.median(diag_iw)), n_iw)
    rec("h", "diag_median_REST", "all_joints", None, float(np.median(diag_re)), n_re)
    off_iw = np.array([float(np.mean(np.abs(m[~np.eye(6, dtype=bool)]))) for m in cx])
    s_iw = [str(s) for s in zx["sessions"][compl]]
    s_re = [str(s) for s in zr["sessions"][complr]]
    rec("h", "diag_IW_q25", "all_joints", None, np.percentile(diag_iw, 25), n_iw)
    rec("h", "diag_IW_q75", "all_joints", None, np.percentile(diag_iw, 75), n_iw)
    rec("h", "offdiag_absmean_median_IW", "all_joints", None, float(np.median(off_iw)), n_iw)
    w_iw = stats.wilcoxon(diag_iw, off_iw)
    rec("h", "wilcoxon_W_IW_diag_vs_offdiag", "all_joints", None, w_iw.statistic, n_iw)
    rec("h", "wilcoxon_p_IW_diag_vs_offdiag", "all_joints", None, w_iw.pvalue, n_iw)
    rec("h", "diag_REST_q25", "all_joints", None, np.percentile(diag_re, 25), n_re)
    rec("h", "diag_REST_q75", "all_joints", None, np.percentile(diag_re, 75), n_re)
    both = [s for s in s_re if s in s_iw]
    w_pair = stats.wilcoxon(np.array([diag_iw[s_iw.index(s)] for s in both]),
                            np.array([diag_re[s_re.index(s)] for s in both]))
    rec("h", "wilcoxon_W_diag_IW_vs_REST_paired", "all_joints", None, w_pair.statistic, len(both))
    rec("h", "wilcoxon_p_diag_IW_vs_REST_paired", "all_joints", None, w_pair.pvalue, len(both))


def main():
    z = np.load(inp("cycle_average_patient.npz"), allow_pickle=True)
    cycle_curves(z["obs_curves"], z["dec_curves"], z["percycle_r"])
    spectra(np.load(inp("spectra_patient.npz"), allow_pickle=True))
    cyclograms(z["obs_curves"], z["dec_curves"])
    correspondence(np.load(inp("correspondence_patient_imagined.npz"), allow_pickle=True),
                   np.load(inp("correspondence_patient_rest_input.npz"), allow_pickle=True))
    p = out_path("patient_reconstruction.csv")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["panel", "quantity", "key", "x", "value", "n_sessions_contributing"])
        w.writerows(ROWS)
    os.replace(tmp, p)
    print("wrote", p, len(ROWS), "rows")


if __name__ == "__main__":
    main()
