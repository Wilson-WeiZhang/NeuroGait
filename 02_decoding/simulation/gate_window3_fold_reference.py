"""Figure 7 — physics simulation and walk-intent gate

gate_window3_fold_reference — within-session 10-fold gate scores whose rest reference is drawn inside each fold from that fold's training rest events.

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import sys

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gate_window3_within_session as M
import gate_features as pgf
import gate_fusion as PF
import gate as GATE

SESSIONS = M.SESSIONS
LAM, BASE_MIN_S, HOP_S = M.LAM, M.BASE_MIN_S, M.HOP_S
SEED, B_BOOT, REF_MIN_S = 0, 1000, 30.0


N_DAMPED = 0


def riemann_mean_damped(Cs, tau=0.5, tol=1e-10, itmax=3000):
    R = np.mean(Cs, axis=0)
    for _ in range(itmax):
        Rm, Rp = PF.powm_sym(R, -0.5), PF.powm_sym(R, 0.5)
        S = np.mean([PF.logm_sym(Rm @ C @ Rm) for C in Cs], axis=0)
        n = float(np.linalg.norm(S, "fro"))
        R = Rp @ PF._expm_sym(tau * S) @ Rp
        R = (R + R.T) / 2.0
        if n < tol:
            return R
    raise RuntimeError(f"damped Karcher mean did not converge (|S|_F {n:.3e})")


def whiteners_safe(covs, lam=LAM):
    """Band whiteners R^-1/2 from the Karcher mean of the shrunk reference covariances (damped solver as a counted fallback)."""
    global N_DAMPED
    out = []
    for b in range(len(pgf.BANDS)):
        Cs = [PF.shrink(c[b], lam) for c in covs]
        try:
            Rb = PF.riemann_mean(Cs)
        except np.linalg.LinAlgError:
            Rb = riemann_mean_damped(Cs)
            N_DAMPED += 1
        out.append(PF.powm_sym(Rb, -0.5))
    return out


def ra_stack(covs_list, whit):
    return np.vstack([PF.ra_feat(c, whit, LAM) for c in covs_list])


def rel(P, Pbase):
    return np.log10((P + 1e-20) / (Pbase + 1e-20))


def draw_reference(rest_names, dur, onset, scheme, rs):
    """Rest events from the training fold until cumulative duration >= REF_MIN_S."""
    if scheme == "random":
        order = list(np.asarray(rest_names)[rs.permutation(len(rest_names))])
    elif scheme == "earliest":
        order = sorted(rest_names, key=lambda n: onset[n])
    else:
        raise ValueError(scheme)
    cum, pick = 0.0, []
    for n in order:
        pick.append(n)
        cum += dur[n]
        if cum >= REF_MIN_S:
            break
    return pick, cum


def run_scheme(D, RAW, T, scheme, want_dense=True):
    """scheme: 'random' | 'earliest' (rest-event order of the in-fold reference)."""
    streams, thr_rows, tiled_sc, models = [], [], {}, {}
    for s in SESSIONS:
        R = RAW[s]
        te_m = D["sess"] == s
        idx_te = np.where(te_m)[0]
        ev_row = D["evn"][idx_te]
        names, folds = M.folds_for(D, s, "within_random")
        onset = {n: D["onset"][(s, n)] for n in names}
        rs = np.random.RandomState(SEED)
        sc_t = {c: np.full(idx_te.size, np.nan) for c in ("fuse2_V1", "bp_lda")}
        mods = {}
        for k, (tr_e, te_e) in enumerate(folds):
            tr_names = [names[i] for i in tr_e]
            te_names = [names[i] for i in te_e]
            tr_rest = [n for n in tr_names if n.startswith("rest")]
            ref_names, ref_dur = draw_reference(tr_rest, R["dur"], onset, scheme, rs)
            refP = np.vstack([R["tiled"][n][0] for n in ref_names])
            refC = [c for n in ref_names for c in R["tiled"][n][1]]
            fit_names = [n for n in tr_names if n not in set(ref_names)]
            Pbase = np.median(refP, axis=0)
            whit = whiteners_safe(refC, LAM)

            Xb = np.vstack([rel(R["tiled"][n][0], Pbase) for n in fit_names])
            Xr = np.vstack([ra_stack(R["tiled"][n][1], whit) for n in fit_names])
            y = np.concatenate([np.full(len(R["tiled"][n][0]), 1 if n.startswith("iw") else 0)
                                for n in fit_names])
            g = np.concatenate([np.full(len(R["tiled"][n][0]), n) for n in fit_names])
            sa = StandardScaler().fit(Xb)
            cla = LDA(solver="lsqr", shrinkage="auto").fit(sa.transform(Xb), y)
            sb = StandardScaler().fit(Xr)
            clb, Cbest = PF.fit_ts(sb.transform(Xr), y, g)

            cA = cla.decision_function(sa.transform(rel(refP, Pbase)))
            cB = clb.decision_function(sb.transform(ra_stack(refC, whit)))
            sdA = float(cla.decision_function(sa.transform(Xb)).std())
            sdB = float(clb.decision_function(sb.transform(Xr)).std())
            mA, mB = float(np.median(cA)), float(np.median(cB))
            fc = ((cA - mA) / sdA + (cB - mB) / sdB) / 2.0
            th = ((0.0 - mA) / sdA + (0.0 - mB) / sdB) / 2.0
            dl = GATE.IQR_K * float(np.percentile(fc, 75) - np.percentile(fc, 25))
            mod = dict(sa=sa, cla=cla, sb=sb, clb=clb, Pbase=Pbase, whit=whit,
                       mA=mA, mB=mB, sdA=sdA, sdB=sdB, theta_sym=th, delta=dl, fold=k)
            for n in te_names:
                mods[n] = mod

            for n in te_names:
                P, C = R["tiled"][n]
                dA = cla.decision_function(sa.transform(rel(P, Pbase)))
                dB = clb.decision_function(sb.transform(ra_stack(C, whit)))
                fu = ((dA - mA) / sdA + (dB - mB) / sdB) / 2.0
                m_rows = ev_row == n
                sc_t["fuse2_V1"][m_rows] = fu
                sc_t["bp_lda"][m_rows] = dA
            thr_rows.append(dict(scheme=scheme, T=T, session=s, fold=k,
                                 n_ref_events=len(ref_names), ref_dur_s=round(ref_dur, 2),
                                 ref_events=";".join(str(x) for x in ref_names), n_ref_win=len(refP),
                                 n_fit_events=len(fit_names), n_fit_win=int(len(y)),
                                 lr_C=Cbest, sd_A=sdA, sd_B=sdB, centre_A=mA, centre_B=mB,
                                 theta_sym=th, delta=dl, calib_iqr=dl / GATE.IQR_K))
        assert not np.isnan(sc_t["fuse2_V1"]).any(), f"{s}: unscored tiled windows"
        tiled_sc[s] = sc_t
        models[s] = mods
        if want_dense:
            for n, (ts, P, C) in R["dense"].items():
                mod = mods[n]
                dA = mod["cla"].decision_function(mod["sa"].transform(rel(P, mod["Pbase"])))
                dB = mod["clb"].decision_function(mod["sb"].transform(ra_stack(C, mod["whit"])))
                fu = ((dA - mod["mA"]) / mod["sdA"] + (dB - mod["mB"]) / mod["sdB"]) / 2.0
                for t, va, vb, vf in zip(ts, dA, dB, fu):
                    streams.append(dict(session=s, event=n,
                                        label=1 if n.startswith("iw") else 0,
                                        t_end_ev_s=round(t, 4),
                                        ev_dur_s=round(R["dur"][n], 4), n_dec_ev=len(ts),
                                        score_lda=float(va), score_ts=float(vb),
                                        fuse_V1=float(vf), fold=mod["fold"],
                                        theta_sym=mod["theta_sym"], delta=mod["delta"],
                                        e_sigma=float((vf - mod["theta_sym"]) / mod["delta"])))
        a1 = M.auc_np(D["y"][te_m], sc_t["fuse2_V1"])
        a0 = M.auc_np(D["y"][te_m], sc_t["bp_lda"])
        print(f"  [{scheme} T={T}] {s}: fuse2 {a1:.6f}  lda {a0:.6f}", flush=True)
    return streams, tiled_sc, thr_rows, models


def cell(D, tiled_sc, clf, boots):
    per, rows, bm = [], [], np.zeros((B_BOOT, len(SESSIONS)))
    for si, s in enumerate(SESSIONS):
        m = D["sess"] == s
        y, x, ev = D["y"][m], tiled_sc[s][clf], D["evn"][m]
        a = M.auc_np(y, x)
        per.append(a)
        ue = np.unique(ev)
        at = M.auc_np(np.array([y[ev == e][0] for e in ue]),
                      np.array([x[ev == e].mean() for e in ue]))
        rows.append(dict(session=s, n_win=int(m.sum()), auc_tiled=a, auc_trial=at))
        idx_iw = [np.where(ev == e)[0] for e in ue if e.startswith("iw")]
        idx_rs = [np.where(ev == e)[0] for e in ue if e.startswith("rest")]
        for b in range(B_BOOT):
            ii = np.concatenate([idx_iw[i] for i in boots[b][si][0]]
                                + [idx_rs[i] for i in boots[b][si][1]])
            bm[b, si] = M.auc_np(y[ii], x[ii])
    mean_auc = float(np.mean(per))
    lo, hi = np.nanpercentile(np.nanmean(bm, axis=1), [2.5, 97.5])
    ci = np.nanpercentile(bm, [2.5, 97.5], axis=0)
    for i, r in enumerate(rows):
        r["ci_lo"], r["ci_hi"] = float(ci[0, i]), float(ci[1, i])
    return rows, mean_auc, float(lo), float(hi)
