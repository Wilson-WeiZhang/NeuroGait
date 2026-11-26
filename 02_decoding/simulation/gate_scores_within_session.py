"""Figure 7 — physics simulation and walk-intent gate

gate_scores_within_session — the gate under within-session 10-fold cross-validation (by event) on the post-operative patient sessions: one rung (loader, Karcher solve, fold scheme, scoring) and the leakage and stream self-checks. Run by gate_scores_causal.py.

Reads:
  the aligned per-session recordings (not distributed)

Part of the simulation and gate stage: it reads decoder output and
writes the rollout or classifier tables the Figure 7 renderers read.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"

import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import karcher_checked as KC
import gate_features as pgf
import gate_fusion as PF
import gate_window3_within_session as M
import gate_window3_fold_reference as FR


SESS11 = ["P1_sess04", "P1_sess05", "P1_sess06", "P1_sess07", "P1_sess08",
          "P1_sess09", "P2_sess03", "P2_sess05", "P2_sess06",
          "P2_sess08", "P2_sess09"]

LAM = PF.LAM_RA
REF_MIN_S = FR.REF_MIN_S
SEED, B_BOOT, NFOLD = M.SEED, M.B_BOOT, M.NFOLD
HOP_S = M.HOP_S

KARCHER_LOG = []


def log(m):
    print(m, flush=True)


def checked_whiteners(covs, lam=LAM):
    """Band whiteners R^-1/2 from the converged Karcher mean of the reference covariances; convergence is recorded."""
    res = KC.band_karcher(covs, lam)
    KARCHER_LOG.extend(res)
    return [PF.powm_sym(r.mean, -0.5) for r in res]


SOLVERS = {"checked": checked_whiteners}


def prepare_raw2(sessions, root, T, base_min_s):
    """FR.prepare_raw, parameterised over (sessions, root, base_min_s) and tolerant of an
    EMPTY calibration list (which is what base_min_s = 0 produces -- the allrest pool)."""
    t0 = time.time()
    pgf.WIN_S = T
    PF.WIN_S = T
    out, anchors, inv = {}, [], []
    for s in sessions:
        X, fs, cal, ordered_eval = PF.session_layout(root, s, base_min_s)
        calP, calC = [], []
        for e in cal:
            for w in pgf.windows(X, e["a"], e["b"], fs):
                calP.append(pgf.bandpower(w, fs))
                calC.append(PF.band_covs(w))
        tiled, dense, dur, order = {}, {}, {}, []
        n_t = n_d = n_zero = 0
        for lab, k, e in ordered_eval:
            name = f"{lab}{k}"
            order.append(name)
            dur[name] = (e["b"] - e["a"]) / fs
            anchors.append(dict(session=s, event=name, role="evaluated",
                                label=1 if lab == "iw" else 0, a=e["a"], b=e["b"],
                                t_start_s=e["a"] / fs, t_stop_s=e["b"] / fs,
                                dur_s=(e["b"] - e["a"]) / fs))
            ws = pgf.windows(X, e["a"], e["b"], fs)
            if ws:
                tiled[name] = (np.vstack([pgf.bandpower(w, fs) for w in ws]),
                               [PF.band_covs(w) for w in ws])
                n_t += len(ws)
            else:
                n_zero += 1
            dw = M.dense_windows(X, e["a"], e["b"], fs, T)
            if dw:
                dense[name] = ([t for t, _w in dw],
                               np.vstack([pgf.bandpower(w, fs) for _t, w in dw]),
                               [PF.band_covs(w) for _t, w in dw])
                n_d += len(dw)
        anom = []
        if abs(fs - 250.0) > 1e-9:
            anom.append(f"fs={fs}")
        if n_zero:
            anom.append(f"events_without_a_{T:.0f}s_window={n_zero} "
                        f"({','.join(n for n in order if n not in tiled)})")
        out[s] = dict(calP=(np.vstack(calP) if calP else None), calC=calC,
                      tiled=tiled, dense=dense, dur=dur, order=order)
        inv.append(dict(session=s, T=T, root=os.path.basename(root), base_min_s=base_min_s,
                        n_iw_ev=sum(1 for n in order if n.startswith("iw")),
                        n_rest_ev=sum(1 for n in order if n.startswith("rest")),
                        n_calib_ev=len(cal), n_calib_win=len(calC),
                        n_events_with_windows=len(tiled), n_tiled_win=n_t, n_dense_dec=n_d,
                        anomalies="; ".join(anom)))
        log(f"  [raw T={T} {os.path.basename(root)} base={base_min_s:g}] {s}: "
            f"{len(order)} events ({len(tiled)} windowable), {n_t} tiled, {n_d} dense"
            + (f"   ANOMALY: {'; '.join(anom)}" if anom else ""))
    log(f"  prepare_raw2 done in {time.time() - t0:.0f} s")
    return out, anchors, inv


def minimal_D(RAW, sessions, T):
    """The (T, y, sess, evn, onset) frame FR.run_scheme and M.folds_for consume."""
    sess, evn, y = [], [], []
    for s in sessions:
        for n in RAW[s]["order"]:
            if n not in RAW[s]["tiled"]:
                continue
            k = len(RAW[s]["tiled"][n][0])
            sess += [s] * k
            evn += [n] * k
            y += [1 if n.startswith("iw") else 0] * k
    onset = {}
    return dict(T=T, y=np.array(y), sess=np.array(sess), evn=np.array(evn), onset=onset)


def attach_onsets(D, anchors):
    for r in anchors:
        D["onset"][(r["session"], r["event"])] = r["a"]
    return D


def fold_table(D, sessions, T, tag):
    """The fold assignment, and the (session, event) -> fold map, from M.folds_for."""
    rows, fmap = [], {}
    for s in sessions:
        names, folds = M.folds_for(D, s, "within_random")
        assert len(folds) == NFOLD, f"{s}: {len(folds)} folds"
        seen = set()
        for k, (tr, te) in enumerate(folds):
            for i in te:
                assert names[i] not in seen, f"{s}: {names[i]} in two test folds"
                seen.add(names[i])
                fmap[(s, names[i])] = k
                rows.append(dict(tag=tag, T=T, session=s, fold=k, event=names[i],
                                 label=1 if names[i].startswith("iw") else 0, role="test"))
            assert set(tr).isdisjoint(set(te)), f"{s} fold {k}: train/test overlap"
            assert len(tr) + len(te) == len(names), f"{s} fold {k}: partition incomplete"
        assert seen == set(names), f"{s}: test folds do not cover the event list"
    return rows, fmap


def run_rung(name, sessions, root, T, solver, base_min_s, scheme, want_dense, chk,
             RAWCACHE, fold_rows=None):
    """One rung of the ladder. Returns (per-session AUC dict, mean, ci_lo, ci_hi, stream,
    thr_rows, D, fmap)."""
    log(f"\n=========== RUNG {name}: {len(sessions)} sess, {os.path.basename(root)}, T={T}, "
        f"solver={solver}, base_min_s={base_min_s:g}, ref={scheme} ===")
    key = (tuple(sessions), root, T, base_min_s)
    if key not in RAWCACHE:
        RAWCACHE[key] = prepare_raw2(sessions, root, T, base_min_s)
    RAW, anchors, inv = RAWCACHE[key]
    D = attach_onsets(minimal_D(RAW, sessions, T), anchors)
    frows, fmap = fold_table(D, sessions, T, name)
    if fold_rows is not None:
        fold_rows.extend(frows)

    n0 = len(KARCHER_LOG)
    FR.SESSIONS = list(sessions)
    M.SESSIONS = list(sessions)
    FR.whiteners_safe = SOLVERS[solver]
    FR.REF_MIN_S = REF_MIN_S
    stream, tsc, thr_rows, _mods = FR.run_scheme(D, RAW, T, scheme, want_dense=want_dense)
    nonconv = sum(1 for r in KARCHER_LOG[n0:] if not r.converged)
    if solver == "checked":
        its = [r.n_iter for r in KARCHER_LOG[n0:]]
        log(f"  [karcher {name}] {len(its)} checked solves, iters "
            f"{min(its)}..{max(its)} (median {int(np.median(its))}), non-converged {nonconv}")
        chk.append(dict(check=f"{name}_karcher", key="non_converged", n=len(its),
                        max_abs_delta=nonconv, tol=0, passed=int(nonconv == 0)))

    boots = M.boot_index(D)
    rows, mean_, lo, hi = FR.cell(D, tsc, "fuse2_V1", boots)
    per = {r["session"]: r["auc_tiled"] for r in rows}
    for r in thr_rows:
        r["rung"] = name
    log(f"[RUNG {name}] fuse2 V1 tiled mean-of-{len(sessions)} {mean_:.6f} [{lo:.4f}, {hi:.4f}]")
    return per, mean_, lo, hi, stream, thr_rows, rows, D, fmap, inv


def check_no_leakage(name, thr_rows, D, sessions, fmap, chk):
    """Protocol 6.5: for every fold, reference and fit events are TRAINING events only."""
    bad = 0
    for r in thr_rows:
        s, k = r["session"], r["fold"]
        refs = [x for x in str(r["ref_events"]).split(";") if x]
        for n in refs:
            if fmap[(s, n)] == k:
                bad += 1
        assert r["n_ref_events"] + r["n_fit_events"] == \
            sum(1 for (ss, nn), kk in fmap.items() if ss == s and kk != k), \
            f"{name} {s} fold {k}: ref+fit != training-event count"
    chk.append(dict(check=f"{name}_no_leakage", key="ref_events_in_test_fold",
                    n=len(thr_rows), max_abs_delta=bad, tol=0, passed=int(bad == 0)))
    assert bad == 0, f"{name}: {bad} reference events came from the test fold"
    log(f"[SELF-CHECK {name} leakage] 0 reference events from a test fold, "
        f"ref+fit == training count on all {len(thr_rows)} folds")


def check_stream(name, stream, RAWinv, fmap, chk):
    """Protocol 6.3 / 6.6: dense row counts and one-fold-per-event."""
    per, evfold = {}, {}
    for r in stream:
        per[r["session"]] = per.get(r["session"], 0) + 1
        k = (r["session"], r["event"])
        evfold.setdefault(k, set()).add(r["fold"])
    worst = 0
    for d in RAWinv:
        got, want = per.get(d["session"], 0), d["n_dense_dec"]
        worst = max(worst, abs(got - want))
        chk.append(dict(check=f"{name}_dense_row_count", key=d["session"], n=got,
                        max_abs_delta=abs(got - want), tol=0, passed=int(got == want)))
        assert got == want, f"{name} {d['session']}: {got} dense rows, loader says {want}"
    multi = [k for k, v in evfold.items() if len(v) != 1]
    assert not multi, f"{name}: events scored from more than one fold: {multi[:5]}"
    mism = [k for k, v in evfold.items() if list(v)[0] != fmap[k]]
    assert not mism, f"{name}: fold column disagrees with the fold map: {mism[:5]}"
    chk.append(dict(check=f"{name}_one_fold_per_event", key="all", n=len(evfold),
                    max_abs_delta=len(multi) + len(mism), tol=0,
                    passed=int(not multi and not mism)))
    log(f"[SELF-CHECK {name} stream] {len(stream)} rows, {len(evfold)} events, "
        f"max |delta| vs loader {worst}, one fold per event, fold column consistent")
