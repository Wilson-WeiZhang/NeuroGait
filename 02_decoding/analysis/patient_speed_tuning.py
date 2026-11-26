"""Speed tuning of gait-phase locking |c| in the patient sessions, the same measure as the healthy curves of speed_tuning.py.

Per session (stage `sessions`), the single-trial projections of speed_tuning.projections (same band rule, scale grid 0.80-1.20 in
steps of 0.02, k1 cluster left + right sensorimotor, k2 cluster midline) on the patient's zero-phase EEG export.  Imagined-walking
trials: the patient rows of the scoring mask (scored or not) in event order whose walker left hip is finite in at least 90 % of
the trial, then at least 2 s; rest: every rest event of the session, borrowing the walker phase of imagined trial (j mod n), at
least 2 s.  Halves: odd = 1st, 3rd, ... imagined trial by time, even = 2nd, 4th, ...; per-session curve = mean of the odd and the
even half curves.
Summary (stage `summary`): median over sessions per scale, SE of a median = 1.253 sd / sqrt(n), band = median +- 1.96 SE.
  pointwise test   imagined (half footing) against yoked rest, paired Wilcoxon, two-sided, per scale; BH over the 21 scales.
  per patient      median over that patient's sessions.
  rest drawn down  (descriptive) rest drawn to the session's imagined half size, 200 draws without replacement, seed
                   [20261110, 152, 1, session digits]; pointwise tests repeated.
  trial-count floor  each healthy session's odd and even halves drawn to n* = floor of the median over the patient sessions of
                   (n_odd + n_even) / 2, without replacement (a half already <= n* is used whole), 200 draws, seed
                   [20261110, 152, 2, session digits]; mean over draws of the session's half-footing curve.

Named inputs: scoring_mask.csv; label patients; patient_prep_triples, patient_prep_triples_repaired (per-session EEG, goniometer
and event exports; the repaired export holds the sessions whose reference channel was repaired), label
repaired_reference_sessions (comma-separated); patient_peak_bands_1.csv .. _3.csv (walker goniometer stride frequency and k bands);
speed_tuning_trials.npz (healthy single-trial projections, from speed_tuning.py).
Named outputs: patient_speed_session.csv, patient_speed_group.csv, patient_speed_pointwise.csv.
Usage: python patient_speed_tuning.py sessions [--workers N] | summary
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import OUT, inp, label, write_table
import speed_tuning as ST

SCALES = ST.SCALES
I08, I10, I12 = (int(np.flatnonzero(np.isclose(SCALES, v))[0]) for v in (0.8, 1.0, 1.2))
NDRAW = 200
SEED = 20261110
PARTS = OUT / "patient_speed_parts"


def digits(s):
    return int("".join(c for c in s if c.isdigit()))


def sessions():
    mask = pd.read_csv(inp("scoring_mask.csv"))
    return sorted(mask[(mask.cohort == label("patients")) & (mask.condition == "IW")].session.unique())


def patient_id(sess_list):
    return {p: f"P{i + 1}" for i, p in enumerate(sorted({s.split("_")[0] for s in sess_list}))}


def root_of(sess):
    repaired = set(label("repaired_reference_sessions").split(","))
    return inp("patient_prep_triples_repaired" if sess in repaired else "patient_prep_triples")


def patient_bands(sess):
    bands = pd.concat([pd.read_csv(inp(f"patient_peak_bands_{i}.csv")) for i in (1, 2, 3)])
    b = bands[bands.session == sess]
    assert sorted(b.harmonic_k.tolist()) == [1, 2, 3], (sess, b.harmonic_k.tolist())
    return b.set_index("harmonic_k")


def patient_trials(sess, ev, gx, glabels):
    """Imagined walking: the mask rows of the session in event order, walker hip finite in >= 90 % of the trial; rest: every rest event."""
    mask = pd.read_csv(inp("scoring_mask.csv"))
    m = mask[(mask.session == sess) & (mask.cohort == label("patients")) & (mask.condition == "IW")].sort_values("ev_trial_idx")
    iw = [ev[int(i) - 1] for i in m.ev_trial_idx]
    assert all(str(e.code) == "gaitimagery" for e in iw), sess
    h_w = gx[:, glabels.index("LHipW X")]
    iw = [e for e in iw if np.isfinite(h_w[int(e.gtstart) - 1: int(e.gtstop)]).mean() >= 0.9]
    rest = [e for e in ev if str(e.code) == "rest"]
    return iw, rest, len(m)


def session_curves(x, fs, labels, gx, gfs, glabels, iw, rest_events, b):
    """speed_tuning.one_session from the EEG to the projections, with the trial lists passed in."""
    x = x.astype(float)
    lab = pd.Index(labels)
    bb1 = [b.loc[1].band_lo, b.loc[1].band_hi]
    f0 = float(b.loc[1].peak_hz)
    phiL = ST.goni_phase(gx, glabels, "LHipW X", *bb1, gfs)
    keep = [e for e in iw if ST.span(e, x.shape[0], len(phiL))[2] >= 2 * fs]
    sp_iw = [ST.span(e, x.shape[0], len(phiL)) for e in keep]
    ph_iw = [np.unwrap(phiL[g0: g0 + L]) for _, g0, L in sp_iw]
    sp_rest, ph_rest = [], []
    for j, t in enumerate(rest_events):
        e0, _, L = ST.span(t, x.shape[0], len(phiL))
        p = ph_iw[j % len(ph_iw)]
        n = min(L, len(p))
        if n >= 2 * fs:
            sp_rest.append((e0, 0, n))
            ph_rest.append(p[:n])
    A = ST.projections(x, fs, lab, f0, {"IW": (sp_iw, ph_iw), "REST": (sp_rest, ph_rest)})
    out = {}
    for (cond, k), a in A.items():
        out[f"A_{cond}_k{k}"] = a
        out[f"c_{cond}_k{k}"] = np.abs(a.mean(axis=1)).mean(axis=1)
    out["rank_iw"] = np.argsort(np.argsort([int(e.etstart) for e in keep]))
    out["f0"] = np.array(f0)
    return out


def half_curves(A, rank):
    """|c| on the odd (rank 0, 2, ...) and even (rank 1, 3, ...) halves of the imagined-walking trials."""
    A = A.astype(complex)
    mo, me = rank % 2 == 0, rank % 2 == 1
    return np.abs(A[:, mo].mean(axis=1)).mean(axis=1), np.abs(A[:, me].mean(axis=1)).mean(axis=1)


def one_session(sess):
    part = PARTS / f"{sess}.npz"
    if part.exists():
        return sess, "skipped", 0.0
    t0 = time.time()
    x, fs, labels, gx, gfs, glabels, ev = ST.load_triple(sess, root_of(sess))
    iw, rest, nmask = patient_trials(sess, ev, gx, glabels)
    o = session_curves(x, fs, labels, gx, gfs, glabels, iw, rest, patient_bands(sess))
    out = {}
    for c in ("IW", "REST"):
        for k in (1, 2):
            out[f"A_{c}_k{k}"] = o[f"A_{c}_k{k}"].astype(np.complex64)
            out[f"c_{c}_k{k}"] = o[f"c_{c}_k{k}"]
    for k in (1, 2):
        co, ce = half_curves(out[f"A_IW_k{k}"], o["rank_iw"])
        out[f"c_odd_k{k}"], out[f"c_even_k{k}"], out[f"c_half_k{k}"] = co, ce, 0.5 * (co + ce)
    rk = o["rank_iw"]
    out.update(rank_iw=rk, f0=o["f0"], n_iw=np.array(out["A_IW_k1"].shape[1]), n_odd=np.array(int((rk % 2 == 0).sum())),
               n_even=np.array(int((rk % 2 == 1).sum())), n_rest=np.array(out["A_REST_k1"].shape[1]))
    tmp = PARTS / f"{sess}.tmp.npz"
    np.savez_compressed(tmp, **out)
    os.replace(tmp, part)
    return sess, "done", time.time() - t0


# ---- summary ---------------------------------------------------------------------------------------------------------
def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok):
        o = ok[np.argsort(p[ok])]
        r = p[o] * len(ok) / np.arange(1, len(ok) + 1)
        q[o] = np.minimum(np.minimum.accumulate(r[::-1])[::-1], 1.0)
    return q


def wil(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return np.nan if np.all(d == 0) else float(stats.wilcoxon(a, b).pvalue)


def se_med(A):
    return 1.253 * A.std(axis=0, ddof=1) / np.sqrt(len(A))


def session_table(P, post):
    pid = patient_id(post)
    rows = []
    for s in post:
        z = P[s]
        for k in (1, 2):
            for i, sc_ in enumerate(SCALES):
                rows.append(dict(session=s, patient=pid[s.split("_")[0]], harmonic_k=k, scale=float(sc_), c_half=z[f"c_half_k{k}"][i], c_odd=z[f"c_odd_k{k}"][i], c_even=z[f"c_even_k{k}"][i],
                                 c_all_trials=z[f"c_IW_k{k}"][i], c_rest=z[f"c_REST_k{k}"][i], n_iw=int(z["n_iw"]), n_odd=int(z["n_odd"]), n_even=int(z["n_even"]), n_rest=int(z["n_rest"]), f0_hz=float(z["f0"])))
    return pd.DataFrame(rows)


def stacked(P, post):
    H = {k: np.array([P[s][f"c_half_k{k}"] for s in post]) for k in (1, 2)}
    Rr = {k: np.array([P[s][f"c_REST_k{k}"] for s in post]) for k in (1, 2)}
    Al = {k: np.array([P[s][f"c_IW_k{k}"] for s in post]) for k in (1, 2)}
    return H, Rr, Al


def group_rows(H, Rr, Al):
    """Group curves of imagined walking (half footing), yoked rest and all imagined trials; pointwise imagined against rest."""
    grp, pw = [], []
    n = len(H[1])
    for k in (1, 2):
        med, se = np.median(H[k], axis=0), se_med(H[k])
        mr, ser = np.median(Rr[k], axis=0), se_med(Rr[k])
        ma = np.median(Al[k], axis=0)
        for nm, m_, s_ in (("SCI imagined (half footing)", med, se), ("SCI rest (yoked)", mr, ser), ("SCI imagined (all trials)", ma, se_med(Al[k]))):
            grp += [dict(group=nm, harmonic_k=k, scale=float(sc_), median_c=float(a), se=float(b), ci_lo=float(a - 1.96 * b), ci_hi=float(a + 1.96 * b), n_sessions=n) for sc_, a, b in zip(SCALES, m_, s_)]
        p_ = np.array([wil(H[k][:, j], Rr[k][:, j]) for j in range(len(SCALES))])
        q_ = bh(p_)
        pw += [dict(group="SCI imagined vs rest", harmonic_k=k, scale=float(sc_), n_pairs=n, median_diff=float(np.median(H[k][:, j] - Rr[k][:, j])), p=float(a), pFDR=float(b), bar=bool(b < 0.05)) for j, (sc_, a, b) in enumerate(zip(SCALES, p_, q_))]
    return grp, pw


def patient_rows(H, Rr, post):
    grp = []
    pid = patient_id(post)
    for who, name in pid.items():
        ix = [i for i, s in enumerate(post) if s.split("_")[0] == who]
        for k in (1, 2):
            m = np.median(H[k][ix], axis=0)
            mr = np.median(Rr[k][ix], axis=0)
            for nm, v in (("SCI imagined (half footing)", m), ("SCI rest (yoked)", mr)):
                grp += [dict(group=f"{name} {nm}", harmonic_k=k, scale=float(sc_), median_c=float(a), se=np.nan, ci_lo=np.nan, ci_hi=np.nan, n_sessions=len(ix)) for sc_, a in zip(SCALES, v)]
    return grp


def drawn_rest_rows(P, post, H, Rr):
    """Descriptive: rest drawn down to the session's imagined half size."""
    grp, pw = [], []
    n = len(post)
    Rd = {k: np.zeros_like(Rr[k]) for k in (1, 2)}
    for i, s in enumerate(post):
        z = P[s]
        nh = int(round(0.5 * (int(z["n_odd"]) + int(z["n_even"]))))
        nr = int(z["n_rest"])
        rng = np.random.default_rng([SEED, 152, 1, digits(s)])
        for k in (1, 2):
            A = z[f"A_REST_k{k}"].astype(complex)
            if nr <= nh:
                Rd[k][i] = z[f"c_REST_k{k}"]
                continue
            acc = np.zeros(len(SCALES))
            for _ in range(NDRAW):
                j = rng.choice(nr, nh, replace=False)
                acc += np.abs(A[:, j].mean(axis=1)).mean(axis=1)
            Rd[k][i] = acc / NDRAW
    for k in (1, 2):
        md = np.median(Rd[k], axis=0)
        grp += [dict(group="SCI rest drawn to the half size (descriptive)", harmonic_k=k, scale=float(sc_), median_c=float(a), se=float(b), ci_lo=float(a - 1.96 * b), ci_hi=float(a + 1.96 * b), n_sessions=n) for sc_, a, b in zip(SCALES, md, se_med(Rd[k]))]
        p_ = np.array([wil(H[k][:, j], Rd[k][:, j]) for j in range(len(SCALES))])
        q_ = bh(p_)
        pw += [dict(group="SCI imagined vs rest drawn to the half size (descriptive)", harmonic_k=k, scale=float(sc_), n_pairs=n, median_diff=float(np.median(H[k][:, j] - Rd[k][:, j])), p=float(a), pFDR=float(c), bar=bool(c < 0.05)) for j, (sc_, a, c) in enumerate(zip(SCALES, p_, q_))]
    return grp, pw


def floor_rows(P, post, H):
    """Healthy curves with each half drawn to the patients' median half size, beside the original healthy and the patient curves."""
    ns = int(np.floor(float(np.median([0.5 * (int(P[s]["n_odd"]) + int(P[s]["n_even"])) for s in post]))))
    Z = np.load(inp("speed_tuning_trials.npz"))
    hs = sorted({k.split("|")[0] for k in Z.files})
    Hh = {k: np.zeros((len(hs), len(SCALES))) for k in (1, 2)}
    Hm = {k: np.zeros((len(hs), len(SCALES))) for k in (1, 2)}
    Hr = {k: np.zeros((len(hs), len(SCALES))) for k in (1, 2)}
    for i, s in enumerate(hs):
        rk = Z[f"{s}|rank_iw"]
        halves = (rk % 2 == 0, rk % 2 == 1)
        rng = np.random.default_rng([SEED, 152, 2, digits(s)])
        AA = {k: Z[f"{s}|A_IW_k{k}"].astype(complex) for k in (1, 2)}
        for k in (1, 2):
            Hr[k][i] = Z[f"{s}|c_REST_k{k}"]
            co, ce = (np.abs(AA[k][:, m].mean(axis=1)).mean(axis=1) for m in halves)
            Hh[k][i] = 0.5 * (co + ce)
        acc = {k: np.zeros(len(SCALES)) for k in (1, 2)}
        idx = [np.flatnonzero(m) for m in halves]
        for _ in range(NDRAW):
            pick = [ix if len(ix) <= ns else rng.choice(ix, ns, replace=False) for ix in idx]
            for k in (1, 2):
                acc[k] += 0.5 * sum(np.abs(AA[k][:, p_].mean(axis=1)).mean(axis=1) for p_ in pick)
        for k in (1, 2):
            Hm[k][i] = acc[k] / NDRAW
    grp = []
    for k in (1, 2):
        mh, mm, ms, mr = np.median(Hh[k], axis=0), np.median(Hm[k], axis=0), np.median(H[k], axis=0), np.median(Hr[k], axis=0)
        for nm, v, A_ in (("Healthy imagined, original half footing", mh, Hh[k]), (f"Healthy imagined, halves drawn to {ns} trials", mm, Hm[k]), ("SCI imagined, half footing", ms, H[k]), ("Healthy rest (yoked)", mr, Hr[k])):
            grp += [dict(group=nm, harmonic_k=k, scale=float(sc_), median_c=float(a), se=float(b), ci_lo=float(a - 1.96 * b), ci_hi=float(a + 1.96 * b), n_sessions=len(A_)) for sc_, a, b in zip(SCALES, v, se_med(A_))]
    return grp


def summary():
    post = sessions()
    P = {s: dict(np.load(PARTS / f"{s}.npz")) for s in post}
    H, Rr, Al = stacked(P, post)
    grp, pw = group_rows(H, Rr, Al)
    grp += patient_rows(H, Rr, post)
    g2, p2 = drawn_rest_rows(P, post, H, Rr)
    grp += g2
    pw += p2
    grp += floor_rows(P, post, H)
    write_table(session_table(P, post), "patient_speed_session.csv", index=False, float_format="%.8g")
    write_table(pd.DataFrame(grp), "patient_speed_group.csv", index=False, float_format="%.8g")
    write_table(pd.DataFrame(pw), "patient_speed_pointwise.csv", index=False, float_format="%.8g")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("sessions", "summary"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "sessions":
        PARTS.mkdir(parents=True, exist_ok=True)
        pending = sessions()
        with Pool(a.workers) as p:
            for i, (s, st, dt) in enumerate(p.imap_unordered(one_session, pending), 1):
                print(f"{i}/{len(pending)} {s} {st} {dt:.1f} s", flush=True)
    else:
        summary()


if __name__ == "__main__":
    main()
