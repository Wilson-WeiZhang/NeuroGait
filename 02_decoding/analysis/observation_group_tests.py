"""Group-level tests of imagined walking against observation only (six participants) and of the patients against rest.

Rows of observation_group_tests.csv, by item:
  G4   |c| minus the yoked-rest null mean per participant, by block (imagined walking with observation, observation only)
       and harmonic: one-sample t against 0 (BH across k), exact sign-flip p, participants above their own rest p95.
  G5   similarity of the six-participant group map (imagined walking, observation only) to the group map of the other
       36 participants (imagined or real walking), against random six-participant rest group maps (B = 10,000,
       one-sided, BH across k).
  G6/G7  imagined walking minus observation only, cycle-label permutation of the per-trial sums: per participant 15 of
       the 30 cycles relabelled, joint draws B = 20,000 (seed [20261008, participant number, 2]); statistics: group-map
       similarity to the imagined / real walking group map, mean per-participant similarity to the leave-participant-out
       maps, central left-right cosine of the group map; two-sided, BH across k.
  G8   phase offsets (Rayleigh across participants, BH across k); "G8 group map": k = 2 group-map offsets with a
       cycle-pair bootstrap 95 % CI (B = 2,000, seed [20261008, participant number, 3], participants fixed).
  G9   decoder output, imagined walking vs observation only: participant means of the k = 1 power share, k = 1 and
       k = 2 agreement and half-stride error fraction; paired t (BH across the four), exact sign-flip, trial-level LME
       (statsmodels mixedlm, value ~ condition, random intercept and slope by participant, REML).
  G10a mean patient-session similarity to the healthy imagined-walking map against random sets of 11 healthy rest
       sessions (B = 100,000, one-sided, BH across k).
  G10b pooled post-operative patient map against random eleven-participant rest group maps (B = 10,000, one-sided).
The random draws of G5, G10a and G10b come in this order from one generator, seed 20261104.
rest_input_absolute.csv: per participant and block, median decoding r on the task trials and on the resting-input
windows (and on the paired subset).

Named inputs: healthy maps (see state_locking.py), observation_phase_maps_by_condition.csv, observation_locking_table.csv,
observation_trial_sums_<participant>.mat, session_similarity_by_state.csv, observation_timing.csv (both from
state_locking.py), observation_dump_<participant> (folder of *_dump.h5), observation_scoring_mask_<participant>.csv,
patient_maps.csv, patient_session_similarity.csv, observation_rest_input_<participant>_<block>.csv; label
post_operative_sessions (comma-separated patient session names).
Named outputs (derived/): observation_group_tests.csv, rest_input_absolute.csv, patient_rest_floor.csv.
"""
import h5py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from analysis_common import inp, label, write_table
from harmonic_agreement import FS, proj, stride_freq, z
from state_locking import BLOCKS, KS, bh, circ, coh, gmean, load_cube, load_states

LEFT = ["FC3", "FC1", "C3", "C1", "CP3", "CP1"]
RIGHT = ["FC4", "FC2", "C4", "C2", "CP4", "CP2"]
PERM_SEED, B_GROUP = 20261008, 20_000
CC, TC = label("condition_code"), label("test_condition_column")   # per-trial condition code; block column of the tables


# ------------------------------------------------------------------ per-trial sums (MATLAB v7.3 file)
def _arr(f, name):
    x = f[name][()]
    return np.transpose(x) if x.ndim > 1 else x   # column-major -> reverse axes


def _chars(f, name):
    x = np.atleast_2d(f[name][()]).T
    return ["".join(chr(c) for c in row).rstrip() for row in x]


def load_trial_sums(path):
    """Per-trial complex sums of one participant: SP [trial, channel, point, k], SA [trial, channel, k], counts."""
    with h5py.File(path, "r") as f:
        d = {k: _arr(f, k) for k in ("sum_point_re", "sum_point_im", "count_point_ch", "sum_all_re", "sum_all_im",
                                     "count_all_ch")}
        for k in ("cycle", CC):
            d[k] = np.asarray(f[k][()]).ravel()
        d["chan_labels"] = _chars(f, "chan_labels")
    n_tr = d[CC].size
    for k in ("sum_point_re", "sum_point_im", "count_point_ch"):
        d[k] = d[k].reshape(n_tr, d[k].shape[1], d[k].shape[2], -1)
    for k in ("sum_all_re", "sum_all_im", "count_all_ch"):
        d[k] = d[k].reshape(n_tr, d[k].shape[1], -1)
    d["SP"] = d["sum_point_re"] + 1j * d["sum_point_im"]
    d["SA"] = d["sum_all_re"] + 1j * d["sum_all_im"]
    return d


def cycle_layout(d):
    """Cycles (30), trial-to-cycle matrix [30 x trial], cycles of imagined walking (15), adjacent cycle pairs (15)."""
    cyc = d["cycle"].astype(int)
    cond = d[CC].astype(int)
    cycles = np.unique(cyc)
    assert cycles.size == 30
    cyc_cond = {}
    for c in cycles:
        a = np.unique(cond[cyc == c])
        assert a.size == 1
        cyc_cond[c] = int(a[0])
    T = (cyc[None, :] == cycles[:, None]).astype(float)
    miao_cyc = np.array([cyc_cond[c] == 1 for c in cycles])
    assert miao_cyc.sum() == 15
    pairs = cycles.reshape(15, 2)
    for p in pairs:
        assert {cyc_cond[p[0]], cyc_cond[p[1]]} == {1, 2}
    return cycles, T, miao_cyc, pairs


def subj_num(part):
    return int(part.split("_")[1])


def primary_draws(part, B, purpose):
    rng = np.random.default_rng([PERM_SEED, subj_num(part), purpose])
    idx = np.argsort(rng.random((B, 30)), axis=1)[:, :15]
    S = np.zeros((B, 30))
    np.put_along_axis(S, idx, 1.0, axis=1)
    return S


# ------------------------------------------------------------------ batched map statistics
def pm_batch(d, ki, W, order):
    """Complex phase maps for trial weights W [n_draw x n_trial]: [n_draw, channel, 9 points], channels in `order`."""
    W = np.atleast_2d(W)
    n_tr = d["SP"].shape[0]
    sp = W @ d["SP"][:, :, :, ki].reshape(n_tr, -1)
    cp = W @ d["count_point_ch"][:, :, :, ki].reshape(n_tr, -1)
    sa = W @ d["SA"][:, :, ki]
    ca = W @ d["count_all_ch"][:, :, ki]
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = sp / cp
        ma = sa / ca
    mp[(cp == 0) | ~np.isfinite(mp)] = np.nan
    ma[(ca == 0) | ~np.isfinite(ma)] = np.nan
    pm = mp.reshape(W.shape[0], ma.shape[1], -1) - ma[:, :, None]
    return pm[:, order, :]


def coh_batch(P, H):
    """Spatial similarity for a batch of maps P [n, channel, point] against one map H, NaN cells dropped per map."""
    ok = np.isfinite(P.real) & np.isfinite(H.real)[None]
    P0 = np.where(ok, P, 0)
    H0 = np.where(ok, H[None], 0)
    num = np.abs(np.sum(np.conj(H0) * P0, axis=(1, 2)))
    return num / np.sqrt(np.sum(np.abs(P0) ** 2, axis=(1, 2)) * np.sum(np.abs(H0) ** 2, axis=(1, 2)))


def cos_lr_batch(G, k, idxL, idxR):
    ph = np.exp(1j * 2 * np.pi * k * np.arange(8) / 8)
    a = np.nanmean(G[:, :, :8] * ph[None, None, :], axis=2)
    aL, aR = np.nanmean(a[:, idxL], axis=1), np.nanmean(a[:, idxR], axis=1)
    return np.real(aL * np.conj(aR)) / (np.abs(aL) * np.abs(aR))


def p2(null, obs):
    return (1 + np.sum(np.abs(null) >= np.abs(obs) - 1e-12)) / (null.size + 1)


def p1(null, obs):
    return (1 + np.sum(null >= obs - 1e-12)) / (null.size + 1)


def signflip(x):
    x = np.asarray(x, float)
    n = len(x)
    m = np.array([np.mean(x * np.array([1 if b >> i & 1 else -1 for i in range(n)])) for b in range(2 ** n)])
    return float(np.mean(np.abs(m) >= abs(x.mean()) - 1e-12))


# ------------------------------------------------------------------ items
def locking_excess():
    b = pd.read_csv(inp("observation_locking_table.csv"))
    b["ex"] = b.abs_c - b.rest_null_mean
    rows = []
    for cond, g in b.groupby(TC):
        ps = [stats.ttest_1samp(g[g.harmonic_k == k].ex, 0).pvalue for k in KS]
        for k, q, p in zip(KS, bh(ps), ps):
            x = g[g.harmonic_k == k].ex.to_numpy()
            t = stats.ttest_1samp(x, 0).statistic
            n_up = int(g[g.harmonic_k == k].above_rest_p95.sum())
            rows.append({"item": "G4", TC: cond, **dict(harmonic_k=k, mean_excess=x.mean(), t=t, p=p, pFDR=q,
                             signflip_p=signflip(x), above_p95=n_up)})
    return rows


def decoder_trials():
    """Per scored trial of the observation sessions: mean over joints of k1 power share, k1 / k2 agreement, swap."""
    tr = []
    for p, (mi, ao, _, _) in BLOCKS.items():
        mask = pd.read_csv(inp(f"observation_scoring_mask_{p}.csv"))
        kp = set(zip(mask[mask.keep == 1][TC], mask[mask.keep == 1].trial_idx))
        for st, blk in (("IW", mi), ("OBS", ao)):
            with h5py.File(inp(f"observation_dump_{p}") / f"{blk}_dump.h5", "r") as h:
                C_, Y, T_ = h["rows/C_pred"][:], h["rows/Y_true"][:], h["rows/time_sec"][:]
                tid, off_, nr = h["trials/trial_idx"][:], h["trials/row_offset"][:], h["trials/n_rows"][:]
            for ti, o, n in zip(tid, off_, nr):
                if (blk, int(ti)) not in kp:
                    continue
                c0, y0, t0 = C_[o:o + n], Y[o:o + n], T_[o:o + n]
                hip = 0 if np.isfinite(y0[:, 0]).all() else 3
                vals = []
                for j in range(6):
                    ok = np.isfinite(c0[:, j]) & np.isfinite(y0[:, j]) & np.isfinite(y0[:, hip])
                    if ok.sum() < 2 * FS:
                        continue
                    c, y, t = c0[ok], y0[ok], t0[ok]
                    f0 = stride_freq(y[:, hip])
                    ap = [proj(z(c[:, j]), k * f0, t) for k in KS]
                    at = [proj(z(y[:, j]), k * f0, t) for k in KS]
                    pw = np.abs(ap) ** 2
                    c1 = float(np.cos(np.angle(ap[0] * np.conj(at[0]))))
                    vals.append((pw[0] / pw.sum(), c1, float(np.cos(np.angle(ap[1] * np.conj(at[1])))), float(c1 < 0)))
                if vals:
                    v = np.mean(vals, axis=0)
                    tr.append(dict(participant=p, state=st, trial_idx=int(ti), frac_pred1=v[0], cos_d1=v[1],
                                   cos_d2=v[2], swap1=v[3]))
    return pd.DataFrame(tr)


def main():
    rng = np.random.default_rng(20261104)
    rows = locking_excess()

    CO, chans = load_states()
    SP, _ = load_cube(inp("observation_phase_maps_by_condition.csv"), channels=chans)
    sess = sorted(CO["IW"])
    subj = np.array([s.split("_sess")[0] for s in sess])
    A = {st: np.stack([CO[st][s] for s in sess]) for st in CO}       # [session, channel, point, k]
    parts = sorted(set(subj))
    drop6 = [BLOCKS[p][2].rstrip("_") for p in BLOCKS]
    keep = ~np.isin(subj, drop6)
    G68 = {st: A[st][keep].mean(axis=0) for st in A}
    S6 = {"IW": gmean([SP[BLOCKS[p][0]] for p in BLOCKS]), "OBS": gmean([SP[BLOCKS[p][1]] for p in BLOCKS])}

    # G5: six-participant group maps against the resting floor
    B5 = 10000
    fl = {r: {k: np.empty(B5) for k in KS} for r in ("IW", "RW")}
    for b in range(B5):
        ch = rng.choice(parts, 6, replace=False)
        rs = [rng.choice(np.where(subj == p)[0]) for p in ch]
        Mr = A["REST"][rs].mean(axis=0)
        kp = ~np.isin(subj, ch)
        for r in fl:
            G = A[r][kp].mean(axis=0)
            for k in KS:
                fl[r][k][b] = coh(Mr[:, :, k - 1], G[:, :, k - 1])
    for r in fl:
        for st in ("IW", "OBS"):
            obs = {k: coh(S6[st][:, :, k - 1], G68[r][:, :, k - 1]) for k in KS}
            ps = [p1(fl[r][k], obs[k]) for k in KS]
            for k, q, p in zip(KS, bh(ps), ps):
                rows.append(dict(item="G5", state=st, reference=r, harmonic_k=k, similarity=obs[k],
                                 floor_median=np.median(fl[r][k]), floor_p95=np.quantile(fl[r][k], .95), p=p, pFDR=q))

    # G6, G7: cycle-label permutation on the maps
    D, lay, S_grp, order = {}, {}, {}, {}
    for p in BLOCKS:
        d = load_trial_sums(inp(f"observation_trial_sums_{p}.mat"))
        D[p] = d
        lay[p] = cycle_layout(d)
        S_grp[p] = primary_draws(p, B_GROUP, 2)
        ch = list(d["chan_labels"])
        order[p] = [ch.index(c) for c in chans]
    idxL, idxR = [chans.index(c) for c in LEFT], [chans.index(c) for c in RIGHT]
    Glpo = {r: {p: A[r][subj != BLOCKS[p][2].rstrip("_")].mean(axis=0) for p in BLOCKS} for r in ("IW", "RW")}

    def stats_for(Sdict, k):
        ki = k - 1
        MA, MB = [], []
        out = {}
        pp = {r: [] for r in Glpo}
        for p in BLOCKS:
            T = lay[p][1]
            a = pm_batch(D[p], ki, Sdict[p] @ T, order[p])
            b = pm_batch(D[p], ki, (1 - Sdict[p]) @ T, order[p])
            MA.append(a)
            MB.append(b)
            for r in Glpo:
                g = Glpo[r][p][:, :, ki]
                pp[r].append(coh_batch(a, g) - coh_batch(b, g))
        GA, GB = np.nanmean(np.stack(MA), axis=0), np.nanmean(np.stack(MB), axis=0)
        for r in Glpo:
            g = G68[r][:, :, ki]
            out[f"sim_group_{r}"] = coh_batch(GA, g) - coh_batch(GB, g)
            out[f"sim_perpart_{r}"] = np.mean(np.stack(pp[r]), axis=0)
        out["parity_central"] = cos_lr_batch(GA, k, idxL, idxR) - cos_lr_batch(GB, k, idxL, idxR)
        return out

    Sobs = {p: lay[p][2].astype(float)[None, :] for p in BLOCKS}
    OBS_STAT = {k: stats_for(Sobs, k) for k in KS}
    NULL = {k: {n: [] for n in OBS_STAT[k]} for k in KS}
    for start in range(0, B_GROUP, 500):
        Sc = {p: S_grp[p][start:start + 500] for p in BLOCKS}
        for k in KS:
            s = stats_for(Sc, k)
            for n in s:
                NULL[k][n].append(s[n])
    NULL = {k: {n: np.concatenate(NULL[k][n]) for n in NULL[k]} for k in KS}
    for n in ("sim_group_IW", "sim_group_RW", "sim_perpart_IW", "sim_perpart_RW", "parity_central"):
        ps = [p2(NULL[k][n], OBS_STAT[k][n][0]) for k in KS]
        for k, q, p in zip(KS, bh(ps), ps):
            rows.append(dict(item="G6/G7", statistic=n, harmonic_k=k, delta=float(OBS_STAT[k][n][0]), p=p, pFDR=q))

    # G8: timing
    T = pd.read_csv(inp("observation_timing.csv"))
    for col in ("iw_minus_obs", "obs_vs_rwgroup", "iw_vs_rwgroup"):
        c = [circ(T[T.harmonic_k == k][col].dropna().to_numpy(), k) for k in KS]
        for k, q, (mu, Rb, p) in zip(KS, bh([x[2] for x in c]), c):
            rows.append(dict(item="G8", statistic=col, harmonic_k=k, mean_pct=mu, R=Rb, p=p, pFDR=q))
    # group-map offset with a cycle-pair bootstrap
    Bb = 2000
    cw = {}
    for p in BLOCKS:
        cycles, _, miao, pairs = lay[p]
        r_ = np.random.default_rng([PERM_SEED, subj_num(p), 3])
        bp = r_.integers(0, 15, size=(Bb, 15))
        pos = {c: i for i, c in enumerate(cycles)}
        w = np.zeros((Bb, 30))
        for j, (c0, c1) in enumerate(pairs):
            n_ = (bp == j).sum(axis=1)
            w[:, pos[c0]] = n_
            w[:, pos[c1]] = n_
        cw[p] = w
    k, ki = 2, 1
    Ma, Mb = [], []
    for p in BLOCKS:
        miao = lay[p][2].astype(float)
        Ma.append(pm_batch(D[p], ki, (cw[p] * miao) @ lay[p][1], order[p]))
        Mb.append(pm_batch(D[p], ki, (cw[p] * (1 - miao)) @ lay[p][1], order[p]))
    GA, GB = np.nanmean(np.stack(Ma), axis=0), np.nanmean(np.stack(Mb), axis=0)

    def off(Gx, Gy):
        zz = np.nansum(np.conj(Gy[:, :, :8]) * Gx[:, :, :8], axis=(1, 2))
        return np.angle(zz) / (2 * np.pi * k) * 100

    Sob = {p: lay[p][2].astype(float) for p in BLOCKS}
    ga = np.nanmean(np.stack([pm_batch(D[p], ki, Sob[p] @ lay[p][1], order[p]) for p in BLOCKS]), axis=0)
    gb = np.nanmean(np.stack([pm_batch(D[p], ki, (1 - Sob[p]) @ lay[p][1], order[p]) for p in BLOCKS]), axis=0)
    grw = G68["RW"][:, :, ki][None]
    for lab, o, bs in (("IW minus OBS", off(ga, gb)[0], off(GA, GB)), ("OBS minus RW group", off(gb, grw)[0], off(GB, grw)),
                       ("IW minus RW group", off(ga, grw)[0], off(GA, grw))):
        w = 50 / k
        dl = ((bs - o + w) % (2 * w)) - w
        lo, hi = np.percentile(dl, [2.5, 97.5]) + o
        rows.append(dict(item="G8 group map", statistic=lab, harmonic_k=k, mean_pct=o, ci_lo=lo, ci_hi=hi))

    # G9: decoder output
    Tr = decoder_trials()
    Es = Tr.groupby(["participant", "state"])[["frac_pred1", "cos_d1", "cos_d2", "swap1"]].mean().reset_index()
    ms = ["frac_pred1", "cos_d1", "cos_d2", "swap1"]
    pt = []
    for m_ in ms:
        x = Es.pivot(index="participant", columns="state", values=m_)
        pt.append(stats.ttest_rel(x.IW, x.OBS))
    for m_, q, t in zip(ms, bh([t.pvalue for t in pt]), pt):
        x = Es.pivot(index="participant", columns="state", values=m_)
        d = Tr.assign(c=np.where(Tr.state == "IW", 0.5, -0.5))
        res = None
        for meth in ("lbfgs", "powell", "cg"):        # first optimiser that converges
            try:
                res = smf.mixedlm(f"{m_} ~ c", d, groups=d.participant, re_formula="~c").fit(reml=True, method=meth)
                if res.converged:
                    break
            except Exception:
                pass
        rows.append(dict(item="G9", statistic=m_, iw=x.IW.median(), obs=x.OBS.median(), t=t.statistic, p=t.pvalue, pFDR=q,
                         signflip_p=signflip((x.IW - x.OBS).to_numpy()), lme_beta=None if res is None else res.params["c"],
                         lme_p=None if res is None else res.pvalues["c"]))

    # G10: patients against the resting floor
    pat, _ = load_cube(inp("patient_maps.csv"), channels=chans)
    post = label("post_operative_sessions").split(",")
    P11 = gmean([pat[s] for s in post])
    pooled = {k: coh(P11[:, :, k - 1], A["IW"].mean(axis=0)[:, :, k - 1]) for k in KS}
    C = pd.read_csv(inp("session_similarity_by_state.csv"))
    F6 = pd.read_csv(inp("patient_session_similarity.csv"))
    ps_ = F6[F6.row_type == "patient_session"]
    BG = 100000
    pa = {}
    for k in KS:
        r = C[(C.state == "REST") & (C.reference == "IW") & (C.harmonic_k == k)].similarity.to_numpy()
        x = ps_[ps_.harmonic_k == k].set_index("session").value.reindex(post).to_numpy()
        nl = np.array([rng.choice(r, 11, replace=False).mean() for _ in range(BG)])
        pa[k] = (x, r, nl, p1(nl, x.mean()))
    for k, q in zip(KS, bh([pa[k][3] for k in KS])):
        x, r, nl, p = pa[k]
        rows.append(dict(item="G10a", harmonic_k=k, patient_mean=x.mean(), rest_mean=r.mean(), rest_p95=np.quantile(r, .95),
                         above_p95=int((x > np.quantile(r, .95)).sum()), p=p, pFDR=q))
    B10 = 10000
    fl11 = {k: np.empty(B10) for k in KS}
    for b in range(B10):
        ch = rng.choice(parts, 11, replace=False)
        rs = [rng.choice(np.where(subj == p)[0]) for p in ch]
        Mr = A["REST"][rs].mean(axis=0)
        G = A["IW"][~np.isin(subj, ch)].mean(axis=0)
        for k in KS:
            fl11[k][b] = coh(Mr[:, :, k - 1], G[:, :, k - 1])
    ps = [p1(fl11[k], pooled[k]) for k in KS]
    for k, q, p in zip(KS, bh(ps), ps):
        rows.append(dict(item="G10b", harmonic_k=k, similarity=pooled[k], floor_median=np.median(fl11[k]),
                         floor_p95=np.quantile(fl11[k], .95), p=p, pFDR=q))
    write_table(pd.DataFrame([dict(harmonic_k=k, floor_median=np.median(fl11[k]), floor_p95=np.quantile(fl11[k], .95))
                              for k in KS]), "patient_rest_floor.csv", index=False)

    # resting input: absolute decoding r on task trials and resting windows
    rr = []
    for p in BLOCKS:
        for blk, st in (("observation", "OBS"), ("imagined", "IW")):
            t = pd.read_csv(inp(f"observation_rest_input_{p}_{blk}.csv"))
            task = t[(t.keep_trial == 1) & t.r_task_input.notna()].r_task_input
            rest = t[(t.keep_rest_window == 1) & t.r_rest_input.notna()].r_rest_input
            pair = t[t.keep_pair == 1]
            rr.append(dict(participant=p, state=st, task_median=task.median(), rest_median=rest.median(), n_task=len(task),
                           n_rest=len(rest), pair_task_median=pair.r_task_input.median(),
                           pair_rest_median=pair.r_rest_input.median(), n_pair=len(pair)))
    write_table(pd.DataFrame(rr), "rest_input_absolute.csv", index=False)
    write_table(pd.DataFrame(rows), "observation_group_tests.csv", index=False)


if __name__ == "__main__":
    main()
