"""Gait-phase locking |c| and map similarity under imagined walking (MI+AO) versus observation only (AO), six
supplementary participants (one session each, conditions alternating by walker cycle).

Locking. For each participant the per-trial phase sums (written by `observation_lock_trial_sums.m`, pooled band of the
session) are recombined exactly into |c| of any trial subset: mean_point = sum_point / count_point_ch,
mean_all = sum_all / count_all_ch, phase_mod = mean_point - mean_all, |c| = mean over channels x cycle points of
|phase_mod|. delta|c| = |c|(MI+AO) - |c|(AO) is tested by a within-participant cycle-label permutation: 15 of the 30
cycles are relabelled MI+AO uniformly at random (both trials of a cycle together), B = 10,000, p = (b + 1) / (B + 1),
one-sided (MI+AO > AO) and two-sided; 95 % CI = percentile bootstrap over the 15 adjacent cycle pairs (B = 2,000).
Cohort: mean of the six participant deltas against B = 20,000 joint draws (each participant relabelled independently);
seeds numpy default_rng([20261008, participant number, purpose]), purpose 1 = participant draws, 2 = joint draws,
3 = bootstrap. Companion: paired t over the six participant differences. Benjamini-Hochberg across k1 / k2 / k3.
The |c| of each condition under its own band (with its yoked-rest null) is carried as extra columns.

Similarity. Per participant, condition and harmonic, the modulus of the complex spatial coherence between the
condition's phase map and the healthy imagined-walking group map without that participant's sessions; null =
1,000 channel shuffles per map (seed 20261014, one generator over participants in the order SUB_09, 01, 12, 23, 29, 35);
percentile among the healthy sessions' own leave-participant-out similarities; paired t over the six differences,
BH across harmonics; group maps of the six against the healthy map without all six participants' sessions.

Named inputs: observation_trial_sums_<key>.mat (six), observation_abs_c_by_condition.csv,
observation_rest_null_by_condition.csv, observation_phase_maps_by_condition.csv, healthy_phase_imagined.csv;
label condition_code (field of the per-trial sums holding the condition: 1 = MI+AO, 2 = AO).
Named output: derived/observation_locking.csv.
"""
import argparse

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from analysis_common import inp, label, out_path

SEED = 20261008
B_PRIMARY, B_BOOT, B_GROUP = 10_000, 2_000, 20_000
N_SHUF, SHUF_SEED = 1000, 20261014
KS = (1, 2, 3)
# participant -> (MI+AO session, AO session, healthy-session prefix, recorded session, key of the per-trial sums)
SESSIONS = {
    "SUB_01": ("subj01_sess05miao", "subj01_sess05ao", "subj01_", "SUB_01_sess05", "sub01"),
    "SUB_09": ("subj09_sess03miao", "subj09_sess03ao", "subj09_", "SUB_09_sess03", "sub09"),
    "SUB_12": ("subj12_sess03_abmiao", "subj12_sess03_abao", "subj12_", "SUB_12_sess03", "sub12"),
    "SUB_23": ("subj23_sess03_abmiao", "subj23_sess03_abao", "subj23_", "SUB_23_sess03", "sub23"),
    "SUB_29": ("subj29_sess03_abcmiao", "subj29_sess03_abcao", "subj29_", "SUB_29_sess03", "sub29"),
    "SUB_35": ("subj35_sess03_abmiao", "subj35_sess03_abao", "subj35_", "SUB_35_sess03", "sub35"),
}
PARTS = list(SESSIONS)                                              # row order of the table
RUN_ORDER = ["SUB_09", "SUB_01", "SUB_12", "SUB_23", "SUB_29", "SUB_35"]   # order of the cohort sums and the shuffle stream
POOLED = "pooled band"
OWN_BAND = "|c| of each condition under its own band"
MAPS = "complex maps of each condition under its own band"
REF_LPO = "healthy imagined-walking map without the participant's own sessions (leave-participant-out)"
REF_SIX = "healthy imagined-walking map without all six participants' sessions"


# ---------- per-trial phase sums ----------
def load_trial_sums(path):
    def arr(f, k):
        x = f[k][()]
        return np.transpose(x) if x.ndim > 1 else x                # MATLAB v7.3 is column-major
    with h5py.File(path, "r") as f:
        d = {k: arr(f, k) for k in ("sum_point_re", "sum_point_im", "count_point_ch", "sum_all_re", "sum_all_im",
                                    "count_all_ch", "kept")}
        d["cycle"] = np.asarray(f["cycle"][()]).ravel()
        d["condition"] = np.asarray(f[label("condition_code")][()]).ravel()
    n_tr = d["condition"].size
    for k in ("sum_point_re", "sum_point_im", "count_point_ch"):
        d[k] = d[k].reshape(n_tr, d[k].shape[1], d[k].shape[2], -1)
    for k in ("sum_all_re", "sum_all_im", "count_all_ch"):
        d[k] = d[k].reshape(n_tr, d[k].shape[1], -1)
    d["kept"] = d["kept"].reshape(n_tr, -1)
    d["SP"] = d["sum_point_re"] + 1j * d["sum_point_im"]
    d["SA"] = d["sum_all_re"] + 1j * d["sum_all_im"]
    return d


def abs_c(d, ki, W):
    """|c| for trial weights W [n_draw x n_trial] (0/1 subsets or bootstrap multiplicities)."""
    W = np.atleast_2d(np.asarray(W, float))
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
    n_ch = d["SA"].shape[1]
    pm = mp.reshape(W.shape[0], n_ch, -1) - ma[:, :, None]
    return np.nanmean(np.abs(pm).reshape(W.shape[0], -1), axis=1)


def cycle_layout(d):
    cyc = d["cycle"].astype(int)
    cond = d["condition"].astype(int)
    cycles = np.unique(cyc)
    assert cycles.size == 30
    cyc_cond = {c: int(np.unique(cond[cyc == c])[0]) for c in cycles}
    T = (cyc[None, :] == cycles[:, None]).astype(float)           # [30 cycles x n_trial]
    miao_cyc = np.array([cyc_cond[c] == 1 for c in cycles])
    assert miao_cyc.sum() == 15
    pairs = cycles.reshape(15, 2)                                 # adjacent cycles: one MI+AO, one AO
    return cycles, T, miao_cyc, pairs


def delta_from_cycle_sets(d, ki, T, S):
    """S [n_draw x 30] 0/1 = cycles labelled MI+AO; |c|(S) - |c|(complement)."""
    return abs_c(d, ki, S @ T) - abs_c(d, ki, (1 - S) @ T)


def relabel_draws(num, B, purpose):
    rng = np.random.default_rng([SEED, num, purpose])
    idx = np.argsort(rng.random((B, 30)), axis=1)[:, :15]
    S = np.zeros((B, 30))
    np.put_along_axis(S, idx, 1.0, axis=1)
    return S


def pvals(null, obs, eps=1e-12):
    return int((null >= obs - eps).sum()), int((np.abs(null) >= abs(obs) - eps).sum())


def bh(p):
    """Benjamini-Hochberg adjusted p over one family (the three harmonics)."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1.0)
    return out


def permutation():
    """Per participant x k: pooled-band |c|, delta, bootstrap CI, permutation p; cohort joint-draw p per k."""
    P, draws, obs_k = {}, {k: [] for k in KS}, {k: [] for k in KS}
    for part in RUN_ORDER:
        d = load_trial_sums(inp(f"observation_trial_sums_{SESSIONS[part][4]}.mat"))
        assert (d["kept"] > 0).all()
        num = int(part.split("_")[1])
        cycles, T, miao_cyc, pairs = cycle_layout(d)
        S_obs = miao_cyc[None, :].astype(float)
        S_pri = relabel_draws(num, B_PRIMARY, 1)
        S_grp = relabel_draws(num, B_GROUP, 2)
        boot_pairs = np.random.default_rng([SEED, num, 3]).integers(0, 15, size=(B_BOOT, 15))
        pos = {c: i for i, c in enumerate(cycles)}
        mult = np.zeros((B_BOOT, 30))                             # cycle multiplicity = multiplicity of its pair
        cnt = np.stack([(boot_pairs == j).sum(axis=1) for j in range(15)], axis=1)
        for j, (c0, c1) in enumerate(pairs):
            mult[:, pos[c0]] = cnt[:, j]
            mult[:, pos[c1]] = cnt[:, j]
        Wm = (mult * miao_cyc[None, :]) @ T
        Wa = (mult * ~miao_cyc[None, :]) @ T
        w_m, w_a = (d["condition"] == 1).astype(float), (d["condition"] == 2).astype(float)
        for ki, k in enumerate(KS):
            obs = float(delta_from_cycle_sets(d, ki, T, S_obs)[0])
            b1, b2 = pvals(delta_from_cycle_sets(d, ki, T, S_pri), obs)
            boot = abs_c(d, ki, Wm) - abs_c(d, ki, Wa)
            draws[k].append(delta_from_cycle_sets(d, ki, T, S_grp))
            obs_k[k].append(obs)
            P[(part, k)] = dict(abs_c_miao=float(abs_c(d, ki, w_m)[0]), abs_c_ao=float(abs_c(d, ki, w_a)[0]),
                                n_miao=int(w_m.sum()), n_ao=int(w_a.sum()), delta=obs,
                                ci_lo=float(np.percentile(boot, 2.5)), ci_hi=float(np.percentile(boot, 97.5)),
                                p_one=(b1 + 1) / (B_PRIMARY + 1), p_two=(b2 + 1) / (B_PRIMARY + 1))
    Gp = {}
    for k in KS:
        obs = float(np.mean(obs_k[k]))
        b1, b2 = pvals(np.mean(np.vstack(draws[k]), axis=0), obs)
        Gp[k] = dict(role="primary" if k == 2 else "secondary", mean_delta=obs, p_one=(b1 + 1) / (B_GROUP + 1),
                     p_two=(b2 + 1) / (B_GROUP + 1), floor=1 / (B_GROUP + 1))
    return P, Gp


# ---------- |c| of each condition under its own band (with the yoked-rest null) ----------
def own_band():
    cs = pd.read_csv(inp("observation_abs_c_by_condition.csv")).set_index(["session", "harmonic_k"]).mean_abs
    nl = pd.read_csv(inp("observation_rest_null_by_condition.csv")).set_index(["session", "harmonic_k"])
    L = {}
    for p in PARTS:
        for c, s in (("MI+AO", SESSIONS[p][0]), ("AO", SESSIONS[p][1])):
            for k in KS:
                v, n = float(cs.loc[(s, k)]), nl.loc[(s, k)]
                L[(p, c, k)] = dict(abs_c=v, rest_null_mean=n.rest_null_mean, rest_null_p95=n.rest_null_p95,
                                    n_resamples=int(n.n_resamples), above=bool(v > n.rest_null_p95),
                                    z=(v - n.rest_null_mean) / n.rest_null_sd)
    return L


# ---------- map similarity ----------
def gmean(A, mask=None):
    sub = A if mask is None else A[mask]
    return np.nanmean(sub.real, axis=0) + 1j * np.nanmean(sub.imag, axis=0)


def coh(P, H):
    p, h = P.ravel(), H.ravel()
    ok = np.isfinite(p.real) & np.isfinite(p.imag) & np.isfinite(h.real) & np.isfinite(h.imag)
    p, h = p[ok], h[ok]
    if p.size < 5:
        return np.nan
    denom = np.linalg.norm(p) * np.linalg.norm(h)
    return float(np.abs(np.vdot(h, p)) / denom) if denom > 0 else np.nan


def load_maps(path, channels=None):
    d = pd.read_csv(path)
    sess = list(dict.fromkeys(d.session))
    chans = channels or list(dict.fromkeys(d.channel))
    fr = sorted(d.cycle_fraction.round(6).unique())
    A = np.full((len(sess), len(chans), len(fr), 3), np.nan, complex)
    si = {s: i for i, s in enumerate(sess)}
    ci = {c: i for i, c in enumerate(chans)}
    fi = {f: i for i, f in enumerate(fr)}
    d = d[d.channel.isin(chans)]
    A[d.session.map(si).values, d.channel.map(ci).values, d.cycle_fraction.round(6).map(fi).values,
      d.harmonic_k.values - 1] = d.real.values + 1j * d.imag.values
    return sess, chans, A


def similarity():
    sess6, ch6, A6 = load_maps(inp("observation_phase_maps_by_condition.csv"))
    hs, _, HA = load_maps(inp("healthy_phase_imagined.csv"), channels=ch6)
    assert len(hs) == 80
    subj = np.array([x.split("_sess")[0] for x in hs])
    HL = {k: np.full(len(hs), np.nan) for k in KS}               # healthy leave-participant-out similarity
    for i in range(len(hs)):
        G = gmean(HA, subj != subj[i])
        for k in KS:
            HL[k][i] = coh(HA[i, :, :, k - 1], G[:, :, k - 1])
    idx6 = {s: i for i, s in enumerate(sess6)}
    H80 = gmean(HA)
    rng = np.random.default_rng(SHUF_SEED)

    def shuffled(M, Gk):
        return np.array([coh(M[rng.permutation(M.shape[0])], Gk) for _ in range(N_SHUF)])

    S, NULL = {}, {k: [] for k in KS}
    for p in RUN_ORDER:
        mi, ao, pref = SESSIONS[p][:3]
        mask = np.array([not x.startswith(pref) for x in hs])
        G = gmean(HA, mask)
        for k in KS:
            Gk = G[:, :, k - 1]
            rec = dict(n_ref=int(mask.sum()))
            for c, sname in (("MI+AO", mi), ("AO", ao)):
                M = A6[idx6[sname], :, :, k - 1]
                v = coh(M, Gk)
                null = shuffled(M, Gk)
                NULL[k].append(null)
                rec[c] = dict(sim=v, full80=coh(M, H80[:, :, k - 1]), p95=float(np.quantile(null, 0.95)),
                              p=float((1 + (null >= v).sum()) / (N_SHUF + 1)), pct=float(100 * (HL[k] < v).mean()))
            Mm, Ma = A6[idx6[mi], :, :, k - 1], A6[idx6[ao], :, :, k - 1]
            v = coh(Ma, Mm)
            null = shuffled(Ma, Mm)
            rec["own"] = dict(sim=v, p95=float(np.quantile(null, 0.95)), p=float((1 + (null >= v).sum()) / (N_SHUF + 1)))
            S[(p, k)] = rec

    Gc = {"MI+AO": gmean(A6[[idx6[SESSIONS[p][0]] for p in RUN_ORDER]]),
          "AO": gmean(A6[[idx6[SESSIONS[p][1]] for p in RUN_ORDER]])}
    mask6 = np.array([not any(x.startswith(SESSIONS[p][2]) for p in RUN_ORDER) for x in hs])
    G68 = gmean(HA, mask6)
    GR = {}
    for k in KS:
        for c in ("MI+AO", "AO"):
            M = Gc[c][:, :, k - 1]
            v = coh(M, G68[:, :, k - 1])
            null = shuffled(M, G68[:, :, k - 1])
            GR[(c, k)] = dict(sim=v, p95=float(np.quantile(null, 0.95)), p=float((1 + (null >= v).sum()) / (N_SHUF + 1)),
                              abs_c=float(np.nanmean(np.abs(M))), n_ref=int(mask6.sum()))
        GR[("AO vs MI+AO", k)] = dict(sim=coh(Gc["AO"][:, :, k - 1], Gc["MI+AO"][:, :, k - 1]))

    band = {}
    for k in KS:
        v = np.concatenate(NULL[k])
        band[k] = dict(zip(("p10", "q25", "median", "q75", "p90", "p95"),
                           (float(np.quantile(v, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95))), n=v.size)
    T = {k: paired([S[(p, k)]["MI+AO"]["sim"] for p in PARTS], [S[(p, k)]["AO"]["sim"] for p in PARTS]) for k in KS}
    for k, q in zip(KS, bh([T[k]["p_two"] for k in KS])):
        T[k]["pfdr_two"] = q
    return S, GR, band, T


def paired(m, a):
    m, a = np.asarray(m, float), np.asarray(a, float)
    d = m - a
    t = stats.ttest_rel(m, a)
    ci = t.confidence_interval(0.95)
    return dict(mean_delta=d.mean(), ci_lo=ci.low, ci_hi=ci.high, t=t.statistic, df=len(d) - 1, p_two=t.pvalue,
                n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n=len(d))


# ---------- table ----------
def build_table(P, Gp, L, SM):
    G = {}
    for k in KS:
        G[("own", k)] = paired([L[(p, "MI+AO", k)]["abs_c"] for p in PARTS], [L[(p, "AO", k)]["abs_c"] for p in PARTS])
        G[("pooled", k)] = paired([P[(p, k)]["abs_c_miao"] for p in PARTS], [P[(p, k)]["abs_c_ao"] for p in PARTS])
        G[("perm", k)] = dict(Gp[k])
    for rec, key, out in (("perm", "p_one", "pfdr_one"), ("perm", "p_two", "pfdr_two"),
                          ("pooled", "p_two", "pfdr_two"), ("own", "p_two", "pfdr_two")):
        for k, q in zip(KS, bh([G[(rec, k)][key] for k in KS])):
            G[(rec, k)][out] = q

    rows = []
    for p in PARTS:
        mi, ao, _, sess, _ = SESSIONS[p]
        for k in KS:
            q = P[(p, k)]
            for c, vs in (("MI+AO", mi), ("AO", ao)):
                r = L[(p, c, k)]
                rows.append(dict(
                    row_type="participant_condition", participant=p, session=sess, condition=c, virtual_session=vs,
                    harmonic_k=k, input=POOLED, abs_c=q["abs_c_miao"] if c == "MI+AO" else q["abs_c_ao"],
                    n_trials=q["n_miao"] if c == "MI+AO" else q["n_ao"], abs_c_asrun=r["abs_c"],
                    asrun_rest_null_mean=r["rest_null_mean"], asrun_rest_null_p95=r["rest_null_p95"],
                    asrun_rest_null_draws=r["n_resamples"], asrun_above_rest_p95=r["above"],
                    asrun_z_vs_rest_null=r["z"]))
            rows.append(dict(
                row_type="participant_difference", participant=p, session=sess, condition="MI+AO minus AO",
                harmonic_k=k, input=POOLED, delta_abs_c=q["delta"], ci95_lo=q["ci_lo"], ci95_hi=q["ci_hi"],
                ci_method="percentile bootstrap over the 15 cycle pairs, B 2000",
                perm_p_one_sided=q["p_one"], perm_p_two_sided=q["p_two"], perm_B=B_PRIMARY,
                delta_abs_c_asrun=L[(p, "MI+AO", k)]["abs_c"] - L[(p, "AO", k)]["abs_c"]))
    for k in KS:
        g, t = G[("perm", k)], G[("pooled", k)]
        rows.append(dict(row_type="cohort_cycle_label_permutation", test_role="main", condition="MI+AO minus AO",
                         harmonic_k=k, input=POOLED, harmonic_role=g["role"], delta_abs_c=t["mean_delta"],
                         ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"], ci_method="t interval over the six participant differences",
                         perm_p_one_sided=g["p_one"], perm_pfdr_one_sided=g["pfdr_one"], perm_p_two_sided=g["p_two"],
                         perm_pfdr_two_sided=g["pfdr_two"], perm_B=B_GROUP, p_floor=g["floor"], n_positive=t["n_pos"],
                         n_negative=t["n_neg"], n=6))
        rows.append(dict(row_type="cohort_paired_t", test_role="companion", condition="MI+AO minus AO", harmonic_k=k,
                         input=POOLED, delta_abs_c=t["mean_delta"], ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"], t=t["t"],
                         df=t["df"], p_two_sided=t["p_two"], pfdr_two_sided=t["pfdr_two"], n_positive=t["n_pos"],
                         n_negative=t["n_neg"], n=t["n"]))
        a = G[("own", k)]
        rows.append(dict(row_type="cohort_paired_t_asrun", test_role="extra (band of each condition)",
                         condition="MI+AO minus AO", harmonic_k=k, input=OWN_BAND, delta_abs_c=a["mean_delta"],
                         ci95_lo=a["ci_lo"], ci95_hi=a["ci_hi"], t=a["t"], df=a["df"], p_two_sided=a["p_two"],
                         pfdr_two_sided=a["pfdr_two"], n_positive=a["n_pos"], n_negative=a["n_neg"], n=a["n"]))
    S, GR, band, T = SM
    for p in PARTS:
        mi, ao, _, sess, _ = SESSIONS[p]
        for k in KS:
            r = S[(p, k)]
            for c, vs in (("MI+AO", mi), ("AO", ao)):
                q = r[c]
                rows.append(dict(
                    row_type="participant_similarity", participant=p, session=sess, condition=c, virtual_session=vs,
                    harmonic_k=k, input=MAPS, reference=REF_LPO, n_healthy_sessions_ref=r["n_ref"],
                    similarity=q["sim"], similarity_full80=q["full80"], shuffle_p95=q["p95"], shuffle_p=q["p"],
                    above_shuffle_p95=bool(q["sim"] > q["p95"]), n_shuffle=N_SHUF, shuffle_seed=SHUF_SEED,
                    healthy_percentile=q["pct"]))
            rows.append(dict(
                row_type="participant_similarity_ao_vs_own_miao", participant=p, session=sess,
                condition="AO vs own MI+AO", harmonic_k=k, input=MAPS, reference="the same participant's MI+AO map",
                similarity=r["own"]["sim"], shuffle_p95=r["own"]["p95"], shuffle_p=r["own"]["p"],
                above_shuffle_p95=bool(r["own"]["sim"] > r["own"]["p95"]), n_shuffle=N_SHUF, shuffle_seed=SHUF_SEED))
            rows.append(dict(
                row_type="participant_similarity_difference", participant=p, session=sess, condition="MI+AO minus AO",
                harmonic_k=k, input=MAPS, reference=REF_LPO, delta_similarity=r["MI+AO"]["sim"] - r["AO"]["sim"]))
    for k in KS:
        t = T[k]
        rows.append(dict(row_type="cohort_paired_t_similarity", test_role="panel d brackets", condition="MI+AO minus AO",
                         harmonic_k=k, input=MAPS, reference=REF_LPO, delta_similarity=t["mean_delta"],
                         ci95_lo=t["ci_lo"], ci95_hi=t["ci_hi"], ci_method="t interval over the six participant differences",
                         t=t["t"], df=t["df"], p_two_sided=t["p_two"], pfdr_two_sided=t["pfdr_two"],
                         n_positive=t["n_pos"], n_negative=t["n_neg"], n=t["n"]))
        b = band[k]
        rows.append(dict(row_type="similarity_null_band", condition="pooled MI+AO and AO", harmonic_k=k, input=MAPS,
                         reference=REF_LPO, null_p10=b["p10"], null_q25=b["q25"], null_median=b["median"],
                         null_q75=b["q75"], null_p90=b["p90"], null_p95=b["p95"], n_shuffle=b["n"], shuffle_seed=SHUF_SEED))
        for c in ("MI+AO", "AO"):
            g = GR[(c, k)]
            rows.append(dict(row_type="group_map_similarity", condition=c, harmonic_k=k, input=MAPS, reference=REF_SIX,
                             n_healthy_sessions_ref=g["n_ref"], similarity=g["sim"], shuffle_p95=g["p95"],
                             shuffle_p=g["p"], above_shuffle_p95=bool(g["sim"] > g["p95"]), n_shuffle=N_SHUF,
                             shuffle_seed=SHUF_SEED, groupmap_abs_c=g["abs_c"]))
        rows.append(dict(row_type="group_map_similarity", condition="AO group map vs MI+AO group map", harmonic_k=k,
                         input=MAPS, reference="the six-session MI+AO group map", similarity=GR[("AO vs MI+AO", k)]["sim"]))
    return pd.DataFrame(rows)


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    P, Gp = permutation()
    df = build_table(P, Gp, own_band(), similarity())
    p = out_path("observation_locking.csv")
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.12g", lineterminator="\n")
    tmp.replace(p)
    print("wrote", p, f"({len(df)} rows)")


if __name__ == "__main__":
    main()
