"""Group gait-phase x frequency maps of induced power and their cluster tests.

Input per session: per-trial sums of the spectral coefficients over gait-phase bins (S1 = complex sum, S2 = power sum,
cnt = samples per bin; frequencies 3-40 Hz, 24 phase bins), trial labels and cycle numbers, from the per-trial EEG stage.
The normalized induced map of a set of trials is 10 log10 of the induced power (mean power minus the power of the
phase-locked mean, averaged over channels) divided by its mean over the phase bins, per frequency.

Observation experiment (six participants, imagined walking vs observation only, 30 + 30 trials, two trials per cycle):
statistic = group mean of the imagined-walking minus observation-only map. Null: within-participant cycle-label
permutation, 15 of the 30 cycles relabelled at random with both trials of a cycle together, B = 5,000, seed
[20261103, participant number, 1]; the phase-locked part is recomputed for each relabelling. Each pixel is divided by
its null SD (z). Clusters of |z| > 1.96 of one sign, contiguous in frequency and circularly in phase; FWER p from the
largest cluster mass per draw. Mean power per frequency (10 log10 of mean trial power) is tested by the same draws, BH
across frequencies.
Healthy sessions (80): imagined walking and real walking against yoked rest; t map of the session differences,
sign-flip over sessions (B = 5,000, seed [20261103, 80, 1]), clusters of |t| > t(0.975, 79), same FWER rule.

Named inputs: phase_frequency_observation (folder of <session>.npz, six sessions), phase_frequency_healthy (folder of
<session>.npz, 80 sessions).
Named outputs (derived/): observation_phase_frequency.npz, healthy_phase_frequency.npz, phase_frequency_clusters.csv
(observation rows, then healthy rows).
Usage: python phase_frequency_maps.py
"""
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from analysis_common import inp, label, write_npz, write_table

SEED, B = 20261103, 5000
OBS_SESSIONS = {"subj01_sess05": 1, "subj09_sess03": 9, "subj12_sess03_ab": 12, "subj23_sess03_ab": 23,
                "subj29_sess03_abc": 29, "subj35_sess03_ab": 35}
FOUR = ndimage.generate_binary_structure(2, 1)


def norm_map(S1, S2, cnt, w, chunk=100):
    """Normalized induced maps (dB) for trial-weight rows w [draw x trial]: [draw, frequency, phase bin]."""
    w = np.atleast_2d(w).astype(np.float32)
    n_tr, nf, nb, nch = S1.shape
    a1, a2 = S1.reshape(n_tr, -1), S2.reshape(n_tr, -1)
    out = np.zeros((w.shape[0], nf, nb), np.float32)
    c = cnt.astype(np.float32)
    for s in range(0, w.shape[0], chunk):
        ww = w[s: s + chunk]
        N = ww @ c
        T = (ww @ a2).reshape(-1, nf, nb, nch) / N[:, None, :, None]
        L = (ww @ a1).reshape(-1, nf, nb, nch) / N[:, None, :, None]
        P = (T - (L.real ** 2 + L.imag ** 2)).mean(axis=3)
        out[s: s + chunk] = 10 * np.log10(P / P.mean(axis=2, keepdims=True))
    return out


def clusters(z, thr):
    """Signed clusters above thr, contiguous in frequency (axis 0) and circular in phase (axis 1): (mass, sign, mask)."""
    res = []
    for sign in (1, -1):
        m = sign * z > thr
        if not m.any():
            continue
        lab, n = ndimage.label(m, structure=FOUR)
        for f in np.flatnonzero(m[:, 0] & m[:, -1]):              # join across the wrap of the gait cycle
            a, b = lab[f, 0], lab[f, -1]
            if a != b:
                lab[lab == b] = a
        for u in np.unique(lab[lab > 0]):
            q = lab == u
            res.append((float(np.abs(z[q]).sum()), sign, q))
    return res


def max_mass(z, thr):
    return max((x[0] for x in clusters(z, thr)), default=0.0)


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    r = np.empty(len(p))
    r[o] = np.minimum(q, 1)
    return r


def table(label, z, thr, null_mass, freqs, effect):
    rows = []
    nb = z.shape[1]
    for mass, sign, q in sorted(clusters(z, thr), key=lambda t: -t[0]):
        fq = np.flatnonzero(q.any(axis=1))
        ph = np.flatnonzero(q.any(axis=0))
        pk = np.unravel_index(np.argmax(np.abs(z) * q), z.shape)
        rows.append(dict(contrast=label, sign=sign, mass=mass, p_fwer=(1 + (null_mass >= mass).sum()) / (len(null_mass) + 1),
                         n_pixels=int(q.sum()), f_lo=freqs[fq[0]], f_hi=freqs[fq[-1]], phase_bins=" ".join(map(str, ph)),
                         peak_f=freqs[pk[0]], peak_phase_pct=(pk[1] + 0.5) / nb * 100, peak_stat=z[pk], peak_db=effect[pk]))
    return rows


def observation():
    D = {s: np.load(inp("phase_frequency_observation") / f"{s}.npz") for s in OBS_SESSIONS}
    freqs = D["subj12_sess03_ab"]["freqs"]
    obs_iw, obs_ob, null, lp_obs, lp_null, hips = {}, {}, [], [], [], []
    for s, num in OBS_SESSIONS.items():
        d = D[s]
        cond, cyc = d[label("test_condition_column")], d["cycle"]
        cycles = np.unique(cyc)
        assert cycles.size == 30
        assert all(np.unique(cond[cyc == c]).size == 1 for c in cycles)
        n_tr = np.array([(cyc == c).sum() for c in cycles])
        assert (cond == 1).sum() == (cond == 2).sum() == 30 and (n_tr == 2).all()
        S1, S2, cnt = d["S1"], d["S2"], d["cnt"]
        T = (cyc[None, :] == cycles[:, None]).astype(np.float32)
        rng = np.random.default_rng([SEED, num, 1])
        pick = np.argsort(rng.random((B, 30)), axis=1)[:, :15]
        Sd = np.zeros((B, 30), np.float32)
        np.put_along_axis(Sd, pick, 1.0, axis=1)
        obs_iw[s] = norm_map(S1, S2, cnt, (cond == 1).astype(np.float32))[0]
        obs_ob[s] = norm_map(S1, S2, cnt, (cond == 2).astype(np.float32))[0]
        null.append(norm_map(S1, S2, cnt, Sd @ T) - norm_map(S1, S2, cnt, (1 - Sd) @ T))
        lp = 10 * d["lp"]                                          # dB, per trial and frequency
        lp_obs.append(lp[cond == 1].mean(axis=0) - lp[cond == 2].mean(axis=0))
        W = Sd @ T
        lp_null.append(W @ lp / 30 - (1 - W) @ lp / 30)
        hips.append(d["hip"].sum(axis=0) / np.maximum(cnt.sum(axis=0), 1))
    ss = list(OBS_SESSIONS)
    I, O = np.stack([obs_iw[s] for s in ss]), np.stack([obs_ob[s] for s in ss])
    diff = I.mean(axis=0) - O.mean(axis=0)
    nul = np.mean(null, axis=0)
    sd = nul.std(axis=0)
    z = diff / sd
    zn = nul / sd
    mm = np.array([max_mass(zn[i], 1.96) for i in range(B)])
    mz = np.abs(zn).max(axis=(1, 2))
    rows = table("IW-OBS n=6", z, 1.96, mm, freqs, diff)
    lo = np.mean(lp_obs, axis=0)
    ln = np.mean(lp_null, axis=0)
    pl = (1 + (np.abs(ln) >= np.abs(lo)).sum(axis=0)) / (B + 1)
    write_npz("observation_phase_frequency.npz", freqs=freqs, sessions=np.array(ss), map_iw=I, map_ob=O, diff=diff, z=z,
              null_sd=sd, null_mass=mm, null_maxz=mz, hip=np.stack(hips), lp_diff=np.array(lp_obs), lp_p=pl, lp_q=bh(pl))
    return rows


def signflip(Ds, label, freqs, rng):
    n = Ds.shape[0]
    flat = Ds.reshape(n, -1).astype(np.float64)
    ss = (flat ** 2).sum(axis=0)

    def tmap(mean):
        var = (ss / n - mean ** 2) * n / (n - 1)
        return mean / np.sqrt(var / n)

    t0 = tmap(flat.mean(axis=0)).reshape(Ds.shape[1:])
    thr = stats.t.ppf(0.975, n - 1)
    mm = np.zeros(B)
    for s in range(0, B, 250):
        F = rng.choice([-1.0, 1.0], (min(250, B - s), n))
        tn = tmap(F @ flat / n).reshape(-1, *Ds.shape[1:])
        for i in range(tn.shape[0]):
            mm[s + i] = max_mass(tn[i], thr)
    return t0, table(label, t0, thr, mm, freqs, Ds.mean(axis=0))


def healthy():
    files = sorted(inp("phase_frequency_healthy").glob("*.npz"))
    assert len(files) == 80
    D = [np.load(f) for f in files]
    freqs = D[0]["freqs"]
    M = {c: np.stack([d[f"map_{c}"] for d in D]) for c in ("IW", "REST", "RW")}
    L = {c: 10 * np.stack([d[f"lp_{c}"] for d in D]) for c in ("IW", "REST", "RW")}
    rng = np.random.default_rng([SEED, 80, 1])
    rows, out = [], {}
    for lab, a in (("IW-REST n=80", "IW"), ("RW-REST n=80", "RW")):
        t0, r = signflip(M[a] - M["REST"], lab, freqs, rng)
        rows += r
        out[f"t_{a}"] = t0
        d = L[a] - L["REST"]
        F = rng.choice([-1.0, 1.0], (B, 80))
        nl = (F @ d) / 80
        pl = (1 + (np.abs(nl) >= np.abs(d.mean(axis=0))).sum(axis=0)) / (B + 1)
        out[f"lp_diff_{a}"], out[f"lp_p_{a}"], out[f"lp_q_{a}"] = d.mean(axis=0), pl, bh(pl)
    write_npz("healthy_phase_frequency.npz", freqs=freqs, sessions=np.array([f.stem for f in files]),
              mean_IW=M["IW"].mean(axis=0), mean_REST=M["REST"].mean(axis=0), mean_RW=M["RW"].mean(axis=0),
              hip_IW=np.stack([d["hip_IW"] for d in D]), hip_RW=np.stack([d["hip_RW"] for d in D]),
              n_iw=np.array([int(d["n_iw"]) for d in D]), n_rest=np.array([int(d["n_rest"]) for d in D]),
              n_rw=np.array([int(d["n_rw"]) for d in D]), n_nonfinite=np.array([int(d["n_nonfinite"]) for d in D]), **out)
    return rows


def main():
    rows = observation() + healthy()
    write_table(pd.DataFrame(rows), "phase_frequency_clusters.csv", index=False)


if __name__ == "__main__":
    main()
