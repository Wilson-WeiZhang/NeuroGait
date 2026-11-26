"""Gait-phase locking of the scalp maps in real walking, imagined walking, observation only and rest.

Maps are the per-session complex harmonic phase maps (channel x 9 gait-cycle points x harmonic k = 1-3), from the
80 healthy sessions (real walking, imagined walking, yoked rest) and from the six observation-experiment sessions
(imagined walking and observation only in all six, real walking in five).
  locking_by_state.csv          |c| = mean over channels and points of the map modulus, per state x session x harmonic.
                                For the observation sessions, rest is the mean of the two yoked-rest null means
                                (imagined-walking and observation-only blocks).
  session_similarity_by_state.csv  per healthy session, the complex spatial coherence of each state's map with the
                                leave-participant-out group map of every state.
  observation_timing.csv        per observation participant and harmonic, the global phase offset (% of the gait cycle,
                                negative = first map leads) of imagined walking against observation only, of each
                                against the leave-participant-out real-walking group map and against the participant's
                                own real-walking map.
Named inputs: healthy_phase_imagined.csv, healthy_phase_real.csv, healthy_phase_rest.csv, observation_phase_maps_by_condition.csv,
observation_real_maps.csv, observation_rest_null.csv. Named outputs: the three tables above (derived/).
"""
import numpy as np
import pandas as pd

from analysis_common import inp, write_table

# participant -> imagined-walking block, observation-only block, healthy-session prefix, real-walking block (None = none)
BLOCKS = {"SUB_09": ("subj09_sess03miao", "subj09_sess03ao", "subj09_", "subj09_sess03"),
        "SUB_01": ("subj01_sess05miao", "subj01_sess05ao", "subj01_", "subj01_sess05"),
        "SUB_12": ("subj12_sess03_abmiao", "subj12_sess03_abao", "subj12_", "subj12_sess03_ab"),
        "SUB_23": ("subj23_sess03_abmiao", "subj23_sess03_abao", "subj23_", "subj23_sess03_ab"),
        "SUB_29": ("subj29_sess03_abcmiao", "subj29_sess03_abcao", "subj29_", "subj29_sess03_abc"),
        "SUB_35": ("subj35_sess03_abmiao", "subj35_sess03_abao", "subj35_", None)}
KS = (1, 2, 3)
STATE_MAPS = {"IW": "healthy_phase_imagined.csv", "RW": "healthy_phase_real.csv", "REST": "healthy_phase_rest.csv"}


def load_cube(path, channels=None):
    """{session: complex array[channel, point (9), harmonic (3)]}; channel order sorted, or as given."""
    d = pd.read_csv(path, usecols=["session", "harmonic_k", "cycle_fraction", "channel", "real", "imag"])
    chans = channels or sorted(d.channel.unique())
    fr = sorted(d.cycle_fraction.round(6).unique())
    ci, fi = {c: i for i, c in enumerate(chans)}, {f: i for i, f in enumerate(fr)}
    out = {}
    for s, g in d.groupby("session"):
        A = np.full((len(chans), len(fr), 3), np.nan, complex)
        g = g[g.channel.isin(chans)]
        A[g.channel.map(ci).to_numpy(), g.cycle_fraction.round(6).map(fi).to_numpy(), g.harmonic_k.to_numpy() - 1] = \
            g.real.to_numpy() + 1j * g.imag.to_numpy()
        out[s] = A
    return out, chans


def load_states():
    """Healthy maps of the three states (channel order of the imagined-walking table) and the channel list."""
    CO = {}
    CO["IW"], chans = load_cube(inp(STATE_MAPS["IW"]))
    for st in ("RW", "REST"):
        CO[st], _ = load_cube(inp(STATE_MAPS[st]), channels=chans)
    return CO, chans


def gmean(arrs):
    a = np.stack(arrs)
    return np.nanmean(a.real, axis=0) + 1j * np.nanmean(a.imag, axis=0)


def coh(p, h):
    """Spatial similarity: modulus of the complex spatial coherence over channels and points."""
    p, h = p.ravel(), h.ravel()
    return float(np.abs(np.vdot(h, p)) / (np.linalg.norm(p) * np.linalg.norm(h)))


def abs_c(m):
    return float(np.nanmean(np.abs(m)))


def delta(m, g, k):
    """angle(sum M conj(G)) over the 8 unique points (the 9th repeats the first), in % of the gait cycle."""
    return float(np.angle(np.vdot(g[:, :8], m[:, :8])) / (2 * np.pi * k) * 100)


def circ(values_pct, k):
    """Circular mean (% of the cycle), resultant length and Rayleigh p of offsets at harmonic k."""
    z = np.exp(1j * 2 * np.pi * k * np.asarray(values_pct) / 100)
    n, Rb = len(z), abs(z.mean())
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * Rb) ** 2)) - (1 + 2 * n))
    return float(np.angle(z.mean()) / (2 * np.pi * k) * 100), float(Rb), float(min(p, 1.0))


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = np.empty(len(p))
    m_ = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q[o] = np.minimum(m_, 1)
    return q


def main():
    CO, chans = load_states()
    SP, _ = load_cube(inp("observation_phase_maps_by_condition.csv"), channels=chans)
    SR, _ = load_cube(inp("observation_real_maps.csv"), channels=chans)
    sess = sorted(CO["IW"])
    assert all(sorted(CO[s]) == sess for s in CO) and len(sess) == 80
    subj = {s: s.split("_sess")[0] for s in sess}

    # locking |c| by state x harmonic
    rows = []
    for st in CO:
        for s in sess:
            for k in KS:
                rows.append(dict(source="cohort", state=st, session=s, harmonic_k=k, abs_c=abs_c(CO[st][s][:, :, k - 1])))
    null = pd.read_csv(inp("observation_rest_null.csv")).set_index(["session", "harmonic_k"])
    for p, (mi, ao, _, rw) in BLOCKS.items():
        for k in KS:
            rows.append(dict(source="supp", state="IW", session=p, harmonic_k=k, abs_c=abs_c(SP[mi][:, :, k - 1])))
            rows.append(dict(source="supp", state="OBS", session=p, harmonic_k=k, abs_c=abs_c(SP[ao][:, :, k - 1])))
            rows.append(dict(source="supp", state="REST", session=p, harmonic_k=k,
                             abs_c=float(np.mean([null.loc[(mi, k)].rest_null_mean, null.loc[(ao, k)].rest_null_mean]))))
            if rw:
                rows.append(dict(source="supp", state="RW", session=p, harmonic_k=k, abs_c=abs_c(SR[rw][:, :, k - 1])))
    write_table(pd.DataFrame(rows), "locking_by_state.csv", index=False)

    def group(st, drop):
        return gmean([CO[st][s] for s in sess if not any(s.startswith(d) for d in drop)])

    # per-session similarity to the leave-participant-out group maps of every state
    rows = []
    for s in sess:
        G = {st: group(st, [subj[s] + "_"]) for st in CO}
        for st in CO:
            for rs in CO:
                for k in KS:
                    rows.append(dict(session=s, state=st, reference=rs, harmonic_k=k,
                                     similarity=coh(CO[st][s][:, :, k - 1], G[rs][:, :, k - 1])))
    write_table(pd.DataFrame(rows), "session_similarity_by_state.csv", index=False)

    # timing of the observation-experiment maps
    rows = []
    for p, (mi, ao, pref, rw) in BLOCKS.items():
        G = group("RW", [pref])
        for k in KS:
            m_i, m_o = SP[mi][:, :, k - 1], SP[ao][:, :, k - 1]
            rec = dict(participant=p, harmonic_k=k, iw_minus_obs=delta(m_i, m_o, k),
                       iw_vs_rwgroup=delta(m_i, G[:, :, k - 1], k), obs_vs_rwgroup=delta(m_o, G[:, :, k - 1], k))
            if rw:
                rec["iw_vs_ownrw"] = delta(m_i, SR[rw][:, :, k - 1], k)
                rec["obs_vs_ownrw"] = delta(m_o, SR[rw][:, :, k - 1], k)
            rows.append(rec)
    write_table(pd.DataFrame(rows), "observation_timing.csv", index=False)


if __name__ == "__main__":
    main()
