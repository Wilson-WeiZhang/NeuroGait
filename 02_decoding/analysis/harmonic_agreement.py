"""Per-harmonic phase agreement between decoded and reference joint trajectories, healthy sessions.

For every scored trial and joint, the decoded and the reference trajectory are z-scored over the samples where both (and
the contralateral reference joint and the reference hip) are finite, and projected onto Hann-weighted complex sinusoids
at k * f0 (k = 1, 2, 3; f0 = stride frequency of the reference hip of that trial, peak of the Hann-windowed spectrum in
0.45-1.15 Hz, left hip unless it has missing samples, then right hip). Per harmonic the table keeps
cos(angle(alpha_pred * conj(alpha_true))) (agreement with the same joint), the same against the contralateral joint, and
the projected powers. cos_d1 < 0 marks a half-stride error (the decoded joint follows the opposite leg). Session table:
mean agreement per harmonic, fraction of half-stride errors, mean share of each harmonic in the summed power of k = 1-3,
median full-trajectory r, number of trials; for imagined walking the responder flag and the session locking values are
joined.

Named inputs: scoring_mask.csv; healthy_imagined_dump, healthy_real_dump, healthy_rest_dump (one *_dump.h5 per session);
responder_table.csv; session_locking_table.csv (session, excess_k1, excess_k2, r_v10 columns).
Named outputs (derived/): trial_joint_power.csv (imagined walking, one row per trial x joint);
session_agreement_imagined.csv, session_agreement_real.csv, session_agreement_rest_input.csv (one row per session).
Usage: python harmonic_agreement.py [IW|RW|REST ...]   (default: all three)
"""
import sys

import h5py
import numpy as np
import pandas as pd

from analysis_common import inp, write_table

FS = 250.0
JOINTS = ["LHip", "LKnee", "LAnk", "RHip", "RKnee", "RAnk"]
DUMP = {"IW": "healthy_imagined_dump", "RW": "healthy_real_dump", "REST": "healthy_rest_dump"}
SESSION_OUT = {"IW": "session_agreement_imagined.csv", "RW": "session_agreement_real.csv",
               "REST": "session_agreement_rest_input.csv"}


def stride_freq(x):
    x = x - x.mean()
    n = 1 << 15
    f = np.fft.rfftfreq(n, 1 / FS)
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)), n)) ** 2
    band = (f >= 0.45) & (f <= 1.15)
    return f[band][np.argmax(p[band])]


def proj(x, f, t):
    w = np.hanning(len(x))
    return np.sum(w * x * np.exp(-1j * 2 * np.pi * f * t))


def z(x):
    return (x - x.mean()) / x.std()


def trial_joint_rows(cond):
    m = pd.read_csv(inp("scoring_mask.csv"))
    m = m[(m.cohort == "healthy") & (m.condition == cond) & (m.keep == 1)]
    keep = set(zip(m.session, m.trial_idx))
    rows = []
    for fp in sorted(inp(DUMP[cond]).glob("*_dump.h5")):
        with h5py.File(fp, "r") as h:
            s = h.attrs["session"]
            s = s.decode() if isinstance(s, bytes) else str(s)
            C, Y, T = h["rows/C_pred"][:], h["rows/Y_true"][:], h["rows/time_sec"][:]
            tid, off, nr = h["trials/trial_idx"][:], h["trials/row_offset"][:], h["trials/n_rows"][:]
        for ti, o, n in zip(tid, off, nr):
            if (s, int(ti)) not in keep:
                continue
            c0, y0, t0 = C[o:o + n], Y[o:o + n], T[o:o + n]
            hip = 0 if np.isfinite(y0[:, 0]).all() else 3          # stride frequency from a usable reference hip
            for j in range(6):
                jc = (j + 3) % 6
                ok = np.isfinite(c0[:, j]) & np.isfinite(y0[:, j]) & np.isfinite(y0[:, jc]) & np.isfinite(y0[:, hip])
                if ok.sum() < 2 * FS:
                    continue
                c, y, t = c0[ok], y0[ok], t0[ok]
                f0 = stride_freq(y[:, hip])
                rec = dict(session=s, trial_idx=int(ti), joint=JOINTS[j], f0=f0, n=int(ok.sum()))
                for k in (1, 2, 3):
                    ap, at, ac = proj(z(c[:, j]), k * f0, t), proj(z(y[:, j]), k * f0, t), proj(z(y[:, jc]), k * f0, t)
                    rec[f"cos_d{k}"] = float(np.cos(np.angle(ap * np.conj(at))))
                    rec[f"cos_contra{k}"] = float(np.cos(np.angle(ap * np.conj(ac))))
                    rec[f"pow_pred{k}"] = float(abs(ap) ** 2)
                    rec[f"pow_true{k}"] = float(abs(at) ** 2)
                rec["r_full"] = float(np.corrcoef(c[:, j], y[:, j])[0, 1])
                rows.append(rec)
    return pd.DataFrame(rows)


def run(cond):
    L = trial_joint_rows(cond)
    if cond == "IW":
        write_table(L, "trial_joint_power.csv", index=False)
    for k in (1, 2, 3):
        L[f"frac_pred{k}"] = L[f"pow_pred{k}"] / (L.pow_pred1 + L.pow_pred2 + L.pow_pred3)
        L[f"frac_true{k}"] = L[f"pow_true{k}"] / (L.pow_true1 + L.pow_true2 + L.pow_true3)
    L["swap1"] = (L.cos_d1 < 0).astype(float)
    S = L.groupby("session").agg(cos_d1=("cos_d1", "mean"), cos_d2=("cos_d2", "mean"), cos_d3=("cos_d3", "mean"),
                                 swap1=("swap1", "mean"), frac_pred1=("frac_pred1", "mean"),
                                 frac_pred2=("frac_pred2", "mean"), frac_true1=("frac_true1", "mean"),
                                 frac_true2=("frac_true2", "mean"), r_full=("r_full", "median"),
                                 n_trials=("trial_idx", "nunique"))
    if cond == "IW":
        rn = pd.read_csv(inp("responder_table.csv")).set_index("session")[["resp"]]
        ms = pd.read_csv(inp("session_locking_table.csv")).set_index("session")[["excess_k1", "excess_k2", "r_v10"]]
        S = S.join(rn).join(ms)
    write_table(S, SESSION_OUT[cond])


def main(conds):
    for c in conds or ["IW", "RW", "REST"]:
        run(c)


if __name__ == "__main__":
    main(sys.argv[1:])
