"""Power spectra of the reference goniometer angles during imagined walking, healthy sessions.

For every session and every imagined-walking trial whose event quality flag is "good" or "ok", each walker sagittal
goniometer channel (left and right hip, knee, ankle) is cut to the trial window (gtstart..gtstop), demeaned, and its Welch
power spectrum is computed (Hamming window of min(4 s, max(2 s, n/2)) samples, 50 % overlap, nfft 8192, density
scaling). Each spectrum is divided by its area over 0.05-4 Hz and interpolated onto a 0-4 Hz grid in 0.01 Hz steps.
Traces shorter than 2 s or flat are skipped. Spectra are averaged within session per joint group (hip, knee, ankle; left
and right pooled), then across sessions: mean, SEM over sessions (ddof 1) and the 95 % band mean +- 1.96 SEM.

Named inputs: healthy_sessions.csv (column session); healthy_exports (one folder per session holding
<session>_goni.mat (HDF5: goni/x, goni/s, goni/c) and <session>_event.mat (struct array event with code, qc, gtstart,
gtstop)).
Named output (derived/): goniometer_power_summary.csv (one row per frequency bin).
"""
import csv
import os

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from analysis_common import inp, out_path

CONDITION = "gaitimagery"
ROLE, AXIS = "W", "X"
FMAX, FREQ_STEP = 4.0, 0.01
JOINTS = ("hip", "knee", "ankle")
JOINT_STEMS = {"hip": "Hip", "knee": "Kne", "ankle": "Ank"}


def matlab_char(dataset):
    arr = np.asarray(dataset).squeeze()
    return "".join(chr(int(x)) for x in arr.ravel() if int(x))


def load_goni(path):
    with h5py.File(path, "r") as h5:
        x = np.asarray(h5["goni/x"], dtype=float)
        srate = float(np.asarray(h5["goni/s"]).squeeze())
        labels = [matlab_char(h5[ref]) for ref in np.asarray(h5["goni/c"]).ravel()]
    if x.shape[0] == len(labels) and x.shape[1] != len(labels):
        x = x.T
    return x, srate, labels


def as_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(value.astype(str).ravel())
        if value.size == 1:
            return as_text(value.item())
    return str(value)


def normalized_psd(y, fs, freq_grid):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < round(2 * fs) or np.nanstd(y) <= 0:
        return None
    y = y - np.nanmean(y)
    win = min(round(4 * fs), max(round(2 * fs), y.size // 2))
    freqs, pxx = welch(y, fs=fs, window="hamming", nperseg=int(win), noverlap=int(win // 2), nfft=8192,
                       detrend=False, scaling="density")
    keep = (freqs >= 0.05) & (freqs <= FMAX)
    denom = np.trapezoid(pxx[keep], freqs[keep])
    if not np.isfinite(denom) or denom <= 0:
        return None
    return np.interp(freq_grid, freqs, pxx / denom, left=0.0, right=0.0)


def session_psd(session, root, freq_grid):
    """Per joint group: mean normalized spectrum over the session's traces, and the number of traces."""
    base = root / session
    goni_x, fs, labels = load_goni(base / f"{session}_goni.mat")
    events = np.atleast_1d(loadmat(base / f"{session}_event.mat", squeeze_me=True, struct_as_record=False)["event"])
    traces = {joint: [] for joint in JOINTS}
    for event in events:
        if as_text(getattr(event, "code", "")).lower() != CONDITION:
            continue
        if as_text(getattr(event, "qc", "good")).lower() not in {"good", "ok"}:
            continue
        start = max(int(getattr(event, "gtstart")) - 1, 0)
        stop = min(int(getattr(event, "gtstop")), goni_x.shape[0])
        if stop <= start:
            continue
        for joint in JOINTS:
            for label in (f"L{JOINT_STEMS[joint]}{ROLE} {AXIS}", f"R{JOINT_STEMS[joint]}{ROLE} {AXIS}"):
                if label not in labels:
                    continue
                psd = normalized_psd(goni_x[start:stop, labels.index(label)], fs, freq_grid)
                if psd is not None:
                    traces[joint].append(psd)
    means = {joint: np.mean(np.vstack(v), axis=0) for joint, v in traces.items() if v}
    return means, {joint: len(v) for joint, v in traces.items()}


def main():
    with open(inp("healthy_sessions.csv"), newline="", encoding="utf-8-sig") as f:
        sessions = [r["session"].strip() for r in csv.DictReader(f) if r.get("session", "").strip()]
    root = inp("healthy_exports")
    freq_grid = np.arange(0.0, FMAX + FREQ_STEP / 2, FREQ_STEP)
    curves = {joint: [] for joint in JOINTS}
    n_traces = {joint: 0 for joint in JOINTS}
    for i, session in enumerate(sessions, 1):
        means, counts = session_psd(session, root, freq_grid)
        for joint in JOINTS:
            n_traces[joint] += counts[joint]
            if joint in means:
                curves[joint].append(means[joint])
        print(f"[{i:02d}/{len(sessions):02d}] {session}: " + " ".join(f"{j}={counts[j]}" for j in JOINTS))

    summary = {}
    for joint in JOINTS:
        arr = np.vstack(curves[joint])
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        summary[joint] = (mean, sem, 1.96 * sem, arr.shape[0])

    header = ["freq_hz"]
    for joint in JOINTS:
        header += [f"{joint}_mean", f"{joint}_sem", f"{joint}_ci95_low", f"{joint}_ci95_high"]
    header += [f"{joint}_n_sessions" for joint in JOINTS] + [f"{joint}_n_traces" for joint in JOINTS]
    p = out_path("goniometer_power_summary.csv")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, freq in enumerate(freq_grid):
            row = [f"{freq:.5f}"]
            for joint in JOINTS:
                mean, sem, ci95, _ = summary[joint]
                row += [f"{mean[i]:.10g}", f"{sem[i]:.10g}", f"{mean[i] - ci95[i]:.10g}", f"{mean[i] + ci95[i]:.10g}"]
            row += [f"{summary[joint][3]}" for joint in JOINTS] + [f"{n_traces[joint]}" for joint in JOINTS]
            w.writerow(row)
    os.replace(tmp, p)
    print("wrote", p, "traces:", n_traces)


if __name__ == "__main__":
    main()
