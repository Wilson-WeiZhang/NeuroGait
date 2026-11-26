"""Deep baselines of Supplementary Table 2: EEGNet-8,2, DeepConvNet and EEG2GAIT, each re-trained on one session.

Each network is trained per session on the strictly causal EEG (the recording after the causal 1000 -> 250 Hz downsampling and
nothing else; 59 recorded channels) with the front end of its own publication, every filter applied forward in time, and is
evaluated like the decoder: the same trials, the same random ten-fold split, the same goniometer targets (angle minus its mean over
the first 0.2 s of the trial, six joints), scored from 4.00 s after trial onset. The trials that start after the session's
calibration segment are selected later, by `suptable02_deep_baselines.py`; training uses every trial of the fold map, as the decoder.

Shared with the decoder (`baseline_common.py`): the trial selection (imagined walking, quality control good, in recording order),
the fold map (guard: fold-map trial count == selected trial count), the per-fold split of the training trials into training and
validation (seed + 1009 x fold + sum of the character codes of the session label, 15 % validation), the z-scoring of the targets
from the training windows, the per-trial correlation (`corr_1d`), `rpt` (mean of per-trial values) and the output schema.

Scoring: r_4s is the per-trial correlation over the predictions whose window ends at least 4.00 s after trial onset (target sample
j >= ceil(4.0 x fs): 400 at 100 Hz, 512 at 128 Hz, 1000 at 250 Hz). It is the reported score (fold table rpt_deep / r_deep,
per-trial table r and r_4s). r_2s uses every window (the first window ends at 2 s - 1 sample).

Front end, applied to the continuous session signal before the trials are cut (every filter is forward only, so a prediction uses
EEG up to the predicted sample; channels that are all NaN, such as the online reference, are dropped and logged; any other
non-finite sample stops the run):
  eegnet    Lawhern et al. 2018, J Neural Eng:     band-pass 0.1-40 Hz minimum-phase FIR (EEGLAB pop_eegfiltnew design, 8251
            window 2 s = 256 samples at 128 Hz,     taps at 250 Hz) -> causal resampling 250 -> 128 Hz -> per-channel z-score
            kernel length 64 = fs / 2               with the statistics of the training trials
  deep4     Schirrmeister et al. 2017, Hum Brain    band-pass 0.1-38 Hz, forward Butterworth of order 3 per edge -> electrode-wise
            Mapp: window 2 s = 500 samples at       exponential moving standardisation (decay 0.999, initial statistics from the
            250 Hz                                  first 1,000 samples)
  eeg2gait  the authors' code (eeg2gait_team_adapter.py): band-pass 0.1-48 Hz minimum-phase FIR (EEGLAB design, 8251 taps) ->
            causal resampling 250 -> 100 Hz; no re-referencing, no Laplacian, no EEG standardisation; window 2 s = 200 samples
Causal resampling: one polyphase filter (scipy.signal.upfirdn) whose taps are `resample_poly`'s default design (Kaiser beta 5, cutoff
at the lower Nyquist, 20 x max(up, down) + 1 taps) converted to minimum phase (real cepstrum, EEGLAB minphaserceps); output sample k
depends on input samples at times <= k / fs_out. The zero-phase `resample_poly` is not used on the EEG. Start-up: the signal before
the first sample, and before each break listed in the session's boundary table (joins of recording parts and raw data breaks), is
taken as that sample held constant; the band-pass and the resampler restart at every break. The exponential moving
standardisation runs on across breaks: it is an adaptive estimate with a time constant of 1,000 samples, and re-initialising it on
a stretch's first 1,000 samples would use samples after the break.
Trial cut: EEG sample k0 + j pairs with target sample j, k0 = floor(et0 x fs_out / eeg_fs), so a window never ends after its target
sample (the EEG lags the target by 0 to < 1 sample: up to 7.8 ms at 128 Hz, 10 ms at 100 Hz, none at 250 Hz).

Networks:
  eegnet    EEGNet-8,2 (layer-for-layer port of the reference Keras implementation) with a linear 6-output head with max-norm 0.25 in
            place of the softmax; kernel length 64.
  deep4     DeepConvNet: conv (10 x 1, 25) -> conv over electrodes (25) -> BN -> ELU -> max-pool 3 / 3; three blocks [dropout 0.5 ->
            conv 10 x 1 (50 / 100 / 200) -> BN -> ELU -> max-pool 3 / 3]; conv classifier with 6 linear outputs. Layer structure and
            parameter names follow the braindecode implementation (Deep4Net), written out here.
  eeg2gait  the authors' GaitGraph_tmp, preset "paper", with the neighbour table as adjacency prior.
Training:
  eegnet / deep4: Adam, learning rate 1e-3 (default betas, no weight decay, no schedule), masked MSE, at most 50 epochs, early
    stopping on the validation loss (patience 10), batch 64, training stride 0.05 s (6 samples at 128 Hz, 12 at 250 Hz; windows are
    cut on the fly from index arrays), best-validation state restored.
  eeg2gait: the authors' recipe (eeg2gait_team_adapter.TEAM_RECIPE): the model's own loss (MSE plus FFT over the six joints), AdamW
    (weight decay 0.05) with a per-iteration cosine schedule (2 warm-up epochs, 5e-4 -> 1e-4), 200 epochs, batch 128 of consecutive
    windows (stride 1, no shuffling, last incomplete batch dropped), mixed precision, the epoch with the best validation R^2 kept.
  Test stride 1. Seeds: the numpy split / shuffle generator and torch.manual_seed both take the fold seed (logged per fold).
Gates (exit 1 and the outputs stay *.partial): fold-map trial count == selected trial count; every fold-map trial usable; with
--decoder-dump, per session the (fold, trial) set equals the decoder's; with --decoder-dump and --plan, the finite (fold, joint)
cells of r_4s on the scored trials (trial_idx > n_cal) equal the decoder's.

Named inputs (default values of the options): step1_triples (folder with <session>/<session>_{eeg,goni,event}.mat and the
optional <session>_boundaries.csv), fold_map.csv, eeg2gait_team_code (for eeg2gait only).
Run: python train_baseline.py --model {eegnet,deep4,eeg2gait} --session <session> --out <folder>/<tag>_<session>.csv [--device cuda:0]
Writes: <out> (session x fold x joint), <stem>_trial_r.csv (per test trial x joint), <stem>.meta.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.signal as signal
import torch
import torch.nn as nn

import baseline_common as C
import eeg2gait_team_adapter as TEAM
from eeg2gait_channel_neighbours import team_adjacency

INPUT_TAG = "step1_causal_triple"
SCORE_FROM_SEC = 4.0
EMS_DECAY, EMS_INIT, EMS_EPS = 0.999, 1000, 1e-4

SPECS = {
    "eegnet": dict(arch="eegnet_8_2_lawhern2018", fs=128, win_sec=2.0, batch=64, patience=10, epochs=50,
                   stride_sec=0.05, early="val_loss", bp=("fir_minphase", 0.1, 40.0), car=False, norm="zscore",
                   recipe="paper",
                   impl="port of the reference Keras EEGNet-8,2 (kernel length 64)"),
    "deep4": dict(arch="deep_convnet_schirrmeister2017", fs=250, win_sec=2.0, batch=64, patience=10, epochs=50,
                  stride_sec=0.05, early="val_loss", bp=("butter3", 0.1, 38.0), car=False, norm="ems",
                  recipe="paper", impl="paper implementation (Deep4Net layer structure)"),
    "eeg2gait": dict(arch="eeg2gait_team_gaitgraph_paper", fs=100, win_sec=2.0, batch=TEAM.TEAM_RECIPE["batch"],
                     patience=0, epochs=TEAM.TEAM_RECIPE["epochs"], stride_sec=None, early="val_rsq (best epoch)",
                     bp=("fir_minphase", 0.1, 48.0), car=False, norm="none", recipe="team",
                     impl="authors' GaitGraph_tmp, preset paper (eeg2gait_team_adapter.py)"),
}


# ------------------------------------------------------------------------------------------------- filters
def eeglab_bandpass_fir(fs: float, lo: float, hi: float) -> np.ndarray:
    """EEGLAB pop_eegfiltnew(EEG, 'locutoff', lo, 'hicutoff', hi) default design (firfilt): Hamming-windowed sinc,
    transition width df = min(max(lo * 0.25, 2), min(lo, fs/2 - hi)), order = ceil(3.3 / (df / fs) / 2) * 2, -6 dB
    cutoffs lo - df/2 and hi + df/2, b = lowpass(hi + df/2) - lowpass(lo - df/2) with each kernel at unity DC gain
    (firws). 0.1-40 Hz at 250 Hz: df 0.1 Hz, 8251 taps (= the band-pass of the causal preprocessing)."""
    nyq = fs / 2.0
    df = min(max(lo * 0.25, 2.0), min(lo, nyq - hi))
    order = int(math.ceil(3.3 / (df / fs) / 2.0) * 2)
    m = np.arange(-order // 2, order // 2 + 1, dtype=np.float64)
    w = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(order + 1) / order)

    def kernel(fc_hz):
        f = fc_hz / fs                         # cycles / sample (firws: f / fNyquist / 2)
        b = np.empty_like(m)
        nz = m != 0
        b[~nz] = 2.0 * np.pi * f
        b[nz] = np.sin(2.0 * np.pi * f * m[nz]) / m[nz]
        b *= w
        return b / b.sum()

    return kernel(hi + df / 2.0) - kernel(lo - df / 2.0)


def minphase_rceps(b: np.ndarray) -> np.ndarray:
    """EEGLAB firfilt minphaserceps: minimum-phase FIR of the same length and magnitude response via the real cepstrum
    (nFFT = next power of 2 >= 1000 * length, spectrum clipped at 1e-8)."""
    n = len(b)
    nfft = 1 << int(math.ceil(math.log2(n * 1000)))
    s = np.abs(np.fft.fft(b, nfft))
    s[s < 1e-8] = 1e-8
    c = np.real(np.fft.ifft(np.log(s)))
    c = np.concatenate([[c[0]], 2.0 * c[1:nfft // 2], [c[nfft // 2]], np.zeros(nfft // 2 - 1)])
    return np.real(np.fft.ifft(np.exp(np.fft.fft(c))))[:n]


def _stretches(n: int, starts) -> list[tuple[int, int]]:
    s = sorted({0, *[int(v) for v in starts if 0 < int(v) < n]})
    return list(zip(s, s[1:] + [n]))


def fir_causal(x: np.ndarray, b: np.ndarray, starts=()) -> np.ndarray:
    """Forward FIR filtering (FFT overlap-add) of x (N, C) float64; each stretch starts from its first sample held
    constant (len(b) - 1 samples of history), so output n depends on input samples <= n only."""
    out = np.empty_like(x)
    pad = len(b) - 1
    for s0, s1 in _stretches(len(x), starts):
        seg = x[s0:s1]
        xp = np.concatenate([np.repeat(seg[:1], pad, axis=0), seg], axis=0)
        y = signal.oaconvolve(xp, b[:, None], mode="full", axes=0)
        out[s0:s1] = y[pad:pad + (s1 - s0)]
    return out


def butter_causal(x: np.ndarray, sos: np.ndarray, starts=()) -> np.ndarray:
    """Forward IIR (second-order sections); initial state = steady state for the stretch's first sample."""
    out = np.empty_like(x)
    zi0 = signal.sosfilt_zi(sos)                       # (n_sections, 2)
    for s0, s1 in _stretches(len(x), starts):
        seg = x[s0:s1]
        zi = zi0[:, :, None] * seg[0][None, None, :]
        out[s0:s1], _ = signal.sosfilt(sos, seg, axis=0, zi=zi)
    return out


def resampler_taps(up: int, down: int) -> np.ndarray:
    """resample_poly's default polyphase filter (firwin, Kaiser beta 5, cutoff 1 / max(up, down) of the upsampled
    Nyquist, 20 * max + 1 taps, gain = up), converted to minimum phase (same magnitude)."""
    mx = max(up, down)
    h = signal.firwin(2 * 10 * mx + 1, 1.0 / mx, window=("kaiser", 5.0)) * up
    return minphase_rceps(h)


def causal_resample(x: np.ndarray, fs_in: float, fs_out: float, h: np.ndarray | None = None,
                    starts=()) -> np.ndarray:
    """Causal rational resampling of x (N, C) float64: upfirdn with the minimum-phase filter; output k <-> input time
    k / fs_out (one global grid) and uses inputs at times <= k / fs_out. Each stretch (starts) is resampled on its own,
    its past = its first sample held constant, so the filter restarts at every break."""
    g = math.gcd(int(round(fs_in)), int(round(fs_out)))
    up, down = int(round(fs_out)) // g, int(round(fs_in)) // g
    if up == down:
        return x.copy()
    if h is None:
        h = resampler_taps(up, down)
    c = int(math.ceil(math.ceil(len(h) / up) / down))
    out = np.empty((-(-len(x) * up // down), x.shape[1]))
    for s0, s1 in _stretches(len(x), starts):
        p0 = (s0 // down - c) * down                    # padded start on the grid: p0 * up / down is an integer
        xp = np.concatenate([np.repeat(x[s0:s0 + 1], s0 - p0, axis=0), x[s0:s1]], axis=0)
        y = signal.upfirdn(h, xp, up, down, axis=0)
        base = p0 * up // down                           # global output index of y[0]
        m0, m1 = -(-s0 * up // down), -(-s1 * up // down)
        assert m0 - base >= 0 and m1 - base <= len(y), (s0, s1, len(y))
        out[m0:m1] = y[m0 - base:m1 - base]
    return out


def ems(x: np.ndarray, decay=EMS_DECAY, init=EMS_INIT, eps=EMS_EPS) -> np.ndarray:
    """Schirrmeister 2017 A.7 eq. 4-6, electrode-wise exponential moving standardisation on the continuous signal:
    mu_t = (1 - decay) x_t + decay mu_{t-1}; s2_t = (1 - decay)(x_t - mu_t)^2 + decay s2_{t-1};
    x'_t = (x_t - mu_t) / sqrt(s2_t); the first `init` values of mu and s2 = mean and variance of the first `init`
    samples (A.7). sqrt(s2) floored at eps (braindecode's value) for flat channels. Forward recursion: x'_t uses
    samples <= max(t, init - 1)."""
    a = 1.0 - decay
    mu0, v0 = x[:init].mean(0), x[:init].var(0)
    mu = np.empty_like(x)
    mu[:init] = mu0
    mu[init:] = signal.lfilter([a], [1.0, -decay], x[init:], axis=0, zi=(decay * mu0)[None, :])[0]
    d = x - mu
    var = np.empty_like(x)
    var[:init] = v0
    var[init:] = signal.lfilter([a], [1.0, -decay], d[init:] ** 2, axis=0, zi=(decay * v0)[None, :])[0]
    return d / np.maximum(np.sqrt(var), eps)


def _blockwise(fn, x: np.ndarray, block=8) -> np.ndarray:
    """Apply fn (float64 in, float64 out, any length) to channel blocks of x (N, C) float32; result float32."""
    outs = [fn(x[:, c0:c0 + block].astype(np.float64)).astype(np.float32) for c0 in range(0, x.shape[1], block)]
    return np.concatenate(outs, axis=1)


# ---------------------------------------------------------------------------------------------- front end
def front_end(model: str, eeg: np.ndarray, fs: float, labels: list[str], starts=()):
    """Continuous causal front end of `model` on eeg (N, C) at fs. Returns (x (N', C') float32 at the model rate,
    fs_out, info, spatial). spatial (eeg2gait only, else {}): adjacency = the team's adjacency prior."""
    sp = SPECS[model]
    allnan = np.all(np.isnan(eeg), axis=0)
    keep = np.flatnonzero(~allnan)
    dropped = [labels[i] for i in np.flatnonzero(allnan)]
    x = np.ascontiguousarray(eeg[:, keep], dtype=np.float32)
    if not np.isfinite(x).all():
        bad = [labels[keep[i]] for i in np.flatnonzero(~np.isfinite(x).all(0))]
        raise SystemExit(f"non-finite samples in channels that are not all-NaN: {bad} -- STOP")
    lab = [labels[i] for i in keep]
    info = dict(n_chan_in=len(labels), n_chan=len(keep), dropped_channels=dropped,
                flat_channels=[lab[i] for i in np.flatnonzero(np.ptp(x, axis=0) == 0)],
                fs_in=fs, boundaries=[int(v) for v in starts])
    kind, lo, hi = sp["bp"]
    if kind == "fir_minphase":
        b = minphase_rceps(eeglab_bandpass_fir(fs, lo, hi))
        x = _blockwise(lambda v: fir_causal(v, b, starts), x)
        info["filter_desc"] = (f"band-pass {lo:g}-{hi:g} Hz minimum-phase FIR (EEGLAB pop_eegfiltnew design, Hamming, "
                               f"{len(b)} taps at {fs:g} Hz, minphaserceps), forward")
        info["filter_mode"] = "minphase_fir"
        info["bp_taps"] = len(b)
    else:
        sos = signal.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")
        x = _blockwise(lambda v: butter_causal(v, sos, starts), x)
        info["filter_desc"] = f"band-pass {lo:g}-{hi:g} Hz Butterworth order 3 per edge (sos), forward"
        info["filter_mode"] = "butter3_forward"
    if sp["car"]:
        x = x - x.mean(axis=1, keepdims=True)
        info["filter_desc"] += f"; CAR over {x.shape[1]} channels"
    fs_out = float(sp["fs"])
    if abs(fs_out - fs) > 1e-9:
        g = math.gcd(int(round(fs)), int(round(fs_out)))
        h = resampler_taps(int(round(fs_out)) // g, int(round(fs)) // g)
        x = _blockwise(lambda v: causal_resample(v, fs, fs_out, h, starts), x)
        info["filter_desc"] += f"; causal resampling {fs:g} -> {fs_out:g} Hz (upfirdn, minimum-phase {len(h)} taps)"
        info["resample_taps"] = len(h)
    if sp["norm"] == "ems":
        x = _blockwise(ems, x)
        info["filter_desc"] += f"; exponential moving standardisation decay {EMS_DECAY}, init {EMS_INIT} samples"
    elif sp["norm"] == "zscore":
        info["filter_desc"] += "; per-channel z-score from training trials"
    else:
        info["filter_desc"] += "; no EEG standardisation (team pipeline)"
    spatial = {}
    if sp["recipe"] == "team":
        try:
            adj = team_adjacency(lab)
        except ValueError as e:
            raise SystemExit(f"{e} -- STOP")
        spatial = dict(adjacency=adj)
        nn_ = adj.sum(1) - 1
        info.update(adjacency="neighbour table (FuXi1999 56b4625 = team sbj_dep_train), symmetric, diagonal 1",
                    adj_neighbours={lab[k]: int(nn_[k]) for k in range(len(lab))}, adj_edges=int(nn_.sum()) // 2,
                    adj_neighbours_min=int(nn_.min()), adj_neighbours_max=int(nn_.max()))
        info["filter_desc"] += f"; GCN adjacency prior = neighbour table ({int(nn_.sum()) // 2} edges)"
    info["front_end"] = f"{model}_{INPUT_TAG}"
    return x, fs_out, info, spatial


def cut_trials(x: np.ndarray, fs_out: float, eeg_fs: float, goni: np.ndarray, goni_fs: float, goni_labels,
               events, time_step: int, start_window_sec: float):
    """Trials of the front-ended continuous EEG. Target rule = the decoder's (n samples, theta0 over the first
    start_window_sec, resample_to per trial); EEG = x[k0 : k0 + len(y)], k0 = floor(et0 * fs_out / eeg_fs). The list
    index equals the MATLAB trial index (None = unusable trial)."""
    jidx = [goni_labels.index(lab) if lab in goni_labels else None for lab in C.TARGET_LABELS]
    base_n = max(1, int(round(start_window_sec * goni_fs)))
    trials = []
    for e in events:
        et0, et1, gt0, gt1 = e["etstart"], e["etstop"], e["gtstart"], e["gtstop"]
        if et1 < et0 or gt1 < gt0:
            trials.append(None)
            continue
        n = min(et1 - et0 + 1, gt1 - gt0 + 1)
        if n <= 4:
            trials.append(None)
            continue
        y = np.full((n, len(C.TARGET_LABELS)), np.nan, dtype=np.float32)
        base_end = min(gt1, gt0 + base_n - 1)
        for j, gi in enumerate(jidx):
            if gi is None:
                continue
            bv = goni[gt0:base_end + 1, gi].astype(np.float32)
            if np.any(np.isfinite(bv)):
                y[:, j] = goni[gt0:gt0 + n, gi].astype(np.float32) - float(np.nanmean(bv))
        y_rs = C.resample_to(y, goni_fs, fs_out)
        k0 = int(math.floor(et0 * fs_out / eeg_fs + 1e-9))
        e_tr = x[k0:k0 + len(y_rs)]
        m = min(len(e_tr), len(y_rs))
        trials.append(C.Trial(e_tr[:m], y_rs[:m]) if m >= time_step + 2 else None)
    return trials


# ----------------------------------------------------------------------------------------------------- models
class Rearr(nn.Module):
    """(B, C, T) -> (B, 1, T, C): braindecode's Ensure4d + Rearrange('batch C T 1 -> batch 1 T C')."""

    def forward(self, x):
        return x.permute(0, 2, 1).unsqueeze(1)


class TimeSpat(nn.Module):
    """Split first layer: temporal conv (with bias), then conv over all electrodes (no bias, BN follows). The braindecode
    implementation merges the two kernels into one conv, numerically equivalent."""

    def __init__(self, n_chans, n_time_filt, n_spat_filt, k):
        super().__init__()
        self.conv_time = nn.Conv2d(1, n_time_filt, (k, 1), bias=True)
        self.conv_spat = nn.Conv2d(n_time_filt, n_spat_filt, (1, n_chans), bias=False)

    def forward(self, x):
        return self.conv_spat(self.conv_time(x))


class _Squeeze(nn.Module):
    def forward(self, x):
        return x[:, :, :, 0].squeeze(-1) if x.shape[2] == 1 else x[:, :, :, 0]


def _final_layer(n_in, n_out, length):
    m = nn.Sequential()
    m.add_module("conv_classifier", nn.Conv2d(n_in, n_out, (length, 1), bias=True))
    m.add_module("squeeze", _Squeeze())
    return m


def _xavier_init(model: nn.Module):
    """braindecode init: xavier_uniform (gain 1) on conv weights, conv biases 0, BN weight 1 / bias 0."""
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            nn.init.xavier_uniform_(mod.weight, gain=1)
            if mod.bias is not None:
                nn.init.constant_(mod.bias, 0)
        elif isinstance(mod, nn.BatchNorm2d):
            nn.init.constant_(mod.weight, 1)
            nn.init.constant_(mod.bias, 0)


class DeepConvNet(nn.Sequential):
    """DeepConvNet (Schirrmeister et al. 2017; Deep4Net defaults, parameter names as in braindecode). Input (B, C, T), T = 500."""

    def __init__(self, n_chans, n_times, n_outputs):
        super().__init__()
        self.add_module("dimshuffle", Rearr())
        self.add_module("conv_time_spat", TimeSpat(n_chans, 25, 25, 10))   # Fig. 1: 25 temporal (10), 25 spatial
        self.add_module("bnorm", nn.BatchNorm2d(25, momentum=0.1, affine=True, eps=1e-5))
        self.add_module("conv_nonlin", nn.ELU())                           # § 2.4.2: ELU
        self.add_module("pool", nn.MaxPool2d((3, 1), stride=(3, 1)))       # Fig. 1: max pool 3, stride 3
        self.add_module("pool_nonlin", nn.Identity())
        prev = 25
        for k, nf in ((2, 50), (3, 100), (4, 200)):                        # Fig. 1: blocks 2-4, 50 / 100 / 200
            self.add_module(f"drop_{k}", nn.Dropout(p=0.5))                # § 2.4.4: dropout on inputs of conv 2-4
            self.add_module(f"conv_{k}", nn.Conv2d(prev, nf, (10, 1), bias=False))
            self.add_module(f"bnorm_{k}", nn.BatchNorm2d(nf, momentum=0.1, affine=True, eps=1e-5))
            self.add_module(f"nonlin_{k}", nn.ELU())
            self.add_module(f"pool_{k}", nn.MaxPool2d((3, 1), stride=(3, 1)))
            self.add_module(f"pool_nonlin_{k}", nn.Identity())
            prev = nf
        with torch.no_grad():
            self.eval()
            t_left = self(torch.zeros(1, n_chans, n_times)).shape[2]
            self.train()
        self.add_module("final_layer", _final_layer(200, n_outputs, t_left))
        _xavier_init(self)


class EEGNet128(nn.Module):
    """EEGNet-8,2: layer-for-layer port of the reference Keras implementation (Lawhern et al. 2018) with a linear max-norm-0.25
    head, kernLength as an argument: 64 = half of 128 Hz (§ 2.2.1)."""

    F1, D, F2 = 8, 2, 16

    def __init__(self, n_chan: int, n_time: int, n_out: int, kern_length: int = 64, dropout: float = 0.5):
        super().__init__()
        k = kern_length
        f1, fd = self.F1, self.F1 * self.D
        self.block1 = nn.Sequential(
            nn.ZeroPad2d(((k - 1) // 2, k // 2, 0, 0)),
            nn.Conv2d(1, f1, (1, k), bias=False),
            nn.BatchNorm2d(f1),
            C.Conv2dC(f1, fd, (n_chan, 1), groups=f1, bias=False, max_norm=1.0),
            nn.BatchNorm2d(fd),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=dropout),
        )
        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(fd, fd, (1, 16), groups=fd, bias=False),
            nn.Conv2d(fd, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(p=dropout),
        )
        with torch.no_grad():
            feat = self.block2(self.block1(torch.zeros(1, 1, n_chan, n_time)))
        self.head = C.LinearC(feat.numel(), n_out, max_norm=0.25)

    def forward(self, x):                       # x (B, C, T)
        return self.head(torch.flatten(self.block2(self.block1(x.unsqueeze(1))), 1))


def build_model(model: str, n_chan: int, n_time: int, adjacency=None) -> nn.Module:
    n_out = len(C.TARGET_LABELS)
    if model == "eegnet":
        return EEGNet128(n_chan, n_time, n_out, kern_length=SPECS["eegnet"]["fs"] // 2)
    if model == "deep4":
        return DeepConvNet(n_chan, n_time, n_out)
    return TEAM.TeamEEG2GAIT(n_chan, n_time, n_out, adjacency)


# ---------------------------------------------------------------------------------------------- windows
class Windows:
    """Window table of a trial set: window k ends at global sample `end[k]` of the device EEG tensor (C, total); y / m
    / tid / pos as the decoder's windows (target z-units, mask, trial list index, window-end sample within the
    trial). EEG crops are gathered per batch, never materialised."""

    def __init__(self, trials, idx, offsets, ymu, ysd, time_step, stride):
        ends, ys, ms, ts, ps = [], [], [], [], []
        for ti in idx:
            pos = np.arange(time_step - 1, len(trials[ti].y), stride)
            if not len(pos):
                continue
            yv = (trials[ti].y[pos] - ymu) / ysd
            ms.append(np.isfinite(yv).astype(np.float32))
            ys.append(np.nan_to_num(yv, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
            ts.append(np.full(len(pos), ti, np.int64))
            ps.append(pos)
            ends.append(offsets[ti] + pos)
        self.n = int(sum(len(p) for p in ps))
        if self.n:
            self.end, self.y, self.m = np.concatenate(ends), np.concatenate(ys), np.concatenate(ms)
            self.tid, self.pos = np.concatenate(ts), np.concatenate(ps)


def gather(xdev: torch.Tensor, ends: np.ndarray, time_step: int) -> torch.Tensor:
    """(B, C, T) crops ending at `ends` from the device tensor xdev (C, total)."""
    e = torch.as_tensor(ends, device=xdev.device)
    idx = e[:, None] - time_step + 1 + torch.arange(time_step, device=xdev.device)[None, :]
    return xdev[:, idx].permute(1, 0, 2)


@torch.no_grad()
def predict(model, xdev, w: Windows, time_step, bs=512, ctx=None):
    """Eval-mode predictions as float32 numpy; ctx = the team recipe's autocast factory (None = full precision)."""
    model.eval()
    out = []
    for s in range(0, w.n, bs):
        with ctx() if ctx else torch.autocast("cpu", enabled=False):
            out.append(model(gather(xdev, w.end[s:s + bs], time_step)).float().cpu().numpy())
    model.train()
    return np.vstack(out)


# --------------------------------------------------------------------------------------------------- fold
def _train_paper(model, xdev, tr, va, ts, ymu, ysd, cfg, device, rng):
    """eegnet / deep4: Adam, masked MSE, shuffled batches, early stopping on validation loss."""
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    yt_all, mt_all = torch.from_numpy(tr.y).to(device), torch.from_numpy(tr.m).to(device)
    best, best_state, best_ep, bad, hist, keep = -np.inf, None, -1, 0, [], {}
    for ep in range(cfg.epochs):
        model.train()
        p = rng.permutation(tr.n)
        batches = [p[s:s + cfg.batch_size] for s in range(0, tr.n, cfg.batch_size)]
        tl = 0.0
        for bidx in batches:
            bi = torch.as_tensor(bidx, device=device)
            loss = C.masked_mse(model(gather(xdev, tr.end[bidx], ts)), yt_all[bi], mt_all[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += float(loss.detach())
        pv = predict(model, xdev, va, ts)
        vloss = float(((pv - va.y) ** 2 * va.m).sum() / max(va.m.sum(), 1.0))
        _, vr = C.score(pv, va.y, va.m, va.tid, ymu, ysd)
        vr = float(np.nanmean(vr)) if np.any(np.isfinite(vr)) else -np.inf
        hist.append((ep + 1, tl / max(len(batches), 1), vloss, vr))
        if -vloss > best or best_state is None:
            best, best_ep, bad = -vloss, ep + 1, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            keep = dict(best_val_loss=vloss, best_val_r=vr)
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return dict(best_epoch=best_ep, epochs_run=len(hist), hist=hist, **keep), None


def _train_team(model, xdev, tr, va, ts, ymu, ysd, cfg, device):
    """eeg2gait: the team's recipe (eeg2gait_team.TEAM_RECIPE): AdamW from the model's configure_optimizers, per-
    iteration cosine lr, consecutive batches (no shuffle, drop_last), autocast, the model's own loss object, the epoch
    with the best validation R^2 kept (no early stopping)."""
    R = TEAM.TEAM_RECIPE
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True            # sbj_dep_train.init
        torch.backends.cudnn.allow_tf32 = True
    ctx, amp_dtype = TEAM.amp_context(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    opt = model.net.configure_optimizers(R["weight_decay"], R["lr"], R["betas"], device.type)
    steps = tr.n // cfg.batch_size                                 # drop_last
    warm = min(R["warmup_epochs"], max(cfg.epochs - 1, 0))      # = 2 in the recipe; < 2 only for runs under 3 epochs
    sched = TEAM.cosine_scheduler(R["lr"], R["min_lr"], cfg.epochs, steps, warmup_epochs=warm)
    yt_all, mt_all = torch.from_numpy(tr.y).to(device), torch.from_numpy(tr.m).to(device)
    best, best_state, best_ep, hist, keep, it = -np.inf, None, -1, [], {}, 0
    for ep in range(cfg.epochs):
        model.train()
        tl = 0.0
        for b in range(steps):                                     # windows in recording order (shuffle False)
            bidx = np.arange(b * cfg.batch_size, (b + 1) * cfg.batch_size)
            for g in opt.param_groups:
                g["lr"] = sched[it]
            bi = torch.as_tensor(bidx, device=device)
            with ctx():
                loss = TEAM.team_loss(model.loss_fn, model(gather(xdev, tr.end[bidx], ts)), yt_all[bi], mt_all[bi])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tl += float(loss.detach())
            it += 1
        pv = predict(model, xdev, va, ts, ctx=ctx)
        with torch.no_grad():
            vloss = float(TEAM.team_loss(model.loss_fn, torch.from_numpy(pv), torch.from_numpy(va.y),
                                         torch.from_numpy(va.m)))
        vr2 = TEAM.r2_mean(pv, va.y, va.m)
        _, vr = C.score(pv, va.y, va.m, va.tid, ymu, ysd)
        vr = float(np.nanmean(vr)) if np.any(np.isfinite(vr)) else -np.inf
        hist.append((ep + 1, tl / max(steps, 1), vloss, vr, vr2))
        if vr2 > best or best_state is None:
            best, best_ep = vr2, ep + 1
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            keep = dict(best_val_loss=vloss, best_val_r=vr, best_val_rsq=vr2)
    model.load_state_dict(best_state)
    return dict(best_epoch=best_ep, epochs_run=len(hist), hist=hist, amp=str(amp_dtype), steps_per_epoch=steps,
                **keep), ctx


def run_fold(model_name, trials, test_idx, spatial, cfg, device, seed):
    """One outer fold. Split rule = the decoder's (same generator, same n_val); training per SPECS recipe.
    spatial: front_end's {} or {adjacency} (eeg2gait)."""
    pool = [i for i, t in enumerate(trials) if t is not None and i not in set(test_idx)]
    test_idx = [i for i in test_idx if trials[i] is not None]
    if len(pool) < 3 or not test_idx:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool)
    n_val = max(1, int(round(cfg.val_frac * len(perm))))
    val_idx, train_idx = sorted(perm[:n_val].tolist()), sorted(perm[n_val:].tolist())
    if len(train_idx) < 2:
        return None
    ts = cfg.time_step
    sp = SPECS[model_name]
    emu, esd, ymu, ysd = C.fit_scalers(trials, train_idx, ts, cfg.train_stride)
    if sp["norm"] != "zscore":     # ems: standardised on the continuous signal (A.7); none: team pipeline
        emu, esd = np.zeros_like(emu), np.ones_like(esd)
    used = sorted(set(train_idx) | set(val_idx) | set(test_idx))
    offsets, chunks, o = {}, [], 0
    for ti in used:
        offsets[ti] = o
        chunks.append(((trials[ti].eeg - emu) / esd).astype(np.float32))
        o += len(trials[ti].eeg)
    xdev = torch.from_numpy(np.ascontiguousarray(np.concatenate(chunks).T)).to(device)
    tr = Windows(trials, train_idx, offsets, ymu, ysd, ts, cfg.train_stride)
    va = Windows(trials, val_idx, offsets, ymu, ysd, ts, cfg.train_stride)
    te = Windows(trials, test_idx, offsets, ymu, ysd, ts, 1)
    if not (tr.n and va.n and te.n):
        return None

    torch.manual_seed(seed)
    model = build_model(model_name, xdev.shape[0], ts, spatial.get("adjacency")).to(device)
    if sp["recipe"] == "team":
        fit, ctx = _train_team(model, xdev, tr, va, ts, ymu, ysd, cfg, device)
    else:
        fit, ctx = _train_paper(model, xdev, tr, va, ts, ymu, ysd, cfg, device, rng)

    pred_te = predict(model, xdev, te, ts, ctx=ctx)
    keep4 = (te.pos >= cfg.first4).astype(np.float32)[:, None]
    keep2 = (te.pos >= ts - 1).astype(np.float32)[:, None]
    r, rpt = C.score(pred_te, te.y, te.m * keep4, te.tid, ymu, ysd)
    r2, rpt2 = C.score(pred_te, te.y, te.m * keep2, te.tid, ymu, ysd)
    t4 = {(t, j): (rv, n) for t, j, rv, n in C.trial_scores(pred_te, te.y, te.m * keep4, te.tid, ymu, ysd)}
    t2 = {(t, j): (rv, n) for t, j, rv, n in C.trial_scores(pred_te, te.y, te.m * keep2, te.tid, ymu, ysd)}
    trows = [(t, j, t2[(t, j)], t4[(t, j)]) for (t, j) in sorted(t4)]
    state = {k: t.detach().cpu() for k, t in model.state_dict().items()} if cfg.save_weights else None
    return dict(r=r, rpt=rpt, r_2s=r2, rpt_2s=rpt2, n_scored=int(keep4.sum()), trows=trows, state=state,
                n_train=len(train_idx), n_val=len(val_idx), n_test=len(test_idx), n_test_samples=te.n,
                n_train_windows=tr.n, n_params=sum(q.numel() for q in model.parameters()), seed=seed,
                build_log=getattr(model, "build_log", ""), **fit)


def fold_seed(base_seed: int, fold: int, session: str) -> int:
    return base_seed + 1009 * fold + sum(ord(c) for c in session)


# ------------------------------------------------------------------------------------------------- session io
def read_session(session: str, triple_root: str, condition: str = "MI") -> dict:
    base = Path(triple_root) / session
    eeg, eeg_fs, labels, xyz, src = C.read_h5_matrix(base / f"{session}_eeg.mat", "eeg")
    goni, goni_fs, goni_labels, _, _ = C.read_h5_matrix(base / f"{session}_goni.mat", "goni")
    ev = C.trial_filter(C.read_events(base / f"{session}_event.mat"), condition)
    return dict(eeg=eeg, eeg_fs=eeg_fs, labels=labels, xyz=xyz, source_set=src, goni=goni, goni_fs=goni_fs,
                goni_labels=goni_labels, events=ev)


def read_boundaries_csv(path: str) -> list[int]:
    """Stretch starts from the session's <label>_boundaries.csv: column
    boundary_sample_250hz = 0-based sample on the triple axis of the first sample after a break (part join or raw data
    break; `source` says which); header-only = one continuous stretch. Every listed break restarts the filters."""
    if not path:
        return []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if "boundary_sample_250hz" not in (rd.fieldnames or []):
            raise SystemExit(f"{path}: no boundary_sample_250hz column (header {rd.fieldnames}) -- STOP")
        return [int(r["boundary_sample_250hz"]) for r in rd]


def resolve_cfg(args) -> argparse.Namespace:
    sp = SPECS[args.model]
    cfg = argparse.Namespace(**vars(args))
    cfg.fs = float(sp["fs"])
    cfg.time_step = int(round(sp["win_sec"] * sp["fs"]))
    cfg.batch_size = args.batch_size or sp["batch"]
    cfg.patience = args.patience or sp["patience"]
    cfg.epochs_arg = args.epochs
    cfg.epochs = args.epochs or sp["epochs"]
    cfg.train_stride = args.train_stride or (1 if sp["stride_sec"] is None else int(round(sp["stride_sec"] * sp["fs"])))
    cfg.first4 = int(math.ceil(SCORE_FROM_SEC * sp["fs"] - 1e-9))   # 400 / 512 / 1000
    return cfg


def run_session(session: str, sd: dict, folds: dict, cfg, device, log=print):
    """Front end + trial cut + gates G-map / G-usable + all requested folds. Returns (results by fold, info)."""
    t0 = time.time()
    starts = sorted(set(cfg.boundaries or []) | set(read_boundaries_csv(cfg.boundaries_csv)))
    x, fs_out, info, spatial = front_end(cfg.model, sd["eeg"], sd["eeg_fs"], sd["labels"], starts)
    info.update(source_set=sd["source_set"], eeg_fs=sd["eeg_fs"], goni_fs=sd["goni_fs"],
                n_samples_in=int(len(sd["eeg"])), n_samples_out=int(len(x)), front_end_seconds=time.time() - t0)
    log(f"  {session}: front end {info['filter_desc']}; dropped {info['dropped_channels'] or 'none'}; "
        f"{info['n_chan']} ch; {time.time() - t0:.1f}s")
    trials = cut_trials(x, fs_out, sd["eeg_fs"], sd["goni"], sd["goni_fs"], sd["goni_labels"], sd["events"],
                        cfg.time_step, cfg.start_window_sec)
    info.update(n_trials=len(trials), n_usable=sum(t is not None for t in trials),
                missing_joints=[lab for lab in C.TARGET_LABELS if lab not in sd["goni_labels"]])
    if cfg.model == "deep4":
        first_end = min(int(math.floor(e["etstart"] * fs_out / sd["eeg_fs"])) + cfg.time_step - 1
                        for e in sd["events"])
        info["ems_first_window_end"] = first_end
        if first_end < EMS_INIT - 1:
            raise SystemExit(f"{session}: a window ends at sample {first_end} < {EMS_INIT - 1}: the EMS start-up "
                             f"block would use later samples -- STOP")
    n_map = max(max(v) for v in folds.values()) + 1
    if n_map != len(trials):
        raise SystemExit(f"G-map {session}: fold map holds trials 1..{n_map} but trial_filter yields {len(trials)} -- STOP")
    unusable = [i + 1 for i, t in enumerate(trials) if t is None]
    if unusable:
        raise SystemExit(f"G-usable {session}: trials {unusable} unusable -> trial set != decoder -- STOP")
    results = {}
    sel = sorted(folds) if cfg.only_fold == 0 else [cfg.only_fold]
    for fold in sel:
        tf0 = time.time()
        res = run_fold(cfg.model, trials, folds[fold], spatial, cfg, device, fold_seed(cfg.seed, fold, session))
        if res is None:
            log(f"  {session} fold{fold}: skipped (insufficient usable trials)")
            continue
        res["fold_seconds"] = time.time() - tf0
        results[fold] = res
        log(f"  {session} fold{fold}: {res['fold_seconds']:.1f}s epochs {res['epochs_run']} best {res['best_epoch']} "
            f"rpt(>=4s) {np.nanmean(res['rpt']):.4f} rpt(all) {np.nanmean(res['rpt_2s']):.4f}")
    return results, info


def decoder_gates(session: str, results: dict, dump_dir: str, plan_csv: str | None, folds: dict) -> list[str]:
    """G1 (fold, trial) set == the decoder dump's; G5 finite (fold, joint) cells of r_4s on trials > n_cal ==
    the decoder's (rpt_r). Returns failure messages."""
    fails = []
    rows = list(csv.DictReader(open(Path(dump_dir) / f"{session}_trial_r.csv", newline="")))
    dec_set = {(int(r["fold"]), int(r["trial_idx"])) for r in rows}
    map_set = {(f, i + 1) for f, v in folds.items() for i in v}
    if dec_set != map_set:
        fails.append(f"G1 {session}: fold-map (fold, trial) set != decoder dump ({len(map_set)} vs {len(dec_set)})")
    if plan_csv:
        ncal = {r["session"]: int(r["n_cal"]) for r in csv.DictReader(open(plan_csv, newline=""))}[session]
        dec = {}
        for r in rows:
            if int(r["trial_idx"]) > ncal:
                v = float(r["rpt_r"])
                k = (int(r["fold"]), r["joint"])
                dec[k] = dec.get(k, False) or math.isfinite(v)
        net = {}
        for fold, res in results.items():
            for t, j, _, (r4, _) in res["trows"]:
                if t + 1 > ncal:
                    k = (fold, C.TARGET_LABELS[j])
                    net[k] = net.get(k, False) or math.isfinite(r4)
        dc = {k for k, v in dec.items() if v and k[0] in results}
        nc = {k for k, v in net.items() if v}
        if dc != nc:
            fails.append(f"G5 {session}: finite masked cells network {len(nc)} != decoder {len(dc)}")
    return fails


# ------------------------------------------------------------------------------------------------- output
FOLD_COLS = ["session", "fold", "joint", "r_deep", "rpt_deep", "best_val_r", "best_epoch",
             "n_train_trials", "n_val_trials", "n_test_trials", "n_test_samples",
             "target_fs", "time_step", "train_stride", "test_stride", "filter_mode",
             "bp_lo", "bp_hi", "laplacian", "lap_radius_mm", "arch",
             "score_from_sec", "r_deep_from2s", "rpt_deep_from2s", "n_test_samples_scored",
             "fold_seconds", "gpu",
             "input", "front_end", "filter_desc", "norm", "car", "n_chan", "dropped_channels", "impl",
             "early_stop_on", "best_val_loss", "epochs_run", "batch_size", "n_params", "seed", "best_val_rsq", "amp"]
TRIAL_COLS = ["session", "fold", "trial_idx", "joint", "r", "n_samples", "r_2s", "n_r_2s", "r_4s", "n_r_4s",
              "arch", "front_end"]


def write_outputs(out: Path, session: str, results: dict, info: dict, cfg, gpu: str, fails: list[str]):
    """Fold CSV + per-trial CSV + meta.json, each written to *.partial; renamed only when `fails` is empty.
    Per-trial r = r_4s (the reported score), n_samples = n_r_4s."""
    sp = SPECS[cfg.model]
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.with_suffix("")
    tr_out = out.with_name(out.stem + "_trial_r.csv")
    meta = out.with_suffix(".meta.json")
    tmp = {p: p.with_name(p.name + ".partial") for p in (out, tr_out, meta)}
    with open(tmp[out], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FOLD_COLS)
        for fold, res in results.items():
            for j, lab in enumerate(C.TARGET_LABELS):
                w.writerow([session, fold, lab, f"{res['r'][j]:.6f}", f"{res['rpt'][j]:.6f}",
                            f"{res['best_val_r']:.6f}", res["best_epoch"], res["n_train"], res["n_val"],
                            res["n_test"], res["n_test_samples"], cfg.fs, cfg.time_step, cfg.train_stride, 1,
                            info["filter_mode"], sp["bp"][1], sp["bp"][2], 0, "nan", sp["arch"],
                            SCORE_FROM_SEC, f"{res['r_2s'][j]:.6f}", f"{res['rpt_2s'][j]:.6f}", res["n_scored"],
                            f"{res['fold_seconds']:.1f}", gpu,
                            INPUT_TAG, info["front_end"], info["filter_desc"], sp["norm"], int(sp["car"]),
                            info["n_chan"], ";".join(info["dropped_channels"]), sp["impl"], sp["early"],
                            f"{res['best_val_loss']:.6f}", res["epochs_run"], cfg.batch_size, res["n_params"],
                            res["seed"], f"{res.get('best_val_rsq', float('nan')):.6f}", res.get("amp", "none")])
    with open(tmp[tr_out], "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(TRIAL_COLS)
        for fold, res in results.items():
            for t, j, (r2, n2), (r4, n4) in res["trows"]:
                w.writerow([session, fold, t + 1, C.TARGET_LABELS[j], f"{r4:.6f}", n4, f"{r2:.6f}", n2,
                            f"{r4:.6f}", n4, sp["arch"], info["front_end"]])
    if cfg.save_weights:
        wdir = Path(str(stem) + "_weights")
        wdir.mkdir(exist_ok=True)
        for fold, res in results.items():
            torch.save(res["state"], wdir / f"fold{fold}.pt")
    prov = dict(gpu=gpu, torch=torch.__version__, cudnn=torch.backends.cudnn.version(),
                numpy=np.__version__, scipy=scipy.__version__, python=sys.version.split()[0])
    first = results[min(results)] if results else {}
    meta_d = dict(args=vars(cfg), spec=sp, session=session, info=info, gates_failed=fails,
                  folds_written=sorted(results), seeds={f: r["seed"] for f, r in results.items()},
                  history={f: r["hist"] for f, r in results.items()}, input=INPUT_TAG,
                  score_from_sec=SCORE_FROM_SEC, first_scored_sample=cfg.first4,
                  team_recipe=TEAM.TEAM_RECIPE if sp["recipe"] == "team" else None,
                  model_build_log=first.get("build_log", ""), provenance=prov)
    tmp[meta].write_text(json.dumps(meta_d, indent=2, default=str))
    if fails:
        return False
    for p, tp in tmp.items():
        os.replace(tp, p)
    return True


# --------------------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(SPECS))
    p.add_argument("--session", required=True)
    p.add_argument("--triple-root", default=str(C.inp("step1_triples")))
    p.add_argument("--out", required=True, help="fold CSV path; <stem>_trial_r.csv and <stem>.meta.json beside it")
    p.add_argument("--foldmap", default=str(C.inp("fold_map.csv")))
    p.add_argument("--decoder-dump", default="", help="folder of the decoder's per-session <session>_trial_r.csv, for gate G1 (and G5 with --plan)")
    p.add_argument("--plan", default="", help="calibration plan table with the column n_cal, for gate G5")
    p.add_argument("--boundaries", type=int, nargs="*", default=[],
                   help="0-based EEG samples (triple timeline) where a new continuous recording stretch starts; the "
                        "band-pass restarts there (multi-part sessions)")
    p.add_argument("--boundaries-csv", default="", help="<label>_boundaries.csv of the session (see "
                   "read_boundaries_csv); merged with --boundaries")
    p.add_argument("--condition", default="MI", choices=["MI", "EXEC"])
    p.add_argument("--only-fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0, help="0 = the model default (50; eeg2gait 200, authors' recipe)")
    p.add_argument("--patience", type=int, default=0, help="0 = the model default (10; eeg2gait: none, best "
                                                           "validation R^2 of all epochs)")
    p.add_argument("--batch-size", type=int, default=0, help="0 = the model default (64; eeg2gait 128)")
    p.add_argument("--train-stride", type=int, default=0, help="0 = 0.05 s (eeg2gait: 1 sample)")
    p.add_argument("--lr", type=float, default=1e-3, help="eegnet / deep4 (eeg2gait: TEAM_RECIPE)")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--start-window-sec", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--save-weights", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = resolve_cfg(args)
    torch.set_num_threads(cfg.threads)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    fm = C.read_foldmap(Path(cfg.foldmap), [cfg.session])
    if cfg.session not in fm:
        raise SystemExit(f"{cfg.session} not in the fold map {cfg.foldmap}")
    folds = dict(fm[cfg.session])
    sd = read_session(cfg.session, cfg.triple_root, cfg.condition)
    bcsv = Path(cfg.triple_root) / cfg.session / f"{cfg.session}_boundaries.csv"
    if not cfg.boundaries_csv and bcsv.is_file():
        cfg.boundaries_csv = str(bcsv)
    print(f"{cfg.model} {cfg.session}: eeg {sd['eeg'].shape} @ {sd['eeg_fs']:g} Hz (input {INPUT_TAG}), "
          f"{len(sd['events'])} trials, {len(folds)} folds; breaks {read_boundaries_csv(cfg.boundaries_csv)} "
          f"({cfg.boundaries_csv or 'no boundaries csv'}); fs {cfg.fs:g}, window {cfg.time_step}, stride "
          f"{cfg.train_stride}, batch {cfg.batch_size}, patience {cfg.patience}; {gpu}", flush=True)
    results, info = run_session(cfg.session, sd, folds, cfg, device, log=lambda s: print(s, flush=True))
    fails = []
    if cfg.decoder_dump:
        fails += decoder_gates(cfg.session, results, cfg.decoder_dump, cfg.plan or None, folds)
    exp = len(folds) if cfg.only_fold == 0 else 1
    if len(results) != exp:
        fails.append(f"folds written {len(results)} != expected {exp}")
    ok = write_outputs(Path(cfg.out), cfg.session, results, info, cfg, gpu, fails)
    for f in fails:
        print("GATE FAIL", f, flush=True)
    if not ok:
        raise SystemExit(1)
    print(f"wrote {cfg.out} ({len(results)} folds)", flush=True)


if __name__ == "__main__":
    main()
