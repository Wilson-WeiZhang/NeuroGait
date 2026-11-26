"""Adapter that runs the EEG2GAIT network of its authors inside the baseline training loop (Supplementary Table 2).

The network is the authors' code, which this repository does not redistribute (it carries no licence): the public release is
https://github.com/NgHanWei/EEG2Gait (commit f07fce0). Check out that repository and point the named input
`eeg2gait_team_code` at the checkout. The files used are `GaitGraph_model.py` (class `GaitGraph_tmp`, found at the root of the
checkout or in `models/`), `models/DCN_model.py` (`Conv2dWithConstraint`) and `utils/FreDF.py` (`FreDFLossWithEncouraging`).
The model is built with its "paper" preset, which is not the default preset of the public file ("conservative"). Nothing in
those files is edited; the adapter only makes them importable, feeds our windows in and reads the predictions out.

Import: the model file does `from .DCN_model import ...` (so it is loaded as a submodule of a package whose path is the checkout's
`models/` folder), `from utils.FreDF import ...` (a package `utils` pointing at the checkout's `utils/`) and
`from utils.utils import get_metrics` (that module imports packages that are not needed here, so a stub is registered; the model
calls `get_metrics` only when `metrics` contains "r2", which is never passed). `einops` (`rearrange` in `forward`) is a real
dependency.

Model, "paper" preset (read from the file): a temporal layer (zero-pad + conv 1 x 10, 25 filters, no BN or ELU); two graph-
convolution branches (depth 1 and 2, ReLU) on per-time-point channel graphs, each with its own learnable adjacency initialised
from the prior and symmetrised, self-looped and D^-1/2 normalised at every forward; the branch outputs are averaged and added to
the temporal layer's output; a depth-wise (C x 1) conv, BN, ELU, dropout 0.5, average pool 3; a feed-forward stack 25 -> 50 ->
100 -> 200 with kernel 10 and max-pool 3; dropout 0.5; multi-head attention (4 heads) with a residual connection and no
LayerNorm; concatenation along time and a (1, 2 x T // 81) output conv. The adjacency prior is the neighbour table of
`eeg2gait_channel_neighbours.py`. In training mode the model's forward returns its loss; the adapter takes the predictions from
a forward hook on the output conv and computes the loss itself with the model's own loss object (FreDFLossWithEncouraging,
alpha 0.5, beta 0.1, eps 1e-5: MSE plus FFT over the last dimension = the six joints, L1 of the complex difference, each with the
encouraging term). Missing (NaN) target entries: the prediction is replaced by the target there before the loss, so they add no
error to either term (the FFT is linear).

TEAM_RECIPE follows the authors' subject-dependent training script (`sbj_dep_train.py`, gait-decoding task): AdamW from the
model's `configure_optimizers` (weight decay 0.05 on weights of two or more dimensions except the adjacency, betas 0.9 / 0.999),
learning rate 5e-4 with a per-iteration cosine schedule (2 warm-up epochs from 0, minimum 1e-4), 200 epochs, batch 128 of
consecutive windows in recording order (no shuffling, last incomplete batch dropped; a batch may straddle a trial boundary),
autocast bfloat16 where supported (else float16 with a gradient scaler), TF32 on, no gradient clipping, and the epoch with the
best validation R^2 (mean over the six joints, over all validation windows) kept; no early stopping.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import baseline_common as C

TEAM_RECIPE = dict(lr=5e-4, min_lr=1e-4, warmup_epochs=2, epochs=200, batch=128, weight_decay=0.05,
                   betas=(0.9, 0.999), grad_clip=0.0, shuffle=False, drop_last=True, loss_name="freq_reward_mse",
                   preset="paper", select="val_rsq")
_MOD = None


def _team_module():
    global _MOD
    if _MOD is not None:
        return _MOD
    team_dir = C.inp("eeg2gait_team_code")
    model_file = next((p for p in (team_dir / "GaitGraph_model.py", team_dir / "models" / "GaitGraph_model.py") if p.is_file()), None)
    if model_file is None:
        raise FileNotFoundError(f"GaitGraph_model.py not found in {team_dir} or {team_dir / 'models'} (named input eeg2gait_team_code)")
    have = sys.modules.get("utils")
    if have is not None and list(getattr(have, "__path__", [])) != [str(team_dir / "utils")]:
        raise ImportError(f"a different module named 'utils' is already imported: {have}")
    pkg = types.ModuleType("utils")
    pkg.__path__ = [str(team_dir / "utils")]
    stub = types.ModuleType("utils.utils")

    def get_metrics(*a, **k):
        raise RuntimeError("utils.utils.get_metrics is not available here; pass metrics=[]")

    stub.get_metrics = get_metrics
    pkg.utils = stub
    sys.modules["utils"], sys.modules["utils.utils"] = pkg, stub
    tp = types.ModuleType("team_eeg2gait")
    tp.__path__ = [str(team_dir / "models")]
    sys.modules["team_eeg2gait"] = tp
    spec = importlib.util.spec_from_file_location("team_eeg2gait.GaitGraph_model", model_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    _MOD = mod
    return mod


class TeamEEG2GAIT(nn.Module):
    """GaitGraph_tmp (preset "paper") taking (B, C, T) windows and returning (B, n_out) predictions."""

    def __init__(self, n_chan: int, n_time: int, n_out: int, adjacency: np.ndarray):
        super().__init__()
        mod = _team_module()
        cfg = SimpleNamespace(eegnet=SimpleNamespace(
            num_chan_eeg=n_chan, eeg=SimpleNamespace(time_step=n_time), num_chan_kin=n_out,
            preset=TEAM_RECIPE["preset"], loss_name=TEAM_RECIPE["loss_name"],
            adj_matrix=torch.as_tensor(adjacency, dtype=torch.float32)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):                 # the model prints its flags
            self.net = mod.GaitGraph_tmp(cfg, init_adj=False)
        self.build_log = buf.getvalue()
        self.loss_fn = self.net.loss_fn
        self._out = None
        self.net.temporalLastLayer.register_forward_hook(self._grab)

    def _grab(self, module, inp, out):
        self._out = out

    def forward(self, x):
        y0 = torch.zeros(x.shape[0], self.net.nClass, device=x.device, dtype=torch.float32)
        self.net({"EEG": x.unsqueeze(1), "Y": y0}, metrics=[])
        out, self._out = self._out, None
        return out.squeeze(3).squeeze(2)


def team_loss(loss_fn, pred, tgt, mask):
    """The model's loss object on (prediction with masked entries replaced by the target, target)."""
    return loss_fn(torch.where(mask > 0, pred.to(tgt.dtype), tgt), tgt)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Per-iteration learning-rate schedule: linear warm-up from `start_warmup_value`, then cosine decay to `final_value`."""
    warmup_iters = warmup_epochs * niter_per_ep
    warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_epochs > 0 else np.array([])
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    sched = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters))))
                      for i in iters])
    sched = np.concatenate((warmup, sched))
    assert len(sched) == epochs * niter_per_ep
    return sched


def r2_mean(pred, y, mask):
    """R^2 per joint (as sklearn's r2_score, raw values) on the rows whose target is present, averaged over the joints."""
    out = []
    for j in range(y.shape[1]):
        ok = mask[:, j] > 0
        if ok.sum() < 2:
            continue
        yt, yp = y[ok, j].astype(np.float64), pred[ok, j].astype(np.float64)
        ss_tot = ((yt - yt.mean()) ** 2).sum()
        out.append(1.0 - ((yt - yp) ** 2).sum() / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(out)) if out else -np.inf


def amp_context(device: torch.device):
    """Autocast bfloat16 when supported, else float16 (with a gradient scaler); on CPU none."""
    if device.type != "cuda":
        return contextlib.nullcontext, None
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return (lambda: torch.autocast("cuda", dtype=dt)), dt
