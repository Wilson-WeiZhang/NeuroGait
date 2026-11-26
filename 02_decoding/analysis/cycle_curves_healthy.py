"""Cycle-averaged reference and decoded joint trajectories and inter-joint cyclograms, healthy imagined walking.

Input: one cycle-averaged curve per session, joint and series (reference goniometer angle, decoded angle), 50 gait-cycle
bins, z-scored per cycle before averaging (built from the scored trials of each session).

Cycle curves (left hip, knee, ankle): sessions with a complete curve are kept per joint; the table gives the mean over
sessions and the 95 % band mean +- 1.96 SD / sqrt(n).

Cyclograms (hip-knee, hip-ankle, knee-ankle): sessions with complete curves for both joints of the pair in both series
are kept. Per series and joint the session-mean curve is z-scored over the cycle; the 95 % half-width per axis is
1.96 SD / sqrt(n) over sessions in the same z units. The band drawn round the closed loop takes the half-width projected
onto the local loop normal (periodic tangent), caps it at 2.5 times its median and smooths it with a 3-bin periodic
moving average; the table keeps the outer and inner edge points.

Named input: cycle_average_healthy.npz (obs_curves, dec_curves: session x 6 joints x 50 bins; joints 0-2 = left hip,
knee, ankle).
Named outputs (derived/): cycle_curves_healthy.csv (joint x series x bin); cyclogram_trajectories_healthy.csv
(pair x series x bin).
Usage: python cycle_curves_healthy.py [curves|cyclograms ...]   (default: both)
"""
import csv
import os
import sys

import numpy as np

from analysis_common import inp, out_path

NB = 50
NM = {0: "LHip", 1: "LKnee", 2: "LAnk"}
PAIRS = [(0, 1), (0, 2), (1, 2)]


def write_csv(name, header, rows):
    p = out_path(name)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, p)
    print("wrote", p, len(rows), "rows")


def cycle_curves(obs, dec):
    x = np.linspace(0, 100, NB)
    rows = []
    for j in range(3):
        for series, arr in (("reference", obs), ("decoded", dec)):
            a = arr[:, j, :]
            a = a[np.isfinite(a).all(1)]
            m, ci = a.mean(0), 1.96 * a.std(0) / np.sqrt(len(a))
            rows += [[NM[j], series, float(xx), float(mv), float(mv - cv), float(mv + cv)] for xx, mv, cv in zip(x, m, ci)]
    write_csv("cycle_curves_healthy.csv", ["joint", "series", "gait_cycle_pct", "mean", "ci_lower", "ci_upper"], rows)


def tube(mx, my, cx, cy, smooth=3):
    """Outer and inner edges of the 95 % band round a closed loop (periodic tangent, capped, smoothed half-width)."""
    tx = np.roll(mx, -1) - np.roll(mx, 1)
    ty = np.roll(my, -1) - np.roll(my, 1)
    tn = np.hypot(tx, ty) + 1e-9
    nx, ny = -ty / tn, tx / tn
    hw = np.hypot(cx * nx, cy * ny)
    hw = np.minimum(hw, 2.5 * np.median(hw))
    k = np.ones(smooth) / smooth
    hw = np.convolve(np.r_[hw[-smooth:], hw, hw[:smooth]], k, "same")[smooth:-smooth]
    outer = np.column_stack([mx + hw * nx, my + hw * ny])
    inner = np.column_stack([mx - hw * nx, my - hw * ny])
    return outer, inner


def zloop(A, B, n):
    """Session-mean curves of two joints z-scored over the cycle, and their 95 % half-widths in z units."""
    ma, mb = A.mean(0), B.mean(0)
    sa, sb = ma.std() + 1e-9, mb.std() + 1e-9
    return ((ma - ma.mean()) / sa, (mb - mb.mean()) / sb,
            1.96 * A.std(0) / np.sqrt(n) / sa, 1.96 * B.std(0) / np.sqrt(n) / sb)


def cyclograms(obs, dec):
    phase = np.linspace(0, 100, NB)
    rows = []
    for a, b in PAIRS:
        v = (np.isfinite(obs[:, a, :]).all(1) & np.isfinite(obs[:, b, :]).all(1)
             & np.isfinite(dec[:, a, :]).all(1) & np.isfinite(dec[:, b, :]).all(1))
        n = int(v.sum())
        for series, arr in (("reference", obs), ("decoded", dec)):
            mx, my, cx, cy = zloop(arr[v, a], arr[v, b], n)
            outer, inner = tube(mx, my, cx, cy)
            rows += [(NM[a], NM[b], n, series, i, phase[i], mx[i], my[i],
                      outer[i, 0], outer[i, 1], inner[i, 0], inner[i, 1]) for i in range(NB)]
    write_csv("cyclogram_trajectories_healthy.csv",
              ["x_joint", "y_joint", "n_paired_sessions", "series", "cycle_bin", "gait_cycle_pct", "x", "y",
               "ci_outer_x", "ci_outer_y", "ci_inner_x", "ci_inner_y"], rows)


def main():
    parts = sys.argv[1:] or ["curves", "cyclograms"]
    z = np.load(inp("cycle_average_healthy.npz"), allow_pickle=True)
    obs, dec = z["obs_curves"], z["dec_curves"]
    if "curves" in parts:
        cycle_curves(obs, dec)
    if "cyclograms" in parts:
        cyclograms(obs, dec)


if __name__ == "__main__":
    main()
