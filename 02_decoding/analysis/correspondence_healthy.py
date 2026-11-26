"""Joint correspondence matrices of reference and decoded trajectories, healthy imagined walking and resting input.

Element (i, j) of a session matrix is the correlation between reference joint i and joint j of the second series over the
session's concatenated held-out trials: reference joint j (reference matrix) or decoded joint j (decoded matrices). The
resting-input run decodes, with the same trained folds, one contiguous window of the session's own resting EEG in place
of each test trial. Matrices are the mean over sessions (NaN ignored), with display polarity applied: the right hip and
the left knee and ankle goniometers are mounted mirror-image, so their rows and columns are multiplied by -1 (this
changes no same-joint correlation).

Named inputs: correspondence_healthy_imagined.npz, correspondence_healthy_rest_input.npz (sessions, obs_mat, cross_mat:
session x 6 x 6).
Named output (derived/): correspondence_healthy.csv (matrix x row joint x column joint, r).
"""
import csv
import os

import numpy as np

from analysis_common import inp, out_path

LAB = ["LHip", "LKne", "LAnk", "RHip", "RKne", "RAnk"]
POLARITY = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])   # LKne, LAnk, RHip flipped for display
S = np.outer(POLARITY, POLARITY)


def main():
    zi = np.load(inp("correspondence_healthy_imagined.npz"), allow_pickle=True)
    zr = np.load(inp("correspondence_healthy_rest_input.npz"), allow_pickle=True)
    assert [str(s) for s in zi["sessions"]] == [str(s) for s in zr["sessions"]]
    mats = (("reference", np.nanmean(zi["obs_mat"], axis=0) * S, "reference"),
            ("iw_decoded", np.nanmean(zi["cross_mat"], axis=0) * S, "decoded"),
            ("rest_decoded", np.nanmean(zr["cross_mat"], axis=0) * S, "decoded"))
    p = out_path("correspondence_healthy.csv")
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "row_reference_joint", "col_joint", "r"])
        for name, M, collab in mats:
            for i in range(6):
                for j in range(6):
                    w.writerow([name, LAB[i], collab + ":" + LAB[j], f"{M[i, j]:.6f}"])
    os.replace(tmp, p)
    print("wrote", p)


if __name__ == "__main__":
    main()
