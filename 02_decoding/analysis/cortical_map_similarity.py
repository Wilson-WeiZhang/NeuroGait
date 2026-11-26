"""Fig. 6. Correlations over Destrieux parcels, with mirrored hemispheric spins.

Inputs are group_cortical_maps.npz (IW, RW, SCI, P1, P2 vertex maps at k1 and k2)
and source_parcel_anatomy.npz (parcel labels, vertex membership and spherical coordinates).
The upper-tail test uses 5,000 rotations, with BH correction over eight comparisons per harmonic.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests

from analysis_common import inp, write_table

SEED = 20261105


def build():
    Z = np.load(inp("group_cortical_maps.npz"), allow_pickle=True)
    A = np.load(inp("source_parcel_anatomy.npz"), allow_pickle=True)
    labels = [str(x) for x in A["labels"]]
    roi = [A[f"roi_{i}"].astype(int) for i in range(len(labels))]
    sphere = np.asarray(A["sphere"], float)
    side = np.array([0 if lab.endswith(" L") else 1 for lab in labels])
    cen = np.array([sphere[r].mean(axis=0) for r in roi])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    rng = np.random.default_rng(SEED)
    MIR = np.diag([-1.0, 1.0, 1.0])
    tr = [cKDTree(cen[side == i]) for i in (0, 1)]
    idx = [np.flatnonzero(side == i) for i in (0, 1)]
    B = 5000
    Ix = np.empty((B, len(labels)), int)
    for b in range(B):
        q, t = np.linalg.qr(rng.standard_normal((3, 3)))
        q = q @ np.diag(np.sign(np.diag(t)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        for i in (0, 1):
            R_ = q if i == 0 else MIR @ q @ MIR
            _, nn = tr[i].query(cen[side == i] @ R_.T)
            Ix[b, idx[i]] = idx[i][nn]
    G = ("IW", "RW", "SCI", "P1", "P2")
    NAME = dict(IW="Healthy imagined", RW="Healthy real", SCI="SCI imagined", P1="P1 imagined", P2="P2 imagined")
    rows = []
    for k in (1, 2):
        M = {g: np.array([np.nanmean(Z[f"{g}_k{k}"].astype(float)[r]) for r in roi]) for g in G}
        part = []
        for i, a in enumerate(G):
            for b_ in G[i + 1:]:
                if {a, b_} in ({"SCI", "P1"}, {"SCI", "P2"}):
                    continue
                r = np.corrcoef(M[a], M[b_])[0, 1]
                rotated = M[a][Ix]
                rotated -= rotated.mean(axis=1, keepdims=True)
                target = M[b_] - M[b_].mean()
                nul = (rotated @ target) / (np.linalg.norm(rotated, axis=1) * np.linalg.norm(target))
                part.append(dict(harmonic_k=k, a=NAME[a], b=NAME[b_], r=r, null_mean=nul.mean(), null_p95=np.quantile(nul, .95), p_spin=(np.sum(nul >= r) + 1) / (B + 1), n_parcels=len(labels)))
        q = multipletests([x["p_spin"] for x in part], method="fdr_bh")[1]
        for x, y in zip(part, q):
            x["pFDR"] = y
        rows += part
    write_table(pd.DataFrame(rows), "cortical_map_similarity.csv", index=False)


if __name__ == "__main__":
    build()
