"""Fig. 6. Median cortical maps from per-session source-space vertex z scores.

source_vertex_scores.npz contains z_IW_k1, z_RW_k1, z_REST_k1, z_SCI_k1
and the corresponding k2 arrays (sessions by vertices), ordered patient session
labels in sci and healthy session labels in sessions.
"""
import numpy as np

from analysis_common import inp, write_npz


def build():
    cube = np.load(inp("source_vertex_scores.npz"), allow_pickle=True)
    ids = np.array([str(s).split("_")[0] for s in cube["sci"]])
    if set(ids) != {"P1", "P2"}:
        raise ValueError("Expected P1 and P2 in the patient session labels")
    maps = {}
    for k in (1, 2):
        for group in ("IW", "RW", "REST", "SCI"):
            z = cube[f"z_{group}_k{k}"].astype(float)
            centered = z - z.mean(axis=1, keepdims=True)
            maps[f"{group}_k{k}"] = np.median(centered, axis=0).astype(np.float32)
            if group == "SCI":
                for patient in ("P1", "P2"):
                    maps[f"{patient}_k{k}"] = np.median(centered[ids == patient], axis=0).astype(np.float32)
    write_npz("group_cortical_maps.npz", **maps)


if __name__ == "__main__":
    build()
