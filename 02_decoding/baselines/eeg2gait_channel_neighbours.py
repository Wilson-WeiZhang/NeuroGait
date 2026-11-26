"""Channel neighbourhood of the EEG2GAIT baseline (Supplementary Table 2).

CHANNEL_NEIGHBORS is the nearest-neighbour table over the 59 recorded electrodes (up, down, left, right) that the EEG2GAIT authors
define in their training script (https://github.com/NgHanWei/EEG2Gait, `sbj_dep_train.py`); it is reproduced here unchanged so that
the adjacency prior handed to the network is theirs. The channel order is the 59-channel whitelist of the preprocessing, which is
the channel order of the aligned recordings. Entries are kept as written: CP6 lists itself and C4 lists CP6, so 21 listed pairs
are one-directional before symmetrisation. Each channel has 3 to 5 neighbours (mean 3.80).

team_adjacency  the adjacency the training script hands the network: A[i, j] = A[j, i] = 1 for every listed pair, diagonal 1.
"""
from __future__ import annotations

import numpy as np

CHANNEL_NEIGHBORS = {
    # order: up-down-left-right
    "Fp1": ["AF7", "AF3", "AFz"],
    "Fp2": ["AF4", "AF8", "AFz"],
    "AF7": ["Fp1", "AF3", "F7"],
    "AF3": ["Fp1", "F3", "AF7", "AFz"],
    "AFz": ["Fp1", 'Fp2', "Fz", "AF3", "AF4"],
    "AF4": ["Fp2", "F4", "AFz", "AF8"],
    "AF8": ["Fp2", "AF4", "F8"],
    "F7": ["AF7", "FT7", "F5"],
    "F5": ["AF7", "AF3", "FC5", "F7", "F3"],
    "F3": ["AF3", "FC3", "F5", "F1"],
    "F1": ["AF3", "AFz", "FC1", "F3", "Fz"],
    "Fz": ["AFz", "F1", "F2"],
    "F2": ["AFz", "AF4", "FC2", "Fz", "F4"],
    "F4": ["AF4", "FC4", "F2", "F6"],
    "F6": ["AF4", "AF8", "FC6", "F4", "F8"],
    "F8": ["AF8", "FT8", "F6"],
    "FT7": ["F7", "T7", "FC5"],
    "FC5": ["F5", "C5", "FT7", "FC3"],
    "FC3": ["F3", "C3", "FC5", "FC1"],
    "FC1": ["F1", "C1", "FC3"],
    "FC2": ["F2", "C2", "FC4"],
    "FC4": ["F4", "C4", "FC2", "FC6"],
    "FC6": ["F6", "C6", "FC4", "FT8"],
    "FT8": ["F8", "T8", "FC6"],
    "T7": ["FT7", "TP7", "C5"],
    "C5": ["FC5", "CP5", "T7", "C3"],
    "C3": ["FC3", "CP3", "C5", "C1"],
    "C1": ["FC1", "CP1", "C3", "Cz"],
    "Cz": ["CPz", "C1", "C2"],
    "C2": ["FC2", "CP2", "Cz", "C4"],
    "C4": ["FC4", "CP4", "C2", "CP6"],
    "C6": ["FC6", "CP6", "C4", "T8"],
    "T8": ["FT8", "TP8", "C6"],
    "TP7": ["T7", "P7", "CP5"],
    "CP5": ["C5", "P5", "TP7", "CP3"],
    "CP3": ["C3", "P3", "CP5", "CP1"],
    "CP1": ["C1", "P1", "CP3", "CPz"],
    "CPz": ["Cz", "Pz", "CP1", "CP2"],
    "CP2": ["C2", "P2", "CPz", "CP4"],
    "CP4": ["C4", "P4", "CP2", "CP6"],
    "CP6": ["C6", "P6", "CP6", "TP8"],
    "TP8": ["T8", "P8", "CP6"],
    "P7": ["TP7", "PO7", "P5"],
    "P5": ["CP5", "PO7", "PO3", "P7", "P3"],
    "P3": ["CP3", "PO3", "P5", "P1"],
    "P1": ["CP1", "PO3", "POz", "P3", "Pz"],
    "Pz": ["CPz", "POz", "P1", "P2"],
    "P2": ["CP2", "POz", "PO4", "Pz", "P4"],
    "P4": ["CP4", "PO4", "P2", "P6"],
    "P6": ["CP6", "PO4", "PO8", "P4", "P8"],
    "P8": ["TP8", "PO8", "P6"],
    "PO7": ["P7", "O1", "PO3"],
    "PO3": ["P3", "O1", "Oz", "PO7", "POz"],
    "POz": ["Pz", "Oz", "PO3", "PO4"],
    "PO4": ["P4", "Oz", "O2", "POz", "PO8"],
    "PO8": ["P8", "O2", "PO4"],
    "O1": ["PO7", "PO3", "Oz"],
    "Oz": ["POz", "O1", "O2"],
    "O2": ["PO8", "PO4", "Oz"],
}


def _check(labels):
    if set(labels) != set(CHANNEL_NEIGHBORS) or len(labels) != len(CHANNEL_NEIGHBORS):
        raise ValueError(f"channels differ from the neighbour table: missing "
                         f"{sorted(set(CHANNEL_NEIGHBORS) - set(labels))}, extra {sorted(set(labels) - set(CHANNEL_NEIGHBORS))}")
    return {lab: k for k, lab in enumerate(labels)}


def team_adjacency(labels: list[str]) -> np.ndarray:
    """sbj_dep_train.py lines 762-773 in the channel order `labels`: symmetric 0/1 neighbour matrix, diagonal 1."""
    ix = _check(labels)
    a = np.zeros((len(labels), len(labels)), np.float32)
    for ch, nbs in CHANNEL_NEIGHBORS.items():
        for nb in nbs:
            a[ix[ch], ix[nb]] = 1.0
            a[ix[nb], ix[ch]] = 1.0
    np.fill_diagonal(a, 1.0)
    return a


def _check(labels):
    if set(labels) != set(CHANNEL_NEIGHBORS) or len(labels) != len(CHANNEL_NEIGHBORS):
        raise ValueError(f"channels differ from the neighbour table: missing "
                         f"{sorted(set(CHANNEL_NEIGHBORS) - set(labels))}, extra {sorted(set(labels) - set(CHANNEL_NEIGHBORS))}")
    return {lab: k for k, lab in enumerate(labels)}


def team_adjacency(labels: list[str]) -> np.ndarray:
    """Symmetric 0/1 neighbour matrix with diagonal 1, in the channel order `labels`."""
    ix = _check(labels)
    a = np.zeros((len(labels), len(labels)), np.float32)
    for ch, nbs in CHANNEL_NEIGHBORS.items():
        for nb in nbs:
            a[ix[ch], ix[nb]] = 1.0
            a[ix[nb], ix[ch]] = 1.0
    np.fill_diagonal(a, 1.0)
    return a
