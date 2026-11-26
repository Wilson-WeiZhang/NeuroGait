"""Shared helpers of the analysis scripts: named inputs, output folder, atomic table writing.

Inputs are named, not pathed: `inp(name)` resolves a name under $NEUROGAIT_DATA through the optional name -> relative-path
table in $NEUROGAIT_INPUTS (two tab-separated columns). Derived tables are written to $NEUROGAIT_OUT/derived/.
"""
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("NEUROGAIT_DATA", "."))
OUT = Path(os.environ.get("NEUROGAIT_OUT", "analysis_out")) / "derived"
_NAMES = {}
if os.environ.get("NEUROGAIT_INPUTS"):
    with open(os.environ["NEUROGAIT_INPUTS"], encoding="utf-8") as f:
        _NAMES = {r[0]: r[1] for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")}


def inp(name):
    """Path of a named input."""
    return DATA / _NAMES.get(name, name)


def label(name):
    """Value of a named label."""
    return _NAMES.get(name, name)


def out_path(name):
    """Path of a named derived table under the output folder (folder created)."""
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def write_table(df, name, **kw):
    """Write a DataFrame as CSV through a temporary file."""
    p = out_path(name)
    tmp = p.with_name(p.name + ".tmp")
    df.to_csv(tmp, **kw)
    os.replace(tmp, p)
    return p


def write_npz(name, **arrays):
    """Write arrays as a compressed npz through a temporary file."""
    p = out_path(name)
    tmp = p.with_name(p.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, p)
    return p
