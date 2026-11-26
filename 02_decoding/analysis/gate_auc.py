"""Fig. 7f. Mean session AUC from held-out post-calibration event scores.

gate_scores.csv is the scored window stream from gate_scored_events.py.
Tiling selects windows ending at 3, 6, ... seconds after event onset.
The bootstrap samples whole events within session and class, then averages the session AUCs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from analysis_common import inp, write_table


def build():
    data = pd.read_csv(inp("gate_scores.csv"))
    if data.groupby(["session", "event"]).fold.nunique().ne(1).any():
        raise ValueError("A held-out event appears in multiple folds")
    rng = np.random.default_rng(0)
    samples, rows = [], []
    for session, group in data.groupby("session", sort=True):
        windows = group[group.t_end_ev_s % 3 == 0]
        auc = roc_auc_score(windows.label, windows.fuse_V1)
        pools = [[event.loc[event.t_end_ev_s % 3 == 0, "fuse_V1"].to_numpy()
                  for _, event in group[group.label == label].groupby("event", sort=False)]
                 for label in (0, 1)]
        draws = np.empty(1000)
        for b in range(len(draws)):
            scores = [np.concatenate([events[i] for i in rng.integers(len(events), size=len(events))])
                      for events in pools]
            labels = np.repeat([0, 1], [len(values) for values in scores])
            draws[b] = roc_auc_score(labels, np.concatenate(scores))
        samples.append(draws)
        rows.append(dict(session=session, n_events=group.event.nunique(), auc=auc,
                         ci_low=np.percentile(draws, 2.5), ci_high=np.percentile(draws, 97.5)))
    means = np.mean(samples, axis=0)
    rows.append(dict(session="Mean", n_events=sum(row["n_events"] for row in rows),
                     auc=np.mean([row["auc"] for row in rows]), ci_low=np.percentile(means, 2.5),
                     ci_high=np.percentile(means, 97.5)))
    write_table(pd.DataFrame(rows), "gate_auc.csv", index=False)


if __name__ == "__main__":
    build()
