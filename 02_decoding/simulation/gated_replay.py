"""Fig. 7g. Replay a session under three commands: gated walking, available phase, and standing.

The timeline CSV carries session, row_type (event or gate_S1), event, instructed,
t_start_s, t_end_s and state. Times are relative to the concatenated session.
The Fig. 7g example is P1 session 9, selected for near-median kinematic decoding
accuracy and agreement between gate decisions and task instructions.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_io as io
from replay_rig import rollout
from rhythm_metrics import LEVEL


def phase_streams(session, dump, template):
    streams = {}
    for trial in io.h5_session_trials(dump, session):
        start, end = io._longest_finite_run(trial["C_pred"])
        if end - start < 2 or start < io.F7_APRED_WARM_N:
            continue
        warm = io.f7_warm_input(trial["A_pred"], start)
        warm, angles = io.f7_from2s_inputs("decoded", dict(W=warm, Y=trial["C_pred"][start:end],
                                                            trial_idx=trial["trial_idx"]))
        phase = io.f7_causal_phases_warm(warm, angles, template, io.f7_f0_prior(session))
        streams[trial["trial_idx"]] = (float(trial["time_sec"][start]) - io.F7_APRED_WARM_N / io.FS,
                                        float(trial["time_sec"][end-1]), phase)
    return streams


def blend_weights(drive):
    progress = 0.0
    weights = np.zeros(len(drive))
    for i, active in enumerate(drive):
        progress = np.clip(progress + (1 if active else -1) / io.FS, 0.0, 1.0)
        weights[i] = 0.5 * (1 - np.cos(np.pi * progress))
    return weights[:, None]


def replay(session, dump, cycles, timeline):
    template = io.loso_template(cycles, session)
    normative = io.fukuchi_template()
    delta, _ = io.global_delta(template, normative)
    stance = io.device_stance_ankle(io.loso_setpoint(cycles, session)[0])
    streams = phase_streams(session, dump, template)
    rows = pd.read_csv(timeline)
    rows = rows[(rows.session == session) & rows.row_type.isin(["event", "gate_S1"])]
    if rows.empty:
        raise ValueError(f"No event or gate rows for {session}")
    duration = rows.t_end_s.max()
    t = np.arange(round(duration * io.FS)) / io.FS
    phases = np.full((len(t), 6), np.nan)
    gate = np.zeros(len(t), bool)
    next_trial = None
    for event in rows[rows.row_type == "event"].sort_values("t_start_s").itertuples():
        active = (t >= event.t_start_s - 1e-9) & (t < event.t_end_s - 1e-9)
        if event.instructed != "IW":
            gate[active] = True
            continue
        length = event.t_end_s - event.t_start_s
        if next_trial is None:
            start_id = int(event.event[2:]) + 1
            next_trial = next(i for i in range(start_id, 0, -1)
                              if i in streams and abs(streams[i][1] - length) <= 0.01)
        if next_trial not in streams or abs(streams[next_trial][1] - length) > 0.01:
            continue
        onset, _, phase = streams[next_trial]
        start = round((event.t_start_s + onset) * io.FS)
        count = min(len(phase), len(t) - start)
        phases[start:start+count] = phase[:count]
        next_trial += 1
    for decision in rows[rows.row_type == "gate_S1"].itertuples():
        active = (t >= decision.t_start_s - 1e-9) & (t < decision.t_end_s - 1e-9)
        gate[active] = decision.state == "open"
    available = np.isfinite(phases[:, 0])
    held_phase = pd.DataFrame(phases).ffill().fillna(0).to_numpy()
    trajectory = io.phases_to_template(held_phase, normative, delta)
    moving = np.radians((stance + (trajectory - normative.mean(axis=1))) * io.SIGN)
    standing = np.radians(stance * io.SIGN)
    grid = np.arange(0, duration, 0.04)
    result = {"t_s": grid}
    for name, active in [("gate", gate & available), ("available", available), ("hold", np.zeros_like(gate))]:
        weight = blend_weights(active)
        target = (1 - weight) * standing + weight * moving
        run = rollout(target, LEVEL, record_stride=10, full_trace=True)
        trace = run["trace_full"]
        hip = trace[:, 1 + 2 * list(run["site_order"]).index("hip")]
        result[f"x_{name}_m"] = np.interp(grid, trace[:, 0], hip - hip[0])
    return pd.DataFrame(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session", "dump", "cycles", "timeline", "output"):
        parser.add_argument(name)
    args = parser.parse_args()
    table = replay(args.session, args.dump, args.cycles, args.timeline)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
