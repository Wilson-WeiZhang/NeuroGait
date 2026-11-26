# NeuroGait

This repository contains the analysis code for the research paper:

**Cortical phase dynamics encode continuous locomotor kinematics during imagined walking and persist after spinal cord injury**

Wei Zhang<sup>1</sup>, Aung Aung Phyo Wai<sup>1</sup>, Neethu Robinson<sup>1</sup>, Rui Liu<sup>1</sup>, Han Wei Ng<sup>1</sup>, Seng Kwee Wee<sup>2,3,4,5</sup>, Min Wee Phua<sup>2,3</sup>, Wen Li Lui<sup>2,3</sup>, Ananda Sidarta<sup>6</sup>, Lek Syn Lim<sup>6</sup>, Karen Sui Geok Chua<sup>2,3,5</sup>, Kai Rui Wan<sup>7,8</sup>, Zhi Yan Valerie Ng<sup>2,3</sup>, Cuntai Guan<sup>1,9,\*</sup>

<sup>1</sup> College of Computing and Data Science, Nanyang Technological University, Singapore
<sup>2</sup> Department of Rehabilitation Medicine, Tan Tock Seng Hospital, Singapore
<sup>3</sup> Institute of Rehabilitation Excellence, Tan Tock Seng Hospital Rehabilitation Centre, NHG Health, Singapore
<sup>4</sup> Singapore Institute of Technology, Singapore
<sup>5</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore
<sup>6</sup> R3 Institute Singapore, Nanyang Technological University, Singapore
<sup>7</sup> Department of Neurosurgery, National Neuroscience Institute, Tan Tock Seng Hospital, Singapore
<sup>8</sup> Department of Neurosurgery, National Neuroscience Institute, Singapore General Hospital, Singapore
<sup>9</sup> Center of AI in Medicine, Nanyang Technological University, Singapore


## Overview

NeuroGait is the analysis code for a study of cortical phase dynamics in scalp EEG recorded
during imagined walking. Participants imagined walking in time with an observed reference
walker while remaining still, so the reconstruction cannot draw on the participant's own
limb movement. The repository documents the analysis methods: EEG-goniometer alignment and
preprocessing (`01_preprocessing/`), the decoder of continuous lower-limb kinematics, the
statistical analyses that read its output, the deep-network baselines, and the physics
simulation and walk-intent gate in two individuals with spinal cord injury (`02_decoding/`).

The table below connects the main Results to their inputs, entry points and output measures.

## Citation

If you use this code in your research, please cite our paper:

```
Zhang, W., Phyo Wai, A. A., Robinson, N., Liu, R., Ng, H. W., Wee, S. K., Phua, M. W.,
Lui, W. L., Sidarta, A., Lim, L. S., Chua, K. S. G., Wan, K. R., Ng, Z. Y. V., & Guan, C.
Cortical phase dynamics encode continuous locomotor kinematics during imagined walking and persist after spinal cord injury.
```

## Data Availability

Access to the raw EEG and goniometer recordings is governed by data-use agreements
or sponsor restrictions, and they are therefore not publicly available. Requests for
access should be directed to the contacts listed below and will be reviewed against
the governing agreements and the ethics approval.

The processed values behind every figure and table are provided as Source Data files
with the paper.

## Repository Structure

```
01_preprocessing/    healthy/, patient/               EEG-goniometer alignment, goniometer extraction,
                                                      EEG preprocessing and export (MATLAB)
                     causal/                          the causal preprocessing used for the decoder (MATLAB)
02_decoding/         core/, shared/, readout/         the decoder: harmonic bands, features and shared basis (MATLAB)
                     fold_tables/                     the decoder: per-fold harmonic band tables (Python)
                     pyport/                          the decoder: readout (Python)
                     baselines/                       EEGNet-8,2, DeepConvNet and EEG2GAIT trained per session
                                                      on the same trials (Python, PyTorch)
                     analysis/                        statistical analyses (Python, MATLAB)
                     simulation/                      physics simulation and walk-intent gate (Python);
                                                      analysis_config.py holds their shared settings
```

## Analysis Pipeline

Preprocessing (`01_preprocessing/`, MATLAB)

**Step 01**: Align EEG markers to goniometer trigger edges by inter-onset interval
matching (`prep_align.m`, `align_goni_to_eeg.m`)  
**Step 02**: Extract goniometer joint angles and flag trials whose length, amplitude or
waveform shape deviates from the other trials of the session (`extract_goni.m`)  
**Step 03**: Preprocess EEG - band-pass filter, bad-channel interpolation, common
average reference, artefact subspace reconstruction, ICA, ICLabel rejection
(`prep_step2.m`, `prep_step3.m`)  
**Step 04**: Export the aligned EEG, goniometer and event triple per session
(`export_session.m`)

The decoding results use the strictly causal variant of Steps 03-04
(`01_preprocessing/causal/`, one session per call of `healthy/cpc_session.m`). It runs the
stages above with three changes: the 1000 to 250 Hz anti-aliasing low-pass is the same
filter converted to minimum phase (`cpc_step1.m`); bad channels, ICA and ICLabel are fitted
on a calibration copy of the beginning of the recording and applied forward to the whole
recording (`cpc_step2_cal.m`, `cpc_prep_step2.m`); and the 0.1-40 Hz band-pass is the same
FIR converted to minimum phase and applied forward only (`cp_step3.m`, `cpc_step3.m`).
The patient sessions follow the same steps (`01_preprocessing/patient/`). The neural analyses read the zero-phase
preprocessing of Steps 03-04.

Decoding (`02_decoding/`, MATLAB and Python)

**Step 05**: Estimate the gait-frequency harmonic bands from the goniometer spectra
(`core/estimate_goniometer_psd_peaks.m`); for each cross-validation fold the gait frequency
and the band centres are estimated from that fold's training trials only
(`fold_tables/fold_centres.py`, `fold_tables/make_band_tables.py`)  
**Step 06**: Compute the causal features (`core/run_decoder.m`): at each gait harmonic, a
fixed-frequency complex demodulation (first-order low-pass at 0.18 Hz, applied forward
only)  
**Step 07**: Fit the shared principal-component basis on the features of the other
participants (`core/compute_shared_delta_pca_lso_basis.m`)  
**Step 08**: Fit and test the readout (`pyport/run_session.py`, `pyport/decoder.py`, with
`pyport/configs/tenfold_pca30_lpf6.json`): projection onto the leading 30 components of
the shared basis, lagged ridge readout, causal second-order Butterworth low-pass at 6 Hz
on that first-stage output, causal finite impulse response correction fitted on the
low-passed predictions, random 10-fold cross-validation. Every trial trains the decoder;
accuracy is reported on the trials recorded after the session's calibration recording.

Deep baselines (`02_decoding/baselines/`, Python): `train_baseline.py` trains one network
per session on the same trials, folds and targets. EEG2GAIT is the authors' public
implementation (https://github.com/NgHanWei/EEG2Gait), which is not redistributed here.

Statistical analyses (`02_decoding/analysis/`)

| Script | Content |
|---|---|
| `harmonic_phase_maps.m` | gait-cycle harmonic phase maps of the EEG |
| `state_locking.py` | gait-phase locking of the scalp maps in real walking, imagined walking, observation only and rest |
| `decoding_session_medians.py` | per-session decoding accuracy of imagined and real walking, with the paired comparison |
| `harmonic_agreement.py` | per-harmonic phase agreement between decoded and reference joint trajectories |
| `lag_tuning.py` | lag tuning and timing error of the decoded joint trajectories |
| `accuracy_equation.py` | decoding accuracy predicted from harmonic composition and decoded phase agreement |
| `locking_accuracy_stats.py` | session-level statistics linking phase agreement and cortical locking to decoding accuracy |
| `speed_tuning.py` | speed tuning of gait-phase locking |
| `lead_leg_traces.py` | hip angle and the EEG harmonic components by the walker's leading leg |
| `rsa_spin.py` | spatial representational similarity between real and imagined walking, with a spin null |
| `readout_phase.py` | cross-state phase read-out on single trials of imagined walking and rest |
| `phase_frequency_maps.py` | gait-phase by frequency maps of induced power and their cluster tests |
| `mediation.py` | statistical mediation between self-reported imagery control, locking and decoding accuracy |
| `observation_locking.py`, `observation_group_tests.py` | imagined walking against observation only |

Simulation (`02_decoding/simulation/`, Python)

The rollouts drive a MuJoCo biped (`biped_model.py`, `replay_rig.py`, `run_rollouts.py`)
from 2 s into each trial with the gait phase that a causal extractor reads from the
condition's own input: for the decoded condition, the decoder output; for the reference
walker, its goniometer. Outcomes are measured from the onset of the final decoder output
(`rhythm_metrics.py`). The walk-intent gate (`gate_features.py`, `gate.py`) reads the
preprocessed patient EEG.

## Reproducing the main analyses

Figure 1 describes the experimental design. The scripts below are in `02_decoding/analysis/` unless another directory is specified. Start at the row for the research question or run preprocessing and decoding first if you have access to the recordings.

| Results | Starting inputs | Entry points | Output to check |
|---|---|---|---|
| Fig. 2 · Continuous kinematics | Held-out predicted and observed joint angles, cycle averages, trial correlations and calibration mask | `trial_r_tables.py`, `decoding_session_medians.py`, `cycle_curves_healthy.py`, `correspondence_healthy.py`, `patient_reconstruction.py` | One median per session, six-joint correspondence, cycle profiles. Across 80 healthy sessions the medians are 0.465586 during imagined walking and 0.883215 during real walking. |
| Fig. 3 · Timing at stride and step frequencies | Aligned EEG and hip angle, session bands, held-out joint trajectories | `harmonic_phase_maps.m`, `harmonic_agreement.py`, `lag_tuning.py`, `accuracy_equation.py`, `lead_leg_traces.py` | Complex maps at nine gait-cycle positions, decoded phase agreement and phase-error measures, observed and predicted accuracy. |
| Fig. 4 · Walking, observation and rest | Per-session complex phase maps, REST windows, observation-only and imagined-walking trial or fold tables | `state_locking.py`, `rsa_spin.py`, `observation_locking.py`, `observation_group_tests.py`, `observation_transfer.py` | State-specific locking, timing and spatial similarity, rotation-null probabilities, within- and cross-condition decoding. |
| Fig. 5 · Individual variation | Session locking and decoding table, questionnaire responses, participant and session identifiers | `session_master_table.py`, `crosssession_predictor.py`, `locking_accuracy_stats.py`, `speed_tuning.py`, `selfreport_forest.py`, `mediation.py` | Between-session prediction, partial correlations, speed tuning and mediation paths. The indirect standardized effect is 0.362 across 76 sessions. |
| Fig. 6 · Spinal cord injury | Healthy and post-operative complex phase maps, source-space inverse kernel and anatomy, patient accuracy and phase offsets | `patient_spatial_rsa.py`, `src_zmap_session.m`, `cortical_group_maps.py`, `cortical_map_similarity.py`, `patient_case_statistics.py` | Cross-validated electrode distances and spin tests for each patient, session-bootstrap intervals, cortical correlations over 148 parcels, patient-adjusted associations and single-case timing tests. |
| Fig. 7 · Simulated walking and intent | Held-out six-joint predictions, normative joint angles, cadence priors, causal EEG and event labels | In `simulation/`, `run_rollouts.py`, `replay_walker_matched.py`, `build_gate_session_set_causal.py`, `gate_scores_causal.py`, `gate_scored_events.py`. Then `rollout_statistics.py` and `gate_auc.py`. | Trial-matched displacement, cadence and imbalance for four conditions. Gate AUC is calculated per session from 3 s windows after calibration and averaged over 11 sessions. The single-session walk–stand replay is implemented in `simulation/gated_replay.py`. |

### Running an analysis

From the repository root, create a Python environment and install the dependencies. The deep-network baselines additionally require PyTorch appropriate to your CPU or CUDA installation. For EEG2GAIT, provide the upstream implementation as the named input `eeg2gait_team_code`.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export GAIT_PROJECT_ROOT="$PWD"
export NEUROGAIT_DATA=/path/to/study-inputs
export NEUROGAIT_OUT="$PWD/analysis_out"
python 02_decoding/analysis/decoding_session_medians.py
```

For this example, place `iw_trial_r_long.csv` and `rw_trial_r_long.csv` under `NEUROGAIT_DATA`. Both tables contain `session`, `trial_idx`, `joint`, `model` and `rpt_r` for trials after calibration. The output is `analysis_out/derived/decoding_session_medians.csv`. To use a different input layout, set `NEUROGAIT_INPUTS` to a two-column, tab-separated file without a header. Each row gives the script's input name and the corresponding relative path under `NEUROGAIT_DATA`. The same table can supply column labels requested by individual analyses. Intermediate tables written to `derived/` can be mapped as inputs to downstream scripts.

To run one decoder fold, build a shared basis using `compute_shared_delta_pca_lso_basis.m` with `leave_subject_out=true` for healthy participants or a pooled healthy basis for a patient. Use correlation aggregation, sample stride 1 and the causal demodulation mode `cdm_o1_c0p18_causal` for harmonics 1–3. Set `GAIT_TRIPLE_ROOT` and call `export_session_features(session, condition, manifest, bands, basis, destination)` in MATLAB with the selected fold’s training-trial band table. Supply the resulting export and the same basis to Python. From `02_decoding/` run:

```sh
python -m pyport.run_session /path/to/session_export.mat /path/to/basis.mat \
  pyport/configs/tenfold_pca30_lpf6.json /path/to/decoder-output --folds 1 --dump
```

For MATLAB, add the relevant cohort's preprocessing directories, EEGLAB and `02_decoding/shared/` to the path. The two cohorts have functions with the same name, so load one cohort's paths at a time. Supply the acquisition-site registry for raw patient sessions. The registry, recordings, intermediate feature exports, MRI template files, calibration plans and per-session model inputs require separate access. The simulation expects the matched six-joint templates, a session cadence table (`f0_cadence_prior.csv` beside the simulation scripts, or set `GAIT_CADENCE_PRIOR`) and a 50-point normative trajectory file with `hip_deg`, `knee_deg` and `ankle_deg` columns (`GAIT_NORMATIVE_CSV`). For the gate, provide the causal patient exports, labelled EEG events and per-session calibration cuts. For the continuous replay, run `python 02_decoding/simulation/gated_replay.py P1_sess09 DUMP_DIR CYCLES.npz TIMELINE.csv OUTPUT.csv`. The timeline contains the concatenated events and the gate decisions. The example is P1 session 9. Its session-median decoding correlation was 0.50 (post-operative median 0.54); the gate was open at 181 of 196 imagined-walking decisions and closed at 73 of 87 rest decisions. The modelling scripts use `P1_sess04`-style patient session identifiers consistently across these files.

## Requirements

- MATLAB R2024b (https://www.mathworks.com)
- EEGLAB 2024.2 (https://eeglab.org/), with `clean_rawdata`, `firfilt` and `ICLabel`
- Python 3.13.7, with the packages pinned in `requirements.txt`

The cohort: 80 healthy sessions from 42 participants, and two participants with
spinal cord injury contributing 15 retained sessions (4 pre-operative and 11
post-operative).


## Acknowledgements

We thank the healthy volunteers and the participants with spinal cord injury for
their time and engagement across the recording sessions.

## License

MIT License

Copyright (c) 2026 Wei Zhang, Cuntai Guan and co-authors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files, to deal in the Software without
restriction, subject to the conditions in `LICENSE`.

## Contact

For any questions regarding this code repository, please contact:

PI: Prof. Cuntai Guan  
College of Computing and Data Science,  
Nanyang Technological University, Singapore  
Email: ctguan@ntu.edu.sg


or the author: Dr. Wei Zhang  
College of Computing and Data Science,  
Nanyang Technological University, Singapore  
Email: wilson.zhangwei@ntu.edu.sg

