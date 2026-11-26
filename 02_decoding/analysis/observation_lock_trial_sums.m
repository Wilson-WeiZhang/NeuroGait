function observation_lock_trial_sums(key, varargin)
%OBSERVATION_LOCK_TRIAL_SUMS  Per-trial sums of the gait-phase locking |c| of one supplementary participant.
%
%   observation_lock_trial_sums('sub09')                 % one of sub09 sub01 sub12 sub23 sub29 sub35
%   observation_lock_trial_sums('sub09', 'max_threads', 8)
%
% The locking statistic is the one of the main analysis (Fig. 3): the walker's left-hip goniometer is band-passed
% around k1 (2nd-order Butterworth, zero phase) and its Hilbert phase is the gait phase; the EEG is band-passed around
% harmonic k (4th-order Butterworth, zero phase) and reduced to unit phase vectors; for each of 9 points of the cycle
% (half width pi / 8) the unit vectors of the samples whose gait phase falls within the window are summed per channel,
% as are the vectors of all valid samples. The map of a trial set is mean_point - mean_all; |c| = mean |map| over
% channels x points. Because each mean is a plain sum over trials, this script stores the per-trial contributions
% (sum_point, count_point_ch, sum_all, count_all_ch), so that |c| of any subset of trials (a relabelled cycle set,
% a bootstrap sample) is recombined exactly by observation_locking.py.
%
% Trials: the imagine trials of the two condition sessions (MI+AO first, then AO, each in file order; trial_filter
% 'MI'). Every trial uses the pooled band table of the session (one band per harmonic for both conditions). EEG and
% goniometer are read once from the full-session export. Cycle number and event index come from the event-condition
% table of the session (matched by EEG start sample).
%
% Named inputs (resolved through $NEUROGAIT_INPUTS under $NEUROGAIT_DATA): supplementary_exports (folder of the
% session exports, read by load_triple_mat), observation_bands_<key>.csv, observation_event_conditions_<key>.csv.
% Needs load_triple_mat.m and trial_filter.m of the decoder on the path.
% Named output: $NEUROGAIT_OUT/derived/observation_trial_sums_<key>.mat (v7.3), arrays [n_trial x n_ch x 9 x n_k] etc.;
% the condition of each trial (1 = MI+AO, 2 = AO) is stored in the field named by 'condition_field'.
p = inputParser;
p.addParameter('max_threads', 8);
p.addParameter('condition_field', 'condition_code');
p.parse(varargin{:});
try
    maxNumCompThreads(p.Results.max_threads);
catch
end
% key -> full session, MI+AO session, AO session, pooled session
T = {'sub09', 'subj09_sess03',     'subj09_sess03miao',     'subj09_sess03ao',     'subj09_sess03both'
     'sub01', 'subj01_sess05',     'subj01_sess05miao',     'subj01_sess05ao',     'subj01_sess05pool'
     'sub12', 'subj12_sess03_ab',  'subj12_sess03_abmiao',  'subj12_sess03_abao',  'subj12_sess03_abpool'
     'sub23', 'subj23_sess03_ab',  'subj23_sess03_abmiao',  'subj23_sess03_abao',  'subj23_sess03_abpool'
     'sub29', 'subj29_sess03_abc', 'subj29_sess03_abcmiao', 'subj29_sess03_abcao', 'subj29_sess03_abcpool'
     'sub35', 'subj35_sess03_ab',  'subj35_sess03_abmiao',  'subj35_sess03_abao',  'subj35_sess03_abpool'};
row = find(strcmp(T(:, 1), key));
assert(numel(row) == 1, 'unknown participant key %s', key);
full_session = T{row, 2};
cond_sessions = T(row, 3:4);
cond_labels = {'MI+AO', 'AO'};
pooled_session = T{row, 5};
harmonic_list = [1 2 3];
goni_label = 'LHipW X';
lag_ms = 0;
n_cycle_points = 9;

setenv('GAIT_TRIPLE_ROOT', named_input('supplementary_exports'));
band_tbl = readtable(named_input(['observation_bands_' key '.csv']), 'TextType', 'string', 'Delimiter', ',');
prov = readtable(named_input(['observation_event_conditions_' key '.csv']), 'TextType', 'string', 'Delimiter', ',');
harmonics = double(harmonic_list(:)');
cycle_fraction = linspace(0, 1, n_cycle_points);
cycle_phase = cycle_fraction * 2 * pi;
half_width = pi / (n_cycle_points - 1);

S = load_triple_mat(full_session);
lag_samples = round(lag_ms * S.eeg.fs / 1000);
chan_labels = string(S.eeg.labels(:));

ev_all = [];
cond_code = zeros(0, 1);
for ci = 1:numel(cond_sessions)
    A = load_triple_mat(cond_sessions{ci});
    ev = trial_filter(A.events, 'MI');
    ev_all = [ev_all; ev(:)]; %#ok<AGROW>
    cond_code = [cond_code; ci * ones(numel(ev), 1)]; %#ok<AGROW>
    clear A
end
n_tr = numel(ev_all);
etstart = arrayfun(@(e) double(e.etstart), ev_all);
assert(numel(unique(etstart)) == n_tr, 'duplicate trials across conditions');
event_idx = nan(n_tr, 1); cycle = nan(n_tr, 1);
for ti = 1:n_tr
    r = find(double(prov.etstart) == etstart(ti));
    assert(numel(r) == 1, 'etstart %d not unique in the event-condition table', etstart(ti));
    assert(string(prov.cond(r)) == string(cond_labels{cond_code(ti)}), 'condition mismatch at etstart %d', etstart(ti));
    event_idx(ti) = double(prov.event_idx(r)); cycle(ti) = double(prov.cycle(r));
end

n_ch = numel(chan_labels); n_pt = numel(cycle_phase); n_h = numel(harmonics);
out = struct('sum_point_re', zeros(n_tr, n_ch, n_pt, n_h), 'sum_point_im', zeros(n_tr, n_ch, n_pt, n_h), ...
    'count_point_ch', zeros(n_tr, n_ch, n_pt, n_h), 'sum_all_re', zeros(n_tr, n_ch, n_h), ...
    'sum_all_im', zeros(n_tr, n_ch, n_h), 'count_all_ch', zeros(n_tr, n_ch, n_h), ...
    'n_valid_samples', zeros(n_tr, n_h), 'point_counts', zeros(n_tr, n_pt, n_h), 'kept', zeros(n_tr, n_h), ...
    'band_lo', nan(n_tr, n_h), 'band_hi', nan(n_tr, n_h));
opt = struct('goni_label', goni_label);
S.label = pooled_session;                       % band rows of the pooled session for every trial
[phase_map, count_rows, per_trial] = session_cycle_maps_local( ...
    S, ev_all, band_tbl, harmonics, cycle_phase, cycle_fraction, half_width, lag_samples, opt);
for hi = 1:n_h
    q = per_trial(hi);
    out.sum_point_re(:, :, :, hi) = real(q.sum_point);
    out.sum_point_im(:, :, :, hi) = imag(q.sum_point);
    out.count_point_ch(:, :, :, hi) = q.count_point_ch;
    out.sum_all_re(:, :, hi) = real(q.sum_all);
    out.sum_all_im(:, :, hi) = imag(q.sum_all);
    out.count_all_ch(:, :, hi) = q.count_all_ch;
    out.n_valid_samples(:, hi) = q.n_valid_samples;
    out.point_counts(:, :, hi) = q.point_counts;
    out.kept(:, hi) = double(q.kept);
    [lo, hi_] = lookup_harmonic_band_local(band_tbl, S.label, harmonics(hi));
    out.band_lo(:, hi) = lo; out.band_hi(:, hi) = hi_;
    fprintf('k%d band [%g %g]  mean|c| = %.15g  n_trials %d\n', harmonics(hi), lo, hi_, ...
        mean(abs(reshape(phase_map(:, :, hi), [], 1)), 'omitnan'), count_rows.n_trials(hi));
end

out.event_idx = event_idx; out.cycle = cycle; out.etstart = etstart;
out.(p.Results.condition_field) = cond_code;
out.condition_names = char(string(cond_labels(:)));
out.condition_sessions = char(string(cond_sessions(:)));
out.chan_labels = char(chan_labels);
out.harmonics = harmonics; out.cycle_fraction = cycle_fraction;
out.phase_map_re = real(phase_map); out.phase_map_im = imag(phase_map);    % [n_ch x 9 x n_k], all trials
out.count_n_trials = count_rows.n_trials;
out.count_n_valid_samples = count_rows.n_valid_samples;
out.count_min_cycle_point_samples = count_rows.min_cycle_point_samples;
out.full_session = full_session;

out_dir = fullfile(getenv('NEUROGAIT_OUT'), 'derived');
if ~isfolder(out_dir), mkdir(out_dir); end
out_mat = fullfile(out_dir, ['observation_trial_sums_' key '.mat']);
tmp = [out_mat '.tmp.mat'];
save(tmp, '-struct', 'out', '-v7.3');
movefile(tmp, out_mat, 'f');
fprintf('wrote %s  n_trial=%d\n', out_mat, n_tr);
end


function path = named_input(name)
% Path of a named input: $NEUROGAIT_DATA / (relative path of name in the $NEUROGAIT_INPUTS table, or name itself).
rel = name;
tbl = getenv('NEUROGAIT_INPUTS');
if ~isempty(tbl)
    lines = splitlines(string(fileread(tbl)));
    for i = 1:numel(lines)
        f = split(lines(i), sprintf('\t'));
        if numel(f) >= 2 && f(1) == string(name)
            rel = char(f(2));
            break;
        end
    end
end
path = fullfile(getenv('NEUROGAIT_DATA'), rel);
end


% ---- locking statistic of the main analysis, with the per-trial contributions returned ----

function [phase_map, count_rows, per_trial] = session_cycle_maps_local( ...
    S, ev, band_tbl, harmonics, cycle_phase, cycle_fraction, half_width, lag_samples, opt)
n_ch = numel(S.eeg.labels);
n_point = numel(cycle_phase);
n_h = numel(harmonics);
phase_map = nan(n_ch, n_point, n_h);
count_rows = table('Size', [0 5], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double'}, ...
    'VariableNames', {'session', 'harmonic_k', 'n_trials', 'n_valid_samples', 'min_cycle_point_samples'});

goni_idx = find_goni_idx_local(S.goni.labels, opt.goni_label);
[gait_lo, gait_hi] = lookup_harmonic_band_local(band_tbl, S.label, 1);
gait_phase = gait_phase_from_goni_local(S, goni_idx, gait_lo, gait_hi);

for hi = 1:n_h
    harmonic_k = harmonics(hi);
    [bp_lo, bp_hi] = lookup_harmonic_band_local(band_tbl, S.label, harmonic_k);
    [phase_mod, n_trials, n_valid, point_counts, per_trial(hi)] = accumulate_session_cycle_local( ...
        S, ev, gait_phase, bp_lo, bp_hi, cycle_phase, half_width, lag_samples);
    phase_map(:, :, hi) = phase_mod;
    count_rows = [count_rows; {string(S.label), harmonic_k, n_trials, n_valid, min(point_counts)}]; %#ok<AGROW>
end
end

function gait_phase = gait_phase_from_goni_local(S, goni_idx, bp_lo, bp_hi)
fs = S.goni.fs;
g = double(S.goni.x(:, goni_idx));
valid = isfinite(g);
if nnz(valid) < 10
    gait_phase = nan(size(g));
    return;
end
if any(~valid)
    xi = (1:numel(g))';
    g(~valid) = interp1(xi(valid), g(valid), xi(~valid), 'linear', 'extrap');
end
[z, pp, k] = butter(2, [bp_lo bp_hi] / (fs / 2), 'bandpass');
g_band = filtfilt(zp2sos(z, pp, k), 1, g);
gait_phase = angle(hilbert(g_band));
end

function [phase_mod, n_trials, n_valid_samples, point_counts, pt] = accumulate_session_cycle_local( ...
    S, ev, gait_phase, bp_lo, bp_hi, cycle_phase, half_width, lag_samples)
fs = S.eeg.fs;
[z, pp, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
x_band = filtfilt(zp2sos(z, pp, k), 1, double(S.eeg.x)')';
xa = hilbert(x_band')';
phase_unit = xa ./ max(abs(xa), eps);

n_ch = size(xa, 1);
n_point = numel(cycle_phase);
T_total = size(xa, 2);
sum_point = complex(zeros(n_ch, n_point), zeros(n_ch, n_point));
count_point_ch = zeros(n_ch, n_point);
sum_all = complex(zeros(n_ch, 1), zeros(n_ch, 1));
count_all_ch = zeros(n_ch, 1);
point_counts = zeros(1, n_point);
n_trials = 0;
n_valid_samples = 0;
% per-trial contributions (zeros = no contribution; kept marks the trials the loop used)
n_ev = numel(ev);
pt = struct('kept', false(n_ev, 1), 'n_valid_samples', zeros(n_ev, 1), 'point_counts', zeros(n_ev, n_point), ...
    'sum_point', complex(zeros(n_ev, n_ch, n_point), zeros(n_ev, n_ch, n_point)), 'count_point_ch', zeros(n_ev, n_ch, n_point), ...
    'sum_all', complex(zeros(n_ev, n_ch), zeros(n_ev, n_ch)), 'count_all_ch', zeros(n_ev, n_ch));

for ti = 1:numel(ev)
    [out_t, phi] = aligned_time_and_phase_local(S, ev(ti), gait_phase, lag_samples, T_total);
    if isempty(out_t)
        continue;
    end
    good = isfinite(phi);
    if nnz(good) < 5
        continue;
    end
    eeg_cols = out_t + lag_samples;
    good_cols = find(good);
    V_all = phase_unit(:, eeg_cols(good_cols));
    ok_all = isfinite(real(V_all)) & isfinite(imag(V_all));
    V_all(~ok_all) = 0;
    sum_all = sum_all + sum(V_all, 2);
    count_all_ch = count_all_ch + sum(ok_all, 2);
    n_trials = n_trials + 1;
    n_valid_samples = n_valid_samples + numel(good_cols);
    pt.kept(ti) = true; pt.n_valid_samples(ti) = numel(good_cols);
    pt.sum_all(ti, :) = reshape(sum(V_all, 2), 1, []);
    pt.count_all_ch(ti, :) = reshape(sum(ok_all, 2), 1, []);

    for pi = 1:n_point
        d = angle(exp(1i * (phi - cycle_phase(pi))));
        cols = find(good & abs(d) <= half_width);
        if isempty(cols)
            continue;
        end
        V = phase_unit(:, eeg_cols(cols));
        ok = isfinite(real(V)) & isfinite(imag(V));
        V(~ok) = 0;
        sum_point(:, pi) = sum_point(:, pi) + sum(V, 2);
        count_point_ch(:, pi) = count_point_ch(:, pi) + sum(ok, 2);
        point_counts(pi) = point_counts(pi) + numel(cols);
        pt.sum_point(ti, :, pi) = reshape(sum(V, 2), 1, []);
        pt.count_point_ch(ti, :, pi) = reshape(sum(ok, 2), 1, []);
        pt.point_counts(ti, pi) = numel(cols);
    end
end

mean_point = safe_divide_local(sum_point, count_point_ch);
mean_all = safe_divide_local(sum_all, count_all_ch);
phase_mod = mean_point - mean_all;
phase_mod(~isfinite(real(phase_mod)) | ~isfinite(imag(phase_mod))) = NaN;
end

function [out_t, phi] = aligned_time_and_phase_local(S, ev, gait_phase, lag_samples, T_total)
et0 = ev.etstart;
et1 = ev.etstop;
gt0 = ev.gtstart;
gt1 = ev.gtstop;
if et1 < et0 || gt1 < gt0
    out_t = zeros(1, 0); phi = zeros(1, 0); return;
end
Lk = min(et1 - et0 + 1, gt1 - gt0 + 1);
valid_start = max(1, 1 - min(lag_samples)) + et0 - 1;
valid_end = min(T_total - max(lag_samples), et0 + Lk - 1);
if valid_end < valid_start
    out_t = zeros(1, 0); phi = zeros(1, 0); return;
end
out_t = valid_start:valid_end;
local_idx = out_t - et0 + 1;
goni_t = gt0 + local_idx - 1;
goni_t = min(max(goni_t, 1), numel(gait_phase));
phi = double(gait_phase(goni_t));
phi = phi(:)';
end

function [bp_lo, bp_hi] = lookup_harmonic_band_local(tbl, session, harmonic_k)
mask = string(tbl.session) == string(session) & double(tbl.harmonic_k) == double(harmonic_k);
idx = find(mask);
assert(numel(idx) == 1, 'Expected one band row for %s k%d, found %d', session, harmonic_k, numel(idx));
bp_lo = double(tbl.band_lo(idx));
bp_hi = double(tbl.band_hi(idx));
assert(isfinite(bp_lo) && isfinite(bp_hi) && bp_lo > 0 && bp_hi > bp_lo, ...
    'Invalid band for %s k%d: [%g %g]', session, harmonic_k, bp_lo, bp_hi);
end

function idx = find_goni_idx_local(labels, target)
labels = string(labels(:));
idx = find(strcmp(strtrim(labels), strtrim(string(target))), 1);
if isempty(idx)
    idx = find(strcmpi(strtrim(labels), strtrim(string(target))), 1);
end
assert(~isempty(idx), 'Cannot find goni label %s', target);
end

function M = safe_divide_local(Sn, C)
M = Sn ./ C;
M(C == 0 | ~isfinite(real(M)) | ~isfinite(imag(M))) = NaN;
end
