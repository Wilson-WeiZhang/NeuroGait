function harmonic_phase_maps(mode, varargin)
%HARMONIC_PHASE_MAPS Per-session maps of EEG modulation over the gait cycle at the gait harmonics.
%
%   harmonic_phase_maps('imagined')   imagined walking, walker hip goniometer phase
%   harmonic_phase_maps('real')       real walking, the participant's own hip goniometer phase
%   harmonic_phase_maps('rest')       resting EEG windows matched to the imagined-walking trials
%
% For every session, the goniometer hip angle is band-passed around the walking frequency (2nd-order Butterworth,
% zero phase) and its Hilbert phase defines the gait cycle. The EEG (59 channels plus FCz, 250 Hz) is band-passed around
% harmonic k = 1, 2, 3 of the walking frequency (4th-order Butterworth, zero phase) and reduced to unit-length complex
% phasors. Within the good trials of the condition, the phasors are averaged in 9 cycle-phase bins (half-width pi/8,
% bins at 0, 12.5, ... 100 % of the cycle); the mean phasor over all trial samples is subtracted. The result is one
% complex value per (harmonic, cycle bin, channel) and session.
% For 'rest', the gait phase and trial layout of each imagined-walking trial are kept but the EEG samples are read from a
% randomly placed window inside a good resting trial (random generator 'twister', seed 20260717 + 1009 * trial number); sessions
% without a resting trial are skipped.
%
% Named inputs: session_list_<mode>.csv (column "session"), walking_bands_<mode>.csv (columns session, harmonic_k,
% band_lo, band_hi), healthy_exports (folder holding <session>/<session>_{eeg,goni,event}.mat).
% Needs load_triple_mat.m and trial_filter.m on the MATLAB path (folder shared/ of the decoding code).
% Output: $NEUROGAIT_OUT/derived/healthy_phase_<mode>.csv  (mode 'imagined' | 'real' | 'rest'),
%   columns session, harmonic_k, cycle_fraction, cycle_percent, channel, real, imag (+ abs, angle_rad for 'rest').
% Optional name-value arguments: 'sessions' (string array, default all), 'max_threads' (default 8).

p = inputParser;
p.addParameter('sessions', strings(0, 1));
p.addParameter('max_threads', 8);
p.parse(varargin{:});
opt = p.Results;
try, maxNumCompThreads(opt.max_threads); catch, end

switch mode
    case 'imagined', condition = 'MI';   goni_label = 'LHipW X'; yoked = false;
    case 'real',     condition = 'EXEC'; goni_label = 'LHipS X'; yoked = false;
    case 'rest',     condition = 'MI';   goni_label = 'LHipW X'; yoked = true;
    otherwise, error('mode must be imagined, real or rest');
end
harmonics = [1 2 3];
n_cycle_points = 9;
seed = 20260717;
setenv('GAIT_TRIPLE_ROOT', named_input('healthy_exports'));

band_tbl = readtable(named_input(['walking_bands_' mode '.csv']), 'TextType', 'string', 'Delimiter', ',');
T = readtable(named_input(['session_list_' mode '.csv']), 'TextType', 'string', 'Delimiter', ',');
sessions = string(T.session);
if ~isempty(opt.sessions)
    sessions = intersect(sessions, string(opt.sessions(:)), 'stable');
end

cycle_fraction = linspace(0, 1, n_cycle_points);
cycle_phase = cycle_fraction * 2 * pi;
half_width = pi / (n_cycle_points - 1);

out_dir = fullfile(getenv('NEUROGAIT_OUT'), 'derived');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_csv = fullfile(out_dir, ['healthy_phase_' mode '.csv']);
tmp_csv = [out_csv '.tmp'];
fid = fopen(tmp_csv, 'w');
assert(fid > 0, 'Cannot write %s', tmp_csv);
if yoked
    fprintf(fid, 'session,harmonic_k,cycle_fraction,cycle_percent,channel,real,imag,abs,angle_rad\n');
else
    fprintf(fid, 'session,harmonic_k,cycle_fraction,cycle_percent,channel,real,imag\n');
end

chan_labels = strings(0, 1);
for si = 1:numel(sessions)
    session = sessions(si);
    fprintf('[%d/%d] %s\n', si, numel(sessions), session);
    S = load_triple_mat(char(session));
    ev = trial_filter(S.events, condition);
    if isempty(ev)
        warning('No %s events in %s', condition, session);
        continue;
    end
    if yoked
        rest_ev = S.events(strcmp({S.events.code}, 'rest') & strcmp({S.events.qc}, 'good'));
        if isempty(rest_ev)
            warning('No rest trials in %s', session);
            continue;
        end
        has_bands = all(arrayfun(@(kk) nnz(string(band_tbl.session) == string(S.label) & ...
            double(band_tbl.harmonic_k) == kk) == 1, unique([1, harmonics])));
        if ~has_bands
            warning('No band rows for %s', S.label);
            continue;
        end
    else
        rest_ev = [];
    end
    if isempty(chan_labels)
        chan_labels = string(S.eeg.labels(:));
    else
        assert(isequal(chan_labels, string(S.eeg.labels(:))), 'Channel labels mismatch in %s', session);
    end
    lag_samples = 0;

    goni_idx = find_goni_idx(S.goni.labels, goni_label);
    [glo, ghi] = lookup_band(band_tbl, S.label, 1);
    gait_phase = gait_phase_from_goni(S, goni_idx, glo, ghi);

    n_ch = numel(S.eeg.labels);
    phase_map = nan(n_ch, n_cycle_points, numel(harmonics));
    for hi = 1:numel(harmonics)
        [blo, bhi] = lookup_band(band_tbl, S.label, harmonics(hi));
        phase_map(:, :, hi) = accumulate_cycle(S, ev, rest_ev, gait_phase, blo, bhi, ...
            cycle_phase, half_width, lag_samples, yoked, seed);
    end
    for hi = 1:numel(harmonics)
        for pj = 1:n_cycle_points
            for ci = 1:n_ch
                v = phase_map(ci, pj, hi);
                if yoked
                    fprintf(fid, '%s,%d,%.10g,%.6g,%s,%.10g,%.10g,%.10g,%.10g\n', ...
                        char(S.label), harmonics(hi), cycle_fraction(pj), 100 * cycle_fraction(pj), ...
                        char(chan_labels(ci)), real(v), imag(v), abs(v), angle(v));
                else
                    fprintf(fid, '%s,%d,%.10g,%.6g,%s,%.10g,%.10g\n', ...
                        char(S.label), harmonics(hi), cycle_fraction(pj), 100 * cycle_fraction(pj), ...
                        char(chan_labels(ci)), real(v), imag(v));
                end
            end
        end
    end
end
fclose(fid);
movefile(tmp_csv, out_csv, 'f');
fprintf('wrote %s\n', out_csv);
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


function phase_mod = accumulate_cycle(S, ev, rest_ev, gait_phase, bp_lo, bp_hi, cycle_phase, half_width, lag_samples, yoked, seed)
fs = S.eeg.fs;
[z, pp, k] = butter(4, [bp_lo bp_hi] / (fs / 2), 'bandpass');
x_band = filtfilt(zp2sos(z, pp, k), 1, double(S.eeg.x)')';
xa = hilbert(x_band')';
phase_unit = xa ./ max(abs(xa), eps);
n_ch = size(xa, 1);
T_total = size(xa, 2);
n_point = numel(cycle_phase);
sum_point = complex(zeros(n_ch, n_point), zeros(n_ch, n_point));
count_point = zeros(n_ch, n_point);
sum_all = complex(zeros(n_ch, 1), zeros(n_ch, 1));
count_all = zeros(n_ch, 1);

for ti = 1:numel(ev)
    [out_t, phi] = aligned_time_and_phase(ev(ti), gait_phase, lag_samples, T_total);
    if isempty(out_t)
        continue;
    end
    good = isfinite(phi);
    if nnz(good) < 5
        continue;
    end
    if yoked
        sample_t = resting_random_anchor(rest_ev, numel(out_t), lag_samples, T_total, double(seed) + 1009 * ti);
        if isempty(sample_t)
            continue;
        end
        eeg_cols = sample_t + lag_samples;
    else
        eeg_cols = out_t + lag_samples;
    end
    gc = find(good);
    V_all = phase_unit(:, eeg_cols(gc));
    ok_all = isfinite(real(V_all)) & isfinite(imag(V_all));
    V_all(~ok_all) = 0;
    sum_all = sum_all + sum(V_all, 2);
    count_all = count_all + sum(ok_all, 2);
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
        count_point(:, pi) = count_point(:, pi) + sum(ok, 2);
    end
end
mean_point = safe_divide(sum_point, count_point);
mean_all = safe_divide(sum_all, count_all);
phase_mod = mean_point - mean_all;
phase_mod(~isfinite(real(phase_mod)) | ~isfinite(imag(phase_mod))) = NaN;
end


function sample_t = resting_random_anchor(rest_events, n_cols, lag_samples, total_samples, seed)
sample_t = [];
if isempty(rest_events) || n_cols <= 0
    return;
end
min_lag = min(lag_samples);
max_lag = max(lag_samples);
valid = [];
for ri = 1:numel(rest_events)
    r0 = round(double(rest_events(ri).etstart));
    r1 = round(double(rest_events(ri).etstop));
    lo = max(1 - min_lag, r0 - min_lag);
    hi = min(total_samples - max_lag - n_cols + 1, r1 - max_lag - n_cols + 1);
    if hi >= lo
        valid(end + 1).lo = lo; %#ok<AGROW>
        valid(end).hi = hi;
    end
end
if isempty(valid)
    return;
end
rng(seed, 'twister');
ri = randi(numel(valid));
start_t = valid(ri).lo + randi(valid(ri).hi - valid(ri).lo + 1) - 1;
sample_t = start_t:(start_t + n_cols - 1);
end


function gait_phase = gait_phase_from_goni(S, goni_idx, bp_lo, bp_hi)
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


function [out_t, phi] = aligned_time_and_phase(ev, gait_phase, lag_samples, T_total)
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


function [bp_lo, bp_hi] = lookup_band(tbl, session, harmonic_k)
idx = find(string(tbl.session) == string(session) & double(tbl.harmonic_k) == double(harmonic_k));
assert(numel(idx) == 1, 'Expected one band row for %s k%d, found %d', session, harmonic_k, numel(idx));
bp_lo = double(tbl.band_lo(idx));
bp_hi = double(tbl.band_hi(idx));
assert(isfinite(bp_lo) && isfinite(bp_hi) && bp_lo > 0 && bp_hi > bp_lo, ...
    'Invalid band for %s k%d: [%g %g]', session, harmonic_k, bp_lo, bp_hi);
end


function idx = find_goni_idx(labels, target)
labels = string(labels(:));
idx = find(strcmp(strtrim(labels), strtrim(string(target))), 1);
if isempty(idx)
    idx = find(strcmpi(strtrim(labels), strtrim(string(target))), 1);
end
assert(~isempty(idx), 'Cannot find goniometer label %s', target);
end


function M = safe_divide(Sn, C)
M = Sn ./ C;
M(C == 0 | ~isfinite(real(M)) | ~isfinite(imag(M))) = NaN;
end
