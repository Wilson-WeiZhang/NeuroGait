function src_zmap_session(session, triple_root, band_mi, band_exec, qc_flags, kernel_path, channel_path, out_dir, varargin)
%SRC_ZMAP_SESSION  Vertex z of the phase-locked power against random trial rotations, one session (closed-form trial-rotation null).
% Imagined walking and duration-matched resting windows use the same gait phase and bands.  a_t = samples x rotating coefficient of the source analytic signal in trial t (amplitude kept).
% Rotating each trial by an independent random phase gives Da = |sum a_t e^(i th_t)|^2 - sum |a_t|^2 with mean 0 and variance (sum |a_t|^2)^2 - sum |a_t|^4, so
%   z = (|sum a_t|^2 - sum |a_t|^2) / sqrt((sum |a_t|^2)^2 - sum |a_t|^4)
% is the limit of a rotation null with many draws; 'rot_check' stores a_t on every 50th vertex to compare it with 200 draws.
% k2 of imagined walking and rest is also computed with T8 and with Cz replaced by the mean of their 4 nearest electrodes.
p = inputParser;
p.addParameter('harmonics', [1 2]);
p.addParameter('decim', 10);
p.addParameter('seed', 20260717);
p.addParameter('rot_check', false);
p.parse(varargin{:});
opt = p.Results;
maxNumCompThreads(1);
part = fullfile(out_dir, [session '.mat']);
if isfile(part), return; end
setenv('GAIT_TRIPLE_ROOT', triple_root);
setenv('GAIT_GONI_QC_TRIAL_FLAGS', qc_flags);
S = load_triple_mat(session);
kern = load(kernel_path, 'ImagingKernel', 'ChannelOrder');
order = string(kern.ChannelOrder(:));
labels = string(S.eeg.labels(:));
[tf, idx] = ismember(upper(labels), upper(order));
assert(all(tf) && numel(labels) == size(kern.ImagingKernel, 2), 'channels of %s do not match the kernel', session);
K = single(kern.ImagingKernel(:, idx));
C = load(channel_path);
nm = string({C.Channel.Name});
pos = zeros(numel(labels), 3);
for c = 1:numel(labels)
    pos(c, :) = C.Channel(strcmpi(nm, labels(c))).Loc(:, 1)';
end
D = squareform(pdist(pos));
D(1:numel(labels) + 1:end) = inf;
[~, nn] = sort(D, 2);
A = struct('base', eye(numel(labels)));
for e = ["T8", "Cz"]
    i = find(strcmpi(labels, e));
    m = eye(numel(labels));
    m(i, :) = 0;
    m(i, nn(i, 1:4)) = 0.25;
    A.(e) = m;
end
R = struct('session', session, 'decim', opt.decim);
ev_mi = trial_filter(S.events, 'MI');
ev_rw = trial_filter(S.events, 'EXEC');
ev_rest = S.events(strcmp({S.events.code}, 'rest') & strcmp({S.events.qc}, 'good'));
bt_mi = readtable(band_mi, 'TextType', 'string', 'Delimiter', ',');
for k = opt.harmonics
    tag = sprintf('k%d', k);
    variants = {'base'};
    if k == 2, variants = {'base', 'T8', 'Cz'}; end
    [xa, phi] = prepare(S, bt_mi, k, 'LHipW X');
    R.IW.(tag) = accumulate(K, A, variants, xa, S, ev_mi, [], phi, k, opt, 0);
    if ~isempty(ev_rest)
        R.REST.(tag) = accumulate(K, A, variants, xa, S, ev_mi, ev_rest, phi, k, opt, opt.seed);
    end
    if ~isempty(ev_rw) && strlength(string(band_exec)) > 0
        bt_rw = readtable(band_exec, 'TextType', 'string', 'Delimiter', ',');
        [xa, phi] = prepare(S, bt_rw, k, 'LHipS X');
        R.RW.(tag) = accumulate(K, A, {'base'}, xa, S, ev_rw, [], phi, k, opt, 0);
    end
end
tmp = [part '.tmp'];
save(tmp, '-struct', 'R', '-v7');
movefile(tmp, part, 'f');
end


function [xa, gait_phase] = prepare(S, bt, k, goni_label)
[lo1, hi1] = band_of(bt, S.label, 1);
g = double(S.goni.x(:, find(strcmpi(strtrim(string(S.goni.labels(:))), goni_label), 1)));
valid = isfinite(g);
if any(~valid)
    xi = (1:numel(g))';
    g(~valid) = interp1(xi(valid), g(valid), xi(~valid), 'linear', 'extrap');
end
[z, pp, kk] = butter(2, [lo1 hi1] / (S.goni.fs / 2), 'bandpass');
gait_phase = angle(hilbert(filtfilt(zp2sos(z, pp, kk), 1, g)));
[lo, hi] = band_of(bt, S.label, k);
[z, pp, kk] = butter(4, [lo hi] / (S.eeg.fs / 2), 'bandpass');
x_band = filtfilt(zp2sos(z, pp, kk), 1, double(S.eeg.x)')';
xa = single(hilbert(x_band')');
end


function [lo, hi] = band_of(bt, session, k)
i = find(string(bt.session) == string(session) & double(bt.harmonic_k) == k);
assert(numel(i) == 1, 'band row for %s k%d', session, k);
lo = double(bt.band_lo(i));
hi = double(bt.band_hi(i));
end


function out = accumulate(K, A, variants, xa, S, ev, rest_ev, gait_phase, k, opt, seed)
yoked = ~isempty(rest_ev);
T_total = size(xa, 2);
nv = size(K, 1);
for q = 1:numel(variants)
    acc(q) = struct('S', complex(zeros(nv, 1)), 'Q2', zeros(nv, 1), 'Q4', zeros(nv, 1)); %#ok<AGROW>
    KA{q} = K * single(A.(variants{q})); %#ok<AGROW>
end
n_trials = 0;
keep = [];
for ti = 1:numel(ev)
    et0 = ev(ti).etstart;
    Lk = min(ev(ti).etstop - et0 + 1, ev(ti).gtstop - ev(ti).gtstart + 1);
    out_t = et0:min(T_total, et0 + Lk - 1);
    if isempty(out_t) || Lk < 1, continue; end
    goni_t = min(max(ev(ti).gtstart + (out_t - et0), 1), numel(gait_phase));
    phi = double(gait_phase(goni_t));
    phi = phi(:)';
    good = isfinite(phi);
    if nnz(good) < 5, continue; end
    if yoked
        cols = rest_anchor(rest_ev, numel(out_t), T_total, double(seed) + 1009 * ti);
        if isempty(cols), continue; end
    else
        cols = out_t;
    end
    gc = find(good);
    sel = gc(1:opt.decim:end);
    rot = single(exp(1i * k * phi(sel)));
    y = xa(:, cols(sel));
    w = numel(sel);
    for q = 1:numel(variants)
        a = double(mean((KA{q} * y) .* rot, 2)) * w;
        acc(q).S = acc(q).S + a;
        p2 = abs(a) .^ 2;
        acc(q).Q2 = acc(q).Q2 + p2;
        acc(q).Q4 = acc(q).Q4 + p2 .^ 2;
        if q == 1 && opt.rot_check
            keep(:, end + 1) = a(1:50:end); %#ok<AGROW>
        end
    end
    n_trials = n_trials + 1;
end
out = struct('n_trials', n_trials);
for q = 1:numel(variants)
    out.(['z_' variants{q}]) = single((abs(acc(q).S) .^ 2 - acc(q).Q2) ./ sqrt(acc(q).Q2 .^ 2 - acc(q).Q4));
end
if opt.rot_check
    out.a_check = keep;
end
end


function sample_t = rest_anchor(rest_events, n_cols, total_samples, seed)
sample_t = [];
valid = [];
for ri = 1:numel(rest_events)
    r0 = round(double(rest_events(ri).etstart));
    r1 = round(double(rest_events(ri).etstop));
    lo = max(1, r0);
    hi = min(total_samples - n_cols + 1, r1 - n_cols + 1);
    if hi >= lo
        valid(end + 1).lo = lo; %#ok<AGROW>
        valid(end).hi = hi;
    end
end
if isempty(valid), return; end
rng(seed, 'twister');
ri = randi(numel(valid));
start_t = valid(ri).lo + randi(valid(ri).hi - valid(ri).lo + 1) - 1;
sample_t = start_t:(start_t + n_cols - 1);
end
