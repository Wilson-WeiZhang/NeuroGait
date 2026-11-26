function EEG = apply_module_connector_fix(EEG, label, force)
%APPLY_MODULE_CONNECTOR_FIX  Restore the channel order of a session recorded with interchanged module connectors.
%
%   EEG = apply_module_connector_fix(EEG, label)
%   EEG = apply_module_connector_fix(EEG, label, true)   % skip the session-list check
%
%   The two 32-channel amplifier module connectors of the sessions listed in connector_fix_sessions were
%   plugged into each other's sockets, so the sample stored under channel i came from electrode i+32. This
%   function undoes the interchange. It runs immediately after the recordings are loaded and merged, on the
%   raw channel order (63 channels including the electro-oculogram channels), before any channel is dropped,
%   before the 59-channel selection and before bad-channel detection. The common-average reference of
%   the later preprocessing stage is then applied to the restored order.
%
%   Index convention: channel i (1..32) is exchanged with channel i+32 while i+32 does not exceed the number
%   of channels. With 63 channels this exchanges channels 1..31 with 33..63 and leaves channel 32 unchanged.
%
%   The permutation is recorded in EEG.etc.connector_fix_applied (n_swapped, nbchan_at_swap, perm) so that
%   any later file shows whether the repair was applied.

    if nargin < 3 || isempty(force)
        force = false;
    end

    if ~force && ~connector_fix_sessions(label)
        warning('apply_module_connector_fix:notListed', ...
            ['%s is not on the connector-fix session list (%s). Applying the fix to a correctly ' ...
             'wired session scrambles its channel order. Pass force = true only for a session ' ...
             'with a confirmed connector interchange.'], label, strjoin(connector_fix_sessions(), ', '));
    end

    n = EEG.nbchan;
    perm = 1:n;
    for i = 1:32
        j = i + 32;
        if j <= n
            perm([i j]) = perm([j i]);
        end
    end
    n_sw = sum(perm ~= (1:n));

    EEG.data = EEG.data(perm, :);
    EEG.etc.connector_fix_applied = struct('n_swapped', n_sw, 'nbchan_at_swap', n, 'perm', perm);

    fprintf('   [connector_fix] %s: exchanged %d of %d channels (1..%d <-> 33..%d)\n', ...
            label, n_sw, n, min(32, n - 32), min(64, n));
end
