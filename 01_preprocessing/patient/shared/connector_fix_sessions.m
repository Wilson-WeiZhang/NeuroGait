function out = connector_fix_sessions(label)
%CONNECTOR_FIX_SESSIONS  Sessions recorded with the two 32-channel amplifier module connectors interchanged.
%
%   labels = connector_fix_sessions()        1xN cell array of session labels
%   tf     = connector_fix_sessions(label)   true if the label is on the list
%
%   In these sessions the two connectors were plugged into each other's sockets, so the sample stored under
%   channel i came from electrode i+32. The channel order is restored by apply_module_connector_fix
%   (patient/01_align/) right after the recordings are loaded and merged. This function is the only place
%   the session list is defined.

    labels = {'P1_sess04', 'P2_sess01', 'P2_sess02'};

    if nargin < 1 || isempty(label)
        out = labels;
    else
        out = ismember(label, labels);
    end
end
