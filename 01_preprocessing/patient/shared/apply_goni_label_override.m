function [labels, n_applied] = apply_goni_label_override(labels, labels_raw, override, tag)
%APPLY_GONI_LABEL_OVERRIDE  Correct a goniometer sensor that was labelled with the wrong body site at acquisition.
%
%   [labels, n_applied] = apply_goni_label_override(labels, labels_raw, override, tag)
%
%   `override` is an N x 3 cell array of {serial, from_prefix, to_prefix} rows. For every channel whose raw
%   header string (labels_raw, e.g. 'LAnkS <serial> X axis') contains `serial` and whose short label (labels,
%   e.g. 'LAnkS X') starts with `from_prefix`, the prefix is rewritten to `to_prefix`. Exactly two channels,
%   the X and the Y axis of that sensor, must match per row; any other count is an error, because a partial
%   match means the assumption about the recording is wrong.
%
%   The match keys on the serial because the short label is the field that is wrong; the serial in the raw
%   header is the sensor's physical identity. Called by export_session_relabel.

    n_applied = 0;
    if nargin < 4 || isempty(tag), tag = 'goni_label_override'; end
    if isempty(override), return; end

    if isempty(labels_raw)
        error('apply_goni_label_override:noRawLabels', ...
            '%s: override requested but raw header labels are unavailable', tag);
    end
    if numel(labels_raw) ~= numel(labels)
        error('apply_goni_label_override:lengthMismatch', ...
            '%s: labels (%d) and labels_raw (%d) are not aligned', ...
            tag, numel(labels), numel(labels_raw));
    end
    if ~iscell(override) || size(override, 2) ~= 3
        error('apply_goni_label_override:badSpec', ...
            '%s: override must be an N x 3 cell of {serial, from_prefix, to_prefix}', tag);
    end

    for r = 1:size(override, 1)
        serial = override{r, 1};
        fromp  = override{r, 2};
        top    = override{r, 3};

        hit = false(1, numel(labels));
        for c = 1:numel(labels)
            hit(c) = contains(labels_raw{c}, serial) && ...
                     strncmp(strtrim(labels{c}), fromp, numel(fromp));
        end
        k = find(hit);
        if numel(k) ~= 2
            error('apply_goni_label_override:notTwoAxes', ...
                ['%s: serial %s with prefix %s matched %d channels, expected 2 ' ...
                 '(X and Y). Matched: %s'], tag, serial, fromp, numel(k), ...
                strjoin(labels(k), ', '));
        end

        fprintf('  [%s] %s: %s -> %s\n', tag, serial, fromp, top);
        for c = k
            before = strtrim(labels{c});
            labels{c} = [top, before(numel(fromp)+1:end)];
            fprintf('    ch %2d  %-10s -> %-10s   (raw: %s)\n', ...
                c, before, labels{c}, strtrim(labels_raw{c}));
            n_applied = n_applied + 1;
        end
    end
end
