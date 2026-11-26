function registry = build_patient_registry(raw_root) %#ok<STOUT,INUSD>
%BUILD_PATIENT_REGISTRY  Patient session catalogue (not distributed).
%
%   The registry lists, for every patient session, the file layout of the raw recordings, the marker
%   protocol, the goniometer file selection and the per-session processing limits. It describes the
%   storage of identifiable raw data and is not part of the repository.

    error('build_patient_registry:notDistributed', ...
        'The patient session registry describes the raw-data file layout and is not distributed.');
end
