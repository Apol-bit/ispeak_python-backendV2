# Opt-in speech-analysis memory diagnostics

Disabled by default. In a separately approved, isolated staging deployment, set
`ISPEAK_MEMORY_DIAGNOSTICS=true` in the Python process environment. No model,
concurrency, scoring, transcription, timestamp or authentication setting changes.
Set it to `false` or remove it to disable diagnostics. No deployment was performed.

Filter application INFO logs for JSON records with `event=ispeak_memory`.
Each request has a newly generated random `request_id`; no incoming identifier,
account identifier, language, filename, path, transcript, audio or credential is
logged. Only fixed stage/phase/role labels, durations, timings and byte counts are
included. Existing logging settings remain independent; keep detailed speech and
error logging disabled for staging. Apply normal restricted log access/retention.

Stages cover request entry/admission, multipart parsing, upload copies, inference
admission, decoding/validation, repeated signal validation, Whisper with word
timestamps, intensity, pitch, pauses, articulation, filler classification and
reference MFCC/DTW comparison. Both reference and user audio stages are labelled.
Cleanup snapshots cover saved files and then multipart closure; the final request
event is emitted after cleanup, including failures/cancellations. Request-end
events identify outcome only as end/error, never exception details. Normal validation,
response bodies, admission limits, exceptions and cancellation ownership are retained.

Measurements:

- `process_rss_bytes`: current Linux `/proc/self/statm` resident pages or Windows
  working-set bytes. Unsupported/unavailable measurements are null.
- `cgroup_current_bytes`, `cgroup_peak_bytes`, `cgroup_limit_bytes`: common Linux
  cgroup v2/v1 memory counters, with nested membership and namespace-root fallbacks.
  Unavailable/unlimited counters are null. Nonstandard mounts may be unsupported.
  Container counters include other work/processes in the same cgroup; peak is a
  cgroup-lifetime high-water mark, not a resettable per-stage peak.
- `request_temp_file_bytes`: logical sizes of at most two request-owned saved
  uploads, measured with stat only. Missing/deleted files count as zero; failures
  are null. No recursive directory scan or file-content read is performed.
- `multipart_payload_bytes`: logical UploadFile sizes until the spool is closed.
  This is not allocated RAM or total disk usage; it includes in-memory spools.
  Never add it to RSS/cgroup memory as independent physical memory.
- `recording_duration_seconds`, `reference_duration_seconds`: actual decoded
  durations, populated after successful decoding; null before known/on failure.
- `reference_used`: whether the parsed request supplied reference audio; null
  until that metadata is available. It does not mean comparison succeeded.
- `elapsed_ms`: stage wall time, including diagnostic overhead; null for point
  checkpoints. `request_elapsed_ms` locates every snapshot within the request.

Context variables correlate threadpool inference with its request without changing
function arguments or retaining historical request state. A failed sampler or log
handler is best-effort and cannot replace an application error. Disabled diagnostics
perform no measurements, timestamp reads or log writes. Enabled diagnostics read
small fixed-size counter files at stage boundaries, never poll in the background
or collect stack/heap dumps. Their real production overhead has not been measured.

Limitations: boundary snapshots can miss transient peaks, and an OOM kill may leave
only the stage-start event. They do not break down allocations inside Whisper,
ModernBERT or DTW, attribute container peaks to a specific concurrent request,
measure all `/tmp` caches, or profile model startup. They do not collect a delayed
post-request idle sample. Use a controlled single-request staging sequence before
testing overlap; compare warm baselines and container peak changes between requests.

Model-free verification:

```powershell
./.venv/Scripts/python.exe -B -m unittest testing_script.test_memory_diagnostics -v
```

The diagnostic module is included in the main and source-overlay Docker context
allowlists and COPY lists. No image build, model inference or cloud test is needed
for the focused unit tests. Real Linux cgroup permissions/namespace layout and log
delivery must be verified later in an approved staging environment.
