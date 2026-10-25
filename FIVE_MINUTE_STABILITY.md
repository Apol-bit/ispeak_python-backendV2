# Five-minute recovery evidence and release gate

Status: NO-GO for five-minute beta/defense readiness. Full-model results are not
available. This document does not authorize a build, deployment or inference.

## Confirmed production event versus candidate source

The supplied October 25, 2026 event identifies revision ispeak-python-00002-jzg,
timestamp 2026-10-25T08:06:25.984154Z, 16,444 MiB used against 16,384 MiB, and an
88-second recording. Its internal processing stage and reference-used status are
unknown. Production source/model/package equivalence remains unestablished.
The current candidate's Whisper path requests word timestamps; installed
Transformers concatenates decoder cross-attentions before selecting/stacking
alignment heads. This is a suspect, not a measured production root cause.

## Implemented exact reference-DTW storage reduction

compare_reference_enunciation previously requested full pairwise cosine costs,
accumulated costs and predecessor storage from librosa. It only consumes terminal
cost divided by selected path length. bounded_dtw.cosine_dtw_statistics computes
those same values with two cost rows, two path-length rows and one distance row.
The reused feature axis is converted to contiguous float64 once, matching scipy's
conversion. Time remains quadratic; workspace is linear in that axis, plus input
features. Sequence axes are not swapped. Strict-less-than ties retain diagonal,
horizontal, vertical priority. No bands, downsampling, truncation, approximations,
precision reduction or score formula changes are introduced.

## Local measurements (no models or recordings)

Warm synthetic 24-dimensional float32 feature sequences, equal lengths at
16 kHz/hop 256, deterministic random seeds 41/42. tracemalloc peaks below exclude
input creation and are not total RSS/cgroup memory. Timing is local, not Cloud Run.

| Equivalent duration on each axis | Method | Traced peak bytes | Seconds |
| --- | --- | ---: | ---: |
| 30 s / 1,876 frames | Original dense librosa | 84,528,185 | 0.186 |
| 30 s / 1,876 frames | Final bounded implementation | 454,667 | 0.251 |
| 300 s / 18,751 frames | Final bounded implementation | 4,503,107 | 14.401 |

At 30 seconds, terminal cost 1727.1987823042923 and path length 2005 match exactly.
The 300-second feature-pair run completed (cost 17267.74066248664, length 19933),
but no full dense 300-second allocation was attempted. This is NOT five-minute
speech/transcription/scoring validation or a before/after whole-request comparison.
The first bounded prototype repeatedly converted the reused axis; removing that
copy reduced its measured 300-second feature-pair time from 59.065 to 14.401 s.
Numba compilation was warmed before measurements; cold compilation must be timed
in staging. No new software was installed.

Tests compare exact costs, selected path lengths and rounded reference scores on
rectangular random float32/float64 inputs, ties and repeated frames; compare the
complete acoustic reference-comparison output against the original algorithm;
reject invalid/zero-norm features; assert one-row distance allocation for
30/60/120/180/240/300-second frame dimensions; and check retained traced workspace
across 12 sequential calls. Short-other-axis allocation tests do not represent full
five-minute processing. Existing request cleanup/cancellation tests remain required.
Final guarded model-free regression runs: 123 passed with diagnostics disabled and
123 passed with diagnostics enabled, including six new DTW tests. Whisper and
filler-classifier constructors were blocked against accidental model loading.

Whisper, ModernBERT, pitch, fillers, model weights and transcription settings are
unchanged. Their full-duration peak allocations and accuracy remain unverified.

## End-to-end timeout evidence

Read-only local source inspection finds Flutter ApiClient.sendMultipart wraps the
whole exchange in 360 seconds. Node sessionController uses PYTHON_TIMEOUT_MS,
default 300,000 ms, for its Python request. Supplied production Python Cloud Run
timeout is 300 seconds. Deployed Node override, Node ingress/server timeout,
other proxies, and client build equivalence remain unverified. Recording duration
is not processing latency. Measure upload, queue/cold start, analysis, persistence
and response delivery separately and end to end. No timeout was changed.

## Mandatory isolated real-model validation matrix

Use approved synthetic speech, not production recordings. English and Filipino,
30/60/120/180/240/300 seconds, each without reference and with a reference of equal
duration: at least 24 core cases. Include unequal reference lengths, short valid
speech and rejection boundaries for empty/sub-minimum/over-limit inputs. Existing
validation requires at least one second of usable speech; the 0-second recording
start state is not an analyzable speech sample. Do not silently change that rule.

Run short cases first and escalate only with headroom. Repeat 300-second cases
at least three times per language/path; then 20 sequential mixed-duration requests
on a verified same instance. Include cold and warm runs, pauses/fast speech and
filler-rich content. Log only counters, duration, stage times and random IDs.
For the same fixture compare transcript, complete timestamp sequence, filler
evidence, every component and final score against a verified baseline. Preserve
decoding/model settings. Exact equality is required for this DTW-only change;
investigate nondeterminism before accepting a tolerance.

Proposed gates: zero OOM/restarts/timeouts, peak container memory below 12 GiB,
no sustained request-entry/cleanup baseline growth across the sequential run,
all components present, no clipped audio/text, and full exchanges below every
effective deadline with a reviewed safety margin. Stop escalation at 13 GiB or
on monotonic retained growth; abort the isolated experiment at 14 GiB. Boundary
logs/metric delivery cannot guarantee interception of a sudden transient spike.
Cloud Run/client timeout is not an inference kill switch; never retry blindly.

If synchronous five-minute processing cannot fit the actual end-to-end deadlines,
first establish measured per-stage latency. Coordinated timeout increases need
separate approval and bounded cancellation/admission behavior. If the full path
cannot reliably fit synchronous limits, durable asynchronous analysis jobs with
job IDs, idempotency, status polling, persistence and bounded workers are required.
That changes client/Node/API orchestration and needs separate design approval;
it must not be disguised as a Python-only optimization.

## Remaining blockers

Private-image/staging access and security-exception controls; exact deployed
inventory; isolated build/publication/startup approval; real-model profiling of
Whisper/ModernBERT/acoustic stages; complete timeout inventory and full matrix
results. No production changes, staging activation, model inference, Git staging,
commit or push occurred. Prior 117-test model-free evidence alone cannot close
the five-minute gate. Resource and accuracy observations must precede any rollout.
