# Completed-chunk Whisper diagnostic retention

Reported staging observation: 79.893 seconds, no reference audio; container
memory approximately 2.69 GiB before Whisper and 20.42 GiB afterward; Whisper
approximately 160 seconds. This local patch has NOT been measured with models.

In the installed Transformers 5.10.1 implementation, `_set_return_outputs`
forces attentions and scores for token timestamps. Generation accumulates
per-token decoder/cross attentions and vocabulary-sized scores.
`_postprocess_outputs` computes token timestamps, then copies/slices these
diagnostics into per-batch result dictionaries. `generate_with_fallback` uses
scores for its retry/no-speech decisions. `_retrieve_segment` subsequently stores
the complete dictionaries as each segment's `result` in `current_segments`.
Earlier chunks consequently remain reachable while later chunks decode.
The pipeline's word-timestamp path consumes sequences and segment timestamps,
not those scores/attentions. The previous selected-head concatenation fix did
not remove these retained originals.

The instance-scoped hook calls the complete original `generate_with_fallback`
first, then removes scores, logits, attention/hidden-state diagnostics and cache
references from completed long-form word-timestamp result dictionaries. It
retains sequences, timestamps, beam indices, skip/conditioning decisions and
all other metadata. It operates only in a context around iSpeak's existing ASR
pipeline invocation. Other direct generation calls, short-form calls, decoding
settings, model weights, timestamp extraction and scoring are unchanged.
The context resets on exceptions. Version and source fingerprints guard the
generation/fallback/segment/pipeline implementations; unsupported versions
remain unchanged and startup reports activation as false.

Synthetic tests exercise the real upstream fallback loop with a forced retry,
real segment extraction including multiple timestamp boundaries, batch result
retention, positional arguments/defaults, short-form and non-word exclusions,
exception propagation and weak-reference cleanup. Ten small synthetic chunk
results retained 656,040 tensor payload bytes before compaction and 680 after;
all discarded tensor weak references expired. These are reachable tensor bytes,
not process RSS/cgroup measurements. No inference or model construction occurs.

Local validation: 134 model-free tests passed with diagnostics disabled and 134
with diagnostics enabled, including seven new retention regressions. The
installed-version security test was excluded from these passing totals: the
unchanged local `.venv` still has certifi 2026.4.22, filelock 3.29.0 and setuptools
83.0.0, so its three target-version assertions fail. No packages were installed;
the candidate image must independently reverify its approved patched inventory.

This bounds completed-chunk diagnostic retention, NOT the current chunk's peak,
allocator RSS, output sequence length, or total process memory. It does not
prove which tensor family accounts for the observed 17.73 GiB increase, nor
resolve the reported insufficient-speech result. A real-model before/after
comparison remains mandatory: same approved audio and configuration, exact
tokens/word timestamps, transcript/score compatibility, stage/cgroup peaks,
latency and sequential-request cleanup. Include 30/60/120/180/240/300-second
English and Filipino recordings, reference and non-reference paths. Stop if
activation is false, memory is unsafe or results differ.

Normal and v6 security-overlay Docker COPY/context rules already include
`audio_analysis_processing_files/*.py`. A future immutable CI build must also
regenerate its source-hash manifest for the newly approved commit; the existing
1f000522 candidate cannot contain this uncommitted change. No build, Git
mutation, deployment, dependency installation or production change is part of
this patch.
