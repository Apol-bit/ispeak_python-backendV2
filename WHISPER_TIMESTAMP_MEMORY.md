# Selected-head Whisper timestamp allocation reduction

Baseline: b06cf3d8eb9beb95496d5f74e03cb99ffea0f843. Local candidate only;
no model inference, installation, build, publication or deployment performed.

## Confirmed source allocation

Installed Transformers 5.10.1 WhisperGenerationMixin._extract_token_timestamps
concatenates each decoder layer's full head tensors across generation steps,
then stacks only the configured alignment heads. The full-layer concatenations
are avoidable copies: each selected head's values can be viewed first and then
concatenated without changing their order or numerical values.

The new instance-scoped adapter supplies these single-head views to the unchanged
upstream routine through a lightweight proxy. It preserves alignment-head order
and duplicates, sequences, beam indices, time precision, frame cropping and prompt
length. Normalization, median filtering, head averaging, DTW and timestamp output
construction remain upstream. It does not mutate inputs or configuration, copy
model parameters, change generation flags or retain per-request tensors.

Activation requires Transformers 5.10.1 and the SHA-256 fingerprint of the reviewed
upstream extraction method. Unknown versions/methods use upstream unchanged.
The hook is installed after PEFT merge/eval and before pipeline construction;
library files and other model instances are not patched. Verify activation on the
actual loaded model during separately approved staging startup; a source-version
match alone does not establish production-image equivalence.

## Model-free evidence

Synthetic attention fixture: 4 layers, 8 heads, batch 1, 5 total token positions,
40 audio positions, float32; two selected alignment heads. Instrumentation of
torch.cat measures its dim=2 output tensors, not total process/container memory:

- Upstream concatenation output: 25,600 bytes.
- Selected-head concatenation output: 1,600 bytes (16x reduction).

The general concatenation volume falls from L*H*B*T*S*element_size to
K*B*T*S*element_size, where K is the number of selected alignment pairs. This is
not a prediction of total Whisper RAM savings. Attention tensors produced by
generation, selected-head stacking, filtering and alignment allocations remain.

Focused tests compare bit-identical timestamps against the actual installed
upstream method for prompt/prefill lengths including empty alignment output,
head ordering/duplicates, single/multiple batches, integer/list/tuple/tensor frame
lengths, unequal frame lengths and beam gathering. Twelve sequential invocations
verify input identity/content preservation and weak-reference release of request
attention tensors. Unsupported version/method checks preserve upstream behavior.
All five focused tests passed. The guarded model-free regression suite passed
128 tests with diagnostics disabled and 128 with diagnostics enabled. A NumPy
deprecation warning in the unchanged upstream tensor-frame handling remains;
it did not affect timestamp equality or test success.
No model is constructed: owners are SimpleNamespace fixtures and inputs are small
random tensors. This cannot establish real transcription or five-minute stability.

## Unresolved long-form retention

Upstream _postprocess_outputs splits cross/decoder attentions into each result;
_retrieve_segment stores that result in long-form segment records. In CPU mode,
.cpu() can preserve underlying storage instead of creating an independent small
object. Pipeline word-timestamp mode asks for segments. Raw attention results may
therefore remain reachable across segments until generation finishes. Whether
this explains the production 16 GiB event remains unmeasured.

This candidate does NOT discard these results or alter fallback/segment contracts.
It also does not bound the attentions emitted during generation itself. Releasing
unneeded long-form results needs separate consumer/fallback equivalence analysis
and allocation measurements; disabling attentions/timestamps, chunking changes,
precision changes, inference-mode changes and approximate alignment are excluded.

## Isolation and next gates

The module is inside audio_analysis_processing_files, already copied and admitted
by normal, source-overlay and security-overlay Docker recipes. No packaging edits
are required. Existing scores, languages, weights, decoding and API are unchanged.

Before real-model validation: focused release review, approved isolated image and
adequate memory host (not the 7.666 GiB Docker environment), verified hook activation,
then measured English/Filipino 30/60/120/180/240/300-second reference/non-reference
cases, cold/warm and sequential requests. Compare complete transcripts/timestamps
and scores with the unchanged baseline; record RSS/cgroup peaks and time budgets.
The production OOM cause, segment retention impact, full-request savings and full
duration latency remain unverified. No claim of production readiness is made.
