# Hardened v5 rollback image

Built locally with the existing Dockerfile v5 target and current secure application:
`docker build --target v5 -t ispeak-python-v2:latest .`.
The requested bare name ispeak-python-v2 resolves to :latest. For controlled release,
select the immutable digest rather than relying on a mutable tag.

Image ID: `sha256:787831cdcbf055eac5b707226dd648f49bb712724cb1696ef18a24744e157a14`.
Original ispeak-python-v2:local remains unchanged at
`sha256:e0b99b90ff0103d092d8074e93f9e6a5d95fd20e8924d022e98cbcf9ec5768e8`.

The new image selects v5 adapter, trusted schema-1 v5 manifest, existing Whisper-small
base and ModernBERT filler model. All 16 required artifacts verified. CPU-only, one
worker, Hugging Face offline settings and runtime integrity entrypoint remain enabled.
Production private endpoints require a runtime service token; transcriptions and the
test token were absent from logs. Authorized synthetic English, Filipino and Taglish
requests returned valid timestamps, filler availability and acoustic scores. SIGTERM
shutdown exited 0. No model download was possible in the offline test network.

51 Linux regressions passed in their development test configuration; live production
authentication was verified separately. Native suite passed 51 tests. No dependency
versions changed. Accelerate's existing accepted-risk controls remain applicable;
this image does not fix that advisory.

Startup: 30.76 s. Idle: 1.417 GiB. Sampled inference peak: 5.372 GiB. Peak CPU:
408.86% on four CPUs (100% equals one CPU). Use the same safer 8 GiB allocation as
v6. These are local synthetic compatibility measurements, not WER/CER evidence.

Compose rollback requires only PYTHON_IMAGE=ispeak-python-v2:latest, followed by
recreating Python and checking readiness and authenticated analysis. Do not export
v6 model-path variables in the runtime env file: image defaults select the appropriate
adapter and manifest together. Rollback does not replace model files or inference code.
No cloud deployment, image publication, model upload, commit or push was performed.
Private evidence remains under benchmark-results/v6-docker/v5-secure-results.json.
