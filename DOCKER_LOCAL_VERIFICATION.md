# Python V2 Docker/Linux local verification — 2026-10-09

Publication review, 2026-10-14: the later scoped decision in
[ACCELERATE_PRODUCTION_DISPOSITION.md](ACCELERATE_PRODUCTION_DISPOSITION.md)
supersedes this report's earlier statements that Accelerate's disposition is
open. The advisory remains unresolved and is accepted only for controlled thesis
deployment under that document's controls. All other cloud verification gates
remain open. This report records the historical local verification below.

**Packaging status: IMPLEMENTED — CLOUD DEPLOYMENT VERIFICATION PENDING.**
The CPU Linux image built and passed local functional verification. Production
security review remains open for Accelerate. No deployment, commit, push or image
upload was performed. The legacy backend was excluded. Both verification
containers are stopped; the local image and stopped containers remain reviewable.

## A–Z results

| ID | Area | Result |
| --- | --- | --- |
| A | Docker readiness | Local Linux build and inference PASS; cloud verification pending. Production security disposition remains open. |
| B | Exact files changed | `fastapi_backend.py` modified; added `Dockerfile`, `.dockerignore`, `requirements-linux.txt`, `DOCKER.md`, this report, `testing_script/verify_docker_local.py`, `testing_script/test_container_admission.py`. No scoring/model/Node/legacy source changed. |
| C | Dockerfile | CPU Torch, pinned runtime, local models, UID/GID 10001, `/app`, production mode, offline HF, writable temp/cache configuration, health check, `0.0.0.0`, one Uvicorn worker. Shell expansion and `exec` support dynamic PORT and shutdown signals. |
| D | .dockerignore | Default-deny allowlist for runtime modules, manifests and the two required model trees. Excludes Git, Windows venv/startup/editor files, recordings, tests/results, .env/secrets, logs, bytecode, model download caches and training arguments. |
| E | Base image | `python:3.13.12-slim-bookworm`, linux/amd64; resolved base digest `sha256:a58daefb915e1e03ad48f3ca4df8832065412c5c35cacb9d39f4229184de12b6`. |
| F | OS dependencies | FFmpeg, libsndfile1, libgomp1; Debian installs their required libraries. FFmpeg successfully decoded the existing AAC sample. |
| G | Python/dependencies | Python 3.13.12, Torch 2.13.0+cpu, Transformers 5.10.1, PEFT 0.19.1, Librosa 0.11.0, Numba 0.65.1, llvmlite 0.47.0, FastAPI 0.136.1, Uvicorn 0.46.0. Accelerate 1.14.0 and psutil 7.2.2 preserved explicitly. Build and runtime `pip check`: no broken requirements. No CUDA packages. |
| H | Model provisioning | Trusted iSpeak_v5 LoRA/tokenizer, bundled Whisper-small base, and local filler classifier baked into the private image. All 16 included files SHA-256 match the host artifacts. Offline runtime; no model API/download dependency or static file-serving route. |
| I | Image size | Docker inspect content size **1,846,165,107 bytes (1.85 GB / 1.72 GiB)**. Expanded filesystem about **3.84 GB / 3.57 GiB**. Docker Desktop image disk accounting **5.69 GB**, including stored content and expanded layers. Models 1.623 GB / 1.51 GiB; installed Python packages about 1.624 GB / 1.51 GiB. |
| J | Cold start | Four-vCPU/four-thread model-ready startup **17.25 s** after `docker run` returned; restart **18.17 s**. Separate eight-thread startup **14.19 s**. Process-cold measurements on an already built local image; cloud image transfer and fully cold storage were not measured. |
| K | /health | 200 `ready` only after speech and filler loading. Both report available, local Linux paths, no errors; filler `network_allowed=false`. Before lifespan finished the port did not accept requests. Linux readiness regressions confirm 503 for missing speech or filler models. PORT=8080 overrides ISPEAK_PORT=8999; localhost:8000 mapped to 8080. |
| L | Real inference | POST `/transcribe` 200; existing 206,810-byte AAC, decoded duration 16.765 s, 18 timestamped words. iSpeak_v5 LoRA loaded/merged, Whisper executed, contextual filler classifier available, Librosa metrics returned. Scores: pacing 53.7, clarity 94, energy 84, overall 77.4. First inference **49.04 s**, including first-use compilation. Functional evidence only, no accuracy claim. |
| M | Reference audio | Both multipart files accepted; same safe AAC reused as reference for a controlled self-comparison. 200 in **41.22 s**; both analyses and MFCC/delta-MFCC DTW executed. Acoustic similarity 100, transcript similarity 1, comparison applied. No independent reference corpus was evaluated. |
| N | Invalid inputs | Malformed WAV 400; 0.25-second WAV 400 `recording_too_short`; silence and pure tone 400 `insufficient_speech`; 26 MiB upload 413. Short/silent/tone checks took 0.01–0.03 s. Existing Linux mock checks prove these inputs do not call Whisper. |
| O | Admission/queue | Defaults unchanged: 1 active, 4 queued, 30 s timeout. Six simultaneous real requests: 2 completed, 3 timed out near 30 s, 1 rejected immediately with 503. Peak active=1, peak admitted=5; all slots released, health ready. Controlled Linux route tests also verify 500 failure releases its slot and later requests succeed. |
| P | Loaded-model RAM | Docker stats **1.302 GiB** after readiness; raw cgroup **1.304 GiB**. Startup sampled peak about 1.31 GiB. Lower than historical Windows loaded 2.39 GiB; platforms/accounting differ and this excludes first-use inference/JIT allocations. |
| Q | Peak inference RAM | First single inference Docker-style peak **4.212 GiB**; raw kernel peak **4.229 GiB**, close to historical host 4.13 GiB. After settling **3.096 GiB**. Reference peak 4.403 GiB; burst peak 4.519 GiB excluding inactive file cache, **4.722 GiB raw cgroup**. |
| R | CPU | Four-thread cold request mean 177% / peak 404%; warm burst mean 316% / peak 405%, where 100%=one CPU. Four-vCPU warm processing 17.203/15.880 s; eight-vCPU/eight-thread warm processing 15.417/14.453 s, mean CPU 625%, peak 821%. About **9.7% latency improvement** for this sample at roughly twice the CPU use. Four vCPUs are practical for one active inference; broad eight-vCPU benefit is not established. |
| S | Temp/cache | Read-only service passed with writable 512 MiB tmpfs at `/tmp/ispeak`, owned by 10001. TMPDIR, NUMBA_CACHE_DIR and HF_HOME use this location; no package bytecode write dependency. No temporary audio files remained after inference/reference/invalid/queue/restart. Sample upload copies peaked at 206,810 bytes single, 413,620 reference, 1,034,050 burst; retained Numba cache about 2.88 MiB. |
| T | Node→container | Actual installed Node axios/form-data transport and application `usableAnalysis` validator PASS: 200, usable=true, all scores present. PYTHON_BACKEND_URL supplied only to test process. Full authenticated Node HTTP route plus Mongo/session persistence NOT TESTED; no Node config/source changed or Node deployment. |
| U | Restart | Stop/start same container, models reloaded, ready after 18.17 s; POST 200 after 45.95 s including fresh tempfs/JIT cache. Complete response exactly matched initial inference. No prior temporary-file dependency. |
| V | Security | UID 10001, read-only verification filesystem, localhost-only published ports. Artifact inspection found no .env, private keys, logs, recordings, training binaries, Git or download caches in `/app`; allowlist build never copies Node credentials, Mongo/R2/JWT secrets or the legacy backend. Models are root-owned/readable and application paths are not writable to the runtime user. This is an artifact/configuration inspection, not a complete image CVE audit. |
| W | Accelerate | Preserved 1.14.0; required by PEFT metadata and import-time runtime imports. Advisory GHSA-4j2p-28q2-5m79/CVE-2026-69112 remains a production issue. Upstream 1.15.0 exists, but its inspected loader still directly joins untrusted shard names; no verified fix identified. UAT-only exception does not authorize production. |
| X | Railway RAM/CPU | Planning minimum **6 GiB**, safer **8 GiB**, **4 vCPUs**, one worker. Allow at least 120 s startup grace initially, despite local readiness in 14–18 s; first inference was 46–49 s. Reserve capacity for roughly 3.84 GB expanded image and bounded cache/temp space; a paid plan is required for this RAM target. |
| Y | Remaining blockers | Production Accelerate security disposition; actual Railway build/provisioning/private networking/readiness/resource/long-audio verification; full authenticated Node/Mongo flow. Trusted non-Git model files must reach the cloud build context through an explicitly authorized private artifact workflow. Nothing has been uploaded. |
| Z | Packaging P0 | Can change to **IMPLEMENTED — CLOUD DEPLOYMENT VERIFICATION PENDING** for packaging. Overall production deployment remains gated by W/Y; this is not cloud verification or production security approval. |

## Audit and validation details

Before editing, the active repo was clean. The Windows `.venv` used Python
3.13.12 and the same semantic package versions above. `requirements.txt` omitted
Accelerate and psutil even though PEFT requires them; the Linux overlay pins the
already installed versions, without changing the Windows requirements.
`PeftModel` imports Accelerate helpers directly, so removal would break runtime
imports. Current application code does not call the affected checkpoint APIs and
uses trusted local unsharded safetensors. Model-path metadata does not replace the
explicit local base/adapter paths. Runtime source has no absolute Windows drive,
PowerShell startup or `.venv/Scripts` dependency. Existing path handling and temp
cleanup required no edits.

The existing Windows command remains `start_backend.ps1`, which launches
`.venv/Scripts/python.exe -m uvicorn fastapi_backend:app` with ISPEAK_HOST and
ISPEAK_PORT (localhost/8000 defaults). Container startup uses a separate Linux
command with one worker and `0.0.0.0`, supporting PORT first, then ISPEAK_PORT.
The Python configuration mirrors this precedence. API routes/contracts, scoring,
thresholds, transcription settings and model loading/merging were preserved.

Relevant tests ran inside Linux with test sources mounted read-only and network
disabled: **27 tests passed in 40.358 s**. This comprises 25 existing tests across
input validation, admission, readiness and analysis components, plus two new
route-level controlled-worker tests. The controlled tests exercise existing
production code with a short test-only queue timeout; runtime defaults remain 30 s.

Build record: initial attempt failed after 144.03 s due to transient DNS failure
for download.pytorch.org; a retry also encountered registry DNS failure. Host and
Linux DNS subsequently recovered without network-setting changes. The successful
clean Linux dependency/model build took **331.53 s**; final cached configuration
build **1.46 s**. CPU Torch installation and runtime `pip check` both passed.
Dependency layer about 1.62 GB, model layers about 1.62 GB, OS dependency layer
about 466 MB, base about 136 MB. Docker build cache occupies additional local
disk and is not part of the deployable image. No global cache pruning occurred.

Resource sampling used cgroup v2 at approximately 0.5-second intervals, with
`memory.current - inactive_file` for Docker-style used memory, plus raw current
and kernel peak values. Docker stats independently confirmed loaded memory.
Sampling includes the lightweight monitor process. Brief subsecond CPU readings
can exceed the nominal quota; means and sustained activity are more useful.
The eight-thread comparison used a separate container after the main container
stopped, with the same image, audio and memory limit. Both eight-thread warm
responses exactly matched the initial response. Two runs of one short sample
are insufficient to generalize throughput/accuracy or guarantee longer-input
memory limits. Retained allocator/JIT memory explains why post-inference RAM is
higher than initial model-ready RAM.

The verification harness initially used Windows default console decoding for
Docker's UTF-8 progress stderr. Inference/results were unaffected, but log capture
emitted decoder-thread warnings. The harness now explicitly decodes UTF-8 and
archives both stdout and stderr; full container logs were recaptured successfully.

## Sources and next deployment considerations

- [Accelerate advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79): affected through 1.14.0, no patched version listed when checked.
- [Accelerate 1.15.0 release](https://github.com/huggingface/accelerate/releases/tag/v1.15.0) and [its loader source](https://raw.githubusercontent.com/huggingface/accelerate/v1.15.0/src/accelerate/utils/modeling.py): newer release exists, but the inspected `weight_map` join is still unguarded. A version outside the advisory range alone is not evidence of remediation.
- [Railway plan resources](https://docs.railway.com/pricing/plans): currently lists Hobby with 100 GB image/ephemeral-storage limits and sufficient maximum RAM/CPU; Free/Trial memory limits cannot meet this service's requirement. These are plan ceilings, not guaranteed performance or an account configuration check.

Baking models remains practical at the measured size. A persistent volume or
private artifact storage can reduce rebuild transfers later, but adds provisioning
and readiness dependencies; no redesign is needed for initial thesis packaging.
Because weights are not tracked, a Git-only build will not have them. Provision
the trusted files into the authorized build context before any eventual build,
or use a separately approved private image/artifact workflow. Keep the Python
service on private Node→Python networking, configure `/health` readiness and one
worker, and validate representative long/reference inputs before changing limits.

Reviewable local evidence is under `benchmark-results/docker-local/` (ignored):
`results.json`, `samples.json`, `artifact-check.json`, full container logs,
eight-thread results/samples/logs, build logs/results and the exact unittest log.
See `DOCKER.md` for repeatable build/run/verification commands. No secrets or
model weights have been added to Git.
