# Python V2 Linux container

The proposed integrity-enforced provisioning path is documented in
[MODEL_PROVISIONING.md](MODEL_PROVISIONING.md). Docker now verifies the reviewed
local bundle during build and before startup. The integrity guard does not change inference behavior. The current Linux release
target is v6; this documentation does not alter the local Windows launcher. These provisioning changes
are left uncommitted for review; no private image or model artifact was uploaded.

Build from a clean release checkout with separately provisioned, manifest-verified
private artifacts. The final/default Docker target is **v6**; select it explicitly
for release builds. The example tag must be unused:

```powershell
docker build --target v6 -t ispeak-python-v2:v6-release-candidate .
```

Required model layout (weights are deliberately absent from Git):

```text
models/
  iSpeak_v6/
    adapter_config.json, adapter_model.safetensors
    processor_config.json, tokenizer_config.json, tokenizer.json
  iSpeak_v5/
    base_model/
      config.json, generation_config.json, model.safetensors
  filler_classifier/
    config.json, model.safetensors, tokenizer.json, tokenizer_config.json
```

The v6 manifest covers 12 runtime artifacts: five adapter files, three shared
base files, and four filler files. The shared base remains under iSpeak_v5/base_model;
that path does not select the v5 adapter. Both reviewed manifests are source files.
Private weights and the v6 artifact tree remain outside Git; provision them securely
in the build context. A source-only checkout cannot build the model image unaided.

Preserve the legacy ispeak-python-v2:local image and the hardened rollback image
**ispeak-python-v2** (implicitly :latest). Never build, retag, or publish over either.
The hardened rollback's recorded local ID is
`sha256:787831cdcbf055eac5b707226dd648f49bb712724cb1696ef18a24744e157a14`.
Use [SECURE_V5_ROLLBACK.md](SECURE_V5_ROLLBACK.md) for rollback behavior and
[MODEL_PROVISIONING.md](MODEL_PROVISIONING.md) for target-specific provisioning.
Any later v5 build must use --target v5 and a new unused version tag.

Models are baked into the private image. Runtime model loading is local-only and
Hugging Face offline mode is enabled. Do not distribute this image publicly.
There is no static model-file serving route. Download caches, training arguments,
recordings, logs, secrets, Windows virtual environments and the legacy backend
are excluded from the build context.

Start locally with one worker and bounded writable temporary storage:

```powershell
docker run -d --name ispeak-python-v2 --memory 6g --cpus 4 --read-only --tmpfs /tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001 -e ISPEAK_SERVICE_TOKEN -e PORT=8000 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -p 127.0.0.1:8000:8000 ispeak-python-v2:v6-release-candidate
curl.exe http://localhost:8000/health
```

Use 8 GiB RAM as a safer initial allowance; 6 GiB is a planning minimum, not a
guarantee for every accepted recording. Long inputs and reference analysis must
be assessed separately. TMPDIR, Numba cache and Hugging Face cache share
`/tmp/ispeak`; the example limits their combined usage to 512 MiB. A tmpfs counts
toward container memory. Without a tmpfs, the image's precreated directory is
writable by UID 10001, but the runtime must supply a disk quota or monitoring.
Changing the cache directory does not change analysis results or thresholds.

Before starting, privately provision a high-entropy `ISPEAK_SERVICE_TOKEN` of at
least 32 characters in the process/host secret environment, with the same value
as Node `PYTHON_SERVICE_TOKEN`. The command forwards the variable by name; do not
put its value in shell arguments or reports. No production token was generated
by this task. Production startup fails without it. `/transcribe` and `/metrics`
require `X-iSpeak-Service-Token`; minimal health/liveness probes remain unauthenticated.
Keep Python private to Node even with service authentication. No public domain
or TCP proxy is needed for Python.

The image defaults to production mode and an empty browser CORS allowlist.
Private server-to-server Node requests work without browser CORS. Set
`ISPEAK_CORS_ORIGINS` to explicit origins only if browsers must call this service.
Do not configure `*` in production. Any configured production origins must be
exact HTTPS origins. Keep cloud networking private to Node.

`PORT` takes precedence, then `ISPEAK_PORT`, then 8000. The Docker command expands
these variables through a shell and uses `exec` so Uvicorn receives stop signals.
These release instructions do not change the Windows launcher; it continues to use `ISPEAK_PORT`.
Uvicorn binds `0.0.0.0` inside Linux and uses exactly one worker. Its admission
defaults remain one active analysis, four waiting requests and a 30-second queue
timeout. Keep `ISPEAK_MAX_CONCURRENT_INFERENCES=1` initially.

Readiness: `/health` and `/ready` return 200 only when the speech model and filler
classifier are usable; otherwise they return 503. Uvicorn does not begin accepting
connections until lifespan model loading completes. `/live` is a liveness route
once the server is listening. The Docker health check allows a 120-second startup
grace period. Configure the host readiness probe for `/health`, with adequate
cold-start allowance; `EXPOSE 8000` is metadata and does not restrict dynamic PORT.

The base is Python 3.13.12 on Debian Bookworm. OS dependencies are FFmpeg (AAC and
other decoder fallback), libsndfile1 and libgomp1. Python versions remain those in
`requirements.txt`, including PEFT dependencies Accelerate 1.14.0 and psutil 7.2.2;
`requirements-linux.txt` includes those shared pins. Torch 2.13.0 is installed
from the official CPU wheel index. The build runs `pip check`.

Use the current explicit-fixture acceptance commands below for this release.
The older verify_docker_local.py workflow predates the v6 release packaging and
is not the release or hardened-rollback verification procedure. Do not use it
to rebuild or overwrite a preserved image. Tests are excluded from production
images and are mounted read-only by the versioned-image harness.

Accelerate remains an unresolved advisory, retained because PEFT imports it at
runtime. The focused runtime/removal audit supports **ACCEPTED RISK — CONTROLLED
THESIS DEPLOYMENT ONLY**, under the controls and expiry in
[ACCELERATE_PRODUCTION_DISPOSITION.md](ACCELERATE_PRODUCTION_DISPOSITION.md).
This is a separate scoped disposition from the earlier UAT exception, not a fix
or blanket production/deployment approval. If its required controls cannot be
met, the advisory remains a blocker. Public exposure or new model/loading paths
require reassessment.
The security follow-up targeted AnyIO 4.14.2, fsspec 2026.6.0 and urllib3 2.8.0,
with isolated regression/startup/inference verification. Accelerate was not
changed. See the backend `SECURITY_READINESS.md` for the current security gate.
Production transcript/word logging is disabled; development detail requires
`ISPEAK_DEBUG_SPEECH_LOGS=true` and a DEBUG logger. Detailed error traces require
`ISPEAK_DEBUG_ERRORS=true` in development. These flags cannot enable production
speech/error details. No deployment, commit, push or image upload occurred.

## Explicit synthetic fixtures for current acceptance harnesses

Provision WAVs separately, outside Git. Supply --samples with a JSON manifest
whose file paths are relative to its directory. Absolute paths, parent traversal,
missing/invalid WAVs, non-synthetic purposes, and incomplete language coverage are
rejected before model imports or Docker operations. This declaration records the
operator's synthetic-fixture assertion; it cannot prove audio provenance.

```json
[
  {"language":"English","purpose":"synthetic compatibility only","file":"english.wav"},
  {"language":"Filipino","purpose":"synthetic compatibility only","file":"filipino.wav"},
  {"language":"Taglish","purpose":"synthetic compatibility only","file":"taglish.wav"}
]
```

Validate fixtures without inference, containers, model loading, or output files:

```powershell
python -B testing_script/verify_local_activation.py --samples <private-fixtures>/samples.json --validate-only
python -B testing_script/verify_v6_docker.py --samples <private-fixtures>/samples.json --validate-only
```

Omitting --validate-only authorizes expensive local inference when the operator
separately chooses to run the harness; do not run it as a documentation check.
The activation harness uses model-path/manifest variables supplied privately in its
process environment; it does not change the launcher or persistent environment.
The Docker harness requires the recorded existing v6-local image and hardened
ispeak-python-v2 rollback image, checks both immutable local IDs before startup,
and performs no builds or retags. A newly built candidate requires a separately
reviewed identity update before that harness can accept it. Its --network none
containers read fixtures through read-only mounts and write reports only under
ignored benchmark-results. Never provision real user recordings as these fixtures.
