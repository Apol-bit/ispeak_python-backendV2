# Python V2 Linux container

Build from **this directory**, with the trusted local model artifacts provisioned:

```powershell
docker build -t ispeak-python-v2:local .
```

Required model layout (weights are deliberately absent from Git):

```text
models/
  iSpeak_v5/
    adapter_config.json
    adapter_model.safetensors
    added_tokens.json, merges.txt, normalizer.json
    preprocessor_config.json, special_tokens_map.json
    tokenizer_config.json, vocab.json
    base_model/
      config.json, generation_config.json, model.safetensors
  filler_classifier/
    config.json, model.safetensors, tokenizer.json, tokenizer_config.json
```

Models are baked into the private image. Runtime model loading is local-only and
Hugging Face offline mode is enabled. Do not distribute this image publicly.
There is no static model-file serving route. Download caches, training arguments,
recordings, logs, secrets, Windows virtual environments and the legacy backend
are excluded from the build context.

Start locally with one worker and bounded writable temporary storage:

```powershell
docker run -d --name ispeak-python-v2 --memory 6g --cpus 4 --read-only --tmpfs /tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001 -e ISPEAK_SERVICE_TOKEN -e PORT=8000 -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -p 127.0.0.1:8000:8000 ispeak-python-v2:local
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
The Windows launcher remains unchanged and continues to use `ISPEAK_PORT`.
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

Run the local acceptance harness (requires Docker, the existing safe AAC sample,
the local Python launcher and installed Node dependencies):

```powershell
.\.venv\Scripts\python.exe testing_script\verify_docker_local.py
```

It creates `ispeak-python-v2-verify`, refuses to overwrite an existing container,
binds localhost:8000, and stops the container when finished. It leaves the image
and stopped container for inspection. Results and resource samples are written
under ignored `benchmark-results/docker-local/`. Reference verification uses the
same existing safe recording in both fields; this checks the comparison path,
not scoring accuracy. The Node check exercises the actual axios/form-data
transport and application response validator, without saving a database session.
The restart comparison increases the CPU quota to eight but retains four Torch
threads; it is not a complete eight-thread performance benchmark.

For a separate eight-vCPU/eight-thread measurement after the main run stops:

```powershell
.\.venv\Scripts\python.exe testing_script\verify_docker_local.py --eight-cpu-benchmark
```

This uses localhost:8001 and a separate container, performs one cold-cache
inference followed by two warm inferences, archives results in the `eight-cpu/`
subdirectory, and stops the container. In the recorded local comparison, eight
threads improved warm latency by about 9.7% for this one sample while roughly
doubling average CPU use. Start with four vCPUs; repeat with representative
recordings before paying for eight solely to improve latency.

Run the relevant existing Linux regressions with the test directory mounted
read-only (tests are excluded from the production image):

```powershell
docker run --rm --network none --memory 6g --cpus 4 --mount "type=bind,source=$($PWD.Path)\testing_script,target=/app/testing_script,readonly" -e ISPEAK_ENV=development -e ISPEAK_SERVICE_TOKEN= -e ISPEAK_CORS_ORIGINS= -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 ispeak-python-v2:local python -m unittest testing_script.test_input_validation testing_script.test_admission testing_script.test_readiness testing_script.test_analysis_components testing_script.test_container_admission testing_script.test_security_hardening -v
```

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
