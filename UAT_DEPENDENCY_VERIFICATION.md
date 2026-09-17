# Transformers upgrade and controlled-UAT advisory exception

Verified: 2026-09-16. No deployment was performed.

## Change and rollback

- Transformers installed version and requirements pin: `5.5.0` -> `5.10.1`.
- The final resolver dry run proposed only `transformers-5.10.1`.
- Installation used the downloaded wheel with `--no-deps --no-index`.
- Comparing all 88 installed distributions before and after showed exactly one
  version change: Transformers. PyTorch `2.13.0`, PEFT `0.19.1`, Accelerate
  `1.14.0`, tokenizers `0.22.2`, and huggingface-hub `1.31.0` are unchanged.
- No CUDA installation, model, application code, or .env changes were made.
  SHA-256 hashes of all 18 existing model files matched before and after.
- This task changed only the Transformers line in `requirements.txt` and added
  this report. Other pre-existing working-tree changes were preserved.

Before upgrading, the old/new wheels and a copy of the current requirements were
saved outside the repository at:

```text
C:\Users\Apol\AppData\Local\Temp\ispeak-transformers-rollback-1a6ed09b62104272b88c73b1f0aad63c
```

No rollback was needed. While that temporary folder remains available, rollback
from the Python V2 directory is:

```powershell
.\.venv\Scripts\python.exe -m pip --isolated install --no-deps --no-index 'C:\Users\Apol\AppData\Local\Temp\ispeak-transformers-rollback-1a6ed09b62104272b88c73b1f0aad63c\transformers-5.5.0-py3-none-any.whl'
```

Then restore only the Transformers requirement to `transformers==5.5.0`, run
`pip check`, and repeat the affected regression checks. The requirements backup
is `requirements.before.txt`; do not overwrite subsequent unrelated edits with
that entire backup. Temporary folders can be removed by OS cleanup.

## Regression results

| Check | Result | Evidence |
| --- | --- | --- |
| Dependency consistency | PASS | `python -m pip check`: no broken requirements |
| Strict setup | PASS | `python check_setup.py --strict` |
| Existing analysis regressions | PASS | 12 tests in `testing_script.test_analysis_components` |
| Normal backend startup | PASS | `start_backend.ps1` started Uvicorn on localhost |
| HTTP health | PASS | `/health` returned 200 and `ready`; speech and filler models available |
| Whisper base / LoRA adapter loading | PASS | Local iSpeak_v5 adapter and bundled Whisper base loaded on CPU |
| Real transcription | PASS | Existing `temp_audio_folder/temporary_audio.aac`, 18 words, 16.765 seconds |
| English mode | PASS | Real request; complete response identical to pre-upgrade baseline |
| Filipino mode | PASS | Real request; complete response identical to pre-upgrade baseline |
| Taglish mode | PASS | Real request; complete response identical to pre-upgrade baseline |
| Word timestamps | PASS | Nonempty, finite, ordered, start <= end; identical to baseline |
| Filler classification | PASS | Local contextual-classifier test and real-inference response |
| Speech metrics | PASS | Finite 0-100 scores; pacing, articulation and vocal metrics present; full responses unchanged |
| Concurrency / queue / error recovery | PASS | Five controlled ASGI regression checks described below |
| Model preservation | PASS | All 18 file hashes unchanged |
| Installed-package preservation | PASS | Only Transformers changed |
| Raw pip-audit | FAIL / accepted UAT exception | Exactly one advisory remains, in Accelerate; Transformers advisory is gone |

Baseline requests ran on 5.5.0, followed by the same requests on 5.10.1. SHA-256
hashes of sorted complete JSON responses matched for each language:

| Mode | Response SHA-256 (same before and after) |
| --- | --- |
| English | `9e38855a7cc84a7474d3a875182d6c59222123905b6be9892f07f9bb29e16db7` |
| Filipino | `596d3ca5951c74b928cf20e8d1f1eef72a37ee26b9374f1f2cd715212a1207c5` |
| Taglish | `084ef3a7c321500fac7494763de3c49b6f17fa7a0e5db672057121227fdc391c` |

The concurrency checks used the real `/transcribe` route with controlled analysis
workers, without loading models in that test process. They verified:

1. Default configuration is one active inference and a 30-second queue timeout.
2. Capacity one queues a second request and completes it after release.
3. Capacity two queues a third request, never exceeding two active workers.
4. A blocked request returns HTTP 503 after a test-only one-second timeout;
   the active request succeeds and subsequent requests recover.
5. An analysis exception returns HTTP 500 and releases the slot for the next request.

Concurrency/timeout overrides existed only in the test process. No configuration
file or application source was altered. All five checks passed. Python emitted
multipart `SpooledTemporaryFile` ResourceWarnings in this controlled test; the
test path does not import Transformers, so this is an independent upload-resource
cleanup concern, not an upgrade regression. It was left unchanged under this task.

Limits: all three language modes used the same existing AAC recording. These
checks establish regression compatibility, not accuracy on a representative
English/Filipino/Taglish corpus. Physical-device and user acceptance evaluation
remain separate. GPU/CUDA inference was not exercised.

## Temporary accepted Accelerate advisory — controlled thesis UAT only

- Package kept unchanged: `accelerate==1.14.0`, transitive through PEFT.
- IDs: `PYSEC-2026-3804`, `CVE-2026-69112`, `GHSA-4j2p-28q2-5m79`.
- Issue: traversal / denial of service through untrusted sharded-checkpoint
  `weight_map` paths in `load_checkpoint_in_model` / `load_checkpoint_and_dispatch`.
- Disposition: temporarily accepted for controlled thesis UAT by the task owner's
  explicit instruction; not suppressed in pip-audit and not a claim of remediation.
- Rationale: no verified upstream fixed release was identified. Application code
  does not call those checkpoint-loading functions; current runtime uses existing
  local, unsharded safetensors weights. Audio inputs cannot select a checkpoint.
- Conditions: retain the verified model artifacts and dependency environment;
  prevent UAT participants from replacing model files; do not introduce untrusted
  models, sharded checkpoints, or new model-loading/export workflows during UAT.
- Review this exception before any public/production deployment, model-source or
  loading-path change, dependency reinstall, and when a verified upstream fix is
  available. It expires with the current controlled thesis-UAT phase.
- Source: https://github.com/advisories/GHSA-4j2p-28q2-5m79

The fresh unfiltered scan found one vulnerability in one package and exited 1.
The Transformers advisory `PYSEC-2026-3929` / `CVE-2026-9856` /
`GHSA-xrqw-3rrv-vx5w` no longer appears.

Python V2 can proceed to the next controlled thesis deployment/UAT preparation
task under this exception. This result does not authorize deployment or override
other project readiness gates.
