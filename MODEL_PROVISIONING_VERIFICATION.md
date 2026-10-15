# Model provisioning local verification — 2026-10-14

**IMPLEMENTED LOCALLY — PROVIDER SELECTION / CLOUD VERIFICATION PENDING.**
No model upload, registry authentication, cloud resource creation or deployment
occurred. All Python changes remain uncommitted and unstaged. Node Docker source
publication is separately authorized; its final SHA is in the task report.

## Baseline inventory and unchanged artifacts

| Component | Files | Runtime bytes |
| --- | --- | --- |
| models/iSpeak_v5 adapter/tokenizer | 9 | 53,861,676 |
| models/iSpeak_v5/base_model | 3 | 967,000,915 |
| models/filler_classifier | 4 | 602,034,859 |
| Complete bundle | 16 | 1,622,897,450 (1.623 GB / 1.511 GiB) |

All sixteen SHA-256 values and exact file lengths are recorded in
`model_manifests/ispeak-v5.json`. That file's SHA-256 in this local preparation is
`419e667990169868bce3867324047a391ea82c6de1dd7531e4b4fe3ad6b43e66`.
The complete original bundle passed the same manifest again after all negative
tests and inference. No weight, config or tokenizer file was edited, copied over,
renamed or deleted. v5 remains the active baseline; no v6 was created.
The earlier `ispeak-python-v2:security-verification` image also passed all 16
hashes when checked with the new read-only verifier/manifest mounts, confirming
the new bundle preserves the previously verified model artifacts.

Existing .gitignore excludes weights, base_model, caches and evaluation output;
it now also ignores the future v6 tree. Existing LFS attributes do not authorize
uploading weights: do not force-add weight files or put them in Git/Git LFS.
.dockerignore deliberately permits only the two trusted runtime model trees,
excluding README/training_args.bin/download caches. The new verifier/manifests
are also permitted. New runtime verification utility files are excluded from
the production image and mounted read-only during tests.

## Docker and local inference

| Check | Result |
| --- | --- |
| docker build --check . | PASS, no warnings |
| Fresh current Dockerfile build | PASS, separate local model-provisioning-local tag |
| Native complete integrity | 16/16 PASS |
| Build-time integrity | 16/16 PASS |
| Runtime pre-start integrity | 16/16 PASS, before Uvicorn/model loading |
| Missing bundle overlay | Startup rejected, exit 1 |
| Corrupted weight overlay | Startup rejected, exit 1 |
| Whisper base + LoRA initialization | PASS, speech model ready |
| Filler classifier | PASS, ready and available on returned analyses |
| Private service authentication | Missing token 401; generated matching token accepted |
| Normal inference | HTTP 200, generated spoken WAV, 30 words / 12.192 seconds |
| Librosa/scoring checks | Finite existing overall/pacing/clarity/energy scores |
| Reference inference | HTTP 200 using the same generated WAV in both fields; acoustic comparison available, finite score and reference WPM |
| External downloads at runtime | Impossible in test network=none; existing offline/local-only loaders preserved |
| Startup including integrity | 22.68 seconds on local Docker, excludes cloud image transfer |
| Native Python regressions | 41 passed (33 existing + 8 integrity failure-path tests) |
| Linux Python regressions | 41 passed in newly built image |
| Native/build/runtime pip check | No broken requirements |
| git diff --check | PASS |
| Image exclusion checks | No .env, evaluation_results.json or training_args.bin |
| Token logging | Generated token absent from logs |
| Shutdown / cleanup | Exit 0; disposable runtime container removed |

Normal/reference tests use synthetic speech for functionality, not human accuracy
evaluation. The reference self-comparison does not validate an independent corpus.
Whisper/filler code and acoustic scoring were not changed, and complete synthetic
response values/transcripts were not logged or committed. Missing/corrupt tests
used temporary read-only mount overlays; no original artifact was modified.
No normal database, user audio or cloud R2 bucket was contacted.

New private local image:

- Tag: `ispeak-python-v2:model-provisioning-local`
- Additional local tag: `ispeak-python-v2:verified-v5-419e66799016`
- Image ID: `sha256:b0a1576720969de095161c6dab8e4909026c6125bd115bea3f25dd6fed3f1a1b`
- Docker image Size: **1,846,236,282 bytes** (1.846 GB / 1.719 GiB).
- Model payload: 1.623 GB. Prior verified expanded filesystem was about 3.84 GB;
  use that only as a planning estimate, not a fresh measurement of this build.
- Budget roughly two model/image versions for rollback; retained images may
  share layers, so do not assume exact doubling of registry or local disk charges.
- Python limits remain 6 GiB minimum, 8 GiB safer, four CPU threads/four vCPU
  starting point and one inference worker. No new capacity guarantee is claimed.
- Earlier local/security/removal image tags remain unchanged and were not overwritten.

## Python files changed, all left for review

Modified:

- Dockerfile: copy verifier/manifest, build check, runtime pre-start entrypoint.
- .dockerignore: permit only verifier and manifest JSON additions.
- .gitignore: exclude future models/iSpeak_v6.
- DOCKER.md: link the proposed provisioning procedure.

Added:

- model_integrity.py
- model_manifests/ispeak-v5.json (hashes and lengths only)
- MODEL_PROVISIONING.md
- MODEL_PROVISIONING_VERIFICATION.md
- testing_script/test_model_integrity.py
- testing_script/verify_model_provisioning.py
- testing_script/verify_provisioned_runtime.py

No changes to fastapi_backend.py, model.py, whisper_service.py, admission, audio
validation, filler/acoustic modules, dependency pins, launchers, model artifacts,
or Python API contracts. `evaluation_results.json` remains ignored and preserved
locally. Legacy Python was excluded.

## Security and decision gates

The source/candidate credential/artifact audit found no secrets or model weights
among Git candidates. A manifest provides integrity relative to a trusted
baseline, not provenance, consent, signature verification or protection if both
image and manifest are replaced by a privileged attacker. Keep private registry
access, immutable digests, trusted reviewers and non-root/read-only model files.

Accelerate 1.14.0 remains unresolved and accepted only under the existing
controlled-thesis disposition. The stricter guard rejects additional checkpoint
indexes/pickle/executable artifacts before model loading, but does not remediate
the installed library. Any retrained model/loading change requires reassessment;
the current exception is not advance approval of v6. No vulnerability suppression
or dependency change was applied.

Option A is recommended for implementation simplicity and atomic rollback;
compare B/C and current provider constraints in MODEL_PROVISIONING.md. Railway
private-image deployment requires approved Pro/budget and private-network setup.
Cloud Run requires approved billing/registry/IAM and resolution of its HTTP/1
32 MiB combined-upload limit. Neither provider was exercised or configured.

The exact next action is **review this Python provisioning diff and choose an
approved provider/private registry and budget**. After that, separately authorize
the Python source commit and private image upload, verify registry privacy and
identity permissions, then perform cloud readiness checks. Live Atlas/R2,
production secrets/ingress/CORS, private Python networking and representative
load/drain checks remain open. No cloud deployment is authorized by this work.

Ignored evidence: `benchmark-results/model-provisioning/verification.json`.
