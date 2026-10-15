# Private model provisioning — prepared 2026-10-14

**Recommendation: Option A, a private immutable image containing the trusted
models**, subject to approval of the provider, registry privacy and budget.
No provider account, billing, registry, artifact upload or deployment is assumed
or activated. Python changes remain uncommitted for review.

The verified baseline stays in `models/iSpeak_v5`. This task reads and hashes its
files; it does not replace, move or modify them. Model loading, Whisper/LoRA
merge, filler classification, Librosa scoring and response behavior are unchanged.
The new guard runs outside inference, before container startup.

## Current v6 release and preserved v5 rollback

The current Dockerfile's final/default target is **v6**. Build it explicitly with
--target v6 into a NEW unused version tag, for example
`ispeak-python-v2:v6-release-candidate`. Never overwrite or retag the original
`ispeak-python-v2:local` or the hardened `ispeak-python-v2` (implicit :latest).
The hardened rollback ID is
`sha256:787831cdcbf055eac5b707226dd648f49bb712724cb1696ef18a24744e157a14`;
its verification and model-path precautions remain in
[SECURE_V5_ROLLBACK.md](SECURE_V5_ROLLBACK.md). A future v5 build must explicitly
use --target v5 and another unused tag.

The approved v6 build context requires:

```text
models/iSpeak_v6/
  adapter_config.json
  adapter_model.safetensors
  processor_config.json
  tokenizer_config.json
  tokenizer.json
models/iSpeak_v5/base_model/
  config.json
  generation_config.json
  model.safetensors
models/filler_classifier/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
model_manifests/ispeak-v6.json
model_manifests/ispeak-v5.json
```

The v6 manifest verifies 12 runtime files totaling 1,624,928,628 bytes. The image
selects the v6 adapter and v6 manifest together while reusing the local base and
filler. The v5 target selects its own adapter and 16-file v5 manifest. Both targets
verify during build and at startup, offline, before the single Uvicorn worker.

The current .dockerignore admits those five v6 adapter files, the established v5
and filler artifacts, runtime code and manifests; it excludes secrets, recordings,
caches and training arguments. Git still excludes private weight binaries and the
v6 artifact tree. Only checksum manifests belong in this release batch. Provision
trusted artifacts separately in an isolated build context; do not force-add them.

```powershell
docker build --target v6 -t ispeak-python-v2:v6-release-candidate .
```

See DOCKER.md for explicit synthetic-fixture CLI arguments and validation-only
checks. No model, environment, image, registry, or cloud change is performed by
these instructions. Commands below are retained as a **historical v5 provisioning
record**, not current v6 release steps; historical model-selection/default claims
apply only to that earlier stage. Do not execute its old build examples.

## Historical v5 model inventory and selection

The complete runtime bundle contains 16 files totaling **1,622,897,450 bytes**.
Exact per-file sizes and SHA-256 values are in
`model_manifests/ispeak-v5.json`; hashes are integrity metadata, not model weights.

| Component | Current path | Required runtime files |
| --- | --- | --- |
| Whisper-small base | models/iSpeak_v5/base_model | config.json, generation_config.json, model.safetensors |
| v5 LoRA/tokenizer | models/iSpeak_v5 | adapter_config.json, adapter_model.safetensors, added_tokens.json, merges.txt, normalizer.json, preprocessor_config.json, special_tokens_map.json, tokenizer_config.json, vocab.json |
| Filler classifier | models/filler_classifier | config.json, model.safetensors, tokenizer.json, tokenizer_config.json |

The adapter declares `openai/whisper-small`; the base weights are local, not a
runtime Hugging Face download. LoRA weights are 51,957,816 bytes, base weights
966,995,080 bytes and filler weights 598,449,012 bytes. Tokenizer/config files
account for the remainder. README, training_args.bin and .cache are non-runtime
artifacts and excluded from the image. This integrity baseline records the
team's existing locally verified artifacts; checksums do not establish training
data consent, license rights, or supply-chain authenticity on their own.

Existing application configuration already selects local directories:

- `ISPEAK_MODEL_PATH`: adapter; default models/iSpeak_v5.
- `ISPEAK_BASE_MODEL_PATH`: explicit base override; otherwise the selected
  adapter's base_model when present, then the v5 base fallback.
- `ISPEAK_FILLER_MODEL_PATH`: filler classifier; default models/filler_classifier.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`: existing Docker offline flags.
  Application loaders additionally enforce local_files_only=True.

New provisioning-only `ISPEAK_MODEL_MANIFEST` selects a reviewed integrity
manifest. The current v6 image sets model_manifests/ispeak-v6.json; the v5
image and standalone verifier default use model_manifests/ispeak-v5.json. It is not a
model download URL or user-controlled API field. All selected paths/manifests
must be controlled by the deployment maintainer. Windows development startup
and local model selection remain unchanged.

## Options and compatibility

| Criterion | A: private model image | B: private artifact storage | C: private persistent volume |
| --- | --- | --- | --- |
| Security | One reviewed immutable image/digest; registry readers can extract weights | Scoped storage identity, immutable artifact version and trusted manifest needed | Restrict volume writers; app receives read-only model files |
| Cost | Registry retention/pulls plus compute; platform plan may dominate | Storage/operations/egress plus compute and bootstrap work | Persistent disk/backups plus compute and operational work |
| Startup | Local image files; image pull and SHA scan precede model load | Up to 1.623 GB fetch per cold instance unless cached/mounted | Local attached storage reads; first provisioning still required |
| Deployment simplicity | Closest to the existing verified Docker path | New downloader/IAM/retry/capacity workflow, not implemented here | Initial copy, filesystem ownership, mount and backup workflow |
| Updates | Build/test a new image, retain old digest | New immutable object/version and manifest; no overwrite-in-place | New version directory, verify then change approved paths |
| Railway | Private-image deployment requires approved Pro plan | Possible with separately implemented private bootstrap/network identity | Native volume option; operational constraints need review |
| Cloud Run | Artifact Registry/private-image path | Native read-only Cloud Storage mount or private bootstrap | No direct Railway-style local disk; GCS/NFS design needed |
| Rollback | Restore previous image digest and config | Restore previous artifact version and manifest | Restore previous immutable directory and manifest |
| Size | Models embedded; v5/v6 images may share base layers | Models external; downloaded cache consumes disk/RAM according to host | At least 1.623 GB per complete bundle, plus headroom and backup |
| Thesis fit | Lowest implementation risk; billing gate remains explicit | Candidate if budget rules out A; more security/startup work | Candidate for repeated local model updates; more operations |

Railway documents private-image deployment as a Pro feature. Its current pricing
lists Hobby $5/month and Pro $20/month, with included usage; RAM, CPU and volume
usage are separately metered beyond the allowance. Volume storage is $0.15/GB/month:
one 1.623 GB bundle alone is roughly $0.24/month, excluding headroom/backups and
all compute. Free/trial memory is inadequate for the established Python limit.
There is no approval to incur these charges. See
[Railway private images](https://docs.railway.com/services),
[plans/usage prices](https://docs.railway.com/pricing/plans) and
[volumes](https://docs.railway.com/volumes).

Cloud Run can deploy a private image through Artifact Registry, including an
approved remote repository for supported external private registries. Read-only
Cloud Storage mounts are available but add FUSE caching/latency/memory concerns.
Registry storage and transfer are billed separately; compute depends on region,
billing mode, request time and minimum instances. Scale-to-zero can reduce idle
compute but makes model cold starts recurring. No free deployment or account
activation is assumed. See [private image deployment](https://docs.cloud.google.com/run/docs/deploying),
[storage mounts](https://docs.cloud.google.com/run/docs/configuring/services/cloud-storage-volume-mounts),
[registry pricing](https://cloud.google.com/artifact-registry/pricing) and
[Cloud Run pricing](https://cloud.google.com/run/pricing).

**Cloud Run is conditional compatibility, not verified deployment readiness.**
The current CPU/memory plan (4 vCPU, 6 GiB minimum / 8 GiB safer) fits its documented
memory range. It injects PORT; our existing command supports that and binds
0.0.0.0. Its startup window and probe must allow integrity plus model loading.
The existing HTTP/1 Uvicorn/axios path encounters Cloud Run's 32 MiB request limit;
two allowed 25 MiB files exceed it. Do not silently lower limits or change
contracts: resolve and test ingress/transport before choosing that host. Also,
Cloud Run IAM authentication requires a Google-signed ID token, which the existing
Node shared-service-token header does not provide. Keep Python private; do not
disable IAM/public restrictions merely to obtain connectivity. A reviewed
private-access/IAM design is still required. Sources:
[memory limits](https://docs.cloud.google.com/run/docs/configuring/instances/memory-limits),
[container contract](https://docs.cloud.google.com/run/docs/container-contract),
[request limits](https://docs.cloud.google.com/run/quotas), and
[service identity authentication](https://docs.cloud.google.com/run/docs/authenticating/service-to-service).

## Implemented local procedure

At this historical stage, the standard build used the baseline v5 model trees.
The current default is v6, as specified above.
It now copies the verifier and reviewed manifest, verifies all 16 files during
the build, and verifies again before executing the existing CMD at runtime.
No cloud credential is installed or used by the guard. The manifest/image must
remain immutable to the application identity; replacing both weights and manifest
would defeat integrity checking. Registry/directory provenance must be reviewed
separately. SHA-256 is not a signature.

```powershell
python model_integrity.py
docker build --check .
docker build --target v5 -t ispeak-python-v2:v5-provisioning-candidate .
docker run --rm --network none ispeak-python-v2:v5-provisioning-candidate python model_integrity.py
```

The verifier fails with controlled component/file diagnostics if the selected
bundle is incomplete, has a size/checksum mismatch, extra executables/unknown
artifacts, linked files, pickle weights or sharded checkpoint indexes. It does
not modify the bundle or download replacements. The complete 16-file manifest
is mandatory; partial manifests, duplicate entries and unsafe paths fail.
The reviewed manifest is copied into the private image independently of weights.

Historical manifest-generation example only: both reviewed manifests now exist.
Never regenerate them to make unexplained artifact changes pass. New bundle
manifest creation requires separate approval and a new unused output path:

```powershell
# Set existing model-path variables privately to the approved local candidate.
python model_integrity.py --create-manifest <new-approved-manifest.json> --bundle-id <new-approved-bundle>
```

Manifest creation refuses to overwrite an existing manifest. Do not regenerate
the v5 manifest to make unexplained changes pass. The current guard intentionally
accepts only the verified unsharded Safetensors layout; a changed layout/loading
architecture requires review under the Accelerate disposition.

Run the opt-in functional probe using a generated spoken WAV, never user audio:

```powershell
python testing_script/verify_model_provisioning.py --speech-file <generated-speech.wav>
```

The harness uses no external network or host ports, generates an ephemeral service
token, exercises normal and reference inference through HTTP, verifies readiness,
authentication, Librosa scores and filler availability, and removes its disposable
container. Missing/corruption checks use temporary read-only overlay fixtures,
never edits to the original models. Results live in ignored benchmark-results.

After verification, make an immutable descriptive **local-only** image tag that
includes bundle version/manifest hash and record its image ID. Retain the prior
verified image tags. Before any future publication, select and approve a private
registry, verify its visibility and read/write permissions, privately authenticate,
tag for that destination, and separately authorize upload. No docker login,
docker push, cloud artifact download or provider resource creation runs here.

## Non-destructive v5/v6 promotion and rollback

1. Preserve models/iSpeak_v5 and its manifest as the active baseline. Do not change
   the current model-path environment or existing launcher during this task.
2. Place a future approved candidate in a separate private models/iSpeak_v6 tree
   with the same reviewed artifact layout. .gitignore excludes that directory;
   all weight extensions remain ignored. At that historical stage .dockerignore admitted only v5 and filler. The current
   configuration explicitly admits the five reviewed v6 adapter files as well.
3. Use existing ISPEAK_MODEL_PATH and ISPEAK_BASE_MODEL_PATH in an isolated test
   container, with a read-only candidate mount and its reviewed manifest selected
   through ISPEAK_MODEL_MANIFEST. Set the base explicitly to avoid an unintended
   fallback. Keep filler unchanged unless it has a separately reviewed bundle.
4. For a later immutable v6 image, prepare a separate disposable private build
   context/explicit COPY layout or approved read-only mounted-artifact strategy.
   Never rename or overwrite the working v5 tree to satisfy Docker COPY. Review
   image/bundle contents and retest before publication.
5. Gate promotion on provenance/license review, integrity, pip consistency, native
   and Linux regression tests, normal/reference inference, unchanged API schema,
   invalid-input/admission/auth/privacy checks, representative accuracy evaluation,
   CPU/RAM/latency and the Accelerate reassessment triggers. A checksum alone is
   not evidence of equivalent model quality.
6. Promote only after authorization; pin the image digest, bundle manifest and
   explicit model paths as one release. Keep previous image and config available.
   Roll back all three together; never overwrite artifacts in place.

No v6 directory, new model, new training/evaluation corpus or active model switch
was created in this task. The existing accepted-risk controls remain mandatory:
trusted local unsharded artifacts, non-root/private Python, service authentication,
restricted writers, controlled thesis use only, and all reassessment/review dates
in ACCELERATE_PRODUCTION_DISPOSITION.md. New model artifacts themselves trigger
that review; this preparation is not advance risk acceptance for v6.
