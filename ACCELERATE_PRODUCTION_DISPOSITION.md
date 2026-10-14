# Accelerate production risk disposition — 2026-10-09

**Final decision C: ACCEPTED RISK — CONTROLLED THESIS DEPLOYMENT ONLY.**

This disposition addresses only Accelerate's identified advisory. The vulnerability
is not fixed or removed, and pip-audit still fails. The current task expressly
authorizes this scoped disposition when evidence supports non-exposure; the
earlier UAT exception alone did not authorize it. No deployment was performed.
Other security/deployment gates remain separate and open where previously reported.

## 1–2. Why it is installed and used

Accelerate **1.14.0** is a direct pinned requirement in requirements.txt and also
a runtime dependency of **PEFT 0.19.1**, whose metadata requires accelerate>=0.21.0.
Transformers **5.10.1** has no unconditional base requirement for it, but its Torch,
Accelerate and other optional extras require accelerate>=1.1.0. Installed metadata
and pip show identify PEFT as the current mandatory dependent. No other installed
mandatory dependent was identified by pip show.

It is not training-only. model.load_model constructs the existing Whisper/LoRA
runtime, imports PeftModel, loads the trusted adapter, merges it, and creates the
ASR pipeline. PEFT imports Accelerate helpers unconditionally, including memory,
device and dispatch/hook utilities. Transformers conditionally imports Accelerate
helpers when it is available. The filler classifier remains local-only.

The removal probe captured this concrete import chain:

```text
model loading / PeftModel import
→ peft/__init__.py
→ peft/auto.py
→ peft/config.py
→ peft/utils/__init__.py
→ peft/utils/loftq_utils.py:27
→ from accelerate.utils.memory import clear_device_cache
→ ModuleNotFoundError: accelerate
```

Other PEFT import sites include peft_model.py and utils/other.py. Removing
Accelerate requires changing the model-loading architecture; it is not a safe
dependency-only change. A future premerged Whisper export could be evaluated
separately but was not attempted or claimed compatible here.

## 3. Advisory details

- IDs: **PYSEC-2026-3804 / CVE-2026-69112 / GHSA-4j2p-28q2-5m79**.
- GitHub-reviewed severity: moderate, CVSS v4 6.9; affected range <=1.14.0.
- Vulnerable functions: accelerate.utils.modeling.load_checkpoint_in_model and
  accelerate.big_modeling.load_checkpoint_and_dispatch.
- An attacker-controlled sharded checkpoint index can supply weight_map filenames
  with traversal or absolute paths. The loader does not confine those filenames
  to the checkpoint directory. Out-of-directory reads or named-pipe blocking can
  affect confidentiality and availability.
- Prerequisite: malicious checkpoint/index metadata must reach the affected
  loading path. It can originate from an untrusted downloaded model or local
  artifact replacement. Distributed execution is not required; sharded/index
  loading can also occur in one process.
- The path flaw does not require unsafe pickle deserialization. Changing to
  safetensors alone is not a general remediation for an unsafe shard-path resolver.
- Ordinary remote audio is not a checkpoint index. No iSpeak request selects a
  model directory, weight_map, offload folder or checkpoint loader.

Source: [GitHub advisory](https://github.com/advisories/GHSA-4j2p-28q2-5m79).

## 4. Actual exposure: PRESENT BUT NOT EXERCISED

Both application source inspection and a real offline Docker runtime trace support
this classification. The package is imported and its non-vulnerable helpers are
used, so NOT APPLICABLE would overstate the conclusion.

| Runtime phase | Accelerate state / observed behavior |
| --- | --- |
| fastapi_backend application import | Not imported; 0 modules loaded |
| Model startup | Imported; 49 modules loaded; 343 profiled Accelerate module/helper calls |
| PEFT adapter loading | 4 additional profiled helper calls |
| PEFT merge | Accelerate already imported; no calls attributed to this phase |
| Pipeline creation | Accelerate already imported; no calls attributed to this phase |
| Real inference | Remains imported; no additional Accelerate calls observed |
| Reference comparison | Remains imported; no additional Accelerate calls observed |
| Vulnerable checkpoint loaders | **0 calls across all audited phases** |

Counts include import/module initialization, not 347 model dispatches. The profiler
covered the main thread and newly created analysis/lifespan threads. It records
static module/function names only, never arguments, speech, paths or credentials.
The exact counters are in ignored runtime-result.json.

The fixed runtime loads a trusted Whisper base, trusted local LoRA adapter and
trusted filler classifier with local_files_only=True and container offline flags.
There are **0 sharded-checkpoint index files** in the audited image's model tree.
There is no user-supplied model upload or device_map/distributed checkpoint-loading
configuration in this application path. The audit ran with a read-only root
filesystem and isolated temporary storage, no network, a valid test-only service
token and one application model/analysis worker.

This is evidence for the inspected artifact and CPU execution path, not proof
that every possible future library path, GPU configuration or model variant is
safe. Production private-network placement was not tested against a live host.

## 5–6. Patched availability and upgrade decision

Fresh checks on 2026-10-09: PyPI latest is **1.15.0**, while the advisory still lists
**no patched version**. The 1.15.0 loader still derives checkpoint filenames from
index values and joins them to the folder without a confinement check. A release
outside the advisory's stated range is not sufficient evidence of a fix.

Therefore **upgrade test: NOT RUN — no verified fixed version identified**.
No first-fixed-version or patched-version compatibility claim is made. An
unverified upgrade that merely changes audit classification was not pursued.
No Torch, Transformers, PEFT or other package was upgraded or downgraded.

Sources: [PyPI release metadata](https://pypi.org/pypi/accelerate/json),
[1.15.0 loader source, weight_map handling](https://raw.githubusercontent.com/huggingface/accelerate/v1.15.0/src/accelerate/utils/modeling.py).

## 7–8. Isolated removal and compatibility

A disposable image derived from ispeak-python-v2:security-verification removed
only Accelerate. Package inventories confirmed this was the sole package
difference. The original images and native venv remained unchanged.

| Removal check | Result |
| --- | --- |
| pip check | Exit 1: PEFT 0.19.1 requires Accelerate, which is not installed |
| PeftModel import | ModuleNotFoundError: accelerate; chain above |
| Whisper + LoRA loading / merge | Cannot proceed past required import |
| /health | 503; speech model unavailable |
| Filler classifier | Available independently |
| Authenticated analysis | 503; cannot perform speech analysis |
| Removal conclusion | **UNSAFE — retain Accelerate in runtime requirements** |

The probe itself exited 0 because it correctly asserted these expected failures.
That is not a successful model-startup or removal-readiness result.

The unchanged working image remains compatible with Python 3.13.12,
Torch 2.13.0+cpu, Transformers 5.10.1, PEFT 0.19.1 and Accelerate 1.14.0, including
the previously verified AnyIO/fsspec/urllib3 fixes. No compatibility with a patched
Accelerate release was established because one was not identified.

## 9–11. Fresh verification results

| Check | Result |
| --- | --- |
| Native pip check | No broken requirements; exit 0 |
| Verified Linux image pip check | No broken requirements; exit 0 |
| Fresh unfiltered pip-audit | Exactly 1 advisory in Accelerate 1.14.0; exit 1 |
| Relevant native Python suite | **33 passed** |
| Relevant Docker/Linux suite | **33 passed** |
| Real Docker startup /health | 200; speech and filler ready |
| Missing service-token request | 401; authentication preserved |
| Authenticated normal inference | 200; analysis valid, filler available, 18 words / 16.765 seconds |
| Authenticated reference comparison | 200; valid comparison using the existing controlled sample in both fields |
| Full response comparison | Both normal and reference JSON bodies exactly matched the earlier verified Docker baseline |
| Temporary audio cleanup | 0 audio copies remaining |
| Vulnerable loader calls | 0 |
| git diff --check | Passed |

Canonical response hashes, including intended transcripts/timestamps but without
printing their contents:

```text
Normal:    fa39a17b642caf3e34248d244bd83fef08b0fd4b0fb16d3c15c981fe606c9d84
Reference: 20930b469eace3f04bcdb78ca2e710849b7136975f88a21af2d6f271a9bb3a28
```

The relevant suite covers input validation, Librosa/scoring components, readiness,
service authentication/privacy, bounded queues, cancellation and recovery. No
expensive performance benchmark was rerun. The two response comparisons use one
existing English AAC sample; they establish a bounded compatibility check, not
new accuracy evidence across languages/devices or GPU coverage.

Preserved image identities:

```text
security-verification: sha256:7a439abeb14f5aadc72b1339f7363d0e6120f21c9faf7f8cff4e56fa91df1721
previous local image:  sha256:e0b99b90ff0103d092d8074e93f9e6a5d95fd20e8924d022e98cbcf9ec5768e8
```

The removal image is diagnostic only, not a deployable candidate. All audit
containers used --rm and no audit container remained running.

## 12. Exact files changed by this task

- ACCELERATE_PRODUCTION_DISPOSITION.md — this assessment and scoped acceptance.
- DOCKER.md — reference the new disposition instead of relying on the UAT exception.
- testing_script/audit_accelerate_runtime.py — test-only profiler and baseline comparison.
- testing_script/audit_accelerate_removal.py — expected-failure probe for the disposable image.

No production source, dependency requirement, model weight, API contract, service
token, scoring, queue or admission configuration changed in this task. Previous
working-tree changes were preserved. No files in other repositories were modified.
Ignored evidence is under benchmark-results/accelerate-disposition/: advisory
scan, native/Linux tests, runtime results/log, removal build/probe/pip check and
package inventories. Neither transcripts nor production credentials were printed.

## 13–14. Formal acceptance, controls and recommended action

**Accepted risk for controlled thesis deployment only**, based on the user's
explicit fallback instruction and the source/runtime/removal evidence above.
Retain the exact verified dependency stack and model artifacts. This exception
is separate from the previous UAT-only exception and applies only to this advisory.
It is not an authorization to deploy, bypass other gates, suppress pip-audit,
expose Python publicly or introduce untrusted model workflows.

Risk owner: the iSpeak thesis deployment lead; the team must record the named
maintainer and backup reviewer before launch. Review no later than **2026-11-08**
(30 days), at the end of the controlled thesis phase, or immediately on a trigger
below, whichever is earliest. Renewal requires a fresh recorded assessment.

Required controls for this scoped deployment:

1. Python has private Node-only service access and no normal public domain/TCP
   proxy. Keep shared service-token authentication, server-side secret management
   and authenticated Node access. Verify live network isolation before launch.
2. Use the verified immutable image/model artifacts, non-root runtime and
   read-only code/model filesystem. Limit writable space to isolated temporary
   storage. Restrict image/config/model replacement to authorized maintainers;
   preserve trusted artifact hashes and review provenance at build time.
3. Keep fixed local model paths, offline loading and unsharded safetensors;
   prohibit user-provided models, checkpoint indexes, remote model URLs, new
   dispatch/device_map/offload workflows and runtime pip installs.
4. Limit deployment to the controlled thesis participant/admin group and current
   one-worker CPU topology. Do not generalize acceptance to public service,
   training, distributed inference, arbitrary models or a different hardware path.
5. Monitor readiness, request failures, memory/CPU and unexpected restarts, with
   sanitized logs. Preserve audio/privacy controls. Investigate anomalous startup
   or artifact changes; ordinary request monitoring cannot detect all model
   supply-chain compromises.
6. Check upstream advisory/releases and run unfiltered pip-audit at least weekly
   and before each image rebuild. When a fix is verified, test only that targeted
   change in isolation with the same startup/inference/reference/regression checks,
   then retire this exception if successful. Do not suppress the advisory.

Residual risk: vulnerable code remains installed/importable. A compromised build,
artifact source, maintainer account, writable model/configuration or future loader
change could make the vulnerable path reachable and permit local file access or
DoS. Private networking/service authentication alone cannot prevent malicious
checkpoint metadata. A short runtime trace cannot prove universal absence of
future execution. These risks are accepted only within the constraints above.

Revoke/reassess immediately on a model/dependency/loading-path change, addition
of sharded checkpoints, device_map/distributed/GPU use, public Python exposure,
loss of model immutability, suspected supply-chain compromise, a revised advisory
showing an audio-reachable path, or an established upstream fix. If required
controls cannot be met, classification returns to **BLOCKER REMAINS**.

**Recommended action: retain 1.14.0 and use this scoped disposition for the
controlled thesis candidate; do not call the vulnerability resolved.** Overall
deployment still depends on the separately open ingress, cloud and client gates.
No deployment, commit or push occurred.
