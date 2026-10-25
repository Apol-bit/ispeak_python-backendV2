# Minimal v6 dependency security remediation

Baseline: `6b1dc8a5d5230cc94bb5d50362bc8d4df7820a54`.
Only certifi 2026.4.22 -> 2026.10.19, filelock 3.29.0 -> 4.2.1 and
setuptools 83.0.0 -> 84.1.0 are changed. All model dependencies, weights,
manifests, transcription/scoring settings and application source remain frozen.

## Reproducible installation

`docker/requirements.security.txt` pins the three universal wheels and their
PyPI SHA-256 hashes. Use a disposable Python 3.13 environment, never the working
environment. Install with:

```text
python -m pip install --no-cache-dir --no-deps --only-binary=:all: --require-hashes -r docker/requirements.security.txt
python -m pip check
python -B -m unittest testing_script.test_dependency_security -v
```

Compare complete before/after inventories; require exactly the three approved
version changes, no package additions/removals and no model dependency drift.
Do not resolve/upgrade the entire stack. Source distributions are not permitted.

## Separate security overlay (not built)

`docker/Dockerfile.v6-security-overlay` retains the RC1 OCI index digest
`sha256:a938fb342d2ef69ad0e6110b4031d41b7ddc9628311f2f50788922a2c034c05e`.
It installs only the hash-pinned wheels with no dependency resolution as root,
runs pip check, removes its temporary requirements file, and restores UID/GID
10001. Source COPY operations match the approved source overlay; diagnostics
are included. Model paths, offline settings, entrypoint, healthcheck and one-worker
command are inherited. The Dockerfile-specific context allowlist excludes private
artifacts, model weights, tests, environment files and recordings.

The historical source-only overlay is unchanged. Updating requirements alone
cannot remediate packages inherited from RC1. Old package bytes may remain in
historical lower image layers; verify the effective final filesystem/inventory
and distinguish layer findings from executable installed packages.

No container build, image pull, publication, deployment or inference was performed.
Private RC1 installed inventory, provenance, Linux CPU compatibility and OS-level
advisories still require separately approved image access/verification.

## Local model-free verification (2026-10-25)

A disposable Python 3.13 environment outside this repository reads the unchanged
baseline site-packages through a .pth path. Only the three replacements are
installed locally using verified wheel hashes. The original .venv is not modified.
The disposable environment's older bootstrap pip was removed from that environment
only, so unchanged baseline pip 26.2.1 is used read-only; no pip upgrade is part of
this patch. This is a compatibility test environment, not a standalone Linux image.

- Full inventory comparison: exactly the three approved changes across 88 packages.
- pip check: no broken requirements.
- Seven focused tests: exact versions, SSL trust-bundle parsing, nested locking,
  contention/timeouts/release, soft locks, Hugging Face locking, Torch lock
  subclassing, and setuptools extension interfaces without compilation.
- 117 model-free tests passed with diagnostics disabled and 117 enabled, including
  the seven focused tests and existing timing/cleanup regressions. Whisper and
  filler-classifier constructors were guarded against accidental model loading.
- Public OSV metadata audit across 88 effective packages: only the existing
  Accelerate 1.14.0 advisory (GHSA-4j2p-28q2-5m79 / PYSEC-2026-3804) remains.
  No advisories were suppressed. This is not an image/OS security scan.
- Recipe/context inspection, Python syntax, whitespace and scoped secret/privacy
  checks performed without Docker execution. Existing five excluded files retain
  identical SHA-256 hashes; Git index remains empty.

Validation evidence is outside the repository in the local release-preparation
directory; it is not part of the six-file commit scope.

## Compatibility and residual gates

Filelock 4 changes the SoftReadWriteLock protocol. Never share its lock/cache
directories between old/new participants. Inspected consumers use ordinary
FileLock/SoftFileLock; tests cover those APIs, not every optional locking path.
Setuptools changes build/compiler interfaces: use wheels and prohibit unexpected
source builds. Certifi changes TLS trust; no external endpoint was probed.

The existing Accelerate exception remains separate and unresolved. Its controls,
named reviewers and review deadline must be revalidated; no exception is extended
to other packages. An empty advisory result for Accelerate 1.15.0 is not proof
of a verified fix. Do not silently upgrade it.

Before staging: obtain approved private-image access; verify RC1 inventory and
model hashes; build/scan the new immutable image only with separate approval;
compare Linux CPU inventories; repeat model-free tests. Real startup and synthetic
normal/reference inference require separate resource and execution approval.
No claim of unchanged real-model accuracy is made from model-free tests alone.
