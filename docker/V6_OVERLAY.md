# Source-only v6 release image

Use docker/Dockerfile.v6-overlay with the repository root as build context. Its
adjacent Dockerfile-specific ignore file admits runtime Python source only; it
excludes model weights, environment files, recordings and test artifacts.

The base is the verified private Python RC1 OCI index
sha256:a938fb342d2ef69ad0e6110b4031d41b7ddc9628311f2f50788922a2c034c05e,
from source e3a0ebac3d8926809e98b0b94fbc9817fece4139. Its linux/amd64 manifest is
sha256:58a8238268fe4c3380dc7ed00bdab55d3ad1b425b730dca4d79be0fd75627870.
Keep RC1 unchanged as a historical baseline.

This recipe installs no dependencies and copies no model files. Before a build,
verify the approved release commit, unchanged semantic dependency requirements,
the v6 manifest and inherited runtime defaults against RC1. A dependency, model
or required base-security change needs a separate reviewed recipe. Do not claim
that this overlay fixes inherited dependency advisories.

After source-freeze approval, use a unique LOCAL tag, linux/amd64, this Dockerfile
and a clean isolated checkout. Supply the approved source revision as image
metadata. Preserve resource gates and scoped cancellation controls; source-only
assembly does not authorize a lower RAM threshold. Do not publish or deploy as
part of local validation.

Validate resulting source hashes and model integrity, nonroot execution, pip
check, production authentication, port precedence, health/readiness, bounded
FFmpeg decoding, timestamp/assessment/filler regressions, shutdown and a targeted
speech workflow. Model startup and inference need their own resource preflight.
Record the new final digest; do not label it RC1 or overwrite RC1 tags.
