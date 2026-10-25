"""Manual-only diagnostic mode reusing existing bounded build/cleanup tooling.

This entrypoint never runs the legacy execution path. It can reuse the existing
private publisher only after validation and separate publication approval. Local
resources are always removed; no model/image archive is uploaded as a CI artifact.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'remote'))
from materialize import BASE
from preflight import enforce
from runner import Validation

SOURCE = '1f000522a308ce515db314ef423b32c220c9e32e'
BRANCH = 'refs/heads/ci/validate-python-a76c6f47-20261021'
RECIPE = 'docker/Dockerfile.v6-security-overlay'
MANIFEST = Path(__file__).with_name('context-manifest.json')


def publish_if_approved(v, image):
    if os.environ.get('PUBLICATION_APPROVED') != 'true':
        v.evidence('diagnostic_publication', approved=False, attempted=False)
        return None
    if getattr(v, 'model_free_passed', False) is not True:
        raise RuntimeError('Publication requires completed model-free verification')
    # Reuse the exact existing private-package, unique-tag, image-ID, digest,
    # authenticated-pull and anonymous-denial checks. No permission expansion.
    from publish_image import publish
    publish(v, image)
    digest = getattr(v, 'published_digest', '')
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', digest):
        raise RuntimeError('Verified publication digest unavailable')
    immutable = 'ghcr.io/apol-bit/ispeak-python@' + digest
    v.publication_verified = True
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as output:
            output.write('Verified private diagnostic image: `' + immutable + '`\n\n'
                         'Source: `' + SOURCE + '`; model-free validation only; no deployment.\n')
    v.evidence('diagnostic_image_preserved', immutable=immutable, source_commit=SOURCE,
               image_id=image['Id'], private=True, model_inference=False, deployed=False)
    return immutable


def gate():
    required = {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
                'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64',
                'GITHUB_EVENT_NAME': 'workflow_dispatch',
                'GITHUB_REPOSITORY': 'Apol-bit/ispeak_python-backendV2',
                'GITHUB_ACTOR': 'Apol-bit', 'GITHUB_REF': BRANCH,
                'GITHUB_RUN_ATTEMPT': '1', 'ACCELERATE_STAGING_APPROVED': 'true',
                'PRIVATE_BASE_ACCESS_APPROVED': 'true', 'TESTING_CONTAINMENT_ACCEPTED': 'true'}
    if os.name != 'posix' or any(os.environ.get(k) != v for k, v in required.items()):
        raise RuntimeError('Manual hosted execution and explicit staging approvals required')
    reviewers = [os.environ.get(k, '') for k in ['STAGING_MAINTAINER', 'STAGING_BACKUP_REVIEWER']]
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', v) for v in reviewers) or reviewers[0].lower() == reviewers[1].lower():
        raise RuntimeError('Distinct named maintainer/backup acceptance records required')


def export(v):
    manifest = json.loads(MANIFEST.read_text())
    if manifest['commit'] != SOURCE or manifest['base'] != BASE:
        raise RuntimeError('Diagnostic manifest identity mismatch')
    repo, context, tests = (v.root / name for name in ['source', 'context', 'tests'])
    repo.mkdir()
    v.command(['git', 'init', str(repo)])
    v.command(['git', '-C', str(repo), 'remote', 'add', 'origin',
               'https://github.com/Apol-bit/ispeak_python-backendV2.git'])
    v.command(['git', '-C', str(repo), 'fetch', '--depth=1', 'origin', SOURCE], seconds=120)
    if v.command(['git', '-C', str(repo), 'rev-parse', 'FETCH_HEAD']).decode().strip() != SOURCE:
        raise RuntimeError('Diagnostic source mismatch')
    for group, destination in [('files', context), ('tests', tests)]:
        for item in manifest[group]:
            p = Path(item['path'])
            if p.is_absolute() or '..' in p.parts:
                raise RuntimeError('Unsafe source path')
            content = v.command(['git', '-C', str(repo), 'show', SOURCE + ':' + item['path']])
            if len(content) != item['bytes'] or hashlib.sha256(content).hexdigest() != item['sha256']:
                raise RuntimeError('Diagnostic source hash mismatch')
            target = destination / p
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    # The recipe and wheel manifest are byte-pinned, not generalized to arbitrary RUNs.
    if not (context / RECIPE).read_text().startswith('# Preserve RC1'):
        raise RuntimeError('Unexpected diagnostic recipe')
    return context, tests, manifest


class DiagnosticBuild(Validation):
    def __init__(self):
        super().__init__('python')
        self.image = 'ispeak-python-diagnostic:' + SOURCE[:12] + '-' + self.key
        self.model_free_passed = False
        self.publication_verified = False

    def execute(self):
        gate()  # Before credentials, private pulls, source fetch or Docker calls.
        enforce('python', stable=True)
        self.monitor = True
        context, tests, manifest = export(self)
        self.login()  # Existing protected credential, stripped from subprocess env.
        self.builder_start()
        enforce('python', stable=True)
        started = time.monotonic()
        self.docker('buildx', 'build', '--builder', self.builder, '--platform=linux/amd64',
                    '--progress=plain', '--provenance=false', '--sbom=false', '--load',
                    '--label', 'org.opencontainers.image.revision=' + SOURCE,
                    '--label', self.label, '-t', self.image, '-f', str(context / RECIPE),
                    str(context), seconds=900)
        image = json.loads(self.docker('image', 'inspect', self.image))[0]
        base = json.loads(self.docker('image', 'inspect', BASE))[0]
        if image['Os'] != 'linux' or image['Architecture'] != 'amd64':
            raise RuntimeError('Unexpected platform')
        if image['RootFS']['Layers'][:len(base['RootFS']['Layers'])] != base['RootFS']['Layers']:
            raise RuntimeError('Pinned model base layer chain changed')
        cfg = image['Config']
        secret_names = {'MONGO_URI', 'JWT_SECRET', 'PYTHON_SERVICE_TOKEN', 'ISPEAK_SERVICE_TOKEN',
                        'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'GITHUB_TOKEN', 'GHCR_READ_TOKEN'}
        for item in cfg.get('Env', []):
            key, _, value = item.partition('=')
            if key in secret_names and value:
                raise RuntimeError('Credential embedded in candidate environment')
        if cfg.get('User') != '10001:10001' or not cfg.get('Healthcheck'):
            raise RuntimeError('Nonroot/healthcheck mismatch')
        if cfg.get('Entrypoint') != base['Config'].get('Entrypoint') or cfg.get('Cmd') != base['Config'].get('Cmd'):
            raise RuntimeError('Startup command changed')
        if cfg.get('Labels', {}).get('org.opencontainers.image.revision') != SOURCE:
            raise RuntimeError('Source label mismatch')
        for key in ['Env', 'WorkingDir', 'ExposedPorts']:
            if cfg.get(key) != base['Config'].get(key):
                raise RuntimeError('Inherited runtime configuration changed')
        self.stop_builder()
        from validate_image import validate
        validate(self, tests, manifest)
        self.model_free_passed = True
        immutable = publish_if_approved(self, image)
        self.evidence('diagnostic_candidate', result='PASS', source_commit=SOURCE,
                      image_id=image['Id'], elapsed_seconds=round(time.monotonic() - started, 2),
                      published=bool(immutable), immutable=immutable, deployed=False, model_inference=False,
                      image_retention='private GHCR digest' if immutable else 'ephemeral; removed during scoped cleanup')


def main():
    gate()
    if sys.argv[1:] not in [[], ['cleanup']]:
        raise RuntimeError('Unsupported diagnostic execution mode')
    v = DiagnosticBuild()
    def interrupted(*_):
        raise RuntimeError('Diagnostic run cancelled')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if not sys.argv[1:]:
            v.execute()
    finally:
        try:
            if not sys.argv[1:]:
                v.evidence('diagnostic_publication_state',
                           attempted=getattr(v, 'publication_started', False),
                           verified=v.publication_verified,
                           registry_digest=getattr(v, 'published_digest', None),
                           remote_rollback_or_deletion_performed=False)
        finally:
            v.cleanup()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print(json.dumps({'result': 'STOP', 'reason': 'Diagnostic build/verification/publication failed; consult publication-state evidence; private output withheld'}))
        sys.exit(1)
