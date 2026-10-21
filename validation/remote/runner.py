"""DRAFT: bounded ephemeral-runner validation; never publishes or deploys.

Not executable on a workstation. Every subprocess output stays in a 0700 runner
directory. Only curated evidence reaches stdout; no raw Docker/model logs.
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from materialize import BASE, COMMITS, export
from preflight import enforce, measure, problems


class Validation:
    def __init__(self, role):
        self.role = role
        self.key = os.environ['GITHUB_RUN_ID'] + '-' + os.environ['GITHUB_RUN_ATTEMPT']
        if not re.fullmatch(r'\d+-1', self.key):
            raise RuntimeError('First attempt of a GitHub workflow run required')
        self.root = Path(os.environ['RUNNER_TEMP']).resolve() / ('ispeak-validation-' + self.key)
        if self.root.is_symlink():
            raise RuntimeError('Unexpected symbolic-link temporary directory')
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.builder = 'ispeak-' + self.key
        self.label = 'ispeak.validation.run=' + self.key
        self.image = f'ispeak-{role}-candidate:{COMMITS[role][:12]}-{self.key}'
        self.sequence = 0
        self.monitor = False
        self.deadline = time.monotonic() + (1800 if role == 'node' else 2400)
        self.minimum = {}
        self.children = set()
        self.docker_env = dict(os.environ)
        # Never propagate the job token into git/build/test/container subprocesses.
        self.token = self.docker_env.pop('GHCR_READ_TOKEN', '')
        self.docker_env.pop('GITHUB_TOKEN', None)
        self.docker_env['GIT_TERMINAL_PROMPT'] = '0'
        self.docker_env['DOCKER_CONFIG'] = str(self.root / 'docker-auth')

    def command(self, args, seconds=60, input_data=None, check=True):
        self.sequence += 1
        log = self.root / f'command-{self.sequence}.log'
        with log.open('wb') as output:
            process = subprocess.Popen(args, stdout=output, stderr=subprocess.STDOUT,
                                       stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                                       start_new_session=True, env=self.docker_env)
            self.children.add(process)
            if input_data is not None:
                process.stdin.write(input_data)
                process.stdin.close()
            deadline = min(self.deadline, time.monotonic() + seconds)
            try:
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Bounded subprocess timeout; no retry')
                    if self.monitor:
                        data = measure()
                        self.minimum = data if not self.minimum else {
                            key: min(data[key], self.minimum[key]) for key in data}
                        failed = problems(self.role, data, start=False)
                        if failed:
                            self.evidence('resource_stop', measurements=data, failed_gates=failed)
                            raise RuntimeError('Resource stop condition reached; no retry')
                    time.sleep(1)
                if check and process.returncode:
                    raise RuntimeError('Validation subprocess failed; raw output withheld; no retry')
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                self.children.discard(process)
        data = log.read_bytes()
        if len(data) > 4 * 1024 * 1024:
            raise RuntimeError('Unexpected output volume; withheld from public log')
        return data if check else (process.returncode, data)

    def docker(self, *args, **kwargs):
        return self.command(['docker', *args], **kwargs)

    def run_args(self, name, memory='1g', network='none'):
        return ['run', '--name', name, '--label', self.label, '--restart=no',
                '--memory', memory, '--memory-swap', memory, '--cpus', '2',
                '--pids-limit', '256', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                '--network', network]

    def evidence(self, kind, **fields):
        print(json.dumps({'check': kind, **fields}), flush=True)

    def login(self):
        if os.environ.get('PRIVATE_BASE_ACCESS_APPROVED') != 'true':
            raise RuntimeError('Private base access has not been separately approved')
        if not self.token:
            raise RuntimeError('No ephemeral package-read token was supplied')
        config = Path(self.docker_env['DOCKER_CONFIG'])
        config.mkdir(mode=0o700)
        self.command(['docker', 'login', 'ghcr.io', '-u', os.environ['GITHUB_ACTOR'], '--password-stdin'],
                     input_data=(self.token + '\n').encode())
        self.token = ''
        # Login alone is not package access proof. Exact immutable pull must pass.
        self.docker('pull', '--platform=linux/amd64', BASE, seconds=300)
        self.evidence('private_base_read', result='PASS', base=BASE)

    def builder_start(self):
        image = os.environ.get('BUILDKIT_IMAGE', '')
        if not re.fullmatch(r'moby/buildkit@sha256:[a-f0-9]{64}', image):
            raise RuntimeError('Reviewed immutable BuildKit image is required')
        version_output = self.docker('buildx', 'version').decode()
        version = re.search(r'\bv\d+\.\d+\.\d+\b', version_output)
        engine_output = self.docker('version', '--format', '{{.Server.Version}}').decode().strip()
        if not version or not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?', engine_output):
            raise RuntimeError('Build tool version identity unavailable')
        self.evidence('build_toolchain', buildx=version.group(), docker_engine=engine_output)
        config = self.root / 'buildkitd.toml'
        config.write_text('[worker.oci]\n  enabled = true\n  max-parallelism = 1\n[worker.containerd]\n  enabled = false\n')
        self.assert_builder_absent('before_create')
        registrations = self.docker('buildx', 'ls', '--format', '{{.Name}}').decode().splitlines()
        if self.builder in registrations:
            raise RuntimeError('Run-owned builder name collision')
        # Mark intent before bounded creation, so interrupted setup is cleanable.
        (self.root / 'builder-intent').write_text(self.builder)
        self.docker('buildx', 'create', '--name', self.builder, '--driver', 'docker-container',
                    '--driver-opt', 'image=' + image, '--driver-opt', 'memory=4g',
                    '--driver-opt', 'memory-swap=4g', '--driver-opt', 'cpu-period=100000',
                    '--driver-opt', 'cpu-quota=200000', '--driver-opt', 'restart-policy=no',
                    '--driver-opt', 'env.ISPEAK_RELEASE_RUN=' + self.key,
                    '--buildkitd-config', str(config), seconds=60)
        self.docker('buildx', 'inspect', '--bootstrap', self.builder, seconds=120)
        data = json.loads(self.docker('inspect', 'buildx_buildkit_' + self.builder + '0'))[0]
        host = data['HostConfig']
        if (host['Memory'] != 4 * 1024 ** 3 or host['MemorySwap'] != 4 * 1024 ** 3 or
                host['CpuQuota'] != 200000 or host['CpuPeriod'] != 100000 or
                host['RestartPolicy']['Name'] != 'no' or
                'ISPEAK_RELEASE_RUN=' + self.key not in data['Config']['Env']):
            raise RuntimeError('Builder identity/caps verification failed')
        live = tomllib.loads(self.docker('exec', data['Id'], 'cat', '/etc/buildkit/buildkitd.toml').decode())
        worker = live.get('worker', {})
        if (worker.get('oci', {}).get('enabled') is not True
                or worker.get('oci', {}).get('max-parallelism') != 1
                or worker.get('containerd', {}).get('enabled') is not False
                or not data['State']['Running'] or data['State'].get('OOMKilled')
                or data.get('RestartCount') != 0 or data['Config']['Image'] != image):
            raise RuntimeError('Live builder configuration or running state mismatch')
        pinned = json.loads(self.docker('image', 'inspect', image))[0]
        if pinned['Id'] != data['Image']:
            raise RuntimeError('Builder pinned-image identity mismatch')
        self.evidence('containment_policy', scope='controlled_testing_only',
                      acceptance='Docker configuration and hosted-runner containment',
                      kernel_hierarchy_certified=False, active_export_load_certified=False)
        self.evidence('bounded_builder', id=data['Id'], memory_bytes=host['Memory'], cpus=2,
                      max_parallelism=1, builder_image=image)

    def assert_builder_absent(self, phase):
        name = 'buildx_buildkit_' + self.builder + '0'
        containers = self.docker('container', 'ls', '--all', '--quiet', '--filter', 'name=^/' + name + '$')
        volumes = self.docker('volume', 'ls', '--quiet', '--filter', 'name=^' + name + '_state$')
        self.evidence('builder_absence', phase=phase, container_absent=not bool(containers.strip()),
                      state_volume_absent=not bool(volumes.strip()))
        if containers.strip() or volumes.strip():
            raise RuntimeError('Run-owned builder state remains; no broad removal')

    def stop_builder(self):
        name = 'buildx_buildkit_' + self.builder + '0'
        ids = self.docker('container', 'ls', '--all', '--quiet', '--filter', 'name=^/' + name + '$').decode().split()
        if not ids:
            return
        if len(ids) != 1:
            raise RuntimeError('Ambiguous owned builder')
        data = json.loads(self.docker('inspect', ids[0]))[0]
        if (data['Name'] != '/' + name or
                'ISPEAK_RELEASE_RUN=' + self.key not in data['Config'].get('Env', [])):
            raise RuntimeError('Refuse to stop unowned builder')
        self.docker('stop', '--time=3', data['Id'], seconds=15)
        if json.loads(self.docker('inspect', data['Id']))[0]['State']['Running']:
            raise RuntimeError('Owned builder remains running')
        volumes = [row.get('Name') for row in data.get('Mounts', []) if row.get('Type') == 'volume']
        if volumes != [name + '_state']:
            raise RuntimeError('Unexpected builder volumes; refuse removal')

    def cleanup(self):
        self.monitor = False
        self.deadline = time.monotonic() + 90
        for child in tuple(self.children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        self.stop_builder()
        ids = self.docker('container', 'ls', '--all', '--quiet', '--filter', 'label=' + self.label).decode().split()
        if ids:
            self.docker('container', 'rm', '--force', *ids, seconds=20)
        intent = self.root / 'builder-intent'
        if intent.exists():
            if intent.read_text() != self.builder:
                raise RuntimeError('Unexpected builder cleanup intent')
            registered = self.docker('buildx', 'ls', '--format', '{{.Name}}').decode().splitlines()
            if self.builder in registered:
                self.docker('buildx', 'rm', self.builder, seconds=20)
        self.assert_builder_absent('after_cleanup')
        if self.docker('container', 'ls', '--all', '--quiet', '--filter', 'label=' + self.label).strip():
            raise RuntimeError('Owned validation containers remain')
        images = self.docker('image', 'ls', '--quiet', '--no-trunc', self.image).decode().split()
        if images:
            metadata = json.loads(self.docker('image', 'inspect', self.image))[0]
            if metadata['Config'].get('Labels', {}).get('ispeak.validation.run') != self.key:
                raise RuntimeError('Refuse to remove unowned candidate image')
            self.docker('image', 'rm', self.image, seconds=15)
        expected_parent = Path(os.environ['RUNNER_TEMP']).resolve()
        if (self.root.is_symlink() or self.root.resolve().parent != expected_parent
                or self.root.name != 'ispeak-validation-' + self.key):
            raise RuntimeError('Refuse unsafe temporary-directory cleanup')
        shutil.rmtree(self.root)
        self.evidence('cleanup', result='PASS', scoped_containers_absent=True,
                      builder_and_state_absent=True, private_temp_removed=True)

    def execute(self):
        if os.environ.get('AUTOMATION_REVIEW_APPROVED') != 'true':
            raise RuntimeError('Reviewed automation activation required')
        if os.environ.get('TESTING_CONTAINMENT_ACCEPTED') != 'true':
            raise RuntimeError('Controlled Testing containment policy not acknowledged')
        if self.role == 'python' and os.environ.get('PRIVATE_BASE_ACCESS_APPROVED') != 'true':
            raise RuntimeError('Private base access remains disabled')
        if self.role == 'python':
            from node_gate import require_node_success
            require_node_success()  # Before private pulls, builder startup, or inference.
            if not (shutil.which('espeak-ng') or shutil.which('espeak')):
                raise RuntimeError('Synthetic speech tool unavailable; no automatic installation')
        enforce(self.role, stable=True)  # Before source fetch, registry pull, builder bootstrap.
        self.monitor = True
        repo, context = export(self.role, self.root, self.command)
        if self.role == 'python':
            self.login()
        self.builder_start()
        enforce(self.role, stable=True)  # Preserve after-builder stable headroom requirement.
        recipe = 'Dockerfile' if self.role == 'node' else 'docker/Dockerfile.v6-overlay'
        started = time.monotonic()
        output = self.docker('buildx', 'build', '--builder', self.builder, '--platform=linux/amd64',
                             '--progress=plain', '--provenance=false', '--sbom=false', '--load',
                             '--label', 'org.opencontainers.image.revision=' + COMMITS[self.role],
                             '--label', self.label, '-t', self.image, '-f', str(context / recipe),
                             str(context), seconds=900 if self.role == 'node' else 600)
        image = json.loads(self.docker('image', 'inspect', self.image))[0]
        self.evidence('image_built', source_commit=COMMITS[self.role], image_id=image['Id'],
                      image_size_bytes=image['Size'], resource_minima=self.minimum, published=False)
        if image['Config']['Labels'].get('org.opencontainers.image.revision') != COMMITS[self.role]:
            raise RuntimeError('Candidate source label mismatch')
        if image['Architecture'] != 'amd64' or image['Os'] != 'linux':
            raise RuntimeError('Candidate platform mismatch')
        if image['Config'].get('User') in ('', '0', 'root', '0:0') or not image['Config'].get('Healthcheck'):
            raise RuntimeError('Missing nonroot user or healthcheck')
        private_keys = {'MONGO_URI', 'JWT_SECRET', 'PYTHON_SERVICE_TOKEN', 'ISPEAK_SERVICE_TOKEN',
                        'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY', 'GITHUB_TOKEN', 'GHCR_READ_TOKEN'}
        for item in image['Config'].get('Env', []):
            key, _, value = item.partition('=')
            if key in private_keys and value:
                raise RuntimeError('Unexpected credential/configuration embedded in candidate ENV')
        if self.role == 'node':
            references = set(re.findall(rb'docker.io/library/node:24\.21\.0-bookworm-slim@sha256:[a-f0-9]{64}', output))
            if len(references) != 1:
                raise RuntimeError('Resolved Node base identity not established')
            self.evidence('resolved_node_base', reference=references.pop().decode())
        if self.role == 'python':
            base = json.loads(self.docker('image', 'inspect', BASE))[0]
            layers = base['RootFS']['Layers']
            if image['RootFS']['Layers'][:len(layers)] != layers:
                raise RuntimeError('RC1 dependency/model layer chain changed')
            from validate_python import validate
        else:
            from validate_node import validate
        # Release the builder's RAM before runtime/model initialization. No rebuild
        # follows; the loaded local candidate remains available to Docker.
        self.stop_builder()
        stopped = json.loads(self.docker('inspect', 'buildx_buildkit_' + self.builder + '0'))[0]
        if stopped['State']['Running']:
            raise RuntimeError('Builder did not stop before runtime validation')
        validate(self, repo, context)
        self.evidence('candidate', result='PASS', source_commit=COMMITS[self.role], image_id=image['Id'],
                      image_size_bytes=image['Size'], build_seconds=round(time.monotonic() - started, 2),
                      cached_progress_lines=sum(line.endswith(b'CACHED') for line in output.splitlines()),
                      build_log_sha256=hashlib.sha256(output).hexdigest(),
                      resource_minima=self.minimum, published=False, cloud_integration_tested=False)


def main():
    if os.name != 'posix' or os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('RUNNER_ENVIRONMENT') != 'github-hosted':
        raise SystemExit('DRAFT SAFETY STOP: requires a GitHub-hosted Linux workflow job')
    role = os.environ.get('ROLE')
    if role not in COMMITS:
        raise SystemExit('Invalid validation role')
    expected_repo = 'Apol-bit/' + ('ispeak_backend' if role == 'node' else 'ispeak_python-backendV2')
    expected_branch = 'ci/validate-' + ('node-dfa3138f' if role == 'node' else 'python-a76c6f47') + '-20261021'
    if (os.environ.get('GITHUB_REPOSITORY') != expected_repo
            or os.environ.get('GITHUB_ACTOR') != 'Apol-bit'
            or os.environ.get('GITHUB_EVENT_NAME') != 'push'
            or os.environ.get('GITHUB_REF') != 'refs/heads/' + expected_branch):
        raise SystemExit('Unexpected validation execution identity')
    validator = Validation(role)
    def interrupted(_signal, _frame):
        raise RuntimeError('Workflow cancellation requested')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = 0
    try:
        if sys.argv[1:] != ['cleanup']:
            if sys.argv[1:] != [role]:
                raise RuntimeError('Unexpected execution mode')
            validator.execute()
    except Exception as error:
        # Never emit traceback, command arguments, response bodies or private model logs.
        print(json.dumps({'result': 'STOP', 'reason': str(error) if type(error) is RuntimeError else type(error).__name__,
                          'published': False}), flush=True)
        result = 1
    finally:
        validator.evidence('validation_summary', result='PASS' if result == 0 else 'STOP',
                           source_commit=COMMITS[role], resource_minima=validator.minimum,
                           cleanup_pending=True, published=False)
        try:
            validator.cleanup()
        except Exception:
            print(json.dumps({'result': 'STOP', 'reason': 'Scoped cleanup incomplete; no retry', 'published': False}), flush=True)
            result = 1
    return result


if __name__ == '__main__':
    sys.exit(main())
