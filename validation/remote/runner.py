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
from pathlib import Path

from materialize import BASE, COMMITS, export
from preflight import DOCKER_FREE, EMERGENCY_RAM, SYSTEM_FREE, enforce, measure


class Validation:
    def __init__(self, role):
        self.role = role
        self.key = os.environ['GITHUB_RUN_ID'] + '-' + os.environ['GITHUB_RUN_ATTEMPT']
        if not re.fullmatch(r'\d+-1', self.key):
            raise RuntimeError('First attempt of a GitHub workflow run required')
        self.root = Path(os.environ['RUNNER_TEMP']).resolve() / ('ispeak-validation-' + self.key)
        self.root.mkdir(mode=0o700, exist_ok=True)
        self.builder = 'ispeak-' + self.key
        self.label = 'ispeak.validation.run=' + self.key
        self.image = f'ispeak-{role}-candidate:{COMMITS[role][:12]}-{self.key}'
        self.sequence = 0
        self.monitor = False
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
            deadline = time.monotonic() + seconds
            try:
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError('Bounded subprocess timeout; no retry')
                    if self.monitor:
                        data = measure()
                        self.minimum = data if not self.minimum else {
                            key: min(data[key], self.minimum[key]) for key in data}
                        if (data['available_ram_bytes'] < EMERGENCY_RAM or
                                data['docker_free_bytes'] < DOCKER_FREE or
                                data['system_free_bytes'] < SYSTEM_FREE):
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
        config = self.root / 'buildkitd.toml'
        config.write_text('[worker.oci]\n  max-parallelism = 1\n')
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
        self.evidence('bounded_builder', id=data['Id'], memory_bytes=host['Memory'], cpus=2,
                      max_parallelism=1, builder_image=image)

    def cancel_probe(self):
        """Exercise cancellation of active work on the exact dedicated builder.

        This deliberately makes no claim that cancelling an active Docker-daemon
        image import was empirically covered. That remains an activation gate.
        """
        if os.environ.get('EXPORT_LOAD_CONTROLS_APPROVED') != 'true':
            raise RuntimeError('Export/load cancellation coverage still requires separate review')
        # Actual probe is a bounded non-model sleep; no package/model installation.
        base = BASE if self.role == 'python' else 'node:24.21.0-bookworm-slim'
        executable = 'python' if self.role == 'python' else 'node'
        code = 'import time; print("ISPEAK_CANCEL_ACTIVE", flush=True); time.sleep(600)' if self.role == 'python' else \
               'console.log("ISPEAK_CANCEL_ACTIVE");setTimeout(()=>{},600000)'
        probe = self.root / 'cancel-probe'
        probe.mkdir()
        (probe / 'Dockerfile').write_text('FROM ' + base + '\nRUN ' +
                                        json.dumps([executable, '-c' if self.role == 'python' else '-e', code]) + '\n')
        log = self.root / 'cancel-probe.log'
        with log.open('wb') as output:
            child = subprocess.Popen(['docker', 'buildx', 'build', '--builder', self.builder,
                                      '--progress=plain', '--output=type=cacheonly', str(probe)],
                                     stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=self.docker_env)
            self.children.add(child)
            active = False
            try:
                for _ in range(120):
                    if child.poll() is not None:
                        break
                    data = measure()
                    if data['available_ram_bytes'] < EMERGENCY_RAM or data['docker_free_bytes'] < DOCKER_FREE:
                        raise RuntimeError('Resource stop during control probe')
                    if b'ISPEAK_CANCEL_ACTIVE' in log.read_bytes():
                        # Docker progress also echoes RUN source. Require stdout step prefix
                        # followed directly by the marker, not the instruction text.
                        if re.search(rb'#\d+ [0-9.]+ ISPEAK_CANCEL_ACTIVE\s', log.read_bytes()):
                            active = True
                            break
                    time.sleep(1)
                self.stop_builder()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                if not active or child.returncode == 0:
                    raise RuntimeError('Active-work cancellation probe did not establish failure/stop')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
                self.children.discard(child)
        state = json.loads(self.docker('inspect', 'buildx_buildkit_' + self.builder + '0'))[0]
        if state['State']['Running']:
            raise RuntimeError('Cancellation left builder running')
        self.docker('buildx', 'inspect', '--bootstrap', self.builder, seconds=120)
        self.evidence('active_work_cancellation', result='PASS', export_load_scope='separate review required')

    def stop_builder(self):
        name = 'buildx_buildkit_' + self.builder + '0'
        try:
            data = subprocess.check_output(['docker', 'inspect', name], stderr=subprocess.DEVNULL,
                                           timeout=10, env=self.docker_env)
            entry = json.loads(data)[0]
            if 'ISPEAK_RELEASE_RUN=' + self.key in entry['Config'].get('Env', []):
                subprocess.run(['docker', 'stop', '--time=3', entry['Id']], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10, env=self.docker_env, check=False)
        except (subprocess.SubprocessError, ValueError):
            pass  # The always() cleanup retries only cleanup, never a build.

    def cleanup(self):
        self.monitor = False
        self.stop_builder()
        for child in tuple(self.children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
        try:
            ids = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'label=' + self.label],
                                          timeout=10, env=self.docker_env).decode().split()
            if ids:
                subprocess.run(['docker', 'rm', '-f', *ids], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20, env=self.docker_env, check=False)
            subprocess.run(['docker', 'buildx', 'rm', self.builder], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=15, env=self.docker_env, check=False)
        except subprocess.SubprocessError:
            pass
        # Local unique candidate only; never RC1 names or remote tags/digests.
        subprocess.run(['docker', 'image', 'rm', self.image], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=15, env=self.docker_env, check=False)
        expected_parent = Path(os.environ['RUNNER_TEMP']).resolve()
        if self.root.parent == expected_parent and self.root.name == 'ispeak-validation-' + self.key:
            shutil.rmtree(self.root)

    def execute(self):
        if os.environ.get('AUTOMATION_REVIEW_APPROVED') != 'true':
            raise RuntimeError('Reviewed automation activation required')
        if self.role == 'python':
            from node_gate import require_node_success
            require_node_success()  # Before private pulls, builder startup, or inference.
        enforce(self.role, stable=True)  # Before source fetch, registry pull, builder bootstrap.
        repo, context = export(self.role, self.root, self.command)
        self.monitor = True
        if self.role == 'python':
            self.login()
        self.builder_start()
        self.cancel_probe()
        enforce(self.role, stable=True)  # Preserve after-builder stable headroom requirement.
        recipe = 'Dockerfile' if self.role == 'node' else 'docker/Dockerfile.v6-overlay'
        started = time.monotonic()
        output = self.docker('buildx', 'build', '--builder', self.builder, '--platform=linux/amd64',
                             '--progress=plain', '--provenance=false', '--sbom=false', '--load',
                             '--label', 'org.opencontainers.image.revision=' + COMMITS[self.role],
                             '--label', self.label, '-t', self.image, '-f', str(context / recipe),
                             str(context), seconds=900 if self.role == 'node' else 600)
        image = json.loads(self.docker('image', 'inspect', self.image))[0]
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
    validator = Validation(role)
    def interrupted(_signal, _frame):
        raise RuntimeError('Workflow cancellation requested')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if sys.argv[1:] != ['cleanup']:
            if sys.argv[1:] != [role]:
                raise RuntimeError('Unexpected execution mode')
            validator.execute()
    except Exception as error:
        # Never emit traceback, command arguments, response bodies or private model logs.
        print(json.dumps({'result': 'STOP', 'reason': str(error) if type(error) is RuntimeError else type(error).__name__,
                          'published': False}), flush=True)
        return 1
    finally:
        validator.cleanup()
    return 0


if __name__ == '__main__':
    sys.exit(main())
