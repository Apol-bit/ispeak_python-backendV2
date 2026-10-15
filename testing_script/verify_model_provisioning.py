"""Opt-in local verification of the integrity-enforced private image. No uploads."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import tempfile
import time
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--speech-file', required=True, help='Generated synthetic WAV only')
parser.add_argument('--image', default='ispeak-python-v2:model-provisioning-local')
options = parser.parse_args()
repo = Path(__file__).resolve().parent.parent
speech = Path(options.speech_file).resolve()
if not speech.is_file():
    raise SystemExit('Generated synthetic speech file is required')
name = 'ispeak-model-probe-' + uuid.uuid4().hex[:12]
environment = dict(os.environ, ISPEAK_SERVICE_TOKEN=secrets.token_hex(32))
report = {'imageTag': options.image}


def docker(*args, timeout=60, required=True):
    result = subprocess.run(['docker', *args], env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if required and result.returncode:
        raise RuntimeError('Local Docker verification operation failed')
    return result


started = False
try:
    with tempfile.TemporaryDirectory(prefix='ispeak-integrity-negative-') as directory:
        fixture = Path(directory)
        (fixture / 'model.safetensors').write_bytes(b'corrupted synthetic fixture; not a real weight')
        for case, mount in [
            ('missingBundleRejected', f'type=bind,source={fixture},target=/app/models/filler_classifier,readonly'),
            ('corruptWeightRejected', f'type=bind,source={fixture / "model.safetensors"},target=/app/models/iSpeak_v5/base_model/model.safetensors,readonly'),
        ]:
            result = docker('run', '--rm', '--network', 'none', '--mount', mount, '-e', 'ISPEAK_SERVICE_TOKEN', options.image, required=False)
            assert result.returncode == 1 and 'Model integrity failed:' in result.stderr
            assert environment['ISPEAK_SERVICE_TOKEN'] not in result.stdout + result.stderr
            report[case] = 'PASS'
    started_at = time.monotonic()
    docker('run', '-d', '--name', name, '--network', 'none', '--memory', '6g', '--cpus', '4', '--read-only', '--tmpfs', '/tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001',
           '--mount', f'type=bind,source={speech},target=/verification/generated-speech.wav,readonly',
           '--mount', f'type=bind,source={repo / "testing_script" / "verify_provisioned_runtime.py"},target=/verification/verify_provisioned_runtime.py,readonly',
           '-e', 'ISPEAK_SERVICE_TOKEN', '-e', 'PORT=8087', '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4', options.image)
    started = True
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        health = docker('exec', name, 'python', '-c', "import urllib.request;assert urllib.request.urlopen('http://127.0.0.1:8087/health',timeout=3).status==200", required=False, timeout=10)
        if health.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError('Provisioned model readiness timeout')
    report['startupSecondsIncludingIntegrity'] = round(time.monotonic() - started_at, 2)
    probe = docker('exec', name, 'python', '/verification/verify_provisioned_runtime.py', timeout=600)
    report.update(json.loads(probe.stdout.strip().splitlines()[-1]))
    report['pipCheck'] = 'PASS' if docker('exec', name, 'python', '-m', 'pip', 'check').returncode == 0 else 'FAIL'
    docker('exec', name, 'sh', '-c', 'test ! -e /app/.env && test ! -e /app/evaluation_results.json && test ! -e /app/models/iSpeak_v5/training_args.bin')
    report['imageExclusions'] = 'PASS'
    log = docker('logs', name)
    assert environment['ISPEAK_SERVICE_TOKEN'] not in log.stdout + log.stderr
    report['secretAbsentFromLogs'] = 'PASS'
    docker('stop', '-t', '15', name)
    assert docker('inspect', '--format', '{{.State.ExitCode}}', name).stdout.strip() == '0'
    report['shutdown'] = 'PASS'
except Exception as error:
    report['failureCategory'] = type(error).__name__
finally:
    if started:
        report['containerRemoved'] = docker('rm', '-f', name, required=False).returncode == 0
    report['imageId'] = docker('image', 'inspect', options.image, '--format', '{{.Id}}').stdout.strip()
    report['imageSizeBytes'] = int(docker('image', 'inspect', options.image, '--format', '{{.Size}}').stdout.strip())
    (repo / 'benchmark-results' / 'model-provisioning').mkdir(parents=True, exist_ok=True)
    (repo / 'benchmark-results' / 'model-provisioning' / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if 'failureCategory' in report:
    raise SystemExit(1)
