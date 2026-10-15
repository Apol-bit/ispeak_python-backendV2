"""Offline Linux candidate smoke test using read-only overlays on an existing image."""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
import uuid

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True)
parser.add_argument('--speech-file', required=True)
parser.add_argument('--output', required=True)
options = parser.parse_args()
root = Path(__file__).resolve().parent.parent
candidate = Path(options.candidate).resolve()
speech = Path(options.speech_file).resolve()
output = Path(options.output).resolve()
if not output.is_relative_to(root / 'benchmark-results'):
    raise SystemExit('Output must remain in ignored benchmark-results')
name = 'ispeak-v6-linux-' + uuid.uuid4().hex[:12]
environment = dict(os.environ, ISPEAK_SERVICE_TOKEN=secrets.token_hex(32))
report = {}


def docker(*args, required=True, timeout=60):
    result = subprocess.run(['docker', *args], env=environment, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)
    if required and result.returncode:
        raise RuntimeError('Offline Linux candidate check failed')
    return result


started = False
try:
    mounts = []
    for source, target in [(candidate, '/candidate'), (root / 'model.py', '/app/model.py'), (root / 'model_integrity.py', '/app/model_integrity.py'), (speech, '/verification/generated.wav')]:
        mounts += ['--mount', f'type=bind,source={source},target={target},readonly']
    docker('run', '-d', '--name', name, '--network', 'none', '--memory', '6g', '--cpus', '4', '--read-only', '--tmpfs', '/tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001', *mounts,
           '-e', 'ISPEAK_SERVICE_TOKEN', '-e', 'PORT=8089', '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
           '-e', 'ISPEAK_MODEL_PATH=/candidate/adapter', '-e', 'ISPEAK_BASE_MODEL_PATH=/candidate/adapter/base_model',
           '-e', 'ISPEAK_FILLER_MODEL_PATH=/candidate/filler', '-e', 'ISPEAK_MODEL_MANIFEST=/candidate/integrity.json',
           'ispeak-python-v2:model-provisioning-local')
    started = True
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        probe = docker('exec', name, 'python', '-c', "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8089/health',timeout=3).status==200", required=False, timeout=10)
        if probe.returncode == 0:
            break
        time.sleep(2)
    else:
        raise RuntimeError('Candidate readiness timeout')
    code = """import json,math,os,pathlib,urllib.request,urllib.error,uuid
url='http://127.0.0.1:8089'
try:
 urllib.request.urlopen(url+'/metrics',timeout=3)
 raise AssertionError('Unauthenticated request accepted')
except urllib.error.HTTPError as e:
 assert e.code==401
boundary='candidate-'+uuid.uuid4().hex
body=('--'+boundary+'\\r\\nContent-Disposition: form-data; name="file"; filename="generated.wav"\\r\\nContent-Type: audio/wav\\r\\n\\r\\n').encode()+pathlib.Path('/verification/generated.wav').read_bytes()+('\\r\\n--'+boundary+'\\r\\nContent-Disposition: form-data; name="language"\\r\\n\\r\\nEnglish\\r\\n--'+boundary+'--\\r\\n').encode()
request=urllib.request.Request(url+'/transcribe',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary,'X-iSpeak-Service-Token':os.environ['ISPEAK_SERVICE_TOKEN']})
with urllib.request.urlopen(request,timeout=300) as response:
 assert response.status==200
 result=json.load(response)
assert result['analysis_valid'] and result['transcription'].strip() and result['word_timestamps'] and result['fillers']['analysis_available']
assert all(math.isfinite(result['scores'][k]) for k in ['overall','pacing','clarity','energy'])
print(json.dumps({'offlineCandidateLoading':'PASS','health':'PASS','privateAuthentication':'PASS','englishInference':'PASS','timestampsAndFiller':'PASS','librosa':'PASS','responseKeys':sorted(result)}))
"""
    result = docker('exec', name, 'python', '-c', code, timeout=360)
    report.update(json.loads(result.stdout))
    logs = docker('logs', name)
    assert environment['ISPEAK_SERVICE_TOKEN'] not in logs.stdout + logs.stderr
    report['secretAbsentFromLogs'] = True
    docker('stop', '-t', '15', name)
    assert docker('inspect', '--format', '{{.State.ExitCode}}', name).stdout.strip() == '0'
    report['shutdown'] = 'PASS'
except Exception as error:
    report['failureCategory'] = type(error).__name__
finally:
    if started:
        report['containerRemoved'] = docker('rm', '-f', name, required=False).returncode == 0
    output.mkdir(parents=True, exist_ok=True)
    (output / 'linux-candidate.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
if 'failureCategory' in report:
    raise SystemExit(1)
