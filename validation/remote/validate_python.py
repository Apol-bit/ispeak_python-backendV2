"""Offline RC1 verification and one bounded synthetic-speech inference."""
import hashlib
import json
import secrets
import shutil
import time

from preflight import enforce


def validate(v, _repo, context):
    enforce('python', stable=True)  # Separate runtime headroom gate.
    speech = shutil.which('espeak-ng') or shutil.which('espeak')
    if not speech:
        raise RuntimeError('Synthetic speech tool unavailable; no installation or human recording fallback')
    wav = v.root / 'synthetic.wav'
    v.command([speech, '-s', '135', '-w', str(wav),
               'This is a short speech recording for a safe system validation.'], seconds=20)
    wav.chmod(0o644)
    code = """import hashlib,json,pathlib,subprocess,sys
expected=json.loads(sys.argv[1])
for name,digest in expected.items():
 assert hashlib.sha256((pathlib.Path('/app')/name).read_bytes()).hexdigest()==digest
assert not any((pathlib.Path('/app')/name).exists() for name in ('.env','.git','recordings'))
subprocess.run([sys.executable,'-m','pip','check'],check=True)
subprocess.run([sys.executable,'/app/model_integrity.py'],check=True)
"""
    hashes = {item.relative_to(context).as_posix(): hashlib.sha256(item.read_bytes()).hexdigest()
              for item in context.rglob('*.py')}
    v.docker(*v.run_args('python-integrity-' + v.key, memory='4g'), '--entrypoint=python',
             v.image, '-c', code, json.dumps(hashes), seconds=180)
    status, diagnostic = v.docker(*v.run_args('python-invalid-' + v.key, memory='4g'),
                         '--entrypoint=python', v.image, '-c', 'import fastapi_backend',
                         seconds=60, check=False)
    if status == 0 or b'ISPEAK_SERVICE_TOKEN' not in diagnostic:
        raise RuntimeError('Python accepted missing production service token')
    env = v.root / 'python-fixture.env'
    env.write_text('ISPEAK_SERVICE_TOKEN=' + secrets.token_hex(32) + '\nISPEAK_ENV=production\nPORT=8080\n'
                   'ISPEAK_CORS_ORIGINS=\nHF_HUB_OFFLINE=1\nTRANSFORMERS_OFFLINE=1\n'
                   'OMP_NUM_THREADS=2\nMKL_NUM_THREADS=2\nISPEAK_DEBUG_ERRORS=false\n')
    env.chmod(0o600)
    name = 'python-health-' + v.key
    v.docker(*v.run_args(name, memory='8000000000'), '--read-only',
             '--tmpfs', '/tmp/ispeak:rw,size=512m,mode=1777', '--env-file', str(env),
             '--mount', f'type=bind,src={wav},dst=/synthetic.wav,readonly', '-d', v.image)
    ready = "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8080/ready',timeout=4).status==200"
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        status, _ = v.docker('exec', name, 'python', '-c', ready, seconds=8, check=False)
        if status == 0:
            break
        state = json.loads(v.docker('inspect', name))[0]['State']
        if not state['Running'] or state.get('OOMKilled'):
            raise RuntimeError('Python startup exited or was OOM-killed')
        time.sleep(2)
    else:
        raise RuntimeError('Python readiness deadline failed')
    inference = '''import json,math,os,urllib.request,urllib.error
url='http://127.0.0.1:8080'
def request(path,data=None,authorized=True,content_type=None):
 headers={}
 if authorized: headers['X-iSpeak-Service-Token']=os.environ['ISPEAK_SERVICE_TOKEN']
 if content_type: headers['Content-Type']=content_type
 try:
  with urllib.request.urlopen(urllib.request.Request(url+path,data=data,headers=headers),timeout=120) as r:
   return r.status,r.read()
 except urllib.error.HTTPError as r:return r.code,r.read()
assert request('/transcribe',b'',False)[0]==401
assert request('/openapi.json')[0]==404
boundary='ispeak-disposable-fixture'
newline=chr(13)+chr(10)
def form(duration,maximum=30):
 parts=[]
 for name,value in [('language','English'),('max_duration_seconds',str(maximum)),('duration_seconds',str(duration))]:
  parts.append(newline.join(['--'+boundary,'Content-Disposition: form-data; name="'+name+'"','',value,'']).encode())
 parts.append(newline.join(['--'+boundary,'Content-Disposition: form-data; name="file"; filename="synthetic.wav"','Content-Type: audio/wav','','']).encode()+open('/synthetic.wav','rb').read()+newline.encode())
 parts.append(('--'+boundary+'--'+newline).encode());return b''.join(parts)
assert request('/transcribe',form(301),content_type='multipart/form-data; boundary='+boundary)[0]==400
# False short metadata must not bypass bounded FFmpeg decoding.
status,body=request('/transcribe',form(0.1,1),content_type='multipart/form-data; boundary='+boundary)
assert status==400 and json.loads(body)['detail']['code']=='recording_too_long'
status,body=request('/transcribe',form(8),content_type='multipart/form-data; boundary='+boundary)
assert status==200
result=json.loads(body)
assert isinstance(result.get('transcription'),str) and result['transcription'].strip()
assert result.get('fillers',{}).get('analysis_available') is True
assert isinstance(result.get('assessment_details'),dict)
assert isinstance(result.get('scores'),dict)
overall=result['scores'].get('overall')
assert isinstance(overall,(float,int)) and not isinstance(overall,bool) and math.isfinite(overall) and 0<=overall<=100
assert request('/ready')[0]==200
'''
    v.docker('exec', name, 'python', '-c', inference, seconds=180)
    state = json.loads(v.docker('inspect', name))[0]['State']
    if state.get('Health', {}).get('Status') != 'healthy' or state.get('OOMKilled'):
        raise RuntimeError('Python Docker health or memory check failed')
    v.docker('stop', '--time=15', name, seconds=25)
    if json.loads(v.docker('inspect', name))[0]['State']['ExitCode'] != 0:
        raise RuntimeError('Python graceful shutdown failed')
    v.evidence('python_runtime', result='PASS', port=8080, checksums=True, dependency_check=True,
               inherited_base_layers=True, synthetic_inference_http=200,
               recording_or_transcript_logged=False, external_network=False)
