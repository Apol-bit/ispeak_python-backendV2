"""Local-only Docker acceptance run. Never deploys, pushes or changes Node config."""
from pathlib import Path
import concurrent.futures
import io
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.request
import urllib.error
import uuid
import wave
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'benchmark-results' / 'docker-local'
OUT.mkdir(parents=True, exist_ok=True)
NAME = 'ispeak-python-v2-verify'
IMAGE = 'ispeak-python-v2:local'
BASE = 'http://127.0.0.1:8000'
results = {}
phase = 'startup'
samples = []
SERVICE_TOKEN = os.getenv('ISPEAK_SERVICE_TOKEN', '')

def docker(*args, check=True):
    p = subprocess.run(['docker', *args], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if check and p.returncode:
        raise RuntimeError(p.stderr + p.stdout)
    return (p.stdout + p.stderr if args and args[0] == 'logs' else p.stdout).strip()

def save():
    (OUT / 'results.json').write_text(json.dumps(results, indent=2))
    (OUT / 'samples.json').write_text(json.dumps(samples, indent=2))

def emit(key, value):
    results[key] = value
    save()
    print(key, json.dumps(value), flush=True)

def health():
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=2) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    except (OSError, TimeoutError):
        return None, None

def ready():
    started = time.perf_counter()
    observed = []
    while time.perf_counter() - started < 300:
        code, body = health()
        if code not in observed:
            observed.append(code)
        if code == 200:
            return {'seconds': time.perf_counter()-started, 'observed_statuses': observed, 'health': body}
        time.sleep(1)
    raise RuntimeError('Readiness deadline exceeded: ' + docker('logs', NAME))

def post(data, reference=False, filename='sample.aac'):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for field in (['file', 'reference_audio'] if reference else ['file']):
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
        body.extend(data)
        body.extend(b'\r\n')
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\nEnglish\r\n--{boundary}--\r\n'.encode())
    req = urllib.request.Request(BASE + '/transcribe', data=body, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'X-iSpeak-Service-Token': SERVICE_TOKEN,
    })
    started = time.perf_counter()
    try:
        response = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        response = e
    with response:
        return {'status': response.code, 'seconds': time.perf_counter()-started,
                'body': json.load(response), 'processing_ms': response.headers.get('X-Analysis-Processing-Ms'),
                'queue_ms': response.headers.get('X-Analysis-Queue-Wait-Ms')}

MONITOR = '''import json,time
from pathlib import Path
root=Path('/sys/fs/cgroup')
previous=None
while True:
 try:
  now=time.monotonic(); cpu=int(dict(x.split() for x in (root/'cpu.stat').read_text().splitlines())['usage_usec'])
  memory=int((root/'memory.current').read_text()); stats=dict(x.split() for x in (root/'memory.stat').read_text().splitlines())
  files=list(Path('/tmp/ispeak').glob('*')); disk=sum(p.stat().st_size for p in files if p.is_file())
  value={'memory_bytes':memory-int(stats.get('inactive_file',0)),'cgroup_memory_bytes':memory,'memory_peak_bytes':int((root/'memory.peak').read_text()),'temp_audio_bytes':disk,'temp_audio_files':len([p for p in files if p.is_file()])}
  if previous:value['cpu_percent']=(cpu-previous[1])/(now-previous[0])/10000
  previous=(now,cpu); print(json.dumps(value),flush=True); time.sleep(.5)
 except Exception as exc: print(json.dumps({'error':str(exc)}),flush=True);time.sleep(1)
'''

def monitor():
    process = subprocess.Popen(['docker','exec',NAME,'python','-u','-c',MONITOR], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8')
    def collect():
        for line in process.stdout:
            value = json.loads(line)
            value.update(phase=phase, wall_time=time.time())
            samples.append(value)
    threading.Thread(target=collect, daemon=True).start()
    return process

def summarize():
    summary = {}
    for p in sorted({s['phase'] for s in samples}):
        selected = [s for s in samples if s['phase']==p and 'memory_bytes' in s]
        if not selected: continue
        cpu = [s['cpu_percent'] for s in selected if 'cpu_percent' in s]
        summary[p] = {'sample_count':len(selected), 'memory_max_gib':max(s['memory_bytes'] for s in selected)/2**30,
                      'memory_last_gib':selected[-1]['memory_bytes']/2**30,
                      'cpu_mean_percent':statistics.mean(cpu) if cpu else None,
                      'cpu_peak_percent':max(cpu) if cpu else None,
                      'temp_peak_bytes':max(s['temp_audio_bytes'] for s in selected)}
    emit('measurements', summary)

def wav(seconds, tone=False):
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000)
        f.writeframes(b''.join(struct.pack('<h',int(5000*math.sin(2*math.pi*220*i/16000)) if tone else 0) for i in range(int(seconds*16000))))
    return buffer.getvalue()

def main():
    global phase
    existing=docker('container','inspect',NAME,check=False)
    if existing and json.loads(existing):
        raise RuntimeError('Verification container already exists; inspect before rerunning.')
    emit('image',json.loads(docker('image','inspect',IMAGE))[0]['Size'])
    docker('run','-d','--name',NAME,'--memory','6g','--cpus','4','--read-only','-e','ISPEAK_SERVICE_TOKEN',
           '--tmpfs','/tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001',
           '-e','PORT=8080','-e','ISPEAK_PORT=8999','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
           '-p','127.0.0.1:8000:8080',IMAGE)
    sampler=monitor()
    try:
        emit('cold_start',ready())
        phase='loaded';time.sleep(3)
        emit('loaded_memory',samples[-1])
        emit('loaded_docker_stats',docker('stats','--no-stream','--format','{{json .}}',NAME))
        emit('runtime',docker('exec',NAME,'python','-c',"import sys,importlib.metadata as m,torch;print(sys.version);print({p:m.version(p) for p in ['torch','transformers','peft','accelerate','librosa','numba','llvmlite','fastapi','uvicorn','psutil']});print('cuda',torch.cuda.is_available())"))
        emit('pip_check',docker('exec',NAME,'python','-m','pip','check'))
        audio=(ROOT/'temp_audio_folder'/'temporary_audio.aac').read_bytes()
        phase='inference_4cpu';emit('inference',post(audio))
        phase='settled';time.sleep(5)
        emit('settled_memory',samples[-1])
        emit('temp_after_inference',docker('exec',NAME,'sh','-c','find /tmp/ispeak -maxdepth 1 -type f; du -sk /tmp/ispeak/*'))
        phase='reference';emit('reference_self_comparison',post(audio,reference=True))
        phase='invalid'
        emit('invalid', {name:post(data,filename='invalid.wav') for name,data in
                         [('malformed',b'not audio'),('short',wav(.25,True)),('silence',wav(3)),('tone',wav(3,True)),('oversized',b'RIFF'+b'0'*(26*1024*1024))]})
        phase='queue'
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            future=[pool.submit(post,audio) for _ in range(6)]
            emit('queue_requests',[f.result() for f in future])
        emit('queue_health',health()[1])
        phase='node'
        node_script="const fs=require('fs'),axios=require('axios'),FormData=require('form-data'); const {usableAnalysis}=require('./services/analysisResponse'); const form=new FormData();form.append('file',fs.createReadStream(process.env.SAMPLE));form.append('language','English');axios.post(process.env.PYTHON_BACKEND_URL+'/transcribe',form,{headers:{...form.getHeaders(),'X-iSpeak-Service-Token':process.env.ISPEAK_SERVICE_TOKEN},maxRedirects:0,timeout:300000,maxBodyLength:51*1024*1024}).then(r=>{console.log(JSON.stringify({status:r.status,usable:usableAnalysis(r.data),scores:r.data.scores}));if(!usableAnalysis(r.data))process.exitCode=1}).catch(e=>{require('./services/safeLogger').logError('docker_transport_failed',e);process.exitCode=1})"
        env=dict(os.environ,PYTHON_BACKEND_URL=BASE,SAMPLE=str(ROOT/'temp_audio_folder'/'temporary_audio.aac'))
        node=subprocess.run(['node','-e',node_script],cwd=ROOT.parent/'ispeak_backend',env=env,capture_output=True,text=True,encoding='utf-8',errors='replace')
        emit('node_transport',{'exit_code':node.returncode,'stdout':node.stdout,'stderr':node.stderr,'scope':'Actual Node axios/form-data transport and usableAnalysis validator; no authenticated database session persisted.'})
        emit('disk',docker('exec',NAME,'sh','-c','du -sk /app/models /usr/local/lib/python3.13/site-packages /tmp/ispeak; id; find /app -maxdepth 2 -type f | sort'))
        (OUT/'first-start.log').write_text(docker('logs',NAME),encoding='utf-8')
        docker('stop',NAME);sampler.wait(timeout=30)
        docker('update','--cpus','8',NAME)
        phase='restart'
        docker('start',NAME);sampler=monitor();emit('restart',ready())
        phase='inference_8cpu';emit('restart_inference',post(audio))
        phase='final_settled';time.sleep(5)
        emit('final_health',health()[1])
        emit('final_temp',docker('exec',NAME,'sh','-c','find /tmp/ispeak -maxdepth 1 -type f; du -sk /tmp/ispeak/*'))
    finally:
        summarize()
        (OUT/'container.log').write_text(docker('logs',NAME,check=False),encoding='utf-8')
        docker('stop',NAME,check=False)
        sampler.wait(timeout=30)
        save()

def benchmark_eight_cpu():
    """Separate eight-thread run so quota-only restart is not mistaken for scaling."""
    global NAME, BASE, OUT, phase
    NAME = 'ispeak-python-v2-eight-cpu'
    BASE = 'http://127.0.0.1:8001'
    OUT = OUT / 'eight-cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    existing = docker('container', 'inspect', NAME, check=False)
    if existing and json.loads(existing):
        raise RuntimeError('Eight-CPU benchmark container already exists.')
    docker('run','-d','--name',NAME,'--memory','6g','--cpus','8','--read-only','-e','ISPEAK_SERVICE_TOKEN',
           '--tmpfs','/tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001',
           '-e','PORT=8080','-e','OMP_NUM_THREADS=8','-e','MKL_NUM_THREADS=8',
           '-p','127.0.0.1:8001:8080',IMAGE)
    sampler = monitor()
    try:
        emit('cold_start', ready())
        audio = (ROOT/'temp_audio_folder'/'temporary_audio.aac').read_bytes()
        phase = 'eight_threads_cold_inference'
        emit('cold_inference',post(audio))
        phase = 'eight_threads_warm_inference'
        emit('warm_inferences',[post(audio),post(audio)])
    finally:
        summarize()
        (OUT/'container.log').write_text(docker('logs',NAME,check=False),encoding='utf-8')
        docker('stop',NAME,check=False)
        sampler.wait(timeout=30)
        save()


if __name__=='__main__':
    if len(SERVICE_TOKEN) < 32:
        raise RuntimeError('Set ISPEAK_SERVICE_TOKEN privately before running the local verification harness')
    if '--eight-cpu-benchmark' in sys.argv:
        benchmark_eight_cpu()
    else:
        main()
