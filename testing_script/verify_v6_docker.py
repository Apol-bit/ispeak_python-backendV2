"""Local offline acceptance test for versioned images; no builds or publishing."""
import argparse
import json
import wave
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'benchmark-results/v6-docker'
env = dict(os.environ)
EXPECTED_IMAGES = {
    'v6': ('ispeak-python-v2:v6-local', 'sha256:41e3779d97354831b6f5aa4a22b813ef5cf1b0d6dd93c3e6f36fe2f882a3857e'),
    'v5': ('ispeak-python-v2', 'sha256:787831cdcbf055eac5b707226dd648f49bb712724cb1696ef18a24744e157a14'),
}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path,
                        help='JSON manifest of separately provisioned synthetic WAVs; paths relative to the manifest')
    parser.add_argument('--validate-only', action='store_true',
                        help='Validate fixtures without Docker, model loading, inference, or output files')
    options = parser.parse_args(argv)
    try:
        manifest = options.samples.expanduser().resolve(strict=True)
        if not manifest.is_file():
            raise ValueError('manifest must be a JSON file')
        samples = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if not isinstance(samples, list) or not samples:
            raise ValueError('manifest must contain a nonempty sample list')
        languages = set()
        for sample in samples:
            if (not isinstance(sample, dict) or sample.get('purpose') != 'synthetic compatibility only'
                    or sample.get('language') not in ('English', 'Filipino', 'Taglish')
                    or not isinstance(sample.get('file'), str) or not sample['file']):
                raise ValueError('each sample needs a language, synthetic purpose, and relative WAV path')
            relative = Path(sample['file'])
            if relative.is_absolute() or '..' in relative.parts or '\\' in sample['file'] or ':' in sample['file']:
                raise ValueError('audio paths must stay relative to the manifest directory')
            audio = (manifest.parent / relative).resolve(strict=True)
            if not audio.is_relative_to(manifest.parent) or audio.suffix.lower() != '.wav':
                raise ValueError('audio must be a WAV inside the manifest directory')
            with wave.open(str(audio), 'rb') as wav:
                if wav.getnframes() <= 0 or wav.getframerate() <= 0:
                    raise ValueError('WAV must contain audio frames')
            languages.add(sample['language'])
        if languages != {'English', 'Filipino', 'Taglish'}:
            raise ValueError('provide synthetic samples for English, Filipino, and Taglish')
    except (OSError, ValueError, TypeError, wave.Error, EOFError):
        parser.error('Invalid fixtures: supply a readable JSON list and existing synthetic WAVs inside its directory for all three languages')
    return options, manifest, samples


def inspect_image(image, version):
    expected_name, expected_id = EXPECTED_IMAGES[version]
    info = json.loads(docker('image', 'inspect', image).stdout)[0]
    if image != expected_name or info['Id'] != expected_id:
        raise RuntimeError('Recorded image identity mismatch; refusing container startup')
    return info



def docker(*args, check=True, timeout=600, stdin=None):
    p = subprocess.run(['docker',*args],env=env,input=stdin,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    if check and p.returncode:
        raise RuntimeError('Docker command failed: '+args[0])
    return p


def stats(name):
    r=docker('stats','--no-stream','--format','{{json .}}',name,check=False,timeout=10)
    return json.loads(r.stdout) if r.returncode==0 and r.stdout.strip() else None


def verify(image,version,manifest):
    name='ispeak-'+version+'-'+uuid.uuid4().hex[:10]
    report={'image':image,'stats':[]}; started=False; finished=threading.Event()
    image_info=inspect_image(image, version)
    report.update(image_id=image_info['Id'],image_bytes=image_info['Size'],repo_digests=image_info['RepoDigests'])
    try:
        start=time.perf_counter()
        docker('run','-d','--name',name,'--network','none','--memory','6g','--cpus','4','--read-only',
               '--tmpfs','/tmp/ispeak:rw,noexec,nosuid,size=512m,uid=10001,gid=10001',
               '--mount',f'type=bind,source={manifest.parent},target=/samples,readonly',
               '--mount',f'type=bind,source={manifest},target=/verification/samples.json,readonly',
               '--mount',f'type=bind,source={ROOT / "testing_script"},target=/app/testing_script,readonly',
               '--mount',f'type=bind,source={ROOT / "model_manifests"},target=/verification/manifests,readonly',
               '-e','ISPEAK_SERVICE_TOKEN','-e','PORT=8089','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',image_info['Id'])
        started=True
        for _ in range(180):
            result=docker('exec',name,'python','-c',"import urllib.request;assert urllib.request.urlopen('http://127.0.0.1:8089/ready',timeout=2).status==200",check=False,timeout=10)
            if result.returncode==0: break
            time.sleep(1)
        else: raise RuntimeError('Startup timeout')
        report['startup_seconds']=time.perf_counter()-start
        report['idle_stats']=stats(name)
        def monitor():
            while not finished.is_set():
                value=stats(name)
                if value: report['stats'].append(value)
                finished.wait(.5)
        thread=threading.Thread(target=monitor); thread.start()
        code=r'''
import json,math,os,pathlib,time,urllib.request,urllib.error,uuid,hashlib
from model_integrity import configured_roots,verify_bundle
roots=configured_roots(); verification=verify_bundle(roots,os.environ['ISPEAK_MODEL_MANIFEST'])
result={'roots':{k:str(v) for k,v in roots.items()},'integrity':verification,'samples':[]}
url='http://127.0.0.1:8089'
result['unauthenticated_endpoints']={}
for endpoint in ['/metrics','/transcribe','/docs','/models/iSpeak_v6/adapter_model.safetensors']:
 try:
  status=urllib.request.urlopen(url+endpoint,timeout=3).status
 except urllib.error.HTTPError as e: status=e.code
 result['unauthenticated_endpoints'][endpoint]=status
 assert status==401
samples=json.loads(pathlib.Path('/verification/samples.json').read_text(encoding='utf-8-sig'))
for sample in samples:
 assert sample['purpose']=='synthetic compatibility only'
 file=pathlib.Path('/samples')/sample['file']
 boundary='test-'+uuid.uuid4().hex
 body=('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+file.read_bytes()+('\r\n--'+boundary+'\r\nContent-Disposition: form-data; name="language"\r\n\r\n'+sample['language']+'\r\n--'+boundary+'--\r\n').encode()
 request=urllib.request.Request(url+'/transcribe',data=body,headers={'Content-Type':'multipart/form-data; boundary='+boundary,'X-iSpeak-Service-Token':os.environ['ISPEAK_SERVICE_TOKEN']})
 start=time.perf_counter()
 with urllib.request.urlopen(request,timeout=360) as response: output=json.load(response)
 assert output['analysis_valid'] and output['transcription'].strip() and output['fillers']['analysis_available']
 assert all(math.isfinite(output['scores'][k]) for k in ['overall','clarity','pacing','energy'])
 words=output['word_timestamps']; assert words and all(0<=w['start']<=w['end']<=output['duration_seconds']+.1 for w in words)
 result['samples'].append({'language':sample['language'],'latency_seconds':time.perf_counter()-start,'transcription':output['transcription'],'timestamps':words,'scores':output['scores'],'response_keys':sorted(output)})
print(json.dumps(result))
'''
        response=docker('exec','-i',name,'python','-',stdin=code,timeout=1200)
        report['inference']=json.loads(response.stdout)
        logs=docker('logs',name)
        text=logs.stdout+logs.stderr
        report['logs_safe']=env['ISPEAK_SERVICE_TOKEN'] not in text and all(s['transcription'] not in text for s in report['inference']['samples'])
        assert report['logs_safe']
        (OUT/(version+'-container.log')).write_text(text,encoding='utf-8')
        if version=='v6':
            tests=docker('exec','-e','ISPEAK_ENV=development','-e','ISPEAK_SERVICE_TOKEN=',name,'python','-B','-m','unittest',
                         'testing_script.test_paired_evaluation','testing_script.test_model_integrity','testing_script.test_v6_candidate','testing_script.test_security_hardening',
                         'testing_script.test_readiness','testing_script.test_input_validation','testing_script.test_admission','testing_script.test_analysis_components','testing_script.test_container_admission')
            (OUT/'linux-integrity-tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
            docker('exec',name,'python','-m','pip','check')
        finished.set(); thread.join(timeout=15)
        docker('stop','-t','30',name)
        report['exit_code']=int(docker('inspect','--format','{{.State.ExitCode}}',name).stdout)
        assert report['exit_code']==0
    finally:
        finished.set()
        if started: docker('rm','-f',name,check=False)
        (OUT/(version+'-results.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'version':version,'startup_seconds':report['startup_seconds'],'image_id':report['image_id'],'inference':'PASS','private_authentication':all(s==401 for s in report['inference']['unauthenticated_endpoints'].values()),'logs_safe':report['logs_safe']}),flush=True)


def main(argv=None):
    options, manifest, samples = parse_arguments(argv)
    if options.validate_only:
        print(json.dumps({'fixtures': 'PASS', 'samples': len(samples), 'inference': 'NOT RUN'}))
        return 0
    # Verify both recorded immutable identities before starting either container.
    for version, (image, _) in EXPECTED_IMAGES.items():
        inspect_image(image, version)
    OUT.mkdir(parents=True, exist_ok=True)
    env['ISPEAK_SERVICE_TOKEN'] = secrets.token_hex(32)
    for version, (image, _) in EXPECTED_IMAGES.items():
        verify(image, version, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
