"""Sequential, offline v5/v6 compatibility comparison on approved synthetic WAVs.
No default environment, original model, or trusted manifest is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
parser = argparse.ArgumentParser()
parser.add_argument('--candidate', required=True)
parser.add_argument('--samples', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--child-model', choices=['v5', 'v6'])
options = parser.parse_args()
candidate = Path(options.candidate).resolve()
output = Path(options.output).resolve()
samples_file = Path(options.samples).resolve()
samples = json.loads(samples_file.read_text(encoding='utf-8-sig'))
if not output.is_relative_to(ROOT / 'benchmark-results'):
    raise SystemExit('Output must remain inside ignored benchmark-results')
output.mkdir(parents=True, exist_ok=True)
if any(sample.get('purpose') != 'synthetic compatibility only' or sample.get('ground_truth_verified') for sample in samples):
    raise SystemExit('This harness accepts only the explicitly approved synthetic compatibility samples')


def child():
    # Hard failure on outbound network attempts, in addition to HF offline flags.
    original_connection = socket.create_connection
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    def local_address(address):
        # Windows asyncio uses a loopback socket pair internally. Allow only
        # literal loopback addresses; cloud/model downloads remain blocked.
        return isinstance(address, tuple) and address[0] in {'127.0.0.1', '::1'}
    def guarded_connection(address, *args, **kwargs):
        if not local_address(address):
            raise RuntimeError('External network disabled during candidate verification')
        return original_connection(address, *args, **kwargs)
    def guarded_connect(sock, address):
        if not local_address(address):
            raise RuntimeError('External network disabled during candidate verification')
        return original_connect(sock, address)
    def guarded_connect_ex(sock, address):
        if not local_address(address):
            raise RuntimeError('External network disabled during candidate verification')
        return original_connect_ex(sock, address)
    socket.create_connection = guarded_connection
    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    from model_integrity import configured_roots, verify_bundle
    verification = verify_bundle(configured_roots(), os.environ['ISPEAK_MODEL_MANIFEST'])
    import numpy as np
    import torch
    np.random.seed(0); torch.manual_seed(0)
    from model import ISpeakWhisper
    started = time.perf_counter()
    model = ISpeakWhisper()
    model_seconds = time.perf_counter() - started
    from audio_analysis_processing_files.filler_words import _default_classifier
    _default_classifier()
    from fastapi.testclient import TestClient
    import fastapi_backend as backend
    # Supply the actually loaded model without triggering a second lifespan load.
    backend.app.state.model = model
    backend.app.state.filler_ready = True
    backend.app.state.model_error = backend.app.state.filler_error = None
    client = TestClient(backend.app)
    assert client.get('/health').status_code == 200
    assert client.get('/metrics').status_code == 401
    results = []
    keys = None
    for sample in samples:
        wav = Path(sample['file']).read_bytes()
        request_started = time.perf_counter()
        response = client.post('/transcribe', files={'file': ('generated.wav', wav, 'audio/wav')}, data={'language': sample['language']}, headers={'X-iSpeak-Service-Token': os.environ['ISPEAK_SERVICE_TOKEN']})
        elapsed = time.perf_counter() - request_started
        body = response.json()
        entry = {'language': sample['language'], 'sampleSha256': hashlib.sha256(wav).hexdigest(), 'status': response.status_code, 'latencySeconds': round(elapsed, 3)}
        if response.status_code != 200:
            entry['result'] = 'FAIL'
            entry['validationCode'] = body.get('detail', {}).get('code') if isinstance(body.get('detail'), dict) else None
            results.append(entry)
            continue
        timestamps = body['word_timestamps']
        valid_times = bool(timestamps) and all(isinstance(word.get('start'), (int, float)) and isinstance(word.get('end'), (int, float)) and 0 <= word['start'] <= word['end'] <= body['duration_seconds'] + 0.1 for word in timestamps)
        monotonic = all(a['start'] <= b['start'] for a, b in zip(timestamps, timestamps[1:]))
        valid = body['analysis_valid'] is True and bool(body['transcription'].strip()) and body['fillers']['analysis_available'] is True and all(np.isfinite(body['scores'][key]) for key in ['overall', 'pacing', 'clarity', 'energy']) and valid_times and monotonic
        entry.update(result='PASS' if valid else 'FAIL', transcription=body['transcription'], wordCount=len(timestamps), durationSeconds=body['duration_seconds'], timestamps=timestamps, timestampsValid=valid_times and monotonic, fillerAvailable=body['fillers']['analysis_available'], scores=body['scores'], responseKeys=sorted(body))
        if keys is None:
            keys = set(body)
        assert set(body) == keys
        results.append(entry)
    client.close()
    report = {'model': options.child_model, 'integrity': verification, 'offlineNetworkBlocked': True, 'processorClass': type(model.processor).__name__, 'tokenizerClass': type(model.processor.tokenizer).__name__, 'peftMerged': not any('lora_A' in name or 'lora_B' in name for name, _ in model.model.named_parameters()), 'modelInitializationSeconds': round(model_seconds, 3), 'torchThreads': torch.get_num_threads(), 'results': results, 'werCer': 'NOT CALCULATED: no independently verified ground truth; synthetic compatibility only'}
    (output / (options.child_model + '.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'model': options.child_model, 'samples': [{'language': item['language'], 'result': item['result'], 'latencySeconds': item['latencySeconds']} for item in results]}), flush=True)
    return 0 if all(item['result'] == 'PASS' for item in results) and report['peftMerged'] else 1


def parent():
    import psutil
    summary = {'candidate': str(candidate), 'models': [], 'groundTruthAvailable': False}
    for version in ['v5', 'v6']:
        adapter = ROOT / 'models' / 'iSpeak_v5' if version == 'v5' else candidate / 'adapter'
        manifest = ROOT / 'model_manifests' / 'ispeak-v5.json' if version == 'v5' else candidate / 'integrity.json'
        # Both processes use the exact same copied base/filler and sample bytes.
        environment = dict(os.environ, ISPEAK_MODEL_PATH=str(adapter), ISPEAK_BASE_MODEL_PATH=str(candidate / 'adapter' / 'base_model'), ISPEAK_FILLER_MODEL_PATH=str(candidate / 'filler'), ISPEAK_MODEL_MANIFEST=str(manifest), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ISPEAK_ENV='production', ISPEAK_SERVICE_TOKEN=secrets.token_hex(32), ISPEAK_CORS_ORIGINS='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=str(output / 'numba-cache'))
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--candidate', str(candidate), '--samples', str(samples_file), '--output', str(output), '--child-model', version]
        peak = 0
        with (output / (version + '.local.log')).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
            monitor = psutil.Process(process.pid)
            deadline = time.monotonic() + 900
            while process.poll() is None:
                try:
                    rss = monitor.memory_info().rss
                    for descendant in monitor.children(recursive=True):
                        rss += descendant.memory_info().rss
                    peak = max(peak, rss)
                except psutil.Error:
                    pass
                if time.monotonic() > deadline:
                    process.terminate(); process.wait(timeout=30)
                    raise RuntimeError('Isolated candidate process exceeded its test deadline')
                time.sleep(0.1)
        entry = {'model': version, 'exitCode': process.returncode, 'sampledPeakRssBytes': peak}
        file = output / (version + '.json')
        if file.is_file():
            report = json.loads(file.read_text(encoding='utf-8'))
            entry.update(initializationSeconds=report['modelInitializationSeconds'], samples=[{key: result.get(key) for key in ['language', 'result', 'latencySeconds', 'wordCount', 'timestampsValid', 'fillerAvailable']} for result in report['results']])
        summary['models'].append(entry)
        print(json.dumps(entry), flush=True)
    if all(model['exitCode'] == 0 for model in summary['models']):
        baseline = json.loads((output / 'v5.json').read_text(encoding='utf-8'))
        candidate_report = json.loads((output / 'v6.json').read_text(encoding='utf-8'))
        summary['sameApiResponseKeys'] = all(a['responseKeys'] == b['responseKeys'] and a['sampleSha256'] == b['sampleSha256'] for a, b in zip(baseline['results'], candidate_report['results']))
        summary['comparison'] = [{'language': a['language'], 'sameTranscript': a['transcription'] == b['transcription'], 'sameTimestamps': a['timestamps'] == b['timestamps']} for a, b in zip(baseline['results'], candidate_report['results'])]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return 0 if all(model['exitCode'] == 0 for model in summary['models']) and summary.get('sameApiResponseKeys') else 1


if __name__ == '__main__':
    try:
        raise SystemExit(child() if options.child_model else parent())
    except Exception as error:
        print(json.dumps({'failureCategory': type(error).__name__}), flush=True)
        raise SystemExit(1)
