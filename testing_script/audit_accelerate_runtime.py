"""Offline, test-only Accelerate call audit; never records speech or credentials.

Run in the verified image with this file, the designated AAC fixture and existing
baseline results mounted read-only. No dependency or runtime model is changed.
"""
import hashlib
import importlib.metadata
import json
import sys
import threading
from collections import Counter
from pathlib import Path

phase = 'application_import'
calls = {}
vulnerable_calls = Counter()
snapshots = {}


def profile(frame, event, arg):
    if event != 'call':
        return
    module = frame.f_globals.get('__name__', '')
    if not module.startswith('accelerate.'):
        return
    name = frame.f_code.co_name
    stage = phase
    ancestor = frame.f_back
    # Only static function/module names are inspected, never arguments/locals.
    while ancestor is not None:
        parent_module = ancestor.f_globals.get('__name__', '')
        parent_name = ancestor.f_code.co_name
        if parent_module.startswith('peft.') and parent_name == 'merge_and_unload':
            stage = 'peft_merge'
            break
        if parent_module.startswith('peft.') and parent_name in {'from_pretrained', 'load_adapter'}:
            stage = 'peft_adapter_loading'
            break
        if parent_module == 'transformers.pipelines' and parent_name == 'pipeline':
            stage = 'pipeline_creation'
            break
        ancestor = ancestor.f_back
    calls.setdefault(stage, Counter())[f'{module}.{name}'] += 1
    if (module, name) in {
        ('accelerate.utils.modeling', 'load_checkpoint_in_model'),
        ('accelerate.big_modeling', 'load_checkpoint_and_dispatch'),
    }:
        vulnerable_calls[f'{module}.{name}'] += 1


def snapshot(stage):
    snapshots[stage] = {'imported': 'accelerate' in sys.modules,
                        'module_count': sum(name == 'accelerate' or name.startswith('accelerate.') for name in sys.modules)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    global phase
    from fastapi.testclient import TestClient
    sys.setprofile(profile)
    threading.setprofile(profile)
    import fastapi_backend as backend
    snapshot(phase)
    phase = 'model_startup'
    baseline = json.loads(Path('/audit-baseline.json').read_text())
    with TestClient(backend.app) as client:
        snapshot(phase)
        health = client.get('/health')
        assert health.status_code == 200, 'Model readiness failed'
        assert health.json()['filler_classifier']['available'] is True
        assert client.post('/transcribe').status_code == 401, 'Service authentication missing'
        report = {'health_ready': True, 'filler_ready': True,
                  'service_authentication_preserved': True, 'analyses': {}}
        for stage, reference, baseline_key in [('inference', False, 'inference'),
                ('reference_comparison', True, 'reference_self_comparison')]:
            phase = stage
            with Path('/audit-audio.aac').open('rb') as audio:
                files = {'file': ('audit.aac', audio, 'audio/aac')}
                if reference:
                    with Path('/audit-audio.aac').open('rb') as ref:
                        files['reference_audio'] = ('reference.aac', ref, 'audio/aac')
                        response = client.post('/transcribe', headers={'X-iSpeak-Service-Token': backend.SERVICE_TOKEN},
                                               files=files, data={'language': 'English'})
                else:
                    response = client.post('/transcribe', headers={'X-iSpeak-Service-Token': backend.SERVICE_TOKEN},
                                           files=files, data={'language': 'English'})
            assert response.status_code == 200, 'Analysis failed'
            body = response.json()
            assert body['analysis_valid'] is True and body['fillers']['analysis_available'] is True
            assert body['word_timestamps']
            expected = baseline[baseline_key]['body']
            report['analyses'][stage] = {'valid': True, 'word_count': len(body['word_timestamps']),
                'duration_seconds': body['duration_seconds'], 'scores': body['scores'],
                'response_sha256': digest(body), 'baseline_sha256': digest(expected),
                'matches_baseline': body == expected}
            assert body == expected, 'Response differs from verified baseline'
            snapshot(stage)
        report['temporary_audio_remaining'] = len([p for p in Path('/tmp/ispeak').iterdir()
            if p.is_file() and p.suffix in {'.wav', '.aac', '.mp3', '.m4a'}])
        assert report['temporary_audio_remaining'] == 0
    sys.setprofile(None)
    threading.setprofile(None)
    report.update({'runtime_versions': {name: importlib.metadata.version(name) for name in
        ['accelerate', 'peft', 'transformers', 'torch', 'anyio', 'fsspec', 'urllib3']},
        'import_snapshots': snapshots, 'accelerate_calls_by_stage': {stage: dict(counts) for stage, counts in calls.items()},
        'vulnerable_loader_calls': dict(vulnerable_calls),
        'sharded_model_indexes': len(list(Path('/app/models').rglob('*.index.json')))})
    assert not vulnerable_calls, 'Vulnerable checkpoint loader was exercised'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
