"""Offline, approval-gated paired ASR evaluation; never changes app defaults."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LANGUAGES = ('English', 'Filipino', 'Taglish')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def normalize(text):
    # Preserve diacritics/apostrophes; punctuation becomes space. CER includes spaces.
    text = unicodedata.normalize('NFC', text).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') and c not in "'’" else c for c in text).split())


def edits(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, a in enumerate(reference, 1):
        nxt = [i]
        for j, b in enumerate(hypothesis, 1):
            nxt.append(min(nxt[-1] + 1, row[j] + 1, row[j-1] + (a != b)))
        row = nxt
    return row[-1]


def read_dataset(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 1 or not data.get('approval_record') or not data.get('samples'):
        raise ValueError('Dataset requires schema 1, approval record, and samples')
    seen = set()
    for s in data['samples']:
        if (s.get('language') not in LANGUAGES or s.get('approved') is not True
                or s.get('reference_verified') is not True or s.get('synthetic') is not False
                or not s.get('reference_reviewer') or not s.get('consent_record')
                or not s.get('speaker_id') or not normalize(s.get('reference', ''))):
            raise ValueError('Every sample needs human-audio approval, consent and verified reference')
        p = (path.parent / s['file']).resolve()
        if digest(p) != s.get('sha256') or s.get('id') in seen or not s.get('id'):
            raise ValueError('Sample checksum mismatch or invalid/duplicate ID')
        seen.add(s['id'])
        s['file'] = str(p)
    if set(s['language'] for s in data['samples']) != set(LANGUAGES):
        raise ValueError('All three language groups are required')
    overlap = {}
    for version in ('v5', 'v6'):
        record = data.get('training_records', {}).get(version)
        if not record:
            overlap[version] = 'UNKNOWN: training inventory unavailable'
            continue
        inventory_path = (path.parent / record['file']).resolve()
        if digest(inventory_path) != record['sha256']:
            raise ValueError('Training inventory checksum mismatch')
        training = json.loads(inventory_path.read_text(encoding='utf-8'))
        hashes = set(training['audio_sha256']); speakers = set(training['speaker_ids'])
        if any(s['sha256'] in hashes or s['speaker_id'] in speakers for s in data['samples']):
            raise ValueError('Evaluation/training audio or speaker overlap detected')
        overlap[version] = {'matched_overlap': False, 'coverage': training['coverage'], 'inventory_sha256': record['sha256']}
    return data, overlap


def roots(version, candidate):
    return {'adapter': ROOT / 'models' / ('iSpeak_v5' if version == 'v5' else 'iSpeak_v6'),
            'base': ROOT / 'models/iSpeak_v5/base_model',
            'filler': ROOT / 'models/filler_classifier'}


def manifest(version, candidate):
    return ROOT / 'model_manifests/ispeak-v5.json' if version == 'v5' else candidate / 'integrity.json'


def worker(args, data):
    # Permit loopback only, for Python's local async primitives; no external calls.
    original = socket.socket.connect
    def guarded(sock, address):
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
            raise RuntimeError('External networking disabled')
        return original(sock, address)
    socket.socket.connect = guarded
    socket.socket.connect_ex = lambda sock, address: guarded(sock, address) or 0
    import torch
    import numpy as np
    import librosa
    from model import ISpeakWhisper
    from audio_analysis_processing_files.filler_words import _default_classifier
    from model_integrity import verify_bundle
    verification = verify_bundle(roots(args.worker, args.candidate), manifest(args.worker, args.candidate))
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(1)
    torch.manual_seed(0); np.random.seed(0); torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model = ISpeakWhisper()
    # Explicitly equalize decoding configuration instead of inheriting adapter defaults.
    model.model.generation_config.do_sample = False
    model.model.generation_config.num_beams = 1
    model.model.generation_config.max_new_tokens = 440
    model.model.generation_config.forced_decoder_ids = None
    _default_classifier()
    initialization = time.perf_counter() - started
    # Fixed discarded warm-up, identical audio for both models.
    warm, _ = librosa.load(data['samples'][0]['file'], sr=16000, mono=True)
    model.transcribe('', audio_data={'array': warm, 'sampling_rate': 16000}, language=data['samples'][0]['language'])
    results = []
    for sample in data['samples']:
        audio, _ = librosa.load(sample['file'], sr=16000, mono=True)
        times = []; outputs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs.append(model.transcribe('', audio_data={'array': audio, 'sampling_rate': 16000}, language=sample['language'], condition_on_previous_text=False))
            times.append(time.perf_counter()-start)
        result = outputs[0]; ref = normalize(sample['reference']); hyp = normalize(result['text'])
        words = [w for seg in result['segments'] for w in seg['words']]
        valid = all(0 <= w['start'] <= w['end'] <= len(audio)/16000 + .1 for w in words)
        valid = valid and all(a['start'] <= b['start'] for a,b in zip(words, words[1:]))
        results.append({'id': sample['id'], 'language': sample['language'], 'reference': sample['reference'],
                        'hypothesis': result['text'], 'word_edits': edits(ref.split(), hyp.split()), 'reference_words': len(ref.split()),
                        'character_edits': edits(ref, hyp), 'reference_characters': len(ref),
                        'timestamps': words, 'timestamps_valid': valid, 'latency_seconds': times,
                        'repeat_transcripts_identical': all(o['text'] == result['text'] for o in outputs)})
    report = {'model': args.worker, 'integrity': verification, 'initialization_seconds': initialization,
              'device': str(model.model.device), 'torch_version': torch.__version__, 'results': results}
    (args.output / (args.worker + '.json')).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--worker', choices=('v5', 'v6'), help=argparse.SUPPRESS)
    a = p.parse_args(); a.dataset = a.dataset.resolve(); a.candidate = a.candidate.resolve(); a.output = a.output.resolve()
    if a.threads < 1 or a.repeats < 1 or not a.output.is_relative_to(ROOT / 'benchmark-results'):
        p.error('Positive threads/repeats and output under benchmark-results required')
    data, overlap = read_dataset(a.dataset)
    if a.worker:
        worker(a, data); return
    from model_integrity import verify_bundle
    checks = {v: verify_bundle(roots(v,a.candidate),manifest(v,a.candidate)) for v in ('v5','v6')}
    a.output.mkdir(parents=True, exist_ok=False)
    import psutil
    summary = {'dataset_sha256': digest(a.dataset), 'training_overlap': overlap, 'integrity': checks,
               'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
               'python': sys.version, 'threads': a.threads, 'repeats': a.repeats,
               'decoding': {'task':'transcribe','num_beams':1,'do_sample':False,'max_new_tokens':440,'condition_on_prev_tokens':False,'English':'en','Filipino':'tl','Taglish':'automatic'},
               'preprocessing': 'librosa mono 16000 Hz; NFC/casefold/punctuation-to-space; CER includes spaces', 'models': {}}
    for v in ('v5','v6'):
        r = roots(v,a.candidate)
        env = dict(os.environ, ISPEAK_MODEL_PATH=str(r['adapter']), ISPEAK_BASE_MODEL_PATH=str(r['base']),
                   ISPEAK_FILLER_MODEL_PATH=str(r['filler']), ISPEAK_MODEL_MANIFEST=str(manifest(v,a.candidate)),
                   CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', ISPEAK_ENV='production',
                   OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), PYTHONDONTWRITEBYTECODE='1',
                   NUMBA_CACHE_DIR=str(a.output/'numba-cache'))
        cmd = [sys.executable,'-B',str(Path(__file__).resolve()),'--dataset',str(a.dataset),'--candidate',str(a.candidate),
               '--output',str(a.output),'--threads',str(a.threads),'--repeats',str(a.repeats),'--worker',v]
        peak = 0
        with (a.output/(v+'.local.log')).open('x',encoding='utf-8') as log:
            process = subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            monitor = psutil.Process(process.pid); deadline = time.monotonic()+3600
            while process.poll() is None:
                try: peak = max(peak,monitor.memory_info().rss + sum(c.memory_info().rss for c in monitor.children(recursive=True)))
                except psutil.Error: pass
                if time.monotonic() > deadline:
                    process.kill(); process.wait(); raise RuntimeError('Evaluation timed out')
                time.sleep(.1)
        if process.returncode: raise RuntimeError('Worker failed; inspect private local log')
        report = json.loads((a.output/(v+'.json')).read_text(encoding='utf-8'))
        metrics = {}
        for language in LANGUAGES:
            rows = [x for x in report['results'] if x['language']==language]
            metrics[language] = {'samples':len(rows),'WER':sum(x['word_edits'] for x in rows)/sum(x['reference_words'] for x in rows),
                                 'CER':sum(x['character_edits'] for x in rows)/sum(x['reference_characters'] for x in rows)}
        summary['models'][v] = {'metrics':metrics,'sampled_peak_rss_bytes':peak,'initialization_seconds':report['initialization_seconds']}
    reports = {v:json.loads((a.output/(v+'.json')).read_text(encoding='utf-8')) for v in ('v5','v6')}
    summary['paired_comparison'] = [{'id':x['id'],'language':x['language'],'same_transcript':x['hypothesis']==y['hypothesis'],
                                    'same_timestamps':x['timestamps']==y['timestamps'],'v5_latency_seconds':x['latency_seconds'],
                                    'v6_latency_seconds':y['latency_seconds']} for x,y in zip(reports['v5']['results'],reports['v6']['results'])]
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Completed. Results remain in ignored benchmark-results.')


if __name__ == '__main__':
    main()
