"""Exercise local launch selection and actual FastAPI lifespan, offline."""
import argparse
import json
import wave
import os
from pathlib import Path
import secrets
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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


def main(argv=None):
    options, manifest, samples = parse_arguments(argv)
    if options.validate_only:
        print(json.dumps({'fixtures': 'PASS', 'samples': len(samples), 'inference': 'NOT RUN'}))
        return 0
    output = ROOT / 'benchmark-results/local-v6-activation'
    original_connect = socket.socket.connect
    def guarded(sock, address):
        if not isinstance(address, tuple) or address[0] not in ('127.0.0.1', '::1'):
            raise RuntimeError('External network disabled')
        return original_connect(sock, address)
    socket.socket.connect = guarded
    socket.socket.connect_ex = lambda sock, address: guarded(sock, address) or 0
    os.environ['ISPEAK_SERVICE_TOKEN'] = secrets.token_hex(32)
    os.environ['ISPEAK_DEBUG_SPEECH_LOGS'] = 'false'
    os.environ['ISPEAK_DEBUG_ERRORS'] = 'false'
    os.environ['NUMBA_CACHE_DIR'] = str(ROOT/'benchmark-results/local-v6-activation/numba-cache')
    from model_integrity import configured_roots, verify_bundle
    integrity = verify_bundle(configured_roots(), os.environ['ISPEAK_MODEL_MANIFEST'])
    from fastapi.testclient import TestClient
    import fastapi_backend as backend
    report = {'integrity':integrity, 'network':'external connections blocked', 'samples':[]}
    with TestClient(backend.app) as client:
        assert client.get('/ready').status_code == 200
        loaded = backend.app.state.model
        report['loaded_adapter'] = str(loaded.adapter_path)
        report['loaded_base'] = str(loaded.base_model_path)
        report['lora_merged'] = not any('lora_A' in name or 'lora_B' in name for name,_ in loaded.model.named_parameters())
        assert report['lora_merged']
        assert loaded.adapter_path.resolve() == configured_roots()['adapter'].resolve()
        assert client.get('/metrics').status_code == 401
        for sample in samples:
            assert sample['purpose'] == 'synthetic compatibility only'
            start=time.perf_counter()
            response=client.post('/transcribe',files={'file':('synthetic.wav',(manifest.parent / sample['file']).read_bytes(),'audio/wav')},data={'language':sample['language']},headers={'X-iSpeak-Service-Token':os.environ['ISPEAK_SERVICE_TOKEN']})
            assert response.status_code == 200
            body=response.json()
            assert body['analysis_valid'] and body['fillers']['analysis_available'] and body['word_timestamps']
            report['samples'].append({'language':sample['language'],'transcription':body['transcription'],'timestamps':body['word_timestamps'],'scores':body['scores'],'fillers':body['fillers'],'latency_seconds':time.perf_counter()-start})
    report['shutdown']='lifespan completed'
    version=loaded.adapter_path.name
    output.mkdir(parents=True, exist_ok=True)
    (output/f'{version}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
