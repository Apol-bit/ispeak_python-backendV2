"""Trusted local model bundle verification; no downloads or inference changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    'adapter': ('adapter_config.json', 'adapter_model.safetensors', 'added_tokens.json',
                'merges.txt', 'normalizer.json', 'preprocessor_config.json',
                'special_tokens_map.json', 'tokenizer_config.json', 'vocab.json'),
    'base': ('config.json', 'generation_config.json', 'model.safetensors'),
    'filler': ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json'),
}
DEFAULT_MANIFEST = ROOT / 'model_manifests' / 'ispeak-v5.json'
UNIFIED_ADAPTER_FILES = ('adapter_config.json', 'adapter_model.safetensors',
                         'processor_config.json', 'tokenizer_config.json', 'tokenizer.json')


class IntegrityError(RuntimeError):
    pass


def configured_roots():
    # Reuse the existing application selection rules, without loading models.
    from model import configured_adapter_path, configured_base_model_path
    adapter = configured_adapter_path()
    filler = Path(os.getenv('ISPEAK_FILLER_MODEL_PATH', '').strip() or ROOT / 'models' / 'filler_classifier').expanduser()
    if not filler.is_absolute():
        filler = ROOT / filler
    return {'adapter': adapter, 'base': configured_base_model_path(adapter_path=adapter), 'filler': filler}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def required_files(adapter_format):
    if adapter_format == 'legacy':
        return REQUIRED
    if adapter_format == 'whisper-unified':
        return {**REQUIRED, 'adapter': UNIFIED_ADAPTER_FILES}
    raise IntegrityError('Model integrity failed: unknown adapter format')


def inventory(roots, adapter_format='legacy'):
    files = []
    for role, required in required_files(adapter_format).items():
        root = Path(roots[role])
        if root.is_symlink() or not root.is_dir():
            raise IntegrityError(f'Model integrity failed: {role} directory missing or linked')
        # Ignore only the established non-runtime training/readme/cache artifacts.
        for path in root.iterdir():
            if path.name in {'.cache', 'README.md', 'training_args.bin'}:
                continue
            if role == 'adapter' and path.name == 'base_model' and path.is_dir() and not path.is_symlink():
                continue
            if path.is_symlink() or path.name not in required:
                raise IntegrityError(f'Model integrity failed: unexpected {role} artifact')
        for name in required:
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise IntegrityError(f'Model integrity failed: missing or linked {role}/{name}')
            size = path.stat().st_size
            if not size:
                raise IntegrityError(f'Model integrity failed: empty {role}/{name}')
            files.append({'component': role, 'file': name, 'bytes': size, 'sha256': digest(path)})
    return files


def create_manifest(roots, output, bundle_id, adapter_format='legacy'):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}', bundle_id):
        raise IntegrityError('Model integrity failed: invalid bundle identifier')
    data = {'schema': 1, 'bundle_id': bundle_id, 'files': inventory(roots, adapter_format)}
    if adapter_format != 'legacy':
        data.update(schema=2, adapter_format=adapter_format)
    # Never replace an existing trusted manifest automatically.
    with Path(output).open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')
    return data


def verify_bundle(roots, manifest):
    try:
        data = json.loads(Path(manifest).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise IntegrityError('Model integrity failed: manifest missing or invalid') from error
    if data.get('schema') not in (1, 2) or not isinstance(data.get('bundle_id'), str) or not isinstance(data.get('files'), list):
        raise IntegrityError('Model integrity failed: invalid manifest structure')
    adapter_format = data.get('adapter_format', 'legacy')
    if (data['schema'] == 1 and adapter_format != 'legacy') or (data['schema'] == 2 and adapter_format != 'whisper-unified'):
        raise IntegrityError('Model integrity failed: incompatible schema/adapter format')
    expected = {(role, name) for role, names in required_files(adapter_format).items() for name in names}
    entries = {}
    for entry in data['files']:
        if not isinstance(entry, dict):
            raise IntegrityError('Model integrity failed: invalid manifest entry')
        key = (entry.get('component'), entry.get('file'))
        if key not in expected or key in entries or not re.fullmatch(r'[a-f0-9]{64}', str(entry.get('sha256', ''))) or type(entry.get('bytes')) is not int or entry['bytes'] <= 0:
            raise IntegrityError('Model integrity failed: unsafe, duplicate or invalid manifest entry')
        entries[key] = entry
    if set(entries) != expected:
        raise IntegrityError('Model integrity failed: incomplete manifest')
    actual = inventory(roots, adapter_format)
    for entry in actual:
        expected_entry = entries[(entry['component'], entry['file'])]
        if entry != expected_entry:
            raise IntegrityError(f"Model integrity failed: checksum or size mismatch for {entry['component']}/{entry['file']}")
    return {'bundle_id': data['bundle_id'], 'verified_files': len(actual), 'bytes': sum(entry['bytes'] for entry in actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=os.getenv('ISPEAK_MODEL_MANIFEST') or str(DEFAULT_MANIFEST))
    parser.add_argument('--create-manifest')
    parser.add_argument('--bundle-id')
    parser.add_argument('--adapter-format', choices=['legacy', 'whisper-unified'], default='legacy',
                        help='Explicit export format when creating a new manifest only')
    parser.add_argument('--exec', dest='command', nargs=argparse.REMAINDER)
    options = parser.parse_args()
    try:
        roots = configured_roots()
        if options.create_manifest:
            if not options.bundle_id or options.command:
                raise IntegrityError('Manifest creation requires a bundle ID and cannot launch a process')
            create_manifest(roots, options.create_manifest, options.bundle_id, options.adapter_format)
            print(json.dumps(verify_bundle(roots, options.create_manifest)))
        else:
            result = verify_bundle(roots, options.manifest)
            print(json.dumps(result), flush=True)
            if options.command:
                os.execvp(options.command[0], options.command)
    except (IntegrityError, OSError, ValueError, TypeError):
        # Only our controlled diagnostics may be printed; no raw filesystem/env error.
        error = sys.exc_info()[1]
        print(str(error) if isinstance(error, IntegrityError) else 'Model integrity failed: operation unavailable', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
