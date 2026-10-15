"""Copy a separately approved local v6 export into a new ignored candidate tree."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from model_integrity import REQUIRED, UNIFIED_ADAPTER_FILES, create_manifest, digest, verify_bundle

parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--candidate', required=True)
options = parser.parse_args()
source = Path(options.source).resolve(strict=True)
candidate = Path(options.candidate).resolve()
if not candidate.is_relative_to(ROOT / '.model-candidates') or candidate.exists():
    raise SystemExit('Candidate must be a new directory inside .model-candidates; overwriting is refused')
candidate.parent.mkdir(exist_ok=True)
candidate.mkdir()
adapter = candidate / 'adapter'
base = adapter / 'base_model'
filler = candidate / 'filler'
adapter.mkdir(); base.mkdir(); filler.mkdir()
receipt = []
for source_root, target_root, names in [
    (source, adapter, (*UNIFIED_ADAPTER_FILES, 'README.md')),
    (ROOT / 'models' / 'iSpeak_v5' / 'base_model', base, REQUIRED['base']),
    (ROOT / 'models' / 'filler_classifier', filler, REQUIRED['filler']),
]:
    for name in names:
        original = source_root / name
        if original.is_symlink() or not original.is_file():
            raise SystemExit('Required source artifact missing or linked; partial candidate retained for review')
        target = target_root / name
        before = digest(original)
        shutil.copy2(original, target)
        assert before == digest(target) == digest(original)
        receipt.append({'source': str(original), 'candidate': str(target.relative_to(candidate)), 'bytes': target.stat().st_size, 'sha256': before})
roots = {'adapter': adapter, 'base': base, 'filler': filler}
manifest = candidate / 'integrity.json'
create_manifest(roots, manifest, 'ispeak-v6-isolated-candidate', 'whisper-unified')
(candidate / 'copy-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps({'candidate': str(candidate), 'modelArtifactBytes': sum(entry['bytes'] for entry in receipt), 'manifest': str(manifest), 'integrity': verify_bundle(roots, manifest)}, indent=2))
