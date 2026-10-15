"""Negative checks for the actual unified format; only synthetic temporary data."""
import json
from pathlib import Path
import tempfile
import unittest

from model import ModelUnavailableError, resolve_adapter_path
from model_integrity import REQUIRED, UNIFIED_ADAPTER_FILES, IntegrityError, create_manifest, verify_bundle


class UnifiedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.roots = {role: self.root / role for role in REQUIRED}
        for role, names in {**REQUIRED, 'adapter': UNIFIED_ADAPTER_FILES}.items():
            self.roots[role].mkdir()
            for name in names:
                (self.roots[role] / name).write_bytes(b'synthetic trusted artifact')
        self.adapter = self.roots['adapter']
        (self.adapter / 'processor_config.json').write_text(json.dumps({'processor_class': 'WhisperProcessor', 'feature_extractor': {'feature_extractor_type': 'WhisperFeatureExtractor', 'sampling_rate': 16000, 'feature_size': 80}}))
        (self.adapter / 'tokenizer.json').write_text(json.dumps({'model': {'type': 'BPE'}}))
        self.manifest = self.root / 'candidate.json'
        create_manifest(self.roots, self.manifest, 'synthetic-v6', 'whisper-unified')

    def tearDown(self):
        self.temp.cleanup()

    def test_actual_format_is_accepted_without_fabricated_legacy_files(self):
        self.assertEqual(resolve_adapter_path(self.adapter), self.adapter)
        self.assertEqual(verify_bundle(self.roots, self.manifest)['verified_files'], 12)
        self.assertFalse((self.adapter / 'vocab.json').exists())

    def test_missing_processor_is_rejected_before_loading(self):
        (self.adapter / 'processor_config.json').unlink()
        with self.assertRaises(ModelUnavailableError):
            resolve_adapter_path(self.adapter)
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_non_whisper_processor_is_rejected(self):
        (self.adapter / 'processor_config.json').write_text('{}')
        with self.assertRaises(ModelUnavailableError):
            resolve_adapter_path(self.adapter)

    def test_corrupted_tokenizer_fails_integrity(self):
        path = self.adapter / 'tokenizer.json'
        data = path.read_bytes(); path.write_bytes(data[:-1] + b'!')
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_candidate_manifest_cannot_downgrade_v5_schema(self):
        data = json.loads(self.manifest.read_text()); data['schema'] = 1
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_unlisted_pickle_or_executable_is_rejected(self):
        (self.adapter / 'pytorch_model.bin').write_bytes(b'untrusted fixture')
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_unified_manifest_cannot_remove_required_artifacts(self):
        data = json.loads(self.manifest.read_text()); data['files'].pop()
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)
