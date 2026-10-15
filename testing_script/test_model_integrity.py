import json
from pathlib import Path
import tempfile
import unittest

from model_integrity import REQUIRED, IntegrityError, create_manifest, verify_bundle


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.roots = {role: root / role for role in REQUIRED}
        for role, names in REQUIRED.items():
            self.roots[role].mkdir()
            for name in names:
                (self.roots[role] / name).write_bytes(b'trusted synthetic fixture')
        self.manifest = root / 'bundle.json'
        create_manifest(self.roots, self.manifest, 'synthetic-test-bundle')

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_bundle_verifies(self):
        self.assertEqual(verify_bundle(self.roots, self.manifest)['verified_files'], 16)

    def test_same_length_corruption_fails(self):
        (self.roots['base'] / 'model.safetensors').write_bytes(b'changed synthetic fixture')
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_missing_model_fails(self):
        (self.roots['filler'] / 'model.safetensors').unlink()
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_sharded_checkpoint_index_is_rejected(self):
        (self.roots['base'] / 'model.safetensors.index.json').write_text('{}')
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_traversal_manifest_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data['files'][0]['file'] = '../outside.safetensors'
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_duplicate_or_incomplete_manifest_is_rejected(self):
        data = json.loads(self.manifest.read_text())
        data['files'].append(data['files'][0])
        self.manifest.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)

    def test_manifest_creation_cannot_overwrite_trusted_manifest(self):
        with self.assertRaises(FileExistsError):
            create_manifest(self.roots, self.manifest, 'replacement')

    def test_unapproved_executable_is_rejected(self):
        (self.roots['adapter'] / 'modeling_custom.py').write_text('raise RuntimeError()')
        with self.assertRaises(IntegrityError):
            verify_bundle(self.roots, self.manifest)
