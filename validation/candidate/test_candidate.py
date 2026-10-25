"""Offline candidate-mode safety tests: never use Docker, credentials or network."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_python as candidate
import validate_image


def approved():
    return {'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
            'RUNNER_OS': 'Linux', 'RUNNER_ARCH': 'X64',
            'GITHUB_EVENT_NAME': 'workflow_dispatch',
            'GITHUB_REPOSITORY': 'Apol-bit/ispeak_python-backendV2',
            'GITHUB_ACTOR': 'Apol-bit', 'GITHUB_REF': candidate.BRANCH,
            'GITHUB_RUN_ATTEMPT': '1', 'ACCELERATE_STAGING_APPROVED': 'true',
            'PRIVATE_BASE_ACCESS_APPROVED': 'true', 'TESTING_CONTAINMENT_ACCEPTED': 'true',
            'STAGING_MAINTAINER': 'maintainer', 'STAGING_BACKUP_REVIEWER': 'reviewer'}


class CandidateSafetyTests(unittest.TestCase):
    def test_failure_evidence_never_exposes_private_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = Mock(root=Path(tmp), sequence=1, phase='model_free_image_checks')
            (Path(tmp) / 'command-1.log').write_text(
                'PRIVATE_TOKEN PRIVATE_RECORDING\nTraceback\nFile "<string>", line 27\n'
                'ModuleNotFoundError: PRIVATE_PATH\n')
            candidate.report_failure(v, RuntimeError('PRIVATE_SECRET'))
            v.evidence.assert_called_once_with('diagnostic_failure', phase='model_free_image_checks',
                                               category='missing_python_module', inline_python_line=27)
            self.assertNotIn('PRIVATE', repr(v.evidence.call_args))

    def test_manual_approval_and_host_gates(self):
        with patch.object(candidate.os, 'name', 'posix'), patch.dict(os.environ, approved(), clear=True):
            candidate.gate()
            for key, value in [('GITHUB_EVENT_NAME', 'push'), ('RUNNER_ENVIRONMENT', 'self-hosted'),
                               ('RUNNER_ARCH', 'ARM64'), ('GITHUB_RUN_ATTEMPT', '2'),
                               ('ACCELERATE_STAGING_APPROVED', 'false'),
                               ('PRIVATE_BASE_ACCESS_APPROVED', 'false'),
                               ('TESTING_CONTAINMENT_ACCEPTED', 'false'),
                               ('STAGING_BACKUP_REVIEWER', 'maintainer')]:
                with self.subTest(key=key), patch.dict(os.environ, {key: value}):
                    with self.assertRaises(RuntimeError):
                        candidate.gate()

    def test_recorded_backup_full_name_preserves_distinct_review(self):
        with patch.object(candidate.os, 'name', 'posix'), patch.dict(os.environ, approved(), clear=True):
            with patch.dict(os.environ, {'STAGING_MAINTAINER': 'Apol-bit',
                                         'STAGING_BACKUP_REVIEWER': 'John Laurence'}):
                candidate.gate()
            for backup in ['', ' ', ' maintainer', 'maintainer ', 'MAINTAINER',
                           'John\nLaurence', 'John\tLaurence']:
                with self.subTest(backup=backup), patch.dict(os.environ, {'STAGING_BACKUP_REVIEWER': backup}):
                    with self.assertRaises(RuntimeError):
                        candidate.gate()

    def test_gate_precedes_all_work(self):
        v = candidate.DiagnosticBuild.__new__(candidate.DiagnosticBuild)
        with patch.object(candidate, 'gate', side_effect=RuntimeError('denied')), \
             patch.object(candidate, 'enforce') as enforce, patch.object(candidate, 'export') as export:
            with self.assertRaises(RuntimeError):
                v.execute()
            enforce.assert_not_called()
            export.assert_not_called()

    def test_workstation_is_rejected(self):
        with patch.object(candidate.os, 'name', 'nt'), patch.dict(os.environ, approved(), clear=True):
            with self.assertRaises(RuntimeError):
                candidate.gate()

    def test_corrupt_source_hash_stops_export(self):
        manifest = {'commit': candidate.SOURCE, 'base': candidate.BASE,
                    'files': [{'path': 'model.py', 'bytes': 5, 'sha256': '0' * 64}], 'tests': []}
        with tempfile.TemporaryDirectory() as tmp:
            v = Mock(root=Path(tmp))
            v.command.side_effect = [b'', b'', b'', candidate.SOURCE.encode(), b'wrong']
            with patch.object(candidate, 'MANIFEST') as path:
                path.read_text.return_value = json.dumps(manifest)
                with self.assertRaisesRegex(RuntimeError, 'source hash mismatch'):
                    candidate.export(v)
            self.assertFalse((Path(tmp) / 'context/model.py').exists())

    def test_dependency_drift_stops_before_regressions(self):
        v = Mock()
        v.key = '123-1'
        v.run_args.return_value = []
        baseline = {'accelerate': '1.14.0', 'certifi': '2026.4.22',
                    'filelock': '3.29.0', 'setuptools': '83.0.0', 'torch': '2.13.0+cpu'}
        modified = dict(baseline, certifi='2026.10.19', filelock='4.2.1', setuptools='84.1.0', torch='unexpected')
        v.docker.side_effect = [json.dumps(baseline).encode(), json.dumps(modified).encode()]
        with self.assertRaisesRegex(RuntimeError, 'dependency drift'):
            validate_image.validate(v, Path('unused'), {})
        self.assertEqual(v.docker.call_count, 2)

    def test_candidate_has_only_gated_existing_publisher_and_no_legacy_execute(self):
        text = Path(candidate.__file__).read_text()
        tree = ast.parse(text)
        self.assertNotIn('validate_python', text)
        self.assertNotIn('super().execute', text)
        self.assertNotIn('--push', text)
        self.assertIn("'--load'", text)
        self.assertIn('from publish_image import publish', text)
        execute = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'execute')
        calls = [n for n in ast.walk(execute) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        validation = next(n.lineno for n in calls if n.func.id == 'validate')
        publication = next(n.lineno for n in calls if n.func.id == 'publish_if_approved')
        self.assertLess(validation, publication)
        self.assertEqual(candidate.RECIPE, 'docker/Dockerfile.v6-security-overlay')
        self.assertEqual(candidate.SOURCE, '412d81d35588bd3737a60691a2f498ea877bf899')

    def test_no_push_trigger_and_existing_environment(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/build-python-diagnostic.yml').read_text()
        self.assertIn('  workflow_dispatch:', workflow)
        self.assertNotIn('  push:', workflow)
        self.assertIn('environment: ispeak-python-private-validation', workflow)
        self.assertIn('runs-on: ubuntu-24.04', workflow)
        legacy = (root / '.github/workflows/validate-python.yml').read_text()
        self.assertIn('paths: [.github/workflows/validate-python.yml, validation/remote/**]', legacy)
        self.assertNotIn('validation/candidate', legacy)
        self.assertIn('      publication_approved:', workflow)
        self.assertIn('PUBLICATION_APPROVED: ${{ inputs.publication_approved }}', workflow)
        publication_input = workflow.split('      publication_approved:', 1)[1].split('      maintainer:', 1)[0]
        self.assertIn('default: false', publication_input)

    def test_missing_or_false_publication_approval_never_calls_publisher(self):
        v = Mock(model_free_passed=True)
        for value in ['', 'false', 'TRUE']:
            with patch.dict(os.environ, {'PUBLICATION_APPROVED': value}), patch('publish_image.publish') as publisher:
                self.assertIsNone(candidate.publish_if_approved(v, {}))
                publisher.assert_not_called()

    def test_approval_cannot_override_missing_validation(self):
        v = Mock(model_free_passed=False)
        with patch.dict(os.environ, {'PUBLICATION_APPROVED': 'true'}), patch('publish_image.publish') as publisher:
            with self.assertRaises(RuntimeError):
                candidate.publish_if_approved(v, {})
            publisher.assert_not_called()

    def test_verified_digest_is_recorded_without_rebuild(self):
        v = Mock(model_free_passed=True, publication_verified=False)
        def existing_publisher(v, image):
            v.published_digest = 'sha256:' + 'a' * 64
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / 'summary.md'
            with patch.dict(os.environ, {'PUBLICATION_APPROVED': 'true', 'GITHUB_STEP_SUMMARY': str(summary)}), \
                 patch('publish_image.publish', side_effect=existing_publisher) as publisher:
                immutable = candidate.publish_if_approved(v, {'Id': 'sha256:tested'})
            publisher.assert_called_once()
            self.assertEqual(immutable, 'ghcr.io/apol-bit/ispeak-python@sha256:' + 'a' * 64)
            self.assertTrue(v.publication_verified)
            self.assertIn(immutable, summary.read_text())
            v.docker.assert_not_called()

    def test_publisher_failure_is_not_reported_as_verified(self):
        v = Mock(model_free_passed=True, publication_verified=False)
        with patch.dict(os.environ, {'PUBLICATION_APPROVED': 'true'}), \
             patch('publish_image.publish', side_effect=RuntimeError('verification failed')):
            with self.assertRaises(RuntimeError):
                candidate.publish_if_approved(v, {})
        self.assertFalse(v.publication_verified)

    def test_validation_failure_never_reaches_publication(self):
        v = candidate.DiagnosticBuild.__new__(candidate.DiagnosticBuild)
        v.model_free_passed = False
        v.key, v.label, v.builder, v.image = '123-1', 'owned', 'owned', 'local-candidate'
        v.login, v.builder_start, v.stop_builder, v.evidence = Mock(), Mock(), Mock(), Mock()
        cfg = {'User': '10001:10001', 'Healthcheck': {'Test': ['CMD', 'health']},
               'Entrypoint': ['python'], 'Cmd': ['server'], 'Env': [],
               'Labels': {'org.opencontainers.image.revision': candidate.SOURCE}}
        image = {'Id': 'sha256:tested', 'Os': 'linux', 'Architecture': 'amd64',
                 'RootFS': {'Layers': ['base']}, 'Config': cfg}
        v.docker = Mock(side_effect=[b'', json.dumps([image]).encode(), json.dumps([image]).encode()])
        with patch.object(candidate, 'gate'), patch.object(candidate, 'enforce'), \
             patch.object(candidate, 'export', return_value=(Path('context'), Path('tests'), {})), \
             patch('validate_image.validate', side_effect=RuntimeError('regression failed')), \
             patch.object(candidate, 'publish_if_approved') as publisher:
            with self.assertRaises(RuntimeError):
                v.execute()
            publisher.assert_not_called()
        self.assertFalse(v.model_free_passed)

    def test_manifest_scope(self):
        manifest = json.loads(candidate.MANIFEST.read_text())
        self.assertEqual(manifest['commit'], candidate.SOURCE)
        self.assertEqual(manifest['base'], candidate.BASE)
        names = {v['path'] for v in manifest['files']}
        for path in ['memory_diagnostics.py', 'audio_analysis_processing_files/whisper_timestamps.py',
                     'audio_analysis_processing_files/whisper_retention.py',
                     'audio_analysis_processing_files/articulation/bounded_dtw.py', candidate.RECIPE,
                     'docker/requirements.security.txt']:
            self.assertIn(path, names)
        self.assertFalse(any(p.startswith(('models/', 'testing_script/')) or p.endswith('.wav') for p in names))
        self.assertEqual(len(names), len(manifest['files']))
        self.assertTrue(all(len(v['sha256']) == 64 for v in manifest['files'] + manifest['tests']))

    def test_model_free_checks_override_entrypoint_and_guard_models(self):
        text = Path(validate_image.__file__).read_text()
        self.assertIn("'--entrypoint=python'", text)
        self.assertIn('Model loading forbidden', text)
        self.assertIn('Timestamp optimization inactive', text)
        self.assertIn('Long-form retention optimization inactive', text)
        self.assertIn('whisper_longform_retention', text)
        self.assertIn('whisper_active_chunk', text)
        self.assertIn('Active-chunk optimization inactive', text)
        self.assertIn('chunk._forward_hooks.values()', text)
        self.assertIn('result.testsRun==141', text)
        self.assertIn("len(json.loads(manifest.read_text())['files'])==12", text)


if __name__ == '__main__':
    unittest.main()
