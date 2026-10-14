import os
import unittest
import subprocess
import sys
from unittest.mock import patch
from fastapi.testclient import TestClient
import fastapi_backend as backend
import whisper_service


class SecurityHardeningTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app)

    def tearDown(self):
        self.client.close()

    def test_token_required_before_upload_and_admission(self):
        with patch.object(backend, 'SERVICE_TOKEN', 'test-only-service-token-not-production'):
            for headers in [{}, {'X-iSpeak-Service-Token': 'wrong'}]:
                with patch.object(backend, '_transcribe') as analysis:
                    response = self.client.post('/transcribe', headers=headers)
                    self.assertEqual(response.status_code, 401)
                    analysis.assert_not_called()
                    self.assertEqual(backend.admission.admitted, 0)
            self.assertEqual(self.client.get('/metrics').status_code, 401)
            self.assertEqual(self.client.get('/live').status_code, 200)

    def test_correct_token_reaches_route_and_development_can_be_unconfigured(self):
        token = 'test-only-service-token-not-production'
        with patch.object(backend, 'SERVICE_TOKEN', token):
            self.assertEqual(self.client.get('/metrics', headers={'X-iSpeak-Service-Token': token}).status_code, 200)
        with patch.object(backend, 'SERVICE_TOKEN', ''):
            self.assertEqual(self.client.get('/metrics').status_code, 200)

    def test_production_health_never_exposes_paths_or_model_errors(self):
        with patch.object(backend, 'IS_PRODUCTION', True), patch.object(backend.app.state, 'model_error', 'PRIVATE_PATH_SECRET'):
            response = self.client.get('/health')
            self.assertNotIn('PRIVATE', response.text)
            self.assertNotIn('path', response.text)
            self.assertNotIn('error', response.text)

    def test_error_logs_never_include_content_in_production(self):
        with patch.object(backend, 'IS_PRODUCTION', True), patch.dict(os.environ, {'ISPEAK_DEBUG_ERRORS': 'true'}):
            with self.assertLogs(backend.logger, level='ERROR') as capture:
                backend._log_failure('audio_failed', RuntimeError('PRIVATE_SPEECH_TOKEN_PATH'))
        self.assertNotIn('PRIVATE', '\n'.join(capture.output))
        self.assertIn('RuntimeError', '\n'.join(capture.output))

    def test_speech_logs_require_development_and_explicit_flag(self):
        words = [{'text': 'PRIVATE_SPEECH', 'start': 0, 'end': 1, 'confidence': .9}]
        for mode, flag in [('production', 'true'), ('development', 'false')]:
            with patch.dict(os.environ, {'ISPEAK_ENV': mode, 'ISPEAK_DEBUG_SPEECH_LOGS': flag}):
                with self.assertLogs(whisper_service.logger, level='DEBUG') as capture:
                    whisper_service._log_transcription('PRIVATE_SPEECH', words)
            self.assertNotIn('PRIVATE', '\n'.join(capture.output))
        with patch.dict(os.environ, {'ISPEAK_ENV': 'development', 'ISPEAK_DEBUG_SPEECH_LOGS': 'true'}):
            with self.assertLogs(whisper_service.logger, level='DEBUG') as capture:
                whisper_service._log_transcription('PRIVATE_SPEECH', words)
            self.assertIn('PRIVATE_SPEECH', '\n'.join(capture.output))

    def test_production_startup_rejects_missing_token_and_mixed_wildcard_cors(self):
        for token, origins, message in [('', '', 'ISPEAK_SERVICE_TOKEN'),
                ('security-unit-tests-only-not-production-token', '*,https://admin.example.test', 'ISPEAK_CORS_ORIGINS'),
                ('security-unit-tests-only-not-production-token', 'http://admin.example.test', 'ISPEAK_CORS_ORIGINS')]:
            env = {**os.environ, 'ISPEAK_ENV': 'production', 'ISPEAK_SERVICE_TOKEN': token, 'ISPEAK_CORS_ORIGINS': origins}
            result = subprocess.run([sys.executable, '-c', 'import fastapi_backend'], env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
