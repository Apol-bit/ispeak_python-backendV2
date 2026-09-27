import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import fastapi_backend as backend


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(backend.app)
        self.model = backend.app.state.model
        self.filler = backend.app.state.filler_ready

    def tearDown(self):
        backend.app.state.model = self.model
        backend.app.state.filler_ready = self.filler
        self.client.close()

    def test_liveness_survives_missing_speech_model(self):
        backend.app.state.model = None
        self.assertEqual(self.client.get('/live').status_code, 200)
        self.assertEqual(self.client.get('/health').status_code, 503)
        self.assertEqual(self.client.get('/ready').status_code, 503)
        self.assertEqual(self.client.post('/transcribe').status_code, 503)
        self.assertEqual(backend.admission.admitted, 0)

    def test_filler_failure_is_degraded(self):
        backend.app.state.model = object()
        backend.app.state.filler_ready = False
        response = self.client.get('/ready')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()['filler_classifier']['available'])
        backend.app.state.filler_ready = True
        self.assertEqual(self.client.get('/ready').status_code, 200)

    def test_invalid_audio_never_calls_analysis_and_releases_admission(self):
        backend.app.state.model = object()
        with patch.object(backend, 'generate_full_analysis') as analysis:
            response = self.client.post('/transcribe', files={'file': ('bad.wav', b'not audio')})
        self.assertEqual(response.status_code, 400)
        analysis.assert_not_called()
        self.assertEqual(backend.admission.admitted, 0)
        self.assertEqual(backend.admission.active, 0)


if __name__ == '__main__':
    unittest.main()
