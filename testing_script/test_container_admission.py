"""Route-level failure and queue checks using controlled inference workers."""
import asyncio
import threading
import unittest
from unittest.mock import patch

import httpx
import fastapi_backend as backend
from analysis_admission import AnalysisAdmission

# Valid WAV signature; controlled analysis deliberately does not decode it.
AUDIO = b'RIFF' + b'\0' * 4 + b'WAVE' + b'\0' * 32


class ContainerAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.model = backend.app.state.model
        self.filler = backend.app.state.filler_ready
        self.admission = backend.admission
        backend.app.state.model = object()
        backend.admission = AnalysisAdmission(1, 1, .05)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=backend.app), base_url='http://test')

    async def asyncTearDown(self):
        await self.client.aclose()
        backend.app.state.model = self.model
        backend.app.state.filler_ready = self.filler
        backend.admission = self.admission

    async def request(self):
        return await self.client.post('/transcribe', files={'file': ('test.wav', AUDIO)})

    async def test_analysis_failure_releases_slot(self):
        with patch.object(backend, 'generate_full_analysis', side_effect=RuntimeError('controlled failure')):
            self.assertEqual((await self.request()).status_code, 500)
        self.assertEqual(backend.admission.active, 0)
        self.assertEqual(backend.admission.admitted, 0)
        with patch.object(backend, 'generate_full_analysis', return_value={'fillers': {'analysis_available': True}}):
            self.assertEqual((await self.request()).status_code, 200)

    async def test_bounded_queue_timeout_and_recovery(self):
        entered = threading.Event()
        release = threading.Event()

        def blocked(*args):
            entered.set()
            release.wait(5)
            return {'fillers': {'analysis_available': True}}

        with patch.object(backend, 'generate_full_analysis', side_effect=blocked):
            active = asyncio.create_task(self.request())
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                queued = asyncio.create_task(self.request())
                while backend.admission.admitted < 2:
                    await asyncio.sleep(.001)
                excess = await self.request()
                self.assertEqual(excess.status_code, 503)
                self.assertEqual(excess.headers['Retry-After'], '5')
                self.assertEqual((await queued).status_code, 503)
                self.assertEqual(backend.admission.active, 1)
                self.assertEqual(backend.admission.peak_active, 1)
                self.assertEqual(backend.admission.timed_out, 1)
            finally:
                release.set()
                self.assertEqual((await active).status_code, 200)
        self.assertEqual(backend.admission.admitted, 0)
        with patch.object(backend, 'generate_full_analysis', return_value={'fillers': {'analysis_available': True}}):
            self.assertEqual((await self.request()).status_code, 200)


if __name__ == '__main__':
    unittest.main()
