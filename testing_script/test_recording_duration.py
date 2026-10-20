"""Boundaries use fake sample counts; no long audio or model inference."""
import os
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
import audio_validation as audio
import fastapi_backend as backend
from whisper_service import generate_full_analysis


class RecordingDurationTests(unittest.TestCase):
    def test_bounds_and_metadata(self):
        for seconds in [299, 300, 300.5]:
            audio.check_duration(seconds)
        for seconds in [300.501, 301]:
            with self.assertRaises(audio.SpeechValidationError): audio.check_duration(seconds)
        self.assertIsNone(audio.declared_duration(None, 300))
        self.assertEqual(audio.declared_duration('300', 300), 300)
        for value in ['', '-1', 'NaN', 'Infinity', '1e2', ' 1 ', '300.001']:
            with self.assertRaises(audio.SpeechValidationError): audio.declared_duration(value, 300)
        for value in [301, 0, float('nan'), True]:
            with self.assertRaises(audio.SpeechValidationError): audio.duration_limit(value)

    def test_bounded_compressed_decode_and_exact_limit(self):
        for seconds in [299, 300, 300.5, 300.5000625, 301]:
            fake = SimpleNamespace(size=round(seconds * 16000))
            with patch.object(audio.subprocess, 'run', return_value=SimpleNamespace(stdout=b'fixture')) as decoder, \
                 patch.object(audio.np, 'frombuffer', return_value=fake), patch.object(audio, 'validate_audio') as validate:
                if seconds > 300.5:
                    with self.assertRaises(audio.SpeechValidationError): audio.load_valid_audio('tiny-compressed.aac')
                    validate.assert_not_called()
                else:
                    self.assertEqual(audio.load_valid_audio('tiny-compressed.aac'), (fake, 16000))
                command = decoder.call_args.args[0]
                self.assertIn('300.5000625', command)
                self.assertEqual(decoder.call_args.kwargs['timeout'], 30)
                self.assertEqual(command[command.index('-ac') + 1], '1')
                self.assertEqual(command[command.index('-ar') + 1], '16000')
                self.assertIn('aresample=16000,atrim=end_sample=4808001', command)

    def test_short_real_wav_decodes_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'short.wav'
            t = np.arange(32000) / 16000
            sf.write(path, .1 * np.sin(2*np.pi*220*t) + .04*np.sin(2*np.pi*440*t), 16000)
            samples, rate = audio.load_valid_audio(path)
            self.assertEqual(rate, 16000)
            self.assertEqual(samples.size, 32000)
            for extension in ['mp3', 'aac', 'm4a', 'flac', 'ogg']:
                compressed = Path(directory) / ('short.' + extension)
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path), str(compressed)],
                               check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                decoded, rate = audio.load_valid_audio(compressed)
                self.assertEqual(rate, 16000)
                self.assertGreater(decoded.size, 16000)
                self.assertLess(compressed.stat().st_size, 1024*1024)

    def test_decode_failure_and_timeout_are_controlled(self):
        for failure in [subprocess.CalledProcessError(1, 'ffmpeg'), subprocess.TimeoutExpired('ffmpeg', 30)]:
            with patch.object(audio.subprocess, 'run', side_effect=failure):
                with self.assertRaises(audio.SpeechValidationError): audio.load_valid_audio('invalid.wav')

    def test_spoofed_duration_rejected_before_inference_and_upload_removed(self):
        paths = []
        save = backend._save_upload
        def track(*args):
            path = save(*args); paths.append(path); return path
        model = Mock()
        with patch.object(backend.app.state, 'model', model), \
             patch.object(backend, '_save_upload', side_effect=track), \
             patch.object(audio.subprocess, 'run', return_value=SimpleNamespace(stdout=b'fixture')), \
             patch.object(audio.np, 'frombuffer', return_value=SimpleNamespace(size=301*16000)):
            client = TestClient(backend.app)  # No context manager: no model-loading lifespan.
            try:
                response = client.post('/transcribe', files={'file': ('tiny.aac', b'\xff\xf1' + b'fixture'*100, 'audio/aac')},
                                       data={'duration_seconds': '1'},
                                       headers={'X-iSpeak-Service-Token': backend.SERVICE_TOKEN})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()['detail']['code'], 'recording_too_long')
                model.transcribe.assert_not_called()
            finally:
                client.close()
        self.assertTrue(paths)
        self.assertTrue(all(not os.path.exists(path) for path in paths))

    def test_route_metadata_rejection_precedes_decoding(self):
        with patch.object(backend.app.state, 'model', Mock()), patch.object(backend, '_save_upload') as save:
            client = TestClient(backend.app)
            try:
                for data in [{'duration_seconds': '301'}, {'duration_seconds': 'NaN'},
                             {'max_duration_seconds': '301'}, {'max_duration_seconds': '60', 'duration_seconds': '61'}]:
                    response = client.post('/transcribe', files={'file': ('tiny.wav', b'fixture', 'audio/wav')},
                                           data=data, headers={'X-iSpeak-Service-Token': backend.SERVICE_TOKEN})
                    self.assertEqual(response.status_code, 400)
                save.assert_not_called()
            finally: client.close()


if __name__ == '__main__':
    unittest.main()
