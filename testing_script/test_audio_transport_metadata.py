import hashlib
import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
import fastapi_backend as backend


class AudioTransportMetadataTests(unittest.TestCase):
    def test_upload_copy_preserves_exact_bytes_and_hash(self):
        payload = (b'RIFF' + (36).to_bytes(4, 'little') + b'WAVEfmt '
                   + (16).to_bytes(4, 'little') + (1).to_bytes(2, 'little')
                   + (1).to_bytes(2, 'little') + (16000).to_bytes(4, 'little')
                   + (32000).to_bytes(4, 'little') + (2).to_bytes(2, 'little')
                   + (16).to_bytes(2, 'little') + b'data' + (4).to_bytes(4, 'little')
                   + b'\x01\x00\x02\x00')
        upload = type('Upload', (), {'file': io.BytesIO(payload)})()
        copied = backend._save_upload(upload, '.wav')
        try:
            self.assertEqual(Path(copied).read_bytes(), payload)
            self.assertEqual(backend._file_sha256(copied), hashlib.sha256(payload).hexdigest())
        finally:
            os.remove(copied)

    def test_hash_and_log_use_only_bounded_metadata(self):
        payload = b'RIFF' + (b'private-audio-bytes' * 8)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'private-speaker-name.wav'
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(backend._file_sha256(str(path)), expected)
            with self.assertLogs(backend.logger, level=logging.INFO) as captured:
                backend._log_transport_metadata(
                    file_path=str(path), suffix='.wav', content_type='audio/wave',
                    expected_sha256=expected, declared=37.0, language='English',
                    reference_used=False,
                )
        line = '\n'.join(captured.output)
        self.assertIn(f'sha256={expected}', line)
        self.assertIn('upstream_hash_match=true', line)
        self.assertIn('declared_duration_seconds=37.000', line)
        self.assertNotIn('private-speaker-name', line)
        self.assertNotIn('private-audio-bytes', line)

    def test_untrusted_hash_and_mime_are_not_echoed(self):
        with tempfile.TemporaryDirectory() as root:
            audio = Path(root) / 'fixture.wav'
            audio.write_bytes(b'RIFF-safe-fixture')
            with self.assertLogs(backend.logger, level=logging.INFO) as captured:
                backend._log_transport_metadata(
                    file_path=str(audio), suffix='.wav', content_type='PRIVATE/MIME',
                    expected_sha256='PRIVATE_HASH', declared=None, language='Taglish',
                    reference_used=True,
                )
        line = '\n'.join(captured.output)
        self.assertIn('content_type=unknown', line)
        self.assertIn('upstream_hash_present=false upstream_hash_match=false', line)
        self.assertNotIn('PRIVATE', line)


if __name__ == '__main__':
    unittest.main()
