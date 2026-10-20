import unittest
from model import validate_local_audio_uat


class LocalAudioUatTests(unittest.TestCase):
    def config(self):
        return dict(ISPEAK_LOCAL_AUDIO_UAT='true', ISPEAK_ENV='development',
                    ISPEAK_HOST='127.0.0.1', ISPEAK_PORT='8000',
                    ISPEAK_MODEL_PATH='models/iSpeak_v6',
                    ISPEAK_BASE_MODEL_PATH='models/iSpeak_v5/base_model',
                    ISPEAK_FILLER_MODEL_PATH='models/filler_classifier',
                    ISPEAK_MODEL_MANIFEST='model_manifests/ispeak-v6.json',
                    HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                    ISPEAK_SERVICE_TOKEN='fictional-local-token-at-least-32-characters')

    def test_explicit_v6_loopback_offline_configuration(self):
        self.assertTrue(validate_local_audio_uat(self.config()))
        self.assertFalse(validate_local_audio_uat({}))
        self.assertFalse(validate_local_audio_uat({'ISPEAK_ENV': 'production'}))

    def test_missing_and_malformed_settings_fail_closed(self):
        for key in self.config():
            if key == 'ISPEAK_LOCAL_AUDIO_UAT':
                continue
            env = self.config(); del env[key]
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_local_audio_uat(env)
        for patch in [{'ISPEAK_LOCAL_AUDIO_UAT': 'TRUE'}, {'ISPEAK_HOST': '0.0.0.0'},
                      {'PORT': '9000'}, {'ISPEAK_MODEL_PATH': 'models/iSpeak_v5'},
                      {'ISPEAK_SERVICE_TOKEN': ''}, {'HF_HUB_OFFLINE': '0'}]:
            with self.subTest(patch=patch), self.assertRaises(RuntimeError):
                validate_local_audio_uat({**self.config(), **patch})
