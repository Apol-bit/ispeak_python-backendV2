"""Model-free reference fallback and diagnostic regressions; no real audio."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import whisper_service as service
import audio_validation
from audio_validation import insufficient
from testing_script.test_assessment_details import fixture


def result():
    value = fixture()
    value['transcription'] = 'A valid sample sentence.'
    value['pronunciation'] = {'score': 87}
    value['speaking_rate'] = {'score': 100}
    value['_internal'] = {
        'pacing_stats': {'wpm': 127}, 'pacing_score': 100,
        'energy_score': 100, 'energy_stats': {'is_monotone': False},
        'articulation_stats': copy.deepcopy(value['articulation']),
        'filler_score': 100,
    }
    return value


class ReferenceValidationTests(unittest.TestCase):
    def test_decoded_silence_reports_signal_stage(self):
        with patch.object(audio_validation.subprocess, 'run', return_value=SimpleNamespace(stdout=np.zeros(32000, dtype='<f4').tobytes())), \
             self.assertLogs(service.logger, level='WARNING') as logs:
            with self.assertRaises(type(insufficient())):
                service.generate_full_analysis('private.wav', Mock())
        self.assertIn('role=user stage=signal_validation code=insufficient_speech', str(logs.output))
        self.assertNotIn('private', str(logs.output))

    def test_script_without_reference_retains_normal_assessment(self):
        original = result()
        with patch.object(service, 'load_valid_audio', return_value=(np.ones(2), 16000)), \
             patch.object(service, '_run_analysis', return_value=copy.deepcopy(original)):
            actual = service.generate_full_analysis('user', Mock())
        self.assertEqual(actual['scores'], original['scores'])
        self.assertFalse(actual['assessment_details']['fallback']['referencePacing'])

    def test_invalid_reference_signal_preserves_standard_user_scores(self):
        original = result()
        with patch.object(service, 'load_valid_audio', side_effect=[(np.ones(2), 16000), insufficient()]), \
             patch.object(service, '_run_analysis', return_value=copy.deepcopy(original)) as analyze, \
             self.assertLogs(service.logger, level='WARNING') as logs:
            actual = service.generate_reference_analysis('private-user.wav', 'private-reference.wav', Mock())
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(analyze.call_args.args[0], 'private-user.wav')
        self.assertEqual(actual['scores'], original['scores'])
        self.assertEqual(actual['transcription'], original['transcription'])
        self.assertEqual(actual['fillers'], original['fillers'])
        self.assertTrue(actual['assessment_details']['fallback']['referencePacing'])
        self.assertEqual(actual['assessment_details']['basis']['pacing'], 'target_range')
        self.assertEqual(actual['assessment_details']['completeness'], 'partial')
        self.assertFalse(actual['articulation']['clear_enunciation']['reference_comparison']['available'])
        self.assertNotIn('_internal', actual)
        self.assertIn('role=reference stage=decode code=insufficient_speech', str(logs.output))
        self.assertNotIn('private-', str(logs.output))

    def test_invalid_reference_transcript_falls_back(self):
        with patch.object(service, 'load_valid_audio', return_value=(np.ones(2), 16000)), \
             patch.object(service, '_run_analysis', side_effect=[insufficient(), result()]) as analyze:
            actual = service.generate_reference_analysis('user', 'reference', Mock())
        self.assertEqual(analyze.call_args_list[0].kwargs, {'audio_role': 'reference'})
        self.assertTrue(actual['assessment_details']['fallback']['referencePacing'])

    def test_invalid_user_signal_never_analyzes_reference(self):
        with patch.object(service, 'load_valid_audio', side_effect=insufficient()) as load, \
             patch.object(service, '_run_analysis') as analyze, \
             self.assertLogs(service.logger, level='WARNING') as logs:
            with self.assertRaises(type(insufficient())):
                service.generate_reference_analysis('user', 'reference', Mock())
        self.assertEqual(load.call_count, 1)
        analyze.assert_not_called()
        self.assertIn('role=user stage=decode', str(logs.output))

    def test_invalid_user_transcript_is_not_swallowed_after_reference_failure(self):
        with patch.object(service, 'load_valid_audio', return_value=(np.ones(2), 16000)), \
             patch.object(service, '_run_analysis', side_effect=[insufficient(), insufficient()]):
            with self.assertRaises(type(insufficient())):
                service.generate_reference_analysis('user', 'reference', Mock())

    def test_runtime_reference_failure_is_not_silently_skipped(self):
        with patch.object(service, 'load_valid_audio', return_value=(np.ones(2), 16000)), \
             patch.object(service, '_run_analysis', side_effect=RuntimeError('model unavailable')):
            with self.assertRaises(RuntimeError):
                service.generate_reference_analysis('user', 'reference', Mock())

    def test_valid_reference_keeps_existing_comparison(self):
        with patch.object(service, 'load_valid_audio', return_value=(np.ones(2), 16000)), \
             patch.object(service, '_run_analysis', side_effect=[result(), result()]), \
             patch.object(service, 'compare_reference_enunciation', return_value={'available': True, 'score': 90}):
            actual = service.generate_reference_analysis('user', 'reference', Mock())
        self.assertEqual(actual['speaking_rate']['reference_wpm'], 127)
        self.assertEqual(actual['scores']['pacing'], 100)
        self.assertTrue(actual['assessment_details']['basis']['referenceEnunciationApplied'])
        self.assertFalse(actual['assessment_details']['fallback']['referencePacing'])
        self.assertNotIn('_internal', actual)

    def test_transcript_stage_diagnostics_do_not_disclose_text(self):
        model = Mock()
        model.transcribe.return_value = {'text': 'private', 'segments': []}
        for role in ['user', 'reference']:
            with patch.object(service, 'validate_audio', return_value=2), \
                 self.assertLogs(service.logger, level='WARNING') as logs:
                with self.assertRaises(type(insufficient())):
                    service._run_analysis('private.wav', np.ones(2), 16000, model, audio_role=role)
            self.assertIn(f'role={role} stage=transcript_validation code=insufficient_speech', str(logs.output))
            self.assertNotIn('private', str(logs.output))


if __name__ == '__main__':
    unittest.main()
