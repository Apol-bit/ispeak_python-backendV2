"""Focused bounded-fallback tests; no model weights, audio, or inference."""
import random
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np
from transformers.generation.utils import GenerationMixin
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin as W
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline as P

from audio_analysis_processing_files import whisper_fallback as fallback
from audio_analysis_processing_files.whisper_retention import install_longform_retention
from audio_validation import SpeechValidationError, timestamp_validity_counts, validate_transcript
from model import ISpeakWhisper
from whisper_service import (_extract_word_segments, _log_decode_diagnostics,
                             _validate_decode_windows, _run_analysis)


def pipeline_model(output):
    model = ISpeakWhisper.__new__(ISpeakWhisper)
    model.processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000, n_samples=480000),
        get_prompt_ids=Mock(return_value=torch.tensor([[1, 2, 3]])),
    )
    model.model = SimpleNamespace(device='cpu')
    model.pipe = Mock(return_value=output)
    return model


def generation_output(marker):
    return {
        'sequences': torch.tensor([marker]),
        'token_timestamps': torch.tensor([marker / 100.0]),
    }


class SyntheticWhisper(W):
    def __init__(self, sequences):
        self.config = SimpleNamespace(vocab_size=51865)
        self.sequences = sequences
        self.position = 0

    def _postprocess_outputs(self, **_):
        sequence, output = self.sequences[self.position]
        self.position += 1
        return sequence.unsqueeze(0), [output]


def run_fallback(sequences, *, generation_error=None, diagnostic_ratio_error=False, windows=1):
    model = SyntheticWhisper(sequences)
    if not install_longform_retention(model, P):
        raise AssertionError('Pinned retention hook did not install')
    if not fallback.install_fallback_diagnostics(model):
        raise AssertionError('Pinned fallback diagnostics hook did not install')
    config = SimpleNamespace(
        no_speech_threshold=None,
        compression_ratio_threshold=fallback.COMPRESSION_RATIO_THRESHOLD,
        logprob_threshold=None,
        cache_implementation=None,
        pad_token_id=0,
        eos_token_id=2,
        condition_on_prev_tokens=False,
    )
    original_ratio = model._retrieve_compression_ratio
    ratio_calls = 0

    def retrieve_ratio(*args):
        nonlocal ratio_calls
        ratio_calls += 1
        if diagnostic_ratio_error and ratio_calls % 2 == 0:
            raise RuntimeError('diagnostic ratio unavailable')
        return original_ratio(*args)

    generate_patch = (
        patch.object(GenerationMixin, 'generate', side_effect=generation_error)
        if generation_error is not None
        else patch.object(GenerationMixin, 'generate', return_value={})
    )
    with generate_patch as generate, patch.object(
            model, '_retrieve_compression_ratio', side_effect=retrieve_ratio):
        with fallback.collect_fallback_diagnostics(True) as diagnostics:
            results = []
            for window in range(windows):
                results.append(model.generate_with_fallback(
                    segment_input=torch.zeros(1, 2, 8),
                    decoder_input_ids=torch.tensor([[1]]),
                    cur_bsz=1,
                    seek=torch.tensor([window * 3000]),
                    batch_idx_map=[0],
                    temperatures=fallback.TEMPERATURES,
                    generation_config=config,
                    logits_processor=[],
                    stopping_criteria=None,
                    prefix_allowed_tokens_fn=None,
                    synced_gpus=False,
                    return_token_timestamps=True,
                    do_condition_on_prev_tokens=[False],
                    is_shortform=False,
                    batch_size=1,
                    attention_mask=torch.ones(1, 8),
                    kwargs={},
                ))
        return results[0] if windows == 1 else results, diagnostics, generate.call_count


class WhisperFallbackTests(unittest.TestCase):
    def setUp(self):
        self.repetitive = torch.tensor([42] * 120 + [2])
        randomizer = random.Random(17)
        self.clean = torch.tensor([randomizer.randrange(3, 50000) for _ in range(120)] + [2])
        self.assertGreater(
            W._retrieve_compression_ratio(self.repetitive, 51865),
            fallback.COMPRESSION_RATIO_THRESHOLD,
        )
        self.assertLessEqual(
            W._retrieve_compression_ratio(self.clean, 51865),
            fallback.COMPRESSION_RATIO_THRESHOLD,
        )

    def test_configuration_is_long_form_only(self):
        output = {'text': 'one two three', 'chunks': [
            {'text': 'one', 'timestamp': (0.1001, 0.1004)},
            {'text': 'two', 'timestamp': (0.2, 0.3)},
            {'text': 'three', 'timestamp': (0.4, 0.5)},
        ]}
        short = pipeline_model(output)
        short.transcribe('unused', audio_data=[0.0], language='English',
                         initial_prompt='short-form filler prompt', audio_duration_seconds=30.0)
        short_kwargs = short.pipe.call_args.kwargs['generate_kwargs']
        self.assertNotIn('temperature', short_kwargs)
        self.assertNotIn('compression_ratio_threshold', short_kwargs)
        self.assertNotIn('custom_generate', short_kwargs)
        self.assertTrue(torch.equal(short_kwargs['prompt_ids'], torch.tensor([[1, 2, 3]])))

        long = pipeline_model(output)
        result = long.transcribe('unused', audio_data=[0.0], language='English',
                                 initial_prompt='long-form filler prompt',
                                 audio_duration_seconds=30.001)
        long_kwargs = long.pipe.call_args.kwargs['generate_kwargs']
        self.assertEqual(long_kwargs['temperature'], (0.0, 0.2))
        self.assertEqual(long_kwargs['compression_ratio_threshold'], 1.35)
        self.assertNotIn('custom_generate', long_kwargs)
        self.assertNotIn('prompt_ids', long_kwargs)
        words = _extract_word_segments(result['segments'])
        self.assertEqual((words[0]['start'], words[0]['end']), (0.1001, 0.1004))

    def test_repetitive_decode_retries_at_most_once(self):
        _, diagnostics, calls = run_fallback([
            (self.repetitive, generation_output(1)),
            (self.repetitive, generation_output(2)),
        ])
        self.assertEqual(calls, 2)
        self.assertEqual(diagnostics[0]['retry_count'], 1)
        self.assertGreater(diagnostics[0]['initial_compression_ratio'], 1.35)
        self.assertGreater(diagnostics[0]['final_compression_ratio'], 1.35)
        self.assertTrue(diagnostics[0]['fallback_exhausted'])

    def test_clean_decode_remains_greedy(self):
        _, diagnostics, calls = run_fallback([
            (self.clean, generation_output(1)),
        ])
        self.assertEqual(calls, 1)
        self.assertEqual(diagnostics[0]['retry_count'], 0)
        self.assertLessEqual(diagnostics[0]['initial_compression_ratio'], 1.35)
        self.assertFalse(diagnostics[0]['fallback_exhausted'])

    def test_accepted_retry_timestamps_and_validator_predicate_are_preserved(self):
        accepted_output = generation_output(2)
        result, diagnostics, calls = run_fallback([
            (self.repetitive, generation_output(1)),
            (self.clean, accepted_output),
        ])
        self.assertEqual(calls, 2)
        self.assertEqual(diagnostics[0]['retry_count'], 1)
        self.assertFalse(diagnostics[0]['fallback_exhausted'])
        self.assertIs(result[1][0], accepted_output)
        self.assertTrue(torch.equal(result[1][0]['token_timestamps'],
                                    torch.tensor([0.02])))

        words = [
            {'text': 'one', 'start': 0.1001, 'end': 0.1004},
            {'text': 'two', 'start': 0.2, 'end': 0.3},
            {'text': 'three', 'start': 0.4, 'end': 0.5},
        ]
        snapshot = [dict(word) for word in words]
        self.assertEqual(timestamp_validity_counts(words, 1.0), (3, 3))
        self.assertIsNone(validate_transcript('one two three', words, 1.0))
        self.assertEqual(words, snapshot)

    def test_exhausted_first_window_preserves_valid_later_windows_and_offsets(self):
        failed_output = generation_output(2)
        later_output = {
            'sequences': torch.tensor([50265, 100, 50275]),
            'token_timestamps': torch.tensor([0.0, 0.1, 0.2]),
        }
        later_snapshot = {key: value.clone() for key, value in later_output.items()}
        results, diagnostics, calls = run_fallback([
            (self.repetitive, generation_output(1)),
            (self.repetitive, failed_output),
            (self.clean, later_output),
        ], windows=2)
        self.assertEqual(calls, 3)  # Two attempts on the first, one on the second.
        self.assertEqual([d['fallback_exhausted'] for d in diagnostics], [True, False])
        self.assertEqual([d['retry_count'] for d in diagnostics], [1, 0])
        self.assertIs(results[0][1][0], failed_output)
        self.assertIs(results[1][1][0], later_output)
        for key, value in later_snapshot.items():
            self.assertTrue(torch.equal(later_output[key], value))
        segments, frames = W._retrieve_segment(
            seek_sequence=later_output['sequences'], seek_outputs=[later_output],
            time_offset=torch.tensor([30.0]), timestamp_begin=50265,
            seek_num_frames=torch.tensor([3000]), time_precision=0.02,
            time_precision_features=0.01, input_stride=2, prev_idx=0, idx=0,
            return_token_timestamps=True, decoder_input_ids=torch.empty(1, 0, dtype=torch.long),
        )
        self.assertEqual(float(segments[0]['start']), 30.0)
        self.assertAlmostEqual(float(segments[0]['end']), 30.2)
        self.assertEqual(frames, 3000)
        self.assertTrue(torch.equal(segments[0]['token_timestamps'],
                                    later_snapshot['token_timestamps'] + 30.0))
        snapshot = copy.deepcopy(diagnostics)
        with self.assertLogs('whisper_service', level='WARNING') as logs:
            with self.assertRaises(SpeechValidationError) as caught:
                _validate_decode_windows({'long_form': True, 'segments': diagnostics}, 'user')
        self.assertEqual(caught.exception.code, 'insufficient_speech')
        self.assertIn('rule=exhausted_compression_fallback failed_windows=1', logs.output[0])
        self.assertEqual(diagnostics, snapshot)

    def test_exhaustion_rejects_before_scoring_even_if_global_timestamp_coverage_passes(self):
        _, diagnostics, _ = run_fallback([
            (self.repetitive, generation_output(1)),
            (self.repetitive, generation_output(2)),
            (self.clean, generation_output(3)),
        ], windows=2)
        transcription = {
            'text': 'PRIVATE word one two three four',
            'segments': [{'words': [
                {'word': 'PRIVATE', 'start': 0.1, 'end': 0.1},
                {'word': 'word', 'start': 0.1, 'end': 0.1},
                *[{'word': word, 'start': 30.0 + i, 'end': 30.5 + i}
                  for i, word in enumerate(('one', 'two', 'three', 'four'))],
            ]}],
            '_decoding_diagnostics': {'long_form': True, 'segments': diagnostics},
        }
        snapshot = copy.deepcopy(transcription)
        words = _extract_word_segments(transcription['segments'])
        self.assertEqual(timestamp_validity_counts(words, 86.64), (6, 4))
        self.assertIsNone(validate_transcript(transcription['text'], words, 86.64))
        model = Mock(transcribe=Mock(return_value=transcription))
        with patch('whisper_service.validate_audio', return_value=86.64), \
                patch('whisper_service.analyze_signal_intensity') as score, \
                self.assertLogs('whisper_service', level='INFO') as logs:
            with self.assertRaises(SpeechValidationError):
                _run_analysis('PRIVATE_AUDIO.wav', np.zeros(1), 16000, model)
        score.assert_not_called()
        self.assertEqual(transcription, snapshot)
        self.assertNotIn('PRIVATE', '\n'.join(logs.output))

    def test_recovered_first_window_and_clean_later_window_are_accepted(self):
        _, diagnostics, calls = run_fallback([
            (self.repetitive, generation_output(1)),
            (self.clean, generation_output(2)),
            (self.clean, generation_output(3)),
        ], windows=2)
        self.assertEqual(calls, 3)
        self.assertEqual([d['fallback_exhausted'] for d in diagnostics], [False, False])
        self.assertIsNone(_validate_decode_windows({'long_form': True, 'segments': diagnostics}, 'user'))
        # No new rule is applied to the working short-form path.
        self.assertIsNone(_validate_decode_windows({
            'long_form': False, 'segments': [{'fallback_exhausted': True}],
        }, 'user'))

    def test_exhaustion_decision_survives_optional_telemetry_failures(self):
        with patch.object(fallback, 'perf_counter', side_effect=RuntimeError('clock unavailable')):
            _, diagnostics, calls = run_fallback([
                (self.repetitive, generation_output(1)),
                (self.repetitive, generation_output(2)),
                (self.clean, generation_output(3)),
            ], diagnostic_ratio_error=True, windows=2)
        self.assertEqual(calls, 3)
        self.assertEqual([d['fallback_exhausted'] for d in diagnostics], [True, False])
        self.assertIsNone(diagnostics[0]['initial_compression_ratio'])
        with patch('whisper_service.logger.info', side_effect=RuntimeError('log unavailable')):
            _log_decode_diagnostics({'long_form': True, 'segments': diagnostics}, [], 86.64, 'user')
        with self.assertLogs('whisper_service', level='WARNING'):
            with self.assertRaises(SpeechValidationError):
                _validate_decode_windows({'long_form': True, 'segments': diagnostics}, 'user')

    def test_diagnostics_do_not_log_private_content(self):
        diagnostics = {
            'long_form': True,
            'elapsed_seconds': 1.25,
            'segments': [{
                'segment_index': 0,
                'initial_compression_ratio': 2.0,
                'final_compression_ratio': 1.0,
                'retry_count': 1,
                'elapsed_seconds': 1.0,
            }],
        }
        words = [{'text': 'PRIVATE_TRANSCRIPT', 'start': 0.1, 'end': 0.2}]
        with self.assertLogs('whisper_service', level='INFO') as logs:
            _log_decode_diagnostics(diagnostics, words, 1.0, 'PRIVATE_ROLE')
        output = '\n'.join(logs.output)
        self.assertNotIn('PRIVATE_TRANSCRIPT', output)
        self.assertNotIn('PRIVATE_ROLE', output)
        self.assertIn('role=unknown', output)
        self.assertIn('retry_count=1', output)
        self.assertIn('elapsed_available=true', output)
        self.assertIn('timestamp_total=1 timestamp_usable=1', output)

        with patch('whisper_service.logger.info', side_effect=RuntimeError('log unavailable')):
            self.assertIsNone(_log_decode_diagnostics(diagnostics, words, 1.0, 'user'))

    def test_hook_telemetry_failures_preserve_decode_and_exception(self):
        with patch.object(fallback, 'perf_counter', side_effect=RuntimeError('clock unavailable')):
            result, diagnostics, calls = run_fallback([
                (self.clean, generation_output(1)),
            ], diagnostic_ratio_error=True)
        self.assertEqual(calls, 1)
        self.assertEqual(result[1][0]['sequences'].item(), 1)
        self.assertIsNone(diagnostics[0]['initial_compression_ratio'])
        self.assertIsNone(diagnostics[0]['elapsed_seconds'])
        self.assertIsNone(fallback._CURRENT_SEGMENT.get())

        error = RuntimeError('original decode failure')
        with patch.object(fallback, 'perf_counter', side_effect=RuntimeError('clock unavailable')):
            with self.assertRaises(RuntimeError) as caught:
                run_fallback([
                    (self.clean, generation_output(1)),
                ], generation_error=error)
        self.assertIs(caught.exception, error)
        self.assertIsNone(fallback._CURRENT_SEGMENT.get())


if __name__ == '__main__':
    unittest.main()
