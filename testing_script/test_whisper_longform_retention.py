"""Small synthetic generation outputs; no models, weights, audio or inference."""
import gc
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin as W
from transformers.generation.utils import GenerationMixin
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline as P
from audio_analysis_processing_files import whisper_retention as bounded


def owner():
    obj = SimpleNamespace()
    obj.generate = MethodType(W.generate, obj)
    obj.generate_with_fallback = MethodType(W.generate_with_fallback, obj)
    obj._retrieve_segment = W._retrieve_segment
    return obj


def output():
    return dict(sequences=torch.tensor([1, 100, 3, 4, 105]),
                token_timestamps=torch.arange(5) * .02, beam_indices=torch.tensor([0]),
                **{key: (torch.zeros(32, 64),) for key in bounded._DISCARD})


def retained_bytes(outputs):
    def size(x):
        if isinstance(x, torch.Tensor):
            return x.numel() * x.element_size()
        if isinstance(x, dict):
            return sum(size(v) for v in x.values())
        if isinstance(x, (tuple, list)):
            return sum(size(v) for v in x)
        return 0
    return size(outputs)


class LongformRetentionTests(unittest.TestCase):
    def hooked(self, data, events=None):
        model = owner()
        self.assertTrue(bounded.install_longform_retention(model, P))
        def complete(self, is_shortform=False, return_token_timestamps=True):
            # Simulate outputs AFTER all original decoding/fallback processing.
            if events is not None:
                events.append(tuple(data[0]))
            return (torch.tensor([[100, 3, 4, 105]]), data, [False], [True], dict)
        model._ispeak_retention_original = complete
        return model

    def test_real_upstream_fingerprints_activate_and_install_is_idempotent(self):
        model = owner()
        self.assertTrue(bounded.install_longform_retention(model, P))
        method = model.generate_with_fallback
        self.assertTrue(bounded.install_longform_retention(model, P))
        self.assertIs(method, model.generate_with_fallback)

    def test_only_completed_longform_word_timestamp_outputs_are_compacted(self):
        for active, short, timestamps in [(False, False, True), (True, True, True),
                                           (True, False, False), (True, False, True)]:
            data = [output(), output()]  # includes multi-batch/beam metadata
            events = []
            model = self.hooked(data, events)
            keep = [{k: v for k, v in o.items() if k not in bounded._DISCARD} for o in data]
            with bounded.compact_whisper_segments() if active else nullcontext():
                result = model.generate_with_fallback(short, timestamps)
            self.assertIs(result[1], data)
            self.assertIn('scores', events[0])  # delegate completed before removal
            for o, expected in zip(data, keep):
                for k, v in expected.items():
                    self.assertIs(o[k], v)
                self.assertEqual('scores' in o, not (active and not short and timestamps))

    def test_real_upstream_segment_tokens_offsets_and_timestamps_are_identical(self):
        for sequence in [torch.tensor([100, 3, 4, 105]),
                         torch.tensor([100, 3, 105, 105, 4, 110]),
                         torch.tensor([3, 4, 5])]:
            data = [output()]
            data[0]['token_timestamps'] = torch.arange(len(sequence)+1) * .02
            args = dict(seek_sequence=sequence, time_offset=torch.tensor([2.]), timestamp_begin=100,
                        seek_num_frames=torch.tensor([3000]), time_precision=.02,
                        time_precision_features=.01, input_stride=2, prev_idx=0, idx=0,
                        return_token_timestamps=True, decoder_input_ids=torch.tensor([[1]]))
            before, offset = W._retrieve_segment(seek_outputs=data, **args)
            # Preserve values independently; upstream segments reference the result dict.
            snapshots = [{k: v for k, v in s.items() if k != 'result'} for s in before]
            model = self.hooked(data)
            with bounded.compact_whisper_segments():
                model.generate_with_fallback()
            after, actual_offset = W._retrieve_segment(seek_outputs=data, **args)
            self.assertEqual(actual_offset, offset)
            for expected, actual in zip(snapshots, after):
                for key, value in expected.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.equal(value, actual[key]), key)
                    else:
                        self.assertEqual(value, actual[key])

    def test_diagnostic_storage_does_not_accumulate_across_chunks_or_requests(self):
        before = retained_bytes([output() for _ in range(10)])
        kept = []
        refs = []
        for _ in range(10):
            data = [output()]
            refs.extend(weakref.ref(data[0][k][0]) for k in bounded._DISCARD)
            model = self.hooked(data)
            with bounded.compact_whisper_segments():
                model.generate_with_fallback()
            kept.extend(data)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        after = retained_bytes(kept)
        self.assertLess(after, before / 100)
        print(f'Synthetic 10-chunk retained tensor bytes: {before} -> {after}')
        self.assertFalse(bounded._ACTIVE.get())

    def test_context_and_original_exception_are_preserved(self):
        model = self.hooked([output()])
        error = RuntimeError('synthetic failure')
        def fail(self, is_shortform=False, return_token_timestamps=True):
            raise error
        model._ispeak_retention_original = fail
        with self.assertRaises(RuntimeError) as caught:
            with bounded.compact_whisper_segments():
                model.generate_with_fallback()
        self.assertIs(caught.exception, error)
        self.assertFalse(bounded._ACTIVE.get())

    def test_actual_upstream_fallback_retries_finish_before_release(self):
        class Synthetic(W):
            def __init__(self):
                self.config = SimpleNamespace(vocab_size=128)
                self.decisions = 0

            def _postprocess_outputs(self, **kwargs):
                return torch.tensor([[100, 3, 4, 105, 2]]), [output()]

            def _need_fallback(self, sequence, results, *args):
                assert 'scores' in results[0] and 'cross_attentions' in results[0]
                self.decisions += 1
                return self.decisions == 1, False

        results = []
        for optimized in [False, True]:
            model = Synthetic()
            if optimized:
                self.assertTrue(bounded.install_longform_retention(model, P))
            config = SimpleNamespace(no_speech_threshold=None, cache_implementation=None,
                                     pad_token_id=0, eos_token_id=2, condition_on_prev_tokens=True)
            with patch.object(GenerationMixin, 'generate', return_value={}) as generate:
                with bounded.compact_whisper_segments():
                    result = model.generate_with_fallback(
                        segment_input=torch.zeros(1, 2, 8), decoder_input_ids=torch.tensor([[1]]),
                        cur_bsz=1, seek=torch.tensor([0]), batch_idx_map=[0], temperatures=[0.0, 0.6],
                        generation_config=config, logits_processor=[], stopping_criteria=None,
                        prefix_allowed_tokens_fn=None, synced_gpus=False, return_token_timestamps=True,
                        do_condition_on_prev_tokens=[True], is_shortform=False, batch_size=1,
                        attention_mask=torch.ones(1, 8), kwargs={})
                self.assertEqual(generate.call_count, 2)
            self.assertEqual(model.decisions, 2)
            self.assertEqual('scores' in result[1][0], not optimized)
            results.append(result)
        self.assertTrue(torch.equal(results[0][0][0], results[1][0][0]))
        self.assertTrue(torch.equal(results[0][1][0]['token_timestamps'], results[1][1][0]['token_timestamps']))
        self.assertEqual(results[0][2:], results[1][2:])

    def test_unsupported_implementation_falls_back_without_modification(self):
        for mismatch in ['version', 'method', 'pipeline']:
            model = owner()
            pipeline = P
            if mismatch == 'method':
                model.generate_with_fallback = lambda: None
            if mismatch == 'pipeline':
                pipeline = SimpleNamespace(_forward=lambda: None)
            original = model.generate_with_fallback
            with patch.object(bounded.importlib.metadata, 'version',
                              return_value='other' if mismatch == 'version' else '5.10.1'):
                self.assertFalse(bounded.install_longform_retention(model, pipeline))
            self.assertIs(model.generate_with_fallback, original)


if __name__ == '__main__':
    unittest.main()
