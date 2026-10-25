"""Synthetic attention tensors only; no model construction, weights or audio."""
import gc
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from audio_analysis_processing_files import whisper_timestamps as bounded


ORIGINAL = WhisperGenerationMixin._extract_token_timestamps


def owner():
    obj = SimpleNamespace(config=SimpleNamespace(decoder_layers=4, median_filter_width=3))
    obj._extract_token_timestamps = MethodType(ORIGINAL, obj)
    return obj


def inputs(batch=1, beam=False):
    generator = torch.Generator().manual_seed(71)
    result = bounded._Inputs(
        sequences=torch.zeros((batch, 8), dtype=torch.long),
        cross_attentions=tuple(tuple(torch.rand((batch, 8, n, 40), generator=generator)
                                    for _ in range(4)) for n in [2, 1, 1, 1]),
    )
    if beam:
        result['beam_indices'] = torch.tensor([[0, 0, 0, 0], [1, 1, 1, -1]])
    return result


class WhisperTimestampMemoryTests(unittest.TestCase):
    def test_exact_timestamps_with_crops_prefill_and_duplicate_heads(self):
        for frames in [None, 60, [60], (60,), torch.tensor([60])]:
            for prompt in [0, 1, 2, 5]:
                source = inputs()
                for heads in [[(0, 1), (3, 7)], [(3, 7), (0, 1), (3, 7)]]:
                    original = ORIGINAL(owner(), source, heads, num_frames=frames, num_input_ids=prompt)
                    model = owner()
                    self.assertTrue(bounded.install_selected_head_timestamps(model))
                    actual = model._extract_token_timestamps(source, heads, num_frames=frames, num_input_ids=prompt)
                    self.assertTrue(torch.equal(original, actual))

    def test_batch_variable_frames_and_beam_gathering(self):
        for beam in [False, True]:
            source = inputs(batch=2, beam=beam)
            model = owner()
            bounded.install_selected_head_timestamps(model)
            for frames in [None, [60, 70], torch.tensor([60, 70])]:
                expected = ORIGINAL(owner(), source, [(0, 1), (3, 7)], num_frames=frames, num_input_ids=2)
                actual = model._extract_token_timestamps(source, [(0, 1), (3, 7)], num_frames=frames, num_input_ids=2)
                self.assertTrue(torch.equal(expected, actual))

    def test_cat_copies_only_selected_heads(self):
        source = inputs()
        real_cat = torch.cat
        allocations = []
        def measured(tensors, *args, **kwargs):
            result = real_cat(tensors, *args, **kwargs)
            if kwargs.get('dim') == 2:
                allocations.append(result.numel() * result.element_size())
            return result
        with patch('torch.cat', side_effect=measured):
            ORIGINAL(owner(), source, [(0, 1), (3, 7)], num_input_ids=2)
        before = sum(allocations)
        allocations.clear()
        model = owner()
        bounded.install_selected_head_timestamps(model)
        with patch('torch.cat', side_effect=measured):
            model._extract_token_timestamps(source, [(0, 1), (3, 7)], num_input_ids=2)
        self.assertEqual(sum(allocations), before // 16)

    def test_inputs_unchanged_and_no_request_tensors_retained(self):
        model = owner()
        bounded.install_selected_head_timestamps(model)
        for _ in range(12):
            source = inputs()
            original_attention = source.cross_attentions
            tensor = source.cross_attentions[0][0]
            ref = weakref.ref(tensor)
            snapshot = tensor.clone()
            result = model._extract_token_timestamps(source, [(0, 1)], num_input_ids=2)
            self.assertIs(source.cross_attentions, original_attention)
            self.assertTrue(torch.equal(tensor, snapshot))
            del source, original_attention, tensor, snapshot, result
            gc.collect()
            self.assertIsNone(ref())

    def test_unsupported_version_and_method_remain_unchanged(self):
        model = owner()
        original = model._extract_token_timestamps
        with patch.object(bounded.importlib.metadata, 'version', return_value='different'):
            self.assertFalse(bounded.install_selected_head_timestamps(model))
        self.assertIs(model._extract_token_timestamps, original)
        model._extract_token_timestamps = lambda *args: None
        self.assertFalse(bounded.install_selected_head_timestamps(model))


if __name__ == '__main__':
    unittest.main()
