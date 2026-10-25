"""Synthetic forward outputs only: no model construction or speech inference."""
import gc
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref

import torch
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin as W
from audio_analysis_processing_files import whisper_retention as release
from testing_script.test_whisper_timestamp_memory import inputs, owner
from audio_analysis_processing_files.whisper_timestamps import install_selected_head_timestamps


class ActiveChunkTests(unittest.TestCase):
    def test_real_forward_fingerprint_and_guarded_hook_installation(self):
        model = SimpleNamespace(_ispeak_retention_original=W.generate_with_fallback,
                                register_forward_hook=Mock(return_value=object()))
        model.forward = MethodType(WhisperForConditionalGeneration.forward, model)
        self.assertTrue(release.install_active_chunk_release(model))
        self.assertTrue(release.install_active_chunk_release(model))
        model.register_forward_hook.assert_called_once_with(release._release_self_attentions)
        other = SimpleNamespace(_ispeak_retention_original=W.generate_with_fallback,
                                forward=lambda: None, register_forward_hook=Mock())
        self.assertFalse(release.install_active_chunk_release(other))
        other.register_forward_hook.assert_not_called()

    def test_logits_cross_attention_cache_and_original_output_are_unchanged(self):
        for dtype in [torch.float32, torch.float16, torch.float64]:
            raw = Seq2SeqLMOutput(logits=torch.ones(2, 1, 5, dtype=dtype),
                past_key_values=object(), cross_attentions=(torch.ones(2, 2, 1, 8, dtype=dtype),),
                decoder_attentions=(torch.ones(2, 2, 1, 12, dtype=dtype),))
            self.assertIsNone(release._release_self_attentions(None, (), raw))
            token = release._CHUNK_ACTIVE.set(True)
            try:
                compact = release._release_self_attentions(None, (), raw)
            finally:
                release._CHUNK_ACTIVE.reset(token)
            for key in ['logits', 'past_key_values', 'cross_attentions']:
                self.assertIs(compact[key], raw[key])
            self.assertEqual(raw.decoder_attentions[0].shape, (2, 2, 1, 12))
            self.assertEqual(compact.decoder_attentions[0].shape, (2, 2, 0, 0))
            self.assertEqual(compact.decoder_attentions[0].dtype, dtype)
            self.assertEqual(compact.decoder_attentions[0].device, raw.logits.device)

    def test_real_timestamp_extraction_and_postprocessing_remain_identical(self):
        raw = inputs()
        raw['decoder_attentions'] = tuple(tuple(torch.ones(1, 8, n, n) for _ in range(4))
                                          for n in [2, 1, 1, 1])
        modified = dict(raw)
        modified['decoder_attentions'] = []
        token = release._CHUNK_ACTIVE.set(True)
        try:
            for values in raw.decoder_attentions:
                out = release._release_self_attentions(None, (), Seq2SeqLMOutput(decoder_attentions=values))
                modified['decoder_attentions'].append(out.decoder_attentions)
        finally:
            release._CHUNK_ACTIVE.reset(token)
        modified['decoder_attentions'] = tuple(modified['decoder_attentions'])
        modified = type(raw)(**modified)
        model = owner()
        self.assertTrue(install_selected_head_timestamps(model))
        config = SimpleNamespace(alignment_heads=[(0, 1), (3, 7)], num_frames=None)
        args = dict(decoder_input_ids=torch.tensor([[1, 2]]), return_token_timestamps=True,
                    generation_config=config, is_shortform=False, seek=torch.tensor([0]), batch_idx_map=[0])
        before = W._postprocess_outputs(model, seek_outputs=raw, **args)
        after = W._postprocess_outputs(model, seek_outputs=modified, **args)
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[1][0]['token_timestamps'], after[1][0]['token_timestamps']))

    def test_current_chunk_diagnostic_payload_and_references_are_released(self):
        saved = []
        refs = []
        before = 0
        token = release._CHUNK_ACTIVE.set(True)
        try:
            for step in range(1, 129):
                raw = Seq2SeqLMOutput(decoder_attentions=tuple(torch.ones(1, 2, 1, step) for _ in range(2)))
                before += sum(v.numel() * v.element_size() for v in raw.decoder_attentions)
                refs.extend(weakref.ref(v) for v in raw.decoder_attentions)
                compact = release._release_self_attentions(None, (), raw)
                saved.append(compact.decoder_attentions)
                del raw, compact
        finally:
            release._CHUNK_ACTIVE.reset(token)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        after = sum(v.numel() * v.element_size() for row in saved for v in row)
        self.assertEqual(after, 0)
        print(f'Synthetic active-chunk self-attention payload bytes: {before} -> {after}')

    def test_chunk_scope_is_reset_after_original_failure(self):
        def fail(self, is_shortform=False, return_token_timestamps=True):
            self.seen = release._CHUNK_ACTIVE.get()
            raise ValueError('synthetic')
        model = SimpleNamespace(_ispeak_retention_original=fail)
        with release.compact_whisper_segments(), self.assertRaises(ValueError):
            release._generate_with_fallback(model)
        self.assertTrue(model.seen)
        self.assertFalse(release._CHUNK_ACTIVE.get())

    def test_unusual_attention_outputs_are_left_unchanged(self):
        token = release._CHUNK_ACTIVE.set(True)
        try:
            for output in [{}, {'decoder_attentions': None},
                           {'decoder_attentions': (None,)},
                           {'decoder_attentions': (torch.ones(3),)}]:
                self.assertIsNone(release._release_self_attentions(None, (), output))
        finally:
            release._CHUNK_ACTIVE.reset(token)


if __name__ == '__main__':
    unittest.main()
