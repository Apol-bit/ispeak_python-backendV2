"""Exercise the actual adapter method without importing/loading model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import Mock

from audio_validation import SpeechValidationError, validate_transcript
from whisper_service import _extract_word_segments


tree = ast.parse((Path(__file__).resolve().parents[1] / 'model.py').read_text(encoding='utf-8'))
adapter = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ISpeakWhisper')
method = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == 'transcribe')
namespace = {'Any': Any}
exec(compile(ast.Module(body=[method], type_ignores=[]), 'model.py', 'exec'), namespace)


def convert(chunks, text):
    pipe = Mock(return_value={'text': text, 'chunks': chunks})
    return namespace['transcribe'](SimpleNamespace(pipe=pipe), 'unused', language='English')


class ModelTimestampTests(unittest.TestCase):
    def test_positive_boundaries_are_not_collapsed_by_rounding(self):
        chunks = [{'text': word, 'timestamp': (i + .1001, i + .1004)}
                  for i, word in enumerate(['one', 'two', 'three'])]
        result = convert(chunks, 'one two three')
        words = _extract_word_segments(result['segments'])
        for source, word in zip(chunks, words):
            self.assertEqual((word['start'], word['end']), source['timestamp'])
        validate_transcript(result['text'], words, 3)

    def test_zero_duration_and_out_of_range_are_not_repaired(self):
        for interval in [(1, 1), (40, 41)]:
            with self.subTest(interval=interval):
                result = convert([{'text': 'one two three', 'timestamp': interval}], 'one two three')
                with self.assertRaises(SpeechValidationError):
                    validate_transcript(result['text'], _extract_word_segments(result['segments']), 37.36)

    def test_repeated_chunks_are_preserved_not_deduplicated(self):
        chunks = [{'text': 'one', 'timestamp': (1, 1)} for _ in range(12)]
        result = convert(chunks, 'one ' * 12)
        self.assertEqual(len(_extract_word_segments(result['segments'])), 12)
        with self.assertRaises(SpeechValidationError):
            validate_transcript(result['text'], _extract_word_segments(result['segments']), 37.36)

    def test_latest_reported_counts_reject_even_with_matching_token_counts(self):
        text = ' '.join(['repeated'] * 102 + [f'word{i}' for i in range(230)])
        for invalid in [(3, 3), (40, 41)]:
            chunks = [{'text': word, 'timestamp': (i * .2, i * .2 + .1) if i < 109 else invalid}
                      for i, word in enumerate(text.split())]
            result = convert(chunks, text)
            with self.assertLogs('audio_validation', level='WARNING') as logs:
                with self.assertRaises(SpeechValidationError):
                    validate_transcript(result['text'], _extract_word_segments(result['segments']), 37.36)
            self.assertIn('rule=insufficient_usable_timestamps', logs.output[0])
            self.assertIn('recognized_tokens=332 timestamp_total=332 timestamp_usable=109', logs.output[0])
            self.assertIn('most_common_token_count=102', logs.output[0])


if __name__ == '__main__':
    unittest.main()
