"""Synthetic reproductions of reported counts, not recovered UAT evidence."""
import unittest

from audio_validation import SpeechValidationError, validate_transcript
from whisper_service import _extract_word_segments


class TimestampRejectionTests(unittest.TestCase):
    text = ' '.join(f'word{i}' for i in range(80))
    duration = 42.24

    def test_reported_counts_still_reject_without_lowering_threshold(self):
        # Two different causes can produce the same aggregate diagnostic.
        for invalid in [{'start': 5.0, 'end': 5.0},
                        {'start': 50.0, 'end': 50.1}]:
            with self.subTest(invalid=invalid):
                valid = [{'start': i * .2, 'end': i * .2 + .1} for i in range(132)]
                words = valid + [dict(invalid) for _ in range(314)]
                with self.assertLogs('audio_validation', level='WARNING') as logs:
                    with self.assertRaises(SpeechValidationError):
                        validate_transcript(self.text, words, self.duration, language='English')
                self.assertIn('rule=insufficient_usable_timestamps', logs.output[0])
                self.assertIn('recognized_tokens=80 timestamp_total=446 timestamp_usable=132', logs.output[0])

    def test_valid_eighty_word_timestamps_remain_accepted(self):
        words = [{'start': i * .5, 'end': i * .5 + .3} for i in range(80)]
        self.assertIsNone(validate_transcript(self.text, words, self.duration))

    def test_flattening_does_not_duplicate_or_repair_timestamp_evidence(self):
        source = [{'word': 'same', 'start': 1.0, 'end': 1.0},
                  {'word': 'same', 'start': 1.0, 'end': 1.0},
                  {'word': 'outside', 'start': 50.0, 'end': 51.0},
                  {'word': 'missing', 'start': None, 'end': None}]
        actual = _extract_word_segments([{'words': source}])
        self.assertEqual(len(actual), len(source))
        for original, mapped in zip(source, actual):
            self.assertEqual(mapped['start'], original['start'])
            self.assertEqual(mapped['end'], original['end'])


if __name__ == '__main__':
    unittest.main()
