"""Aggregate diagnostics only; no model loading or audio inference."""
import unittest
from audio_validation import SpeechValidationError, validate_transcript


class TranscriptDiagnosticTests(unittest.TestCase):
    def words(self, count, valid=True):
        return [{'word': 'PRIVATE_WORD', 'start': i * .1,
                 'end': i * .1 + (.08 if valid else 0)} for i in range(count)]

    def rejected(self, text, words, duration, rule, **kwargs):
        with self.assertLogs('audio_validation', level='WARNING') as logs:
            with self.assertRaises(SpeechValidationError) as caught:
                validate_transcript(text, words, duration, **kwargs)
        self.assertEqual(caught.exception.code, 'insufficient_speech')
        self.assertEqual(caught.exception.message,
                         'The recording does not contain enough usable speech. Please record a short sentence in a quiet place and try again.')
        output = '\n'.join(logs.output)
        self.assertIn('rule=' + rule, output)
        self.assertNotIn('PRIVATE', output)
        self.assertNotIn(text, output)
        return output

    def test_each_rule_and_repetition_statistics(self):
        cases = [
            ('PRIVATE alpha', self.words(2), 2, 'minimum_recognized_tokens'),
            ('PRIVATE alpha beta', [], 2, 'missing_word_timestamps'),
            ('PRIVATE ' * 20, self.words(20), 1, 'excessive_token_rate'),
            ('PRIVATE ' * 12, self.words(12), 3, 'single_token_dominance'),
            ('PRIVATE alpha ' * 6, self.words(12), 3, 'repeated_phrase_dominance'),
            ('PRIVATE alpha beta', self.words(3, False), 2, 'insufficient_usable_timestamps'),
        ]
        for text, words, duration, rule in cases:
            with self.subTest(rule=rule):
                output = self.rejected(text, words, duration, rule, language='Filipino')
                self.assertIn('language=Filipino', output)
                self.assertIn(f'timestamp_total={len(words)}', output)
                self.assertIn(f'recognized_tokens={len(text.split())}', output)
                if rule == 'repeated_phrase_dominance':
                    self.assertIn('phrase_size=2 phrase_repeats=6 phrase_fraction=1.000000', output)

    def test_language_and_role_are_allowlisted_and_counts_are_numeric(self):
        output = self.rejected('PRIVATE alpha beta', self.words(3, False), 2,
                               'insufficient_usable_timestamps', language='PRIVATE_SECRET\n', audio_role='PRIVATE_PATH')
        self.assertIn('language=unknown', output)
        self.assertIn('role=unknown', output)
        self.assertIn('duration_seconds=2.000000', output)
        self.assertIn('timestamp_usable=0', output)

    def test_empty_transcript_and_reference_role(self):
        with self.assertLogs('audio_validation', level='WARNING') as logs:
            with self.assertRaises(SpeechValidationError):
                validate_transcript('', [], 2, language='Taglish', audio_role='reference')
        self.assertIn('role=reference', logs.output[0])
        self.assertIn('recognized_tokens=0', logs.output[0])

    def test_valid_transcript_is_unchanged_and_not_logged(self):
        with self.assertNoLogs('audio_validation', level='WARNING'):
            self.assertIsNone(validate_transcript('Three ordinary words', self.words(3), 2, language='English'))


if __name__ == '__main__':
    unittest.main()
