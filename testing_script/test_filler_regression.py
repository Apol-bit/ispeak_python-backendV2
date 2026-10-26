"""Transcript-only regressions; injected predictions test policy, not accuracy."""
import unittest

from audio_analysis_processing_files.filler_words import analyze_fillers
from audio_analysis_processing_files.filler_vocabulary import VOCABULARY


def analyze(text, predictions=()):
    return analyze_fillers(
        [dict(text=w, start=i, end=i + .5) for i, w in enumerate(text.split())],
        classifier=lambda _: list(predictions), full_transcript=text,
    )


class FillerRegressionTests(unittest.TestCase):
    def test_case_punctuation_and_longest_multiword_match(self):
        text = 'UM, You KNOW what I mean; ALAM   MO BA? so ayun.'
        self.assertEqual([m.group().casefold() for m in VOCABULARY.finditer(text)],
                         ['um', 'you know what i mean', 'alam   mo ba', 'so ayun'])
        self.assertEqual(len(analyze(text)['filler_candidates']), 4)
        self.assertEqual(analyze(text)['filler_count'], 0)

    def test_word_boundaries_never_promote_subwords(self):
        for text, fragment in [('pumunta', 'um'), ('someone', 'so'), ('i-share', 'share')]:
            start = text.index(fragment)
            result = analyze(text, [dict(entity_group='FILLER', start=start,
                                        end=start + len(fragment), score=.9964)])
            self.assertEqual(result['filler_count'], 0)
            self.assertEqual(result['filler_score'], 100)
            self.assertEqual(result['filler_candidates'], [])
        self.assertEqual(list(VOCABULARY.finditer('human somehow anong aparang um-like')), [])

    def test_confirmed_present_in_three_languages(self):
        for text, phrase in [('UM, I spoke.', 'UM'), ('Ah, nagsalita ako.', 'Ah'),
                             ('Well, you know, nagsalita ako.', 'you know')]:
            start = text.index(phrase)
            result = analyze(text, [dict(entity_group='FILLER', start=start,
                                        end=start + len(phrase), score=.95)])
            self.assertEqual(result['filler_count'], 1)
            self.assertEqual(len(result['filler_words']), 1)
            self.assertEqual(result['filler_score'], round(max(0, 100 - result['filler_rate'] * 4)))

    def test_grammatical_uses_are_never_penalized_by_vocabulary(self):
        for text in ['I like apples so I bought some.',
                     'Ano ang sagot? Parang bahay ang gusali.',
                     'I like this kasi parang bahay, so we stayed.',
                     'Do you know the answer?', 'Alam mo ba ang sagot?']:
            result = analyze(text)
            self.assertEqual(result['filler_count'], 0)
            self.assertEqual(result['filler_score'], 100)
            self.assertTrue(all(c['confidence'] is None for c in result['filler_candidates']))

    def test_filler_absent_and_audio_omissions_are_not_invented(self):
        for text in ['The project works.', 'Magandang araw sa lahat.', 'We finished ang project.']:
            result = analyze(text)
            self.assertEqual(result['filler_count'], 0)
            self.assertEqual(result['filler_candidates'], [])

    def test_reported_session_terms_are_recognition_only(self):
        # Reported terms only: the 104-second production transcript is not locally
        # available. This must not be represented as a replay of that session.
        result = analyze('ano, parang, ah, so')
        self.assertEqual(result['filler_count'], 0)
        self.assertEqual([c['word'] for c in result['filler_candidates']], ['ano', 'parang', 'ah', 'so'])
        self.assertEqual(result['filler_score'], 100)

    def test_low_confidence_and_grouped_phrase_remain_unpenalized(self):
        text = 'You know, um, hello.'
        result = analyze(text, [dict(entity_group='FILLER', start=0, end=8, score=.59)])
        self.assertEqual(result['filler_count'], 0)
        self.assertEqual(len(result['filler_candidates']), 2)
        self.assertEqual(result['filler_score'], 100)

    def test_language_names_and_codes_do_not_gate_recognition(self):
        for language in ['English', 'Filipino', 'Taglish', 'tl', 'fil', 'en', None]:
            result = analyze_fillers([dict(text='ano,', start=0, end=.5)],
                                     classifier=lambda _: [], language=language)
            self.assertEqual(result['candidate_count'], 1)
            self.assertEqual(result['filler_count'], 0)

    def test_phrase_does_not_bridge_punctuation(self):
        self.assertEqual(list(VOCABULARY.finditer('you, know')), [])
        self.assertEqual(list(VOCABULARY.finditer('alam. mo')), [])


if __name__ == '__main__':
    unittest.main()
