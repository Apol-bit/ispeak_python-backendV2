"""Saved UAT text fixtures and mocked predictions only: no DB, audio or models."""
import copy
from pathlib import Path
import unittest

from audio_analysis_processing_files.word_alignment import reconstruct_words, word_tokens
from audio_analysis_processing_files.filler_words import analyze_fillers
from audio_analysis_processing_files.speaking_rate import analyze_speaking_rate
from whisper_service import _compute_pacing_score, _compute_overall_score


ENGLISH = "good morning everyone today i want to talk about our thesis project i speak um our application helps our users practice their public speaking skills in english and filipino you know we want to help students who'll become more confident speaker."
FILIPINO = "um magalang araw sa yung lahat ngayon gusto kong ipakilala ang amin thesis project na i-speak uh um i-ginawa namin ang application na ito para matulungan ang mga students na mas maging confidence sa public speaking ano, makalaga kasi na matutu tayong mag-sayta ng malinaw sa harap ng maraming top"
TAGLISH = "Good morning everyone today gusto kong i-share yung thesis project namin called i-speak um this application helps students to improve their public speaking skills in english and filipino kasi you know maraming students ang nahirapan mag-salita sa harap nang maraming tao uh so our goal is to help them become more confident speakers"


def segments(text):
    return [dict(text=word, start=i, end=i + .5) for i, word in enumerate(text.split())]


class WordAlignmentTests(unittest.TestCase):
    def test_three_uat_reconstructions_and_original_evidence(self):
        for text, count in [(ENGLISH, 41), (FILIPINO, 50), (TAGLISH, 53)]:
            raw = text.replace('i-speak', 'i -speak').replace('i-ginawa', 'i -ginawa').replace('mag-sayta', 'mag -sayta').replace('i-share', 'i -share').replace('mag-salita', 'mag -salita')
            words = segments(raw)
            before = copy.deepcopy(words)
            captured = []
            result = analyze_fillers(words, classifier=lambda value: captured.append(value) or [], full_transcript=text)
            with self.subTest(text=text[:20]):
                self.assertEqual(captured, [text])
                self.assertEqual(result['coverage'], dict(status='matched', classifier_word_count=count, transcript_word_count=count))
                self.assertEqual(words, before)
                rebuilt, aligned = reconstruct_words(words, text)
                self.assertEqual(len(aligned), len(words))
                for original, mapped in zip(words, aligned):
                    self.assertEqual(rebuilt[mapped['char_start']:mapped['char_end']], original['text'])
                    self.assertEqual((mapped['start'], mapped['end']), (original['start'], original['end']))

    def test_filler_offsets_after_join_and_compound_time_union(self):
        words = segments('i -share um hello')
        def classifier(text):
            self.assertEqual(text, 'i-share um hello')
            return [dict(entity_group='FILLER', start=8, end=10, score=.95)]
        result = analyze_fillers(words, classifier=classifier, full_transcript='i-share um hello')
        self.assertEqual(result['filler_words'][0]['word'], 'um')
        self.assertEqual((result['filler_words'][0]['start'], result['filler_words'][0]['end']), (2, 2.5))
        result = analyze_fillers(words, classifier=lambda _: [dict(entity_group='FILLER', start=0, end=7, score=.95)], full_transcript='i-share um hello')
        self.assertEqual(result['filler_words'][0]['word'], 'i-share')
        self.assertEqual((result['filler_words'][0]['start'], result['filler_words'][0]['end']), (0, 1.5))

    def test_true_omissions_replacements_and_unverified_joins(self):
        for raw, full, expected in [
            ('i -share um', 'i share um', 'i -share um'),
            ('i -share um', 'i-shared um', 'i -share um'),
            ('i -share um', 'i-share missing um', 'i-share um'),
            ('hello um', 'hello extra um', 'hello um'),
            ('hello there', 'hello elsewhere', 'hello there'),
        ]:
            captured = []
            result = analyze_fillers(segments(raw), classifier=lambda value: captured.append(value) or [], full_transcript=full)
            self.assertEqual(captured, [expected])
            # Existing coverage policy ignores standalone punctuation: an
            # unjoined leading hyphen does not itself establish missing words.
            expected_coverage = 'matched' if full == 'i share um' else 'mismatch'
            self.assertEqual(result['coverage']['status'], expected_coverage)
        text, _ = reconstruct_words(segments('i -share'))
        self.assertEqual(text, 'i -share')  # No full-text evidence: no speculative joining.

    def test_punctuation_apostrophes_unicode_and_code_switching(self):
        self.assertEqual(word_tokens("Hello, ANO! i-share don't don’t — … -"), ['hello', 'ano', 'i-share', "don't", "don't"])
        raw = 'HELLO, i -share ano, don’t stop!'
        full = "hello i-share ano don't stop"
        result = analyze_fillers(segments(raw), classifier=lambda _: [], full_transcript=full)
        self.assertEqual(result['coverage']['status'], 'matched')
        self.assertEqual(result['total_words'], 5)
        self.assertEqual(analyze_speaking_rate(segments(raw), 10, full_transcript=full)['wpm'], 30)
        for raw, full in [('a -b -c um', 'a-b-c um'), ('i -share i -share um', 'i-share i-share um')]:
            self.assertEqual(reconstruct_words(segments(raw), full)[0], full)

    def test_shared_counts_and_score_behavior_with_fixed_predictions(self):
        # Previously inspected container durations; arithmetic replay only.
        # A future decoded sample duration/model rerun may differ.
        for text, duration, count, articulation, expected_filler, expected_pace, expected_clarity, expected_overall in [
            (FILIPINO, 26.069333, 50, 92, 84, 95.9, 89, 94.2),
            (TAGLISH, 28.202667, 53, 90.2, 85, 94.0, 88, 93.1),
        ]:
            raw = text.replace('i-speak', 'i -speak').replace('i-ginawa', 'i -ginawa').replace('mag-sayta', 'mag -sayta').replace('i-share', 'i -share').replace('mag-salita', 'mag -salita')
            def predictions(value):
                # Two fixed single-word detections, independent of model accuracy.
                return [dict(entity_group='FILLER', start=match, end=match+2, score=.99) for match in [value.index('um'), value.index('uh')]]
            # Score arithmetic replay uses fluent synthetic timing; pause-gated
            # complementary detection is exercised in test_filler_hybrid.
            fluent = [dict(text=w['text'], start=i * .2, end=i * .2 + .15)
                      for i, w in enumerate(segments(raw))]
            result = analyze_fillers(fluent, classifier=predictions, full_transcript=text)
            self.assertEqual(result['total_words'], count)
            self.assertEqual(result['filler_score'], expected_filler)
            pacing = analyze_speaking_rate(segments(raw), duration, full_transcript=text)
            pace = _compute_pacing_score(pacing)
            clarity = round(articulation * .6 + result['filler_score'] * .4)
            self.assertEqual((pace, clarity, _compute_overall_score(pace, clarity, 100)), (expected_pace, expected_clarity, expected_overall))

    def test_punctuation_only_and_legacy_calls(self):
        result = analyze_fillers(segments('... -'), classifier=lambda _: self.fail('No lexical words'), full_transcript='... -')
        self.assertEqual(result['total_words'], 0)
        self.assertEqual(analyze_speaking_rate(segments('... -'), 2)['wpm'], 0)
        self.assertEqual(analyze_speaking_rate(segments('hello world'), 2)['wpm'], 60)

    def test_packaging_includes_shared_module(self):
        root = Path(__file__).resolve().parents[1]
        self.assertIn('!audio_analysis_processing_files/**', (root / '.dockerignore').read_text())
        self.assertIn('COPY audio_analysis_processing_files/', (root / 'Dockerfile').read_text())
