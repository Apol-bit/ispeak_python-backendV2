"""Model-free rule/contract regressions; real model comparison is a separate run."""
import copy
import json
from pathlib import Path
import unittest

from audio_analysis_processing_files.filler_words import analyze_fillers, ClassifierUnavailable
from audio_analysis_processing_files.filler_context import valid_timing
from audio_analysis_processing_files.filler_vocabulary import VOCABULARY, ENGLISH, FILIPINO

FIXTURES = json.loads((Path(__file__).parent / 'fixtures/filler_hybrid_labeled.json').read_text(encoding='utf-8'))['cases']


def case(identifier):
    return copy.deepcopy(next(c for c in FIXTURES if c['id'] == identifier))


def run(item, predictions=()):
    return analyze_fillers(item['word_segments'], classifier=lambda _: list(predictions),
                          full_transcript=item['transcript'], language=item['language'])


class HybridFillerTests(unittest.TestCase):
    def test_confirmed_restart_in_each_language_without_model_filler(self):
        for identifier, expected in [('E02', 'like'), ('F01', 'ano'), ('F02', 'parang'), ('T01', 'ano'), ('T07', 'parang')]:
            with self.subTest(identifier=identifier):
                item = case(identifier)
                start = item['annotations'][0]['start']
                result = run(item, [dict(entity_group='CONTEXT_WORD', start=start,
                                         end=item['annotations'][0]['end'], score=.99)])
                self.assertEqual(result['filler_count'], 1)
                self.assertEqual(result['filler_words'][0]['word'].casefold(), expected)
                self.assertIsNone(result['filler_words'][0]['confidence'])
                self.assertIn('restart', result['filler_words'][0]['reason'])

    def test_multiword_restart_counts_one_span(self):
        for identifier in ['E03', 'F08', 'T08']:
            result = run(case(identifier))
            self.assertEqual(result['filler_count'], 1)
            self.assertEqual(result['count_unit'], 'grouped_spans')
            self.assertIn(' ', result['filler_words'][0]['word'])

    def test_clear_hesitations_with_separation_and_repetition(self):
        text = 'Well, um, uh, I need time.'
        item = dict(transcript=text, language='English', word_segments=[
            dict(text=w, start=i, end=i+.2) for i,w in enumerate(text.split())])
        result = run(item)
        self.assertEqual([w['word'].casefold() for w in result['filler_words']], ['um', 'uh'])
        self.assertNotEqual(result['filler_words'][0]['start'], result['filler_words'][1]['start'])

    def test_boundary_hesitation_is_not_confirmed_from_one_pause_alone(self):
        result = run(case('E01'))
        self.assertEqual(result['filler_count'], 0)
        self.assertIn('Um', [c['word'] for c in result['filler_candidates']])

    def test_grammatical_controls_with_identical_pause_evidence_not_penalized(self):
        for item in FIXTURES:
            if any(a['label'] == 'grammatical' for a in item['annotations']):
                with self.subTest(identifier=item['id']):
                    self.assertTrue(valid_timing(item['word_segments']))
                    self.assertEqual(run(item)['filler_count'], 0)

    def test_ambiguous_occurrences_remain_candidates(self):
        for item in FIXTURES:
            if any(a['label'] == 'ambiguous' for a in item['annotations']):
                result = run(item)
                self.assertEqual(result['filler_count'], 0)
                self.assertGreater(result['candidate_count'], 0)

    def test_as_in_is_recognized_but_not_automatically_confirmed(self):
        # Even a restart can be emphasis/clarification. Abstention is intentional.
        result = run(case('T02'))
        self.assertEqual(result['filler_count'], 0)
        self.assertIn('as in', [c['word'] for c in result['filler_candidates']])

    def test_no_restart_is_not_enough_for_ambiguous_markers(self):
        for identifier in ['F12', 'T12']:
            result = run(case(identifier))
            self.assertEqual(result['filler_count'], 0)
            self.assertIn('ano', [c['word'] for c in result['filler_candidates']])

    def test_missing_invalid_or_overlapping_timing_disables_complementary_counts(self):
        for field, value in [('start', None), ('start', float('nan')), ('end', float('inf')),
                             ('start', True), ('start', -1), ('end', 0)]:
            item = case('F01')
            item['word_segments'][0][field] = value
            self.assertFalse(valid_timing(item['word_segments']))
            self.assertEqual(run(item)['filler_count'], 0)
        item = case('F01')
        del item['word_segments'][0]['start']
        self.assertFalse(valid_timing(item['word_segments']))
        self.assertEqual(run(item)['filler_count'], 0)
        item = case('F01')
        item['word_segments'][1]['start'] = item['word_segments'][0]['start']
        self.assertFalse(valid_timing(item['word_segments']))
        self.assertEqual(run(item)['filler_count'], 0)

    def test_repetition_or_punctuation_without_pause_gaps_does_not_confirm(self):
        item = case('F01')
        for i, word in enumerate(item['word_segments']):
            word.update(start=i * .2, end=i * .2 + .15)
        self.assertEqual(run(item)['filler_count'], 0)

    def test_existing_pause_threshold_is_required_on_both_sides(self):
        for gap, expected in [(.349, 0), (.35, 1)]:
            item = case('F01')
            words = item['word_segments']
            # Candidate ano is word 2; shift it to change only the left gap.
            words[2]['start'] = words[1]['end'] + gap
            self.assertEqual(run(item)['filler_count'], expected)

    def test_coverage_mismatch_disables_complementary_confirmation(self):
        item = case('F01')
        item['transcript'] = 'missing ' + item['transcript']
        result = run(item)
        self.assertEqual(result['coverage']['status'], 'mismatch')
        self.assertEqual(result['filler_count'], 0)

    def test_model_confirmation_and_vocabulary_do_not_duplicate_occurrence(self):
        item = case('F01'); marker = item['annotations'][0]
        prediction = dict(entity_group='FILLER', start=marker['start'], end=marker['end'], score=.95)
        result = run(item, [prediction, prediction])
        self.assertEqual(result['filler_count'], 1)
        self.assertEqual(result['filler_words'][0]['confidence'], .95)
        self.assertFalse(any(c['word'] == 'ano' for c in result['filler_candidates']))

    def test_uncertain_model_output_is_never_promoted_by_complementary_rule(self):
        item = case('F01'); marker = item['annotations'][0]
        for label, score in [('FILLER', .59), ('UNCERTAIN', .99)]:
            result = run(item, [dict(entity_group=label, start=marker['start'], end=marker['end'], score=score)])
            self.assertEqual(result['filler_count'], 0)
            self.assertEqual(sum(c['word'] == 'ano,' for c in result['filler_candidates']), 1)

    def test_confirmed_model_evidence_precedes_overlapping_uncertain_duplicate(self):
        item = case('F01'); marker = item['annotations'][0]
        result = run(item, [dict(entity_group='UNCERTAIN', start=marker['start'], end=marker['end'], score=.99),
                            dict(entity_group='FILLER', start=marker['start'], end=marker['end'], score=.95)])
        self.assertEqual(result['filler_count'], 1)
        self.assertFalse(any(c['word'] == 'ano,' for c in result['filler_candidates']))

    def test_explicit_mentions_veto_even_injected_model_filler(self):
        for identifier, target in [('E09', 'uh'), ('F10', 'um'), ('T09', 'ano')]:
            item = case(identifier); start = item['transcript'].index("'" + target) + 1
            result = run(item, [dict(entity_group='FILLER', start=start, end=start+len(target), score=.99)])
            self.assertEqual(result['filler_count'], 0)
            self.assertTrue(any('metalinguistic' in c['reason'] for c in result['filler_candidates']))

    def test_recovered_vocabulary_includes_previous_omissions(self):
        for term in ['literally', 'honestly', 'mhm', 'at the end of the day', 'what i mean is',
                     'siguro', 'lang', 'halimbawa', 'ewan ko ba', 'naku naman', 'saka tapos', 'as in']:
            self.assertEqual([m.group() for m in VOCABULARY.finditer(term.upper())], [term.upper()])
        self.assertGreater(len(ENGLISH), 50)
        self.assertGreater(len(FILIPINO), 50)

    def test_boundaries_quotes_case_and_phrase_punctuation(self):
        self.assertEqual(list(VOCABULARY.finditer("human somehow anong um's i-um o'ano")), [])
        self.assertEqual([m.group() for m in VOCABULARY.finditer("'UM' AS   IN")], ['UM', 'AS   IN'])
        self.assertEqual(list(VOCABULARY.finditer('you, know')), [])
        for punctuation in [',', ';', '\u2014']:
            item = case('F01'); item['transcript'] = item['transcript'].replace(',', punctuation)
            for word in item['word_segments']:
                word['text'] = word['text'].replace(',', punctuation)
            self.assertEqual(run(item)['filler_count'], 1)

    def test_language_names_do_not_gate_mixed_vocabulary(self):
        for language in ['English', 'Filipino', 'Taglish', 'tl', 'fil', 'en']:
            item = case('T01'); item['language'] = language
            self.assertEqual(run(item)['filler_count'], 1)

    def test_metalinguistic_guard_does_not_leak_from_previous_sentence(self):
        text = 'A term is defined. Um, I can speak.'
        words = [dict(text=w, start=i * .3, end=i * .3 + .2) for i,w in enumerate(text.split())]
        start = text.index('Um')
        result = analyze_fillers(words, full_transcript=text, classifier=lambda _: [
            dict(entity_group='FILLER', start=start, end=start+2, score=.95)])
        self.assertEqual(result['filler_count'], 1)

    def test_curly_quoted_hesitation_is_review_only(self):
        item = case('E01')
        item['transcript'] = item['transcript'].replace('Um,', '\u2018Um\u2019,')
        item['word_segments'][0]['text'] = '\u2018Um\u2019,'
        start = item['transcript'].index('Um')
        result = run(item, [dict(entity_group='FILLER', start=start, end=start+2, score=.99)])
        self.assertEqual(result['filler_count'], 0)
        self.assertTrue(any('Um' in c['word'] and 'metalinguistic' in c['reason']
                            for c in result['filler_candidates']))

    def test_score_formula_and_contract_unchanged(self):
        result = run(case('F01'))
        self.assertTrue(result['analysis_available'])
        self.assertEqual(result['threshold'], .60)
        self.assertEqual(result['filler_rate'], round(result['filler_count'] / result['total_words'] * 100, 1))
        self.assertEqual(result['filler_score'], round(max(0, 100 - result['filler_rate'] * 4)))
        self.assertEqual(result['candidate_count'], len(result['filler_candidates']))
        self.assertEqual(result['count_unit'], 'grouped_spans')

    def test_classifier_failure_fallback_is_preserved(self):
        item = case('F01')
        def fail(_):
            raise ClassifierUnavailable('private-path')
        result = analyze_fillers(item['word_segments'], classifier=fail, full_transcript=item['transcript'])
        self.assertFalse(result['analysis_available'])
        self.assertIsNone(result['filler_score'])
        self.assertEqual(result['filler_count'], 0)


class MetricAccountingTests(unittest.TestCase):
    def measure(self, item, spans):
        from testing_script.evaluate_filler_hybrid import add_expression_metrics
        evidence = dict(cases=[dict(id=item['id'], language=item['language'], split=item['split'],
            results=dict(hybrid=dict(confirmed=spans)))],
            summaries=[dict(detector='hybrid', group='all', confirmed_spans=len(spans))])
        add_expression_metrics(evidence, dict(cases=[item]))
        return evidence['summaries'][0]

    def test_exact_expression_match(self):
        item = case('F01'); word = item['word_segments'][2]
        result = self.measure(item, [dict(start=word['start'],end=word['end'])])
        self.assertEqual((result['expression_tp'],result['expression_fp'],result['expression_fn']), (1,0,0))

    def test_duplicate_predictions_are_not_multiple_true_positives(self):
        item = case('F01'); word = item['word_segments'][2]
        span = dict(start=word['start'],end=word['end'])
        result = self.measure(item, [span,span])
        self.assertEqual((result['expression_tp'],result['expression_fp'],result['expression_fn']), (1,1,0))

    def test_partial_phrase_is_not_an_exact_expression_match(self):
        item = case('E03'); word = item['word_segments'][2]  # 'you' alone, not 'you know'
        result = self.measure(item, [dict(start=word['start'],end=word['end'])])
        self.assertEqual((result['expression_tp'],result['expression_fp'],result['expression_fn']), (0,1,1))


if __name__ == '__main__':
    unittest.main()
