"""Model-free regression checks. These do not measure classifier accuracy."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('filler_evidence', Path(__file__).resolve().parents[1] / 'audio_analysis_processing_files/filler_words.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)


class FillerEvidenceTests(unittest.TestCase):
    def analyze(self, text, predictions, full=None):
        words = [dict(text=w, start=i, end=i + .5) for i, w in enumerate(text.split())]
        return f.analyze_fillers(words, classifier=lambda _: predictions,
                                 full_transcript=text if full is None else full)

    def test_zero_positive_uncertain_and_context_across_languages(self):
        for text in ['um I like apples', 'ah ano ang sagot', 'um ano this is fine']:
            positive = dict(entity_group='FILLER', start=0, end=2, score=.95)
            r = self.analyze(text, [positive])
            self.assertEqual(r['filler_count'], 1)
            self.assertEqual(r['filler_words'][0]['confidence'], .95)
            self.assertEqual(r['coverage']['status'], 'matched')
            r = self.analyze(text, [{**positive, 'score': .59}])
            if text.startswith('um'):
                # Clear hesitation forms do not depend on model confidence.
                self.assertEqual(r['filler_count'], 1)
                self.assertIsNone(r['filler_words'][0]['confidence'])
                continue
            self.assertEqual(r['filler_count'], 0)
            self.assertGreaterEqual(r['candidate_count'], 1)
            self.assertEqual(r['filler_candidates'][0]['confidence'], .59)
            self.assertEqual(r['filler_score'], 100)
        for word in ['like', 'ano']:
            r = self.analyze(word, [dict(entity_group='CONTEXT_WORD', start=0, end=len(word), score=.99)])
            self.assertEqual(r['filler_count'], 0)
        self.assertEqual(self.analyze('hello there', [])['filler_count'], 0)

    def test_grouped_and_alignment(self):
        r = self.analyze('um um hello', [dict(entity_group='FILLER', start=0, end=5, score=.99)])
        self.assertEqual(r['filler_count'], 1)
        self.assertEqual(r['filler_words'][0]['word'], 'um um')
        self.assertEqual(r['count_unit'], 'grouped_spans')
        self.assertEqual(self.analyze('hello there', [], 'um hello there')['coverage']['status'], 'mismatch')
        self.assertEqual(self.analyze('Hello there', [], 'hello, there!')['coverage']['status'], 'matched')

    def test_failures_are_structured_and_do_not_expose_exceptions(self):
        for error, code in [(f.ClassifierUnavailable('private/path'), 'classifier_unavailable'),
                            (RuntimeError('private/path'), 'classifier_inference_failed')]:
            def fail(_):
                raise error
            r = f.analyze_fillers([dict(text='um', start=0, end=.5)], classifier=fail)
            self.assertFalse(r['analysis_available'])
            self.assertIsNone(r['filler_score'])
            self.assertEqual(r['failure']['code'], code)
            self.assertNotIn('private/path', str(r))


if __name__ == '__main__':
    unittest.main()
