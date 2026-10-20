"""Deterministic scoring evidence tests without model/audio imports."""
import ast
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assessment_details import assessment_details


def fixture():
    return dict(scores=dict(overall=96.8, pacing=100, clarity=92, energy=100,
                            articulation=87, filler_words=100),
                pacing=dict(wpm=127), speaking_rate={},
                articulation=dict(accurate_pronunciation=dict(available=True, pronunciation_accuracy_score=90),
                                  clear_enunciation=dict(enunciation_score=82.5, evidence_level='timing_and_audio')),
                vocal_variety=dict(signal_intensity=dict(loudness_status='Normal volume'),
                                   frequency_pitch=dict(analysis_available=True, pitch_variety_score=90),
                                   temporal_pauses=dict(pause_score=100)),
                fillers=dict(analysis_available=True, candidate_count=0, count=0, coverage=dict(status='matched')))


class AssessmentDetailsTests(unittest.TestCase):
    def test_complete_and_actual_overall_formula(self):
        result = fixture()
        detail = assessment_details(result)
        self.assertEqual(detail['completeness'], 'complete')
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'whisper_service.py').read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_compute_overall_score')
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'score', 'exec'), namespace)
        self.assertEqual(namespace['_compute_overall_score'](100, 92, 100), 96.8)

    def test_unavailable_components_coverage_and_uncertainty(self):
        for missing in ['pronunciation', 'fillers', 'pitch', 'coverage', 'uncertain']:
            r = fixture()
            if missing == 'pronunciation':
                r['articulation']['accurate_pronunciation'] = dict(available=False, pronunciation_accuracy_score=None)
            elif missing == 'fillers':
                r['fillers']['analysis_available'] = False
                r['scores']['filler_words'] = None
            elif missing == 'pitch':
                r['vocal_variety']['frequency_pitch'] = dict(analysis_available=False)
            elif missing == 'coverage':
                r['fillers']['coverage']['status'] = 'mismatch'
            else:
                r['fillers']['candidate_count'] = 2
            d = assessment_details(r)
            self.assertEqual(d['completeness'], 'partial')
            if missing == 'pronunciation':
                self.assertIsNone(d['components']['pronunciation'])
                self.assertEqual(d['weights']['articulation'], dict(pronunciation=0, enunciation=1))
            if missing == 'fillers':
                self.assertIsNone(d['scores']['filler_words'])
                self.assertEqual(d['weights']['clarity'], dict(articulation=1, fillers=0))

    def test_reference_summary_uses_final_adjusted_values(self):
        r = fixture()
        r['speaking_rate']['reference_wpm'] = 130
        r['_internal'] = dict(energy_score=90, energy_rule_base=100)
        r['articulation']['clear_enunciation']['reference_comparison'] = dict(applied=True, base_enunciation_score=85, score=90)
        d = assessment_details(r)
        self.assertEqual(d['basis']['referenceEnergyAdjustment'], 10)
        self.assertTrue(d['basis']['referenceEnunciationApplied'])
        self.assertEqual(d['scores'], r['scores'])


if __name__ == '__main__':
    if '--fixture' in sys.argv:
        r = fixture()
        r['assessment_details'] = assessment_details(r)
        print(json.dumps(r))
    else:
        unittest.main()
