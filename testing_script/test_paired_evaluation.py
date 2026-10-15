import json
from pathlib import Path
import tempfile
import unittest
from testing_script.evaluate_v5_v6 import edits, normalize, read_dataset, digest


class EvaluationTests(unittest.TestCase):
    def test_edit_counts(self):
        self.assertEqual(edits('a b c'.split(), 'a x c d'.split()), 2)
        self.assertEqual(edits('abc', 'ac'), 1)
        self.assertEqual(edits([], ['x']), 1)

    def test_normalization_preserves_language(self):
        self.assertEqual(normalize('  Kumustá, WORLD! '), 'kumustá world')

    def test_approval_hash_and_overlap_gates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); audio = root/'sample.wav'; audio.write_bytes(b'fixture')
            samples = [{'id':str(i), 'language':language, 'file':'sample.wav','sha256':digest(audio),
                        'speaker_id':'speaker','approved':True,'synthetic':False,'consent_record':'reviewed',
                        'reference':'verified words','reference_verified':True,'reference_reviewer':'reviewer'}
                       for i,language in enumerate(('English','Filipino','Taglish'))]
            data = {'schema':1,'approval_record':'reviewed','samples':samples}
            file = root/'dataset.json'
            def save(): file.write_text(json.dumps(data))
            save(); self.assertIn('UNKNOWN', read_dataset(file)[1]['v5'])
            samples[0]['approved'] = False; save()
            with self.assertRaises(ValueError): read_dataset(file)
            samples[0]['approved'] = True; samples[0]['sha256'] = 'wrong'; save()
            with self.assertRaises(ValueError): read_dataset(file)
            samples[0]['sha256'] = digest(audio)
            training = root/'training.json'
            training.write_text(json.dumps({'audio_sha256':[digest(audio)],'speaker_ids':[],'coverage':'complete'}))
            data['training_records'] = {'v5':{'file':'training.json','sha256':digest(training)}}; save()
            with self.assertRaisesRegex(ValueError,'overlap'): read_dataset(file)


if __name__ == '__main__': unittest.main()
