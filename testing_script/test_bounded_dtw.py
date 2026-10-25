"""Exact DTW equivalence and allocation bounds; no models or real recordings."""
import gc
import tracemalloc
import unittest
from unittest.mock import patch

import librosa
import numpy as np
from scipy.spatial.distance import cdist
from audio_analysis_processing_files.articulation import bounded_dtw


class BoundedDtwTests(unittest.TestCase):
    def compare(self, reference, user):
        matrix, path = librosa.sequence.dtw(X=reference, Y=user, metric='cosine')
        cost, length = bounded_dtw.cosine_dtw_statistics(reference, user)
        self.assertEqual(cost, float(matrix[-1, -1]))
        self.assertEqual(length, len(path))
        self.assertEqual(round(float(100 * np.exp(-1.5 * cost / length)), 1),
                         round(float(100 * np.exp(-1.5 * matrix[-1, -1] / len(path))), 1))

    def test_random_rectangular_float32_and_float64(self):
        rng = np.random.default_rng(104)
        for dtype in [np.float32, np.float64]:
            for rows, columns in [(1, 1), (1, 19), (19, 1), (17, 31), (31, 17), (80, 80)]:
                with self.subTest(dtype=dtype, shape=(rows, columns)):
                    self.compare(rng.normal(size=(24, rows)).astype(dtype),
                                 rng.normal(size=(24, columns)).astype(dtype))

    def test_ties_and_repeated_features(self):
        for rows, columns in [(1, 8), (8, 1), (8, 13), (13, 8), (13, 13)]:
            self.compare(np.ones((24, rows)), np.ones((24, columns)))

    def test_invalid_inputs_and_zero_norm_fail(self):
        for reference, user in [(np.empty((24, 0)), np.ones((24, 3))),
                                (np.ones((2, 3)), np.ones((3, 3))),
                                (np.zeros((24, 2)), np.ones((24, 3)))]:
            with self.assertRaises(ValueError):
                bounded_dtw.cosine_dtw_statistics(reference, user)

    def test_five_minute_frame_dimension_never_allocates_pairwise_matrix(self):
        # 300 seconds at 16 kHz / hop 256, centered features: 18,751 frames.
        # A short other axis keeps this a cheap allocation test, not a full workload.
        calls = []
        def measured(x, y, **kwargs):
            result = cdist(x, y, **kwargs)
            calls.append(result.shape)
            return result
        for seconds in [30, 60, 120, 180, 240, 300]:
            frames = 1 + seconds * 16000 // 256
            calls.clear()
            with patch.object(bounded_dtw, 'cdist', side_effect=measured):
                bounded_dtw.cosine_dtw_statistics(np.ones((24, 2)), np.ones((24, frames)))
            self.assertEqual(calls, [(1, frames), (1, frames)])

    def test_reference_score_output_matches_dense_implementation(self):
        from audio_analysis_processing_files.articulation.clear_enunciation import compare_reference_enunciation
        rng = np.random.default_rng(42)
        reference = rng.normal(size=16000).astype(np.float32)
        user = rng.normal(size=18000).astype(np.float32)
        def dense_statistics(x, y):
            matrix, path = librosa.sequence.dtw(X=x, Y=y, metric='cosine')
            return float(matrix[-1, -1]), len(path)
        with patch.object(bounded_dtw, 'cosine_dtw_statistics', side_effect=dense_statistics):
            expected = compare_reference_enunciation(user, reference, 16000)
        actual = compare_reference_enunciation(user, reference, 16000)
        self.assertTrue(actual['available'])
        self.assertEqual(actual, expected)

    def test_sequential_requests_do_not_retain_workspace(self):
        x = np.ones((24, 600))
        bounded_dtw.cosine_dtw_statistics(x, x)  # Warm compilation before measuring.
        tracemalloc.start()
        try:
            baselines = []
            for _ in range(12):
                bounded_dtw.cosine_dtw_statistics(x, x)
                gc.collect()
                baselines.append(tracemalloc.get_traced_memory()[0])
            self.assertLess(max(baselines) - min(baselines), 128 * 1024)
            self.assertLess(tracemalloc.get_traced_memory()[1], 1024 * 1024)
        finally:
            tracemalloc.stop()


if __name__ == '__main__':
    unittest.main()
