"""Model-free checks for the seven speech-delivery components."""

import unittest

import numpy as np

from audio_analysis_processing_files.articulation import (
    analyze_clear_enunciation,
    analyze_pronunciation_accuracy,
    compare_reference_enunciation,
)
from audio_analysis_processing_files.filler_words import analyze_fillers
from audio_analysis_processing_files.speaking_rate import analyze_speaking_rate
from audio_analysis_processing_files.vocal_variety import (
    analyze_frequency_pitch,
    analyze_signal_intensity,
    analyze_temporal_pauses,
)


class AnalysisComponentTests(unittest.TestCase):
    def setUp(self):
        self.sr = 16000
        self.time = np.arange(self.sr * 2) / self.sr

    def test_signal_intensity_distinguishes_quiet_and_loud(self):
        quiet = 0.02 * np.sin(2 * np.pi * 180 * self.time)
        loud = 0.60 * np.sin(2 * np.pi * 180 * self.time)
        self.assertEqual(
            analyze_signal_intensity(quiet, self.sr)["loudness_status"],
            "Too quiet (whispering)",
        )
        self.assertEqual(
            analyze_signal_intensity(loud, self.sr)["loudness_status"],
            "Too loud (shouting)",
        )

    def test_frequency_pitch_detects_monotone_signal(self):
        monotone = np.sin(2 * np.pi * 180 * self.time)
        result = analyze_frequency_pitch(monotone, self.sr)
        self.assertTrue(result["analysis_available"])
        self.assertTrue(result["is_monotone"])

    def test_pitch_variety_is_consistent_across_voice_ranges(self):
        def modulated_voice(base_frequency):
            frequency = base_frequency * (
                1 + 0.30 * np.sin(2 * np.pi * 1.5 * self.time)
            )
            phase = np.cumsum(2 * np.pi * frequency / self.sr)
            return np.sin(phase).astype(np.float32)

        low_voice = analyze_frequency_pitch(modulated_voice(100), self.sr)
        high_voice = analyze_frequency_pitch(modulated_voice(300), self.sr)
        self.assertFalse(low_voice["is_monotone"])
        self.assertFalse(high_voice["is_monotone"])
        self.assertAlmostEqual(
            low_voice["pitch_range_semitones"],
            high_voice["pitch_range_semitones"],
            delta=0.75,
        )

    def test_noise_does_not_get_a_monotone_label(self):
        noise = np.random.default_rng(7).normal(
            0, 0.02, self.time.size
        ).astype(np.float32)
        result = analyze_frequency_pitch(noise, self.sr)
        self.assertFalse(result["analysis_available"])
        self.assertIsNone(result["is_monotone"])

    def test_temporal_pauses_uses_punctuation_as_intent_heuristic(self):
        words = [
            {"text": "finished.", "start": 0.0, "end": 0.5},
            {"text": "Next", "start": 1.0, "end": 1.3},
            {"text": "word", "start": 2.0, "end": 2.3},
        ]
        result = analyze_temporal_pauses(words)
        self.assertEqual(result["likely_intentional_count"], 1)
        self.assertEqual(result["possibly_unintentional_count"], 1)

    def test_pronunciation_does_not_invent_confidence(self):
        result = analyze_pronunciation_accuracy([
            {"text": "hello", "start": 0.0, "end": 0.3, "confidence": None}
        ])
        self.assertFalse(result["available"])
        self.assertIsNone(result["pronunciation_accuracy_score"])

    def test_clear_enunciation_flags_rushed_word(self):
        result = analyze_clear_enunciation([
            {"text": "hello", "start": 0.0, "end": 0.01}
        ])
        self.assertLess(result["enunciation_score"], 50)

    def test_enunciation_accounts_for_syllable_length(self):
        result = analyze_clear_enunciation([
            {"text": "I", "start": 0.0, "end": 0.2},
            {"text": "communicate", "start": 0.2, "end": 1.0},
        ])
        self.assertGreaterEqual(result["enunciation_score"], 85)

    def test_enunciation_uses_relative_word_audibility(self):
        audio = np.concatenate((
            0.10 * np.sin(2 * np.pi * 180 * np.arange(6400) / self.sr),
            0.001 * np.sin(2 * np.pi * 180 * np.arange(6400) / self.sr),
        )).astype(np.float32)
        result = analyze_clear_enunciation(
            [
                {"text": "first", "start": 0.0, "end": 0.4},
                {"text": "second", "start": 0.4, "end": 0.8},
            ],
            y=audio,
            sr=self.sr,
        )
        self.assertEqual(result["evidence_level"], "timing_and_audio")
        self.assertEqual(result["unclear_words"][0]["word"], "second")

    def test_reference_enunciation_identical_audio_scores_high(self):
        speech_like = (
            np.sin(2 * np.pi * 180 * self.time)
            * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * self.time))
        ).astype(np.float32)
        result = compare_reference_enunciation(speech_like, speech_like.copy(), self.sr)
        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["score"], 95)

    def test_speaking_rate_calculates_wpm(self):
        words = [
            {"text": "word", "start": i * 0.3, "end": i * 0.3 + 0.2}
            for i in range(15)
        ]
        self.assertEqual(analyze_speaking_rate(words, 6.0)["wpm"], 150.0)

    def test_local_model_distinguishes_context_from_filler(self):
        words = [
            {"text": "Actually", "start": 0.0, "end": 0.3},
            {"text": "this", "start": 0.3, "end": 0.5},
            {"text": "works", "start": 0.5, "end": 0.8},
            {"text": "um", "start": 0.8, "end": 1.0},
        ]
        def fake_local_model(_text):
            return [
                {
                    "entity_group": "CONTEXT_WORD",
                    "start": 0,
                    "end": 8,
                    "score": 0.99,
                },
                {
                    "entity_group": "FILLER",
                    "start": 20,
                    "end": 22,
                    "score": 0.98,
                },
            ]

        result = analyze_fillers(words, classifier=fake_local_model)
        self.assertTrue(result["analysis_available"])
        self.assertEqual(result["filler_count"], 1)
        self.assertEqual(result["filler_words"][0]["word"], "um")
        self.assertEqual(result["filler_candidates"], [])


if __name__ == "__main__":
    unittest.main()
