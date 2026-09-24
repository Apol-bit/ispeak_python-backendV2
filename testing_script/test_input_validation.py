import unittest
from unittest.mock import Mock, patch
import numpy as np
from audio_validation import SpeechValidationError, validate_audio, validate_transcript
from whisper_service import _run_analysis

class InputValidationTests(unittest.TestCase):
    sr = 16000
    def voice(self, seconds=2):
        t = np.arange(int(self.sr*seconds))/self.sr
        return .1*np.sin(2*np.pi*220*t) + .04*np.sin(2*np.pi*440*t)
    def test_invalid_audio_never_calls_whisper(self):
        samples = [np.zeros(self.sr*3), self.voice(.25),
                   np.sin(2*np.pi*220*np.arange(self.sr*3)/self.sr)*.1,
                   np.full(self.sr*2,.1), np.full(self.sr*2,np.nan)]
        for audio in samples:
            model = Mock()
            with self.subTest(samples=len(audio)), self.assertRaises(SpeechValidationError):
                _run_analysis('unused.wav',audio,self.sr,model)
            model.transcribe.assert_not_called()
    def test_hallucinated_transcript_never_reaches_scoring(self):
        model = Mock()
        model.transcribe.return_value = {
            'text': 'yeah '*30,
            'segments': [{'words': [{'word': 'yeah', 'start': .2, 'end': .2} for _ in range(30)]}],
        }
        with patch('whisper_service.analyze_signal_intensity') as acoustic:
            with self.assertRaises(SpeechValidationError):
                _run_analysis('unused.wav', self.voice(3), self.sr, model)
            acoustic.assert_not_called()

    def test_quiet_harmonic_short_audio_is_not_rejected(self):
        self.assertEqual(validate_audio(self.voice(1.2)*.02,self.sr),1.2)
    def test_three_word_short_speech_is_accepted(self):
        words=[{'start':i*.4,'end':(i+1)*.4,'text':word} for i,word in enumerate(['Good','morning','everyone'])]
        validate_transcript('Good morning everyone',words,1.2)
    def test_runaway_repetition_rejected(self):
        words=[{'start':i*.05,'end':i*.05+.04} for i in range(30)]
        with self.assertRaises(SpeechValidationError):validate_transcript('yeah '*30,words,3)
        with self.assertRaises(SpeechValidationError):validate_transcript('thank you '*10,words,3)
    def test_degenerate_timestamps_rejected(self):
        words=[{'start':.2,'end':.2} for _ in range(5)]
        with self.assertRaises(SpeechValidationError):validate_transcript('this is a normal sentence',words,3)
    def test_ordinary_hesitation_is_not_rejected(self):
        text='I I think we should practice speaking every single day'
        words=[{'start':i*.3,'end':i*.3+.2} for i in range(len(text.split()))]
        validate_transcript(text,words,4)

if __name__=='__main__':unittest.main()
