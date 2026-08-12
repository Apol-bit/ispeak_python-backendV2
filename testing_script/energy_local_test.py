import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import librosa
from model import load_model
from audio_analysis_processing_files.vocal_variety.signal_intensity import analyze_signal_intensity

audio_path = r"C:\Users\ALLAN\AppData\Local\Temp\myapp_uploads\Recording.m4a"
model = load_model()

def test_energy_analysis():
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    energy_stats = analyze_signal_intensity(y, sr)
    from pprint import pprint
    print("=== Signal Intensity Stats ===")
    pprint(energy_stats)

test_energy_analysis()
