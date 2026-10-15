"""Inside-container probe with generated speech; emit status only, no transcripts."""
import json
import math
import os
from pathlib import Path
import urllib.error
import urllib.request
import uuid

base = 'http://127.0.0.1:' + (os.getenv('PORT') or '8000')
health = json.load(urllib.request.urlopen(base + '/health', timeout=5))
assert health['status'] == 'ready'
assert health['speech_model']['available'] and health['filler_classifier']['available']
try:
    urllib.request.urlopen(base + '/metrics', timeout=5)
    raise AssertionError('Unauthenticated metrics accepted')
except urllib.error.HTTPError as error:
    assert error.code == 401


def analyze(reference=False):
    boundary = 'ispeak-' + uuid.uuid4().hex
    speech = Path('/verification/generated-speech.wav').read_bytes()
    chunks = []
    for name in ['file'] + (['reference_audio'] if reference else []):
        chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="generated-speech.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode(), speech, b'\r\n']
    chunks += [f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\nEnglish\r\n--{boundary}--\r\n'.encode()]
    request = urllib.request.Request(base + '/transcribe', data=b''.join(chunks), headers={
        'Content-Type': 'multipart/form-data; boundary=' + boundary,
        'X-iSpeak-Service-Token': os.environ['ISPEAK_SERVICE_TOKEN'],
    })
    with urllib.request.urlopen(request, timeout=300) as response:
        assert response.status == 200
        result = json.load(response)
    assert result['analysis_valid'] is True and result['transcription'].strip()
    assert result['duration_seconds'] >= 1 and result['pacing']['wpm'] > 0
    assert all(math.isfinite(result['scores'][key]) for key in ['overall', 'pacing', 'clarity', 'energy'])
    assert result['fillers']['analysis_available'] is True
    if reference:
        assert result['speaking_rate']['reference_wpm'] > 0
        comparison = result['articulation']['clear_enunciation']['reference_comparison']
        assert comparison['available'] is True and math.isfinite(comparison['score'])
    return result


normal = analyze()
reference = analyze(reference=True)
print(json.dumps({'health': 'PASS', 'speechAndFillerReady': 'PASS', 'serviceAuthentication': 'PASS',
                  'realSyntheticInference': 'PASS', 'librosaScores': 'PASS', 'referenceInference': 'PASS',
                  'wordCount': len(normal['word_timestamps']), 'durationSeconds': normal['duration_seconds']}))
