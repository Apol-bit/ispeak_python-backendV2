"""Compact evidence for existing scoring; this module does not calculate scores."""
def assessment_details(result):
    scores = result.get('scores', {})
    articulation = result.get('articulation', {})
    pronunciation = articulation.get('accurate_pronunciation', {})
    enunciation = articulation.get('clear_enunciation', {})
    vocal = result.get('vocal_variety', {})
    fillers = result.get('fillers', {})
    coverage = (fillers.get('coverage') or {}).get('status', 'unknown')
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    available = {
        'pacing': numeric(result.get('pacing', {}).get('wpm')) and result['pacing']['wpm'] > 0,
        'pronunciation': pronunciation.get('available') is True and numeric(pronunciation.get('pronunciation_accuracy_score')),
        'enunciation': enunciation.get('evidence_level') in ('timing_only', 'timing_and_audio'),
        'audibility': enunciation.get('evidence_level') == 'timing_and_audio',
        'intensity': vocal.get('signal_intensity', {}).get('loudness_status') in (
            'Normal volume', 'Too quiet (whispering)', 'Too loud (shouting)', 'Silence'),
        'pauses': numeric(vocal.get('temporal_pauses', {}).get('pause_score')),
        'pitch': vocal.get('frequency_pitch', {}).get('analysis_available') is True,
        'fillers': fillers.get('analysis_available') is True,
    }
    filler_used = numeric(scores.get('filler_words'))
    accuracy_used = numeric(pronunciation.get('pronunciation_accuracy_score'))
    reference = result.get('speaking_rate', {}).get('reference_wpm') is not None
    reference_pacing = reference and result['speaking_rate']['reference_wpm'] > 0 and available['pacing']
    comparison = enunciation.get('reference_comparison', {})
    reference_applied = comparison.get('applied') is True
    return {
        'version': 1,
        'completeness': 'complete' if all(available.values()) and coverage == 'matched' and not (reference and not reference_pacing)
            and not fillers.get('candidate_count', 0) else 'partial',
        'availability': available,
        'coverage': coverage if coverage in ('matched', 'mismatch') else 'unknown',
        'scores': {key: scores.get(key) for key in ('overall', 'pacing', 'clarity', 'energy', 'articulation', 'filler_words')},
        'components': {
            'pronunciation': pronunciation.get('pronunciation_accuracy_score') if accuracy_used else None,
            'enunciation': enunciation.get('enunciation_score') if available['enunciation'] else None,
            'pitchVariety': vocal.get('frequency_pitch', {}).get('pitch_variety_score') if available['pitch'] else None,
            'pauses': vocal.get('temporal_pauses', {}).get('pause_score') if available['pauses'] else None,
            'energyRuleBase': result.get('_internal', {}).get('energy_rule_base'),
            'baseEnunciation': comparison.get('base_enunciation_score') if reference_applied else None,
            'referenceEnunciation': comparison.get('score') if reference_applied else None,
        },
        'weights': {
            'overall': {'pacing': .35, 'clarity': .40, 'energy': .25},
            'clarity': {'articulation': .60 if filler_used else 1.0, 'fillers': .40 if filler_used else 0.0},
            'articulation': {'pronunciation': .60 if accuracy_used else 0.0, 'enunciation': .40 if accuracy_used else 1.0},
            'energy': {'intensityPitchRules': .80 if available['pauses'] else 1.0, 'pauses': .20 if available['pauses'] else 0.0},
            'referenceEnunciation': {'user': .40 if reference_applied else 1.0, 'reference': .60 if reference_applied else 0.0},
        },
        'fallback': {'pronunciation': not accuracy_used, 'fillers': not filler_used,
                     'pitch': not available['pitch'], 'pacing': not available['pacing'],
                     'referencePacing': reference and not reference_pacing},
        'basis': {
            'pacing': 'reference' if reference_pacing else 'target_range',
            'energy': 'intensity_pitch_rules_then_pauses',
            'enunciationEvidence': enunciation.get('evidence_level', 'unavailable'),
            'referenceEnunciationApplied': enunciation.get('reference_comparison', {}).get('applied') is True,
            'referenceEnergyAdjustment': round(scores.get('energy', 0) - result.get('_internal', {}).get('energy_score', scores.get('energy', 0)), 1) if reference else 0,
        },
    }
