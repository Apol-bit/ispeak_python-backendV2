# Filler analysis correction — 2026-10-26

Local source correction verified; the exact production session remains unverified.
No model, Whisper, scoring formula, threshold, transport, security, production
record, image, or APK was changed.

## Evidence and cause

The active path is `whisper_service.py:324` → `analyze_fillers` →
`whisper_service.py:391` (`fillers` serialization) → Node
`controllers/sessionController.js:153` and `services/fillerAnalysis.js:6` →
`fillerWordCount`, `fillerAnalysisAvailable`, `fillerAnalysis` → Flutter
`lib/pages/result_page.dart`.

Python correctly renames `filler_count/rate/score/words` to the API's
`count/rate/score/words`. Node validates, bounds, maps and persists these fields.
The Flutter subtitle reads that count. There is no demonstrated field-name or
language-normalization defect in this path, and no evidence that a positive
count is dropped by Node. English/Filipino/Taglish language names do not gate
the detector.

Two concrete local defects were reproduced:

1. `audio_analysis_processing_files/filler_words.py:156`: `_classification_item`
   previously accepted any overlapping token prediction, expanding subword
   predictions into whole timestamped words. A real local model run on the
   synthetic sentence `Ah, ano, parang hindi ko alam. So, pumunta kami.`
   returned a FILLER prediction for characters 36–38 (`um` inside `pumunta`),
   confidence 0.99641526. Previously this penalized `pumunta`: count 1,
   rate 11.1%, filler score 56. The new lexical-boundary guard rejects it:
   count 0, filler score 100.
2. Flutter had `lib/models/filler_feedback.dart`, but no runtime caller.
   `lib/pages/result_page.dart:336` now includes its output in Clarity feedback.
   Previously the result showed a concise count without this formatter's
   confirmed-span details, candidate notice, or coverage explanation.

The active V2 detector previously had no vocabulary recognition. It accepted
only the local model's FILLER/UNCERTAIN predictions; CONTEXT_WORD/O predictions
were excluded. The local model classified `Ah, ano` as CONTEXT_WORD in the
synthetic example. Transcript presence therefore did not imply a filler count.
This is evidence of the local classifier's limitations, not proof of what
happened in the reported production assessment.

The old `ispeak_python-backend/audio_analysis_processing_files/clarity_analysis_module/voice_fillerwords_detection.py`
contains English/Filipino lists, phrases and repeat rules. It is not imported
by the active V2 path. Its always-filler list includes ordinary Filipino words
such as `ano`, `parang`, `kasi` and `yung`; enabling it wholesale would cause
unsupported penalties. Taglish requires recognizing both languages together.

## Minimal changes and policy

- Added `filler_vocabulary.py`: both languages, case-insensitive matching,
  longest multiword phrase first, flexible whitespace, lexical boundaries,
  no phrase joining across punctuation. Vocabulary matches are review
  candidates only, with null confidence; they never increase confirmed count.
- Added whole-lexical-span validation before accepting model predictions.
- Retained the 0.60 confidence threshold, grouped-span count, rate formula
  `round(count / total_words * 100, 1)`, filler score
  `round(max(0, 100 - rate * 4))`, all clarity/overall weights and fallback rules.
- Reused existing candidate API fields; Node runtime needs no change.
- Connected Flutter feedback and added clearly labeled review words, which
  may have grammatical uses. Existing stored scores are never recalculated.

More recognized candidates can mark a new assessment partial through the
existing completeness policy. Candidate recognition itself does not change
its numeric scores. Transcript analysis cannot recover sounds omitted by
Whisper, and no missing audio fillers are inferred.

## Recognition versus classification: real local model checks

These are synthetic text checks with the existing local filler model, not
Whisper/audio evaluation and not a replay of the production assessment.

| Text/example | Confirmed spans after fix | Additional review matches |
| --- | --- | --- |
| English: `Um, I like apples. Uh, so we left.` | Um, Uh (2) | like, so |
| Filipino synthetic sentence above | 0 | Ah, ano, parang, So |
| Taglish: `Um, you know, ano, I like this kasi parang bahay.` | Um (1) | you know, ano, like, kasi, parang |
| `Ano ang sagot? Parang bahay ang gusali.` | 0; filler score 100 | Ano, Parang |
| `I like apples so I bought some.` | 0; filler score 100 | like, so |

These checks establish conservative behavior, not general multilingual model
accuracy. They do not establish that all Filipino hesitations will be confirmed.

## Saved-session limitation

The requested assessment is described as October 26, 2026, 09:38 Asia/Manila,
Ordinary Practice, approximately 104 seconds, score 88.9, ID prefix `6adeaf14`.
The configured database could not be reached in either the sandbox or the
approved unrestricted retry (`MongoServerSelectionError`). No account data was
printed, and no production writes were attempted. No screenshot was attached
to this conversation. The full transcript and stored filler payload are thus
unavailable; the session-specific cause and expected count remain unverified.

Tests explicitly distinguish the user-reported terms `ano, parang, ah, so`
from a full-session fixture. Existing saved Filipino and Taglish UAT transcript
fixtures in `test_word_alignment.py` also continue passing; they are different,
shorter assessments. Do not describe either as the requested 104-second replay.

## Verification

- Python: 34 tests passed:
  `.venv/Scripts/python.exe -B -m unittest testing_script.test_filler_regression testing_script.test_filler_evidence testing_script.test_analysis_components testing_script.test_word_alignment testing_script.test_assessment_details`
- Node: 9 tests passed:
  `node --test test/fillerAnalysis.test.js test/analysisResponse.test.js test/assessmentDetails.test.js`
  Includes actual Python output through Node mapping and Mongoose serialization,
  with multilingual review-only candidates and null confidence.
- Flutter: 8 tests passed:
  `flutter test --no-pub test/filler_feedback_test.dart test/assessment_feedback_test.dart`
- Real local classifier checks above passed without loading or changing Whisper.

## Release implication

Python and Flutter source changes are needed for the complete detection-evidence
and display correction. Node needs only the added regression test, no runtime
deployment. The follow-up release authorization on October 26 permits finalizing
with the existing local Filipino, English and Taglish fixtures; inaccessible
production records do not block release preparation. The 104-second production
session remains unverified. Preserve the working Whisper-small-base image model
artifacts/settings and rollback revision `ispeak-python-00005-f2s`. Build and
deployment outcomes belong in the separate release report; this document records
the original local correction and its evidence.
