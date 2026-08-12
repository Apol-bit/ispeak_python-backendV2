# Metrics Compatibility Report

Audit completed: 2026-08-12

## 1. Compatibility Summary

| Component | Status | Result |
|---|---|---|
| Frontend | ✅ Fully Compatible | All current metrics upload, result, history, progress, retry, language, and target-display contracts align. |
| Admin Panel | ✅ Fully Compatible | Metrics statistics, logs, WPM display, processing status, resource language, and target inputs align. |
| Python-Backend V2 | ✅ Fully Compatible | The normal, silence, and reference-relative analysis contracts are stable and compatible with the gateway. |
| Node API and MongoDB support layer | ✅ Fully Compatible | Gateway mapping, validation, resource lookup, persistence, aggregation, and query response shapes align with both UIs. |

Architecture verified:

`Frontend/Admin → Node API (:5000/api) → Python V2 (:8000/transcribe) → Node/MongoDB → Frontend/Admin`

There is no route literally named `metrics`. Python V2 exposes speech metrics through `POST /transcribe`; the Flutter applications consume the flattened and persisted form through the Node session/admin routes.

## 2. Metrics Compatibility Audit

| Endpoint | Used by | Current contract | Status | Resolution |
|---|---|---|---|---|
| `GET :8000/health` | Operations only | Returns overall status, speech-model availability/error, and filler-classifier status. No request body, query, or auth. | ✅ | Already compatible; neither UI depends on it. |
| `POST :8000/transcribe` | Frontend indirectly through upload gateway | Multipart `file` required and `reference_audio` optional. WAV, MP3, M4A, FLAC, OGG, and AAC are accepted. Returns transcription, timestamps, score summary, pacing/speaking rate, pronunciation/articulation, vocal variety, and filler details. | ✅ | Normal and silence paths now expose matching top-level and nested metric keys. |
| `POST /api/upload-audio` | Frontend Practice, Script Practice, Timed Challenge | Multipart `audio`; valid `userId`; language in English/Filipino/Taglish; optional valid `challengeId` or `resourceId`. Returns a flat persisted session. | ✅ | IDs, language, and file types are validated. Script or Challenge reference audio is forwarded to Python. Filler availability is preserved. |
| `GET /api/sessions/:userId` | Frontend Dashboard | Valid ObjectId. Returns sessions newest-first with populated `challengeId`/`resourceId` objects. | ✅ | Invalid IDs return 400 and retry-routing objects are populated. |
| `GET /api/stats/:userId` | Frontend Dashboard, Progress, Profile | Valid ObjectId. Returns `{sessions, overallStats}`; sessions oldest-first. Aggregate fields: `totalSessions`, `avgScore`, `avgPace`, `avgClarity`, `avgEnergy`. | ✅ | Session resource objects and aggregate names match all consumers. |
| `GET /api/admin/stats` | Admin Dashboard | Returns integer-compatible `totalUsers`, `totalSessions`, and rounded `avgAppScore`. | ✅ | Already matched. |
| `GET /api/admin/recent-sessions` | Admin Dashboard | Returns at most 100 sessions newest-first; `userId` includes first name, last name, and email. | ✅ | WPM uses `wpmScore`; processing state uses persisted Pending/Completed/Failed status. |
| `GET /api/admin/ai-logs` | Admin Session Reviews | Alias of recent sessions with the same ordering, limit, populated user, metrics, transcript, feedback, duration, and status. | ✅ | Already matched. |
| `GET /api/resources` | Frontend and Admin, metrics-adjacent | Returns resources newest-first; optional `type` filter. Includes language, `targetMetric`, transcript, and reference-audio metadata. | ✅ | Blank challenge targets fall back to the standard 120–150 WPM display. |
| `POST /api/admin/resources` | Admin, metrics-adjacent | Multipart resource fields with optional `referenceAudio`; accepts the same six audio extensions. | ✅ | Recording resources allow English/Filipino/Taglish; `None` is limited to Guided Tasks. |
| `PUT /api/admin/resources/:id` | Admin, metrics-adjacent | Same field/file contract as create; replaces reference audio when supplied. | ✅ | Same alignment as create. |

## 3. API Contract Mapping and Differences

### Final field mapping

| Python V2 field | Persisted/UI field | Type | Consumer |
|---|---|---|---|
| `scores.pacing` | `paceScore` | Number, 0–100 | Frontend and Admin |
| `scores.clarity` | `clarityScore` | Number, 0–100 | Frontend and Admin |
| `scores.energy` | `energyScore` | Number, 0–100 | Frontend and Admin |
| `scores.overall` | `overallScore` | Number, 0–100 | Frontend and Admin |
| `pacing.wpm` | `wpmScore` | Number | Frontend result and Admin dashboard |
| `fillers.count` | `fillerWordCount` | Integer | Frontend result |
| `fillers.analysis_available` | `fillerAnalysisAvailable` | Boolean | Frontend result |
| `transcription` | `transcription` | String | Frontend model and Admin logs |
| `word_timestamps` | `wordTimestamps` | Array of `{word, start, end}` | Session persistence/teleprompter data |

### Resolved differences

- Silence/no-speech analysis omitted `speaking_rate` and complete filler metadata.
- The gateway discarded filler-analysis availability, making unavailable analysis look like zero fillers.
- Legacy sessions without availability metadata were assumed to have verified filler results; they are now treated conservatively as unavailable.
- Frontend pace feedback used 110–160 WPM while Python V2 uses 120–150 WPM.
- Challenge target examples/defaults used 120 or 120–140 WPM instead of 120–150 WPM.
- Admin WPM consumed the 0–100 `paceScore` rather than raw `wpmScore`.
- Admin inferred pending state from `overallScore == 0`, misclassifying completed silence sessions.
- History/stats returned resource IDs while frontend retry routing requires resource objects.
- Session resource fields referenced nonexistent `Challenge`/`Script` models rather than `LearningResource`.
- Specialized recording screens could send `test_user` and `unknown` sentinel IDs.
- The upload gateway did not enforce Python V2's audio-extension contract.
- Timed Challenges did not use their configured reference audio for relative scoring.
- Recording-resource language allowed `None` at resource creation even though SpeechSession rejects it.
- Timed Challenge ignored configured Taglish and only offered English/Filipino.
- Frontend `flutter_lints` was outside `dev_dependencies`, blocking complete package resolution and analysis.

### Pagination, filtering, sorting, validation, and authentication

- No current session-metrics consumer requests pagination or filtering.
- History is newest-first for dashboard display.
- Stats sessions are oldest-first for trend computation.
- Admin logs are newest-first and capped at 100, matching the current UI.
- Resources support the only current filter, `type`, and are newest-first.
- Upload IDs and language now receive explicit JSON 400 responses on invalid values.
- Node and Python accept the same six audio extensions. Node applies a 100 MB gateway limit.
- Metrics/session/admin routes currently have no authentication middleware, and neither UI sends credentials. The contracts align; adding security is outside this compatibility task.
- `/api/admin/ai-logs` remains an intentional alias, not a deprecated or missing route.

## 4. Database and Backend Verification

- `SpeechSession` supports all current UI fields: user/resource references, language, audio path, duration, status, four scores, WPM, filler count/availability, transcript, feedback, word timestamps, and timestamps.
- Scores/counts are Mongo `Number`; filler availability is `Boolean`; timestamp entries use String/Number/Number.
- Both `challengeId` and `resourceId` correctly reference `LearningResource`. This metadata correction needs no MongoDB migration.
- Existing documents remain readable. Missing legacy filler availability defaults to `false`, avoiding an unsupported “zero fillers” claim.
- Upload mapping preserves all fields consumed by the UI and returns the saved flat document.
- History/stats populate the exact resource objects used by “Practice Again.”
- Stats aggregation matches ObjectId-valued `userId` and exposes all expected aggregate names.
- Admin queries expose the expected user projection and complete session fields.
- Reference audio is resolved for either a Script (`resourceId`) or Challenge (`challengeId`) and forwarded as Python `reference_audio`.
- Python standard and reference-relative scoring return the same public response shape.
- `targetMetric` remains descriptive UI data. Without reference audio, Python uses 120–150 WPM; with reference audio, pacing is scored relative to the reference.

No database migration or destructive data operation was performed.

## 5. Changes Made

| File | Reason |
|---|---|
| `whisper_service.py` | Stabilized silence contract and corrected fixed-range documentation. |
| `../ispeak_backend/controllers/sessionController.js` | Added ID/language validation, preserved filler availability, supported Script/Challenge reference audio, and populated history/stats resources. |
| `../ispeak_backend/models/SpeechSession.js` | Corrected resource refs and added conservative filler-availability persistence. |
| `../ispeak_backend/models/LearningResource.js` | Aligned the documented target example to 120–150 WPM. |
| `../ispeak_backend/routes/sessionRoutes.js` | Aligned upload extensions, size validation, and JSON 400 errors. |
| `../ispeak_frontend/lib/pages/result_page.dart` | Aligned pace range and unavailable-filler feedback/subtitle. |
| `../ispeak_frontend/lib/pages/script_practice_page.dart` | Required real IDs and normalized resource language. |
| `../ispeak_frontend/lib/pages/time_challenge_page.dart` | Required real IDs, aligned configured language including Taglish, and omitted absent optional IDs. |
| `../ispeak_frontend/lib/pages/learning_resources_page.dart` | Aligned blank/default target display. |
| `../ispeak_frontend/lib/pages/guided_task.dart` | Aligned blank/default target display. |
| `../ispeak_frontend/pubspec.yaml` and `pubspec.lock` | Corrected and resolved the existing lint dev dependency so static analysis works. |
| `../ispeak_admin_panel/lib/screens/dashboard_screen.dart` | Corrected WPM binding and persisted processing-status display. |
| `../ispeak_admin_panel/lib/screens/create_resource_screen.dart` | Aligned recording-resource language choices and target example. |

Generated Flutter package metadata was refreshed offline. No SDK/package upgrade, UI redesign, unrelated refactor, secret, environment file, Docker, CI/CD, or Git operation was performed.

## 6. Verification Evidence

Passed:

- Python byte-compilation for the FastAPI entry point and affected metrics modules.
- Static normal-versus-silence response checks for identical public top-level and nested metric keys.
- Speaking-rate boundary checks: 119 slow, 120/150 excellent, 151 fast.
- Node syntax checks for affected controller, models, and routes.
- Runtime Mongoose type/ref/default verification.
- Mocked history/stats ObjectId, population, and ordering checks.
- Runtime inventory of all six session/admin metrics routes.
- Multipart rejection check for unsupported upload extensions.
- Mocked end-to-end Challenge upload proving reference audio is attached, Taglish persists, and filler availability remains false.
- Offline Flutter dependency resolution.
- Full Dart analysis of frontend `lib`: no compile/analyzer errors (one unrelated existing warning and informational lints remain).
- Full Dart analysis of admin `lib`: no compile/analyzer errors (informational lints remain).
- Stale-contract search found no remaining 110–160, 120–140, sentinel-ID, or optimistic filler-fallback usages.
- Backend V2 project-local environment resolves all declared runtime dependencies.
- All 12 Python analysis-component unit tests pass.
- The local `openai/whisper-small` base model and iSpeak_v4 PEFT adapter load and merge successfully on CPU.
- FastAPI lifespan initialization and `GET /health` return ready with iSpeak_v4 available.
- A real multipart `POST /transcribe` request completes iSpeak_v4 inference and returns HTTP 200 with the full metrics contract.

Environment-limited checks:

- A single live Frontend/Admin → Node → MongoDB → Python deployment was not started because no live MongoDB service was placed in scope. Gateway persistence and query behavior were verified with the documented runtime mocks and contract checks instead.

## 7. Final Status

**✅ Fully compatible. No further compatibility action required.**
