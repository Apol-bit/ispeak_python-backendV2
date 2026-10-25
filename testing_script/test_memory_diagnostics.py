"""Small synthetic/mocked diagnostics checks; never start model lifespan/inference."""
import asyncio
from contextlib import ExitStack
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from fastapi.concurrency import run_in_threadpool
import memory_diagnostics as diagnostics
import fastapi_backend as backend
import whisper_service as service


def events(logs):
    return [json.loads(line.split(':', 2)[2]) for line in logs.output]


class MemoryDiagnosticsTests(unittest.TestCase):
    def test_disabled_default_and_nontrue_flags_do_no_sampling_or_logging(self):
        for settings in [{}, {'ISPEAK_MEMORY_DIAGNOSTICS': 'false'},
                         {'ISPEAK_MEMORY_DIAGNOSTICS': 'TRUE'}, {'ISPEAK_MEMORY_DIAGNOSTICS': '1'}]:
            with patch.dict(os.environ, settings, clear=True), \
                 patch.object(diagnostics, '_sample') as sample, \
                 patch.object(diagnostics, '_Request') as request, \
                 patch.object(diagnostics.logger, 'info') as log:
                with diagnostics.memory_request(), diagnostics.memory_stage('pitch'):
                    diagnostics.memory_metadata(audio=object(), sample_rate=0)
                    diagnostics.memory_checkpoint('request_admitted')
                sample.assert_not_called()
                request.assert_not_called()
                log.assert_not_called()

    def test_privacy_schema_duration_and_owned_temporary_sizes(self):
        with tempfile.TemporaryDirectory() as directory:
            private = Path(directory) / 'PRIVATE_RECORDING.wav'
            private.write_bytes(b'PRIVATE_AUDIO')
            upload = SimpleNamespace(file=io.BytesIO(b'PRIVATE_AUDIO'), size=13,
                                     filename='PRIVATE_FILENAME')
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
                 patch.object(diagnostics, '_sample', return_value={'process_rss_bytes': 123}), \
                 self.assertLogs(diagnostics.logger, level='INFO') as logs:
                with diagnostics.memory_request():
                    diagnostics.memory_metadata(path=private, upload=upload,
                                                duration=2.5, reference_used=True)
                    with diagnostics.memory_stage('pitch', 'user'):
                        pass
                    diagnostics.memory_checkpoint('PRIVATE_TRANSCRIPT_TOKEN_PATH')
                    diagnostics.memory_metadata(duration=float('nan'))
                    private.unlink()
                    upload.file.close()
            records = events(logs)
            self.assertNotIn('PRIVATE', str(logs.output))
            pitch = next(e for e in records if e['stage'] == 'pitch')
            self.assertEqual(pitch['request_temp_file_bytes'], 13)
            self.assertEqual(pitch['multipart_payload_bytes'], 13)
            self.assertEqual(pitch['recording_duration_seconds'], 2.5)
            self.assertTrue(pitch['reference_used'])
            self.assertEqual(records[-1]['request_temp_file_bytes'], 0)
            self.assertEqual(records[-1]['multipart_payload_bytes'], 0)
            self.assertIsNone(diagnostics._current.get())
            self.assertEqual(len({e['request_id'] for e in records}), 1)
            self.assertEqual(len(records[0]['request_id']), 32)
            self.assertGreaterEqual(records[-1]['elapsed_ms'], 0)

    def test_unsupported_sensors_emit_null_and_preserve_original_exception(self):
        original = ValueError('PRIVATE_EXCEPTION')
        with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
             patch.object(diagnostics, '_rss_bytes', side_effect=OSError('PRIVATE_PATH')), \
             patch.object(diagnostics, '_cgroup_memory', side_effect=PermissionError('PRIVATE_PATH')), \
             self.assertLogs(diagnostics.logger, level='INFO') as logs:
            with self.assertRaises(ValueError) as raised:
                with diagnostics.memory_request(), diagnostics.memory_stage('pitch'):
                    raise original
        self.assertIs(raised.exception, original)
        records = events(logs)
        self.assertTrue(all(e['process_rss_bytes'] is None for e in records))
        self.assertTrue(all(e['cgroup_current_bytes'] is None for e in records))
        self.assertEqual(records[-1]['phase'], 'error')
        self.assertNotIn('PRIVATE', str(logs.output))
        self.assertIsNone(diagnostics._current.get())

    def test_logging_failure_does_not_change_result_or_exception(self):
        with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
             patch.object(diagnostics.logger, 'info', side_effect=RuntimeError('log unavailable')):
            with diagnostics.memory_request(), diagnostics.memory_stage('pitch'):
                result = 42
            self.assertEqual(result, 42)
            with self.assertRaisesRegex(ValueError, 'original'):
                with diagnostics.memory_request(), diagnostics.memory_stage('pitch'):
                    raise ValueError('original')

    def test_cgroup_v2_nested_membership_and_unlimited_limit(self):
        def read(path, limit=8192):
            return {'/proc/self/cgroup': '0::/isolated\n',
                    '/sys/fs/cgroup/isolated/memory.current': '123',
                    '/sys/fs/cgroup/isolated/memory.peak': '456',
                    '/sys/fs/cgroup/isolated/memory.max': 'max'}[str(path).replace('\\', '/')]
        with patch.object(diagnostics, '_read_text', side_effect=read):
            self.assertEqual(diagnostics._cgroup_memory(), {
                'cgroup_current_bytes': 123, 'cgroup_peak_bytes': 456, 'cgroup_limit_bytes': None})

    def test_cgroup_v1_and_missing_counters(self):
        def read(path, limit=8192):
            values = {'/proc/self/cgroup': '5:memory:/isolated\n',
                      '/sys/fs/cgroup/memory/isolated/memory.usage_in_bytes': '123',
                      '/sys/fs/cgroup/memory/isolated/memory.limit_in_bytes': '999'}
            return values[str(path).replace('\\', '/')]
        with patch.object(diagnostics, '_read_text', side_effect=read):
            self.assertEqual(diagnostics._cgroup_memory(), {
                'cgroup_current_bytes': 123, 'cgroup_peak_bytes': None, 'cgroup_limit_bytes': 999})

    def test_linux_rss_pages_are_current_not_lifetime_peak(self):
        with patch.object(diagnostics.os, 'name', 'posix'), \
             patch.object(diagnostics, '_read_text', return_value='100 12 0'), \
             patch.object(diagnostics.os, 'sysconf', return_value=4096, create=True):
            self.assertEqual(diagnostics._rss_bytes(), 12 * 4096)

    def test_diagnostics_module_has_no_model_dependency(self):
        import subprocess
        import sys
        script = "import sys; import memory_diagnostics; assert not any(n in sys.modules for n in ('torch','transformers','model','numpy'))"
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0)

    def test_mocked_pipeline_has_identical_output_and_transcription_settings(self):
        import numpy as np
        model = Mock()
        model.transcribe.return_value = {'text': 'PRIVATE_TRANSCRIPT one two', 'segments': []}
        replacements = {
            'validate_audio': 2, 'validate_transcript': None,
            'analyze_signal_intensity': {}, 'analyze_frequency_pitch': {'analysis_available': True},
            'analyze_temporal_pauses': {'total_pause_seconds': 0},
            'analyze_speaking_rate': {'wpm': 127},
            'analyze_articulation': {'articulation_score': 90},
            'analyze_fillers': {'filler_score': 100},
        }
        with ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch.object(service, name, return_value=value))
            signal = np.ones(32000, dtype=np.float32)
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'false'}):
                with diagnostics.memory_request():
                    baseline = service._run_analysis('PRIVATE_PATH', signal, 16000, model)
            before = model.transcribe.call_args
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
                 self.assertLogs(diagnostics.logger, level='INFO') as logs:
                with diagnostics.memory_request():
                    actual = service._run_analysis('PRIVATE_PATH', signal, 16000, model)
            self.assertEqual(actual, baseline)
            self.assertEqual(model.transcribe.call_args.args, before.args)
            for key in ('word_timestamps', 'initial_prompt', 'condition_on_previous_text', 'language'):
                self.assertEqual(model.transcribe.call_args.kwargs[key], before.kwargs[key])
            self.assertEqual({e['stage'] for e in events(logs)}, {
                'request', 'signal_validation', 'whisper_word_timestamps', 'intensity',
                'pitch', 'pauses', 'articulation', 'filler_classification'})
            self.assertNotIn('PRIVATE', str(logs.output))

    def test_route_metadata_cleanup_and_response_unchanged(self):
        from testing_script.test_container_admission import AUDIO
        response = {'duration_seconds': 2, 'fillers': {'analysis_available': True}}
        def analysis(*args):
            diagnostics.memory_metadata(duration=2)
            with diagnostics.memory_stage('whisper_word_timestamps', 'user'):
                return response
        with patch.object(backend.app.state, 'model', object()), \
             patch.object(backend, 'SERVICE_TOKEN', ''), \
             patch.object(backend, 'generate_full_analysis', side_effect=analysis), \
             patch.object(backend, 'load_model', side_effect=AssertionError('No model loading')), \
             patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
             self.assertLogs(diagnostics.logger, level='INFO') as logs:
            client = TestClient(backend.app)  # No lifespan/model startup.
            try:
                result = client.post('/transcribe', files={'file': ('PRIVATE.wav', AUDIO)})
            finally:
                client.close()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json(), response)
        records = events(logs)
        stages = {e['stage'] for e in records}
        self.assertTrue({'request_admitted', 'multipart_parse', 'upload_copy', 'inference_admitted',
                         'temporary_audio_cleanup', 'request_cleanup'}.issubset(stages))
        self.assertEqual(len({e['request_id'] for e in records}), 1)  # AnyIO thread context propagation.
        self.assertFalse(records[-1]['reference_used'])
        self.assertEqual(records[-1]['recording_duration_seconds'], 2)
        self.assertEqual(records[-1]['request_temp_file_bytes'], 0)
        self.assertEqual(records[-1]['multipart_payload_bytes'], 0)
        self.assertNotIn('PRIVATE', str(logs.output))

    def test_reference_decode_and_dtw_preserve_complete_assessment(self):
        import copy
        import numpy as np
        from testing_script.test_reference_validation import result
        signal = np.ones(32000, dtype=np.float32)
        with patch.object(service, 'load_valid_audio', return_value=(signal, 16000)), \
             patch.object(service, '_run_analysis', side_effect=lambda *a, **k: copy.deepcopy(result())), \
             patch.object(service, 'compare_reference_enunciation', return_value={'available': True, 'score': 88}):
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'false'}):
                with diagnostics.memory_request():
                    baseline = service.generate_reference_analysis('PRIVATE_USER', 'PRIVATE_REFERENCE', Mock())
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
                 self.assertLogs(diagnostics.logger, level='INFO') as logs:
                with diagnostics.memory_request():
                    diagnostics.memory_metadata(reference_used=True)
                    actual = service.generate_reference_analysis('PRIVATE_USER', 'PRIVATE_REFERENCE', Mock())
        self.assertEqual(actual, baseline)
        records = events(logs)
        self.assertEqual({e['audio_role'] for e in records if e['stage'] == 'decode_validate'}, {'user', 'reference'})
        self.assertEqual([e['phase'] for e in records if e['stage'] == 'reference_dtw'], ['start', 'end'])
        self.assertEqual(records[-1]['recording_duration_seconds'], 2)
        self.assertEqual(records[-1]['reference_duration_seconds'], 2)
        self.assertNotIn('PRIVATE', str(logs.output))

    def test_failed_route_still_cleans_files_and_releases_admission(self):
        from testing_script.test_container_admission import AUDIO
        from analysis_admission import AnalysisAdmission
        with patch.object(backend.app.state, 'model', object()), \
             patch.object(backend, 'SERVICE_TOKEN', ''), \
             patch.object(backend, 'admission', AnalysisAdmission(1, 0, 1)), \
             patch.object(backend, 'generate_full_analysis', side_effect=RuntimeError('PRIVATE_EXCEPTION')), \
             patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
             self.assertLogs(diagnostics.logger, level='INFO') as logs:
            client = TestClient(backend.app)
            try:
                self.assertEqual(client.post('/transcribe', files={'file': ('PRIVATE.wav', AUDIO)}).status_code, 500)
                self.assertEqual(backend.admission.active, 0)
                self.assertEqual(backend.admission.admitted, 0)
            finally:
                client.close()
        last = events(logs)[-1]
        self.assertEqual(last['phase'], 'error')
        self.assertEqual(last['request_temp_file_bytes'], 0)
        self.assertEqual(last['multipart_payload_bytes'], 0)
        self.assertNotIn('PRIVATE', str(logs.output))


class StageFailureIsolationTests(unittest.TestCase):
    """Telemetry failures must never own analysis/cleanup control flow."""

    def test_entry_timing_failure_always_runs_body_and_uses_unknown_elapsed(self):
        for failure in (MemoryError, OSError, RuntimeError):
            with self.subTest(failure=failure.__name__):
                state = Mock()
                token = diagnostics._current.set(state)
                calls = []
                try:
                    with patch.object(diagnostics.time, 'perf_counter', side_effect=failure('PRIVATE_CLOCK')):
                        with diagnostics.memory_stage('pitch'):
                            calls.append('analysis')
                    self.assertEqual(calls, ['analysis'])
                    self.assertEqual(state.emit.call_args.args, ('pitch', 'end', None, None))
                finally:
                    diagnostics._current.reset(token)

    def test_exit_timing_failure_does_not_change_success(self):
        state = Mock()
        token = diagnostics._current.set(state)
        try:
            with patch.object(diagnostics.time, 'perf_counter', side_effect=[10, MemoryError('PRIVATE_CLOCK')]):
                with diagnostics.memory_stage('pitch'):
                    result = 42
            self.assertEqual(result, 42)
            self.assertEqual(state.emit.call_args.args, ('pitch', 'end', None, None))
        finally:
            diagnostics._current.reset(token)

    def test_exit_timing_failure_preserves_original_exception_identity(self):
        state = Mock()
        token = diagnostics._current.set(state)
        original = ValueError('PRIVATE_ORIGINAL_ANALYSIS')
        try:
            with patch.object(diagnostics.time, 'perf_counter', side_effect=[10, MemoryError('PRIVATE_CLOCK')]):
                with self.assertRaises(ValueError) as raised:
                    with diagnostics.memory_stage('pitch'):
                        raise original
            self.assertIs(raised.exception, original)
            self.assertEqual(state.emit.call_args.args, ('pitch', 'error', None, None))
        finally:
            diagnostics._current.reset(token)

    def test_emission_failure_on_entry_and_exit_does_not_skip_body_or_mask_error(self):
        for failure in (MemoryError, OSError, RuntimeError):
            state = Mock()
            state.emit.side_effect = failure('PRIVATE_SAMPLER')
            token = diagnostics._current.set(state)
            original = ValueError('PRIVATE_ANALYSIS')
            try:
                calls = []
                with diagnostics.memory_stage('pitch'):
                    calls.append('analysis')
                self.assertEqual(calls, ['analysis'])
                with self.assertRaises(ValueError) as raised:
                    with diagnostics.memory_stage('pitch'):
                        raise original
                self.assertIs(raised.exception, original)
            finally:
                diagnostics._current.reset(token)

    def test_cleanup_runs_with_both_clock_and_emission_failures(self):
        state = Mock()
        state.emit.side_effect = MemoryError('PRIVATE_SAMPLER')
        token = diagnostics._current.set(state)
        original = ValueError('PRIVATE_ANALYSIS')
        cleaned = []
        try:
            with patch.object(diagnostics.time, 'perf_counter', side_effect=MemoryError('PRIVATE_CLOCK')):
                with self.assertRaises(ValueError) as raised:
                    try:
                        raise original
                    finally:
                        with diagnostics.memory_stage('temporary_audio_cleanup'):
                            cleaned.append('saved files')
                        with diagnostics.memory_stage('request_cleanup'):
                            cleaned.append('multipart files')
            self.assertEqual(cleaned, ['saved files', 'multipart files'])
            self.assertIs(raised.exception, original)
        finally:
            diagnostics._current.reset(token)

    def test_real_collector_clock_failure_preserves_request_and_cleanup(self):
        for failure in (MemoryError, RuntimeError):
            with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
                 patch.object(diagnostics.logger, 'info') as log:
                calls = []
                original = ValueError('PRIVATE_ANALYSIS')
                with self.assertRaises(ValueError) as raised:
                    with diagnostics.memory_request():
                        with patch.object(diagnostics.time, 'perf_counter', side_effect=failure('PRIVATE_CLOCK')):
                            try:
                                with diagnostics.memory_stage('pitch'):
                                    calls.append('analysis')
                                    raise original
                            finally:
                                with diagnostics.memory_stage('request_cleanup'):
                                    calls.append('cleanup')
                self.assertEqual(calls, ['analysis', 'cleanup'])
                self.assertIs(raised.exception, original)
                self.assertIsNone(diagnostics._current.get())
                self.assertNotIn('PRIVATE', str(log.call_args_list))

    def test_base_exception_is_preserved_during_exit_diagnostic_failure(self):
        original = asyncio.CancelledError()
        state = Mock()
        token = diagnostics._current.set(state)
        try:
            with patch.object(diagnostics.time, 'perf_counter', side_effect=[10, OSError('PRIVATE_CLOCK')]):
                with self.assertRaises(asyncio.CancelledError) as raised:
                    with diagnostics.memory_stage('pitch'):
                        raise original
            self.assertIs(raised.exception, original)
        finally:
            diagnostics._current.reset(token)


class ContextIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_requests_have_independent_thread_context(self):
        async def work(duration):
            with diagnostics.memory_request():
                diagnostics.memory_metadata(duration=duration)
                await asyncio.sleep(0)
                await run_in_threadpool(diagnostics.memory_checkpoint, 'inference_admitted')
        with patch.dict(os.environ, {'ISPEAK_MEMORY_DIAGNOSTICS': 'true'}), \
             self.assertLogs(diagnostics.logger, level='INFO') as logs:
            await asyncio.gather(work(1), work(2))
        records = [e for e in events(logs) if e['stage'] == 'inference_admitted']
        self.assertEqual(len({e['request_id'] for e in records}), 2)
        self.assertEqual({e['recording_duration_seconds'] for e in records}, {1, 2})
        self.assertIsNone(diagnostics._current.get())


if __name__ == '__main__':
    unittest.main()
