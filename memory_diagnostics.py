"""Opt-in, best-effort request diagnostics. Never inspect speech or credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import math
import os
from pathlib import Path
import time
import uuid

logger = logging.getLogger(__name__)
_current = ContextVar('ispeak_memory_diagnostics', default=None)
_STAGES = frozenset({
    'request', 'request_admitted', 'admission_rejected', 'multipart_parse',
    'upload_copy', 'inference_admitted', 'decode_validate', 'signal_validation',
    'whisper_word_timestamps', 'intensity', 'pitch', 'pauses', 'articulation',
    'filler_classification', 'reference_dtw', 'temporary_audio_cleanup',
    'request_cleanup',
})


def _read_text(path, limit=8192):
    with open(path, encoding='ascii') as handle:
        return handle.read(limit)


def _rss_bytes():
    if os.name == 'posix':
        return int(_read_text('/proc/self/statm', 128).split()[1]) * os.sysconf('SC_PAGE_SIZE')
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    'peak_working', 'working', 'peak_paged', 'paged',
                    'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(
                wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            return int(counters.working)
    return None


def _cgroup_memory():
    """Common Linux cgroup v2/v1 mounts, including nested memberships."""
    root = Path('/sys/fs/cgroup')
    candidates = []
    try:
        for line in _read_text('/proc/self/cgroup').splitlines():
            _, controllers, membership = line.split(':', 2)
            parts = Path(membership.lstrip('/')).parts
            if '..' in parts:
                continue
            if not controllers:
                candidates.append((root.joinpath(*parts), 'memory.current', 'memory.peak', 'memory.max'))
            elif 'memory' in controllers.split(','):
                candidates.append((root.joinpath('memory', *parts),
                                   'memory.usage_in_bytes', 'memory.max_usage_in_bytes',
                                   'memory.limit_in_bytes'))
    except Exception:
        pass
    candidates.extend([
        (root, 'memory.current', 'memory.peak', 'memory.max'),
        (root / 'memory', 'memory.usage_in_bytes', 'memory.max_usage_in_bytes', 'memory.limit_in_bytes'),
    ])
    for directory, current, peak, limit in candidates:
        try:
            usage = int(_read_text(directory / current, 64).strip())
        except Exception:
            continue
        values = [usage]
        for name in (peak, limit):
            try:
                values.append(int(_read_text(directory / name, 64).strip()))
            except Exception:
                values.append(None)
        return dict(zip(('cgroup_current_bytes', 'cgroup_peak_bytes', 'cgroup_limit_bytes'), values))
    return dict.fromkeys(('cgroup_current_bytes', 'cgroup_peak_bytes', 'cgroup_limit_bytes'))


def _sample():
    try:
        rss = _rss_bytes()
    except Exception:
        rss = None
    try:
        group = _cgroup_memory()
    except Exception:
        group = dict.fromkeys(('cgroup_current_bytes', 'cgroup_peak_bytes', 'cgroup_limit_bytes'))
    return {'process_rss_bytes': rss, **group}


class _Request:
    def __init__(self):
        self.request_id = uuid.uuid4().hex
        self.started = time.perf_counter()
        self.reference_used = None
        self.durations = {'user': None, 'reference': None}
        self.paths = []
        self.uploads = []

    def emit(self, stage, phase, elapsed_ms=None, role=None):
        try:
            if stage not in _STAGES or phase not in ('start', 'end', 'error', 'checkpoint'):
                return
            saved_bytes = 0
            for path in self.paths:
                try:
                    saved_bytes += os.stat(path, follow_symlinks=False).st_size
                except FileNotFoundError:
                    pass
                except Exception:
                    saved_bytes = None
                    break
            multipart_bytes = 0
            try:
                for upload in self.uploads:
                    if not upload.file.closed:
                        if not isinstance(upload.size, int) or upload.size < 0:
                            multipart_bytes = None
                            break
                        multipart_bytes += upload.size
            except Exception:
                multipart_bytes = None
            event = {
                'event': 'ispeak_memory', 'request_id': self.request_id,
                'stage': stage, 'phase': phase,
                'audio_role': role if role in ('user', 'reference') else None,
                'reference_used': self.reference_used,
                'recording_duration_seconds': self.durations['user'],
                'reference_duration_seconds': self.durations['reference'],
                'elapsed_ms': elapsed_ms,
                'request_elapsed_ms': round((time.perf_counter() - self.started) * 1000, 3),
                'request_temp_file_bytes': saved_bytes,
                'multipart_payload_bytes': multipart_bytes,
                **_sample(),
            }
            logger.info('%s', json.dumps(event, allow_nan=False, separators=(',', ':')))
        except Exception:
            # Telemetry must not mask inference/cleanup failures or fail requests.
            pass


@contextmanager
def memory_request():
    state = None
    if os.getenv('ISPEAK_MEMORY_DIAGNOSTICS', 'false') == 'true':
        try:
            state = _Request()
        except Exception:
            pass
    token = _current.set(state)
    try:
        with memory_stage('request'):
            yield
    finally:
        _current.reset(token)


@contextmanager
def memory_stage(stage, role=None):
    state = _current.get()
    if state is None or stage not in _STAGES:
        yield
        return
    started = None
    try:
        started = time.perf_counter()
    except Exception:
        pass
    try:
        state.emit(stage, 'start', role=role)
    except Exception:
        pass
    phase = 'end'
    try:
        yield
    except BaseException:
        phase = 'error'
        raise
    finally:
        elapsed_ms = None
        if started is not None:
            try:
                elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            except Exception:
                pass
        try:
            state.emit(stage, phase, elapsed_ms, role)
        except Exception:
            pass


def memory_checkpoint(stage):
    state = _current.get()
    if state is not None:
        state.emit(stage, 'checkpoint')


def memory_metadata(*, role='user', duration=None, audio=None, sample_rate=None,
                    reference_used=None, path=None, upload=None):
    state = _current.get()
    if state is None:
        return
    try:
        if type(reference_used) is bool:
            state.reference_used = reference_used
        if audio is not None:
            duration = len(audio) / sample_rate
        if role in state.durations and type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
            state.durations[role] = round(duration, 3)
        if path is not None and len(state.paths) < 2:
            state.paths.append(path)
        if upload is not None and len(state.uploads) < 2:
            state.uploads.append(upload)
    except Exception:
        pass
