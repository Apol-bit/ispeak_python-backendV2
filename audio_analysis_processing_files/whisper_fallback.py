"""Bounded Whisper compression fallback diagnostics for the pinned runtime."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import inspect
import math
import textwrap
from time import perf_counter
from types import MethodType


TEMPERATURES = (0.0, 0.2)
COMPRESSION_RATIO_THRESHOLD = 1.35
_NEED_FALLBACK_HASH = '64d630117575e6064708faac57559c0b0bcc4b0df9630012f1ab33889774785e'

_ACTIVE = ContextVar('ispeak_whisper_fallback_diagnostics', default=None)
_CURRENT_SEGMENT = ContextVar('ispeak_whisper_fallback_segment', default=None)


@contextmanager
def collect_fallback_diagnostics(enabled):
    """Collect aggregate decode-window metrics without retaining tokens or text."""
    records = [] if enabled else None
    token = _ACTIVE.set(records)
    try:
        yield records if records is not None else []
    finally:
        _ACTIVE.reset(token)


def _need_fallback(self, seek_sequence, seek_outputs, index, logits_processor,
                   generation_config, vocab_size, temperature):
    result = self._ispeak_fallback_need_original(
        self, seek_sequence, seek_outputs, index, logits_processor,
        generation_config, vocab_size, temperature,
    )
    segment = _CURRENT_SEGMENT.get()
    if segment is not None:
        # Upstream returns the final sequence even if it still needs fallback.
        # Preserve that decision separately from optional ratio/timing telemetry.
        # At each temperature the batch contains only items still being retried;
        # any remaining failure on the last attempt makes this window unusable.
        if segment.get('_decision_temperature') != temperature:
            segment['fallback_exhausted'] = False
        segment['_decision_temperature'] = temperature
        segment['fallback_exhausted'] = segment['fallback_exhausted'] or bool(result[0])
    try:
        if segment is not None:
            ratio = None
            if generation_config.compression_ratio_threshold is not None:
                ratio = float(self._retrieve_compression_ratio(seek_sequence, vocab_size))
                if not math.isfinite(ratio):
                    ratio = None
            segment['attempts'].append({
                'compression_ratio': ratio,
                'temperature': float(temperature) if temperature is not None else None,
                'needs_fallback': bool(result[0]),
            })
    except Exception:
        # Diagnostics must never change the upstream decode decision.
        pass
    return result


def _now():
    try:
        return perf_counter()
    except Exception:
        return None


def _finish_segment(segment, started):
    segment.pop('_decision_temperature', None)
    try:
        finished = _now()
        segment['elapsed_seconds'] = (
            finished - started if finished is not None and started is not None else None
        )
        temperatures = []
        for attempt in segment['attempts']:
            temperature = attempt['temperature']
            if temperature not in temperatures:
                temperatures.append(temperature)
        segment['retry_count'] = max(0, len(temperatures) - 1)
        ratios = [attempt['compression_ratio'] for attempt in segment['attempts']
                  if attempt['compression_ratio'] is not None]
        segment['initial_compression_ratio'] = ratios[0] if ratios else None
        segment['final_compression_ratio'] = ratios[-1] if ratios else None
        # Attempt details contain no private content, but are no longer needed.
        segment.pop('attempts', None)
    except Exception:
        # The decode result or its original exception always takes precedence.
        pass


def _generate_with_diagnostics(self, *args, **kwargs):
    original = self._ispeak_fallback_generate_original
    records = _ACTIVE.get()
    if records is None or kwargs.get('is_shortform') is not False:
        return original(self, *args, **kwargs)

    try:
        segment = {'segment_index': len(records), 'attempts': [], 'fallback_exhausted': False}
        records.append(segment)
        token = _CURRENT_SEGMENT.set(segment)
    except Exception:
        return original(self, *args, **kwargs)

    started = _now()
    try:
        try:
            return original(self, *args, **kwargs)
        finally:
            _finish_segment(segment, started)
    finally:
        try:
            _CURRENT_SEGMENT.reset(token)
        except Exception:
            try:
                _CURRENT_SEGMENT.set(None)
            except Exception:
                pass


def _fingerprint(method):
    source = textwrap.dedent(inspect.getsource(method)).replace('\r\n', '\n')
    return hashlib.sha256(source.encode()).hexdigest()


def install_fallback_diagnostics(model):
    """Install instance-only metrics hooks without altering fallback decisions."""
    if hasattr(model, '_ispeak_fallback_generate_original'):
        return True
    try:
        if importlib.metadata.version('transformers') != '5.10.1':
            return False
        if not hasattr(model, '_ispeak_retention_original'):
            return False
        generate = model.generate_with_fallback
        need = model._need_fallback
        generate_original = generate.__func__
        need_original = need.__func__
        if _fingerprint(need_original) != _NEED_FALLBACK_HASH:
            return False
    except (AttributeError, OSError, TypeError, importlib.metadata.PackageNotFoundError):
        return False

    model._ispeak_fallback_generate_original = generate_original
    model._ispeak_fallback_need_original = need_original
    model.generate_with_fallback = MethodType(_generate_with_diagnostics, model)
    model._need_fallback = MethodType(_need_fallback, model)
    return True
