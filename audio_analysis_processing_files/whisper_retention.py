"""Release completed long-form chunk diagnostics only inside iSpeak's ASR call.

Generation, fallback decisions and timestamp extraction finish before release.
Public model.generate calls outside this context keep their original outputs.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib.metadata
import inspect
import textwrap
from types import MethodType

_ACTIVE = ContextVar('ispeak_compact_whisper_segments', default=False)
_CHUNK_ACTIVE = ContextVar('ispeak_compact_whisper_chunk', default=False)
_FORWARD_HASH = '866de4b20c23de6e99d6eea1cef18f9d52a255181a7a5c970a7ff008e7c64285'
_HASHES = {
    'generate_with_fallback': '5003923fc332e1042984444fdab0f29e1ceb1742543b22133fc5779dfeb09c92',
    'generate': 'f118e84cb0d2ba8952c96b711f7a1704507f9d3e722a0e9324d992b027682a62',
    '_retrieve_segment': '6ee54c5b25bcb4732cc22a9c7c481c55b464aba1ac8ff9004921d956c20cce9f',
}
_PIPELINE_HASH = 'fe180ad749079e0cf1b59831bc231e972dce5746a14ca5f1706e8793f64738c9'
_DISCARD = ('scores', 'logits', 'encoder_attentions', 'decoder_attentions',
            'cross_attentions', 'encoder_hidden_states', 'decoder_hidden_states',
            'past_key_values')


@contextmanager
def compact_whisper_segments():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _fingerprint(method):
    source = textwrap.dedent(inspect.getsource(method)).replace('\r\n', '\n')
    return hashlib.sha256(source.encode()).hexdigest()


def _generate_with_fallback(self, *args, **kwargs):
    original = self._ispeak_retention_original
    # Bind before invoking to preserve positional/keyword support and inspect the
    # exact reviewed call's controls without changing its generation config.
    call = inspect.signature(original).bind(self, *args, **kwargs)
    call.apply_defaults()
    eligible = (_ACTIVE.get() and call.arguments.get('is_shortform') is False
                and call.arguments.get('return_token_timestamps') is True)
    token = _CHUNK_ACTIVE.set(eligible)
    try:
        result = original(self, *args, **kwargs)
    finally:
        _CHUNK_ACTIVE.reset(token)
    if eligible:
        # Tuple: sequences, per-batch completed results, skip, conditioning, type.
        # Scores were already consumed by _need_fallback; attentions were already
        # consumed by _extract_token_timestamps. Keep sequences/beam/timestamps.
        for output in result[1]:
            if isinstance(output, dict):
                for key in _DISCARD:
                    output.pop(key, None)
    return result


def _release_self_attentions(module, inputs, output):
    if not _CHUNK_ACTIVE.get() or not isinstance(output, dict):
        return None
    values = output.get('decoder_attentions')
    if not isinstance(values, tuple) or not values:
        return None
    if any(getattr(value, 'ndim', None) != 4 for value in values):
        return None
    # Return fresh metadata; preserve every prediction/cache/cross-attention
    # tensor by identity. Empty per-layer diagnostics remain batch-indexable in
    # upstream postprocessing, but do not retain the quadratic causal matrices.
    compact = tuple(value.new_empty((value.shape[0], value.shape[1], 0, 0))
                    for value in values)
    return type(output)(**{**output, 'decoder_attentions': compact})


def install_active_chunk_release(model):
    """No attention kernel/config changes: release unused forward diagnostics."""
    if hasattr(model, '_ispeak_chunk_release_handle'):
        return True
    try:
        if (not hasattr(model, '_ispeak_retention_original')
                or importlib.metadata.version('transformers') != '5.10.1'
                or _fingerprint(model.forward.__func__) != _FORWARD_HASH):
            return False
        handle = model.register_forward_hook(_release_self_attentions)
    except (AttributeError, OSError, TypeError, importlib.metadata.PackageNotFoundError):
        return False
    model._ispeak_chunk_release_handle = handle
    return True


def install_longform_retention(model, pipeline_class):
    """Instance-only hook; unsupported upstream implementations remain unchanged."""
    if hasattr(model, '_ispeak_retention_original'):
        return True
    try:
        if importlib.metadata.version('transformers') != '5.10.1':
            return False
        for name, expected in _HASHES.items():
            method = getattr(model, name)
            if _fingerprint(getattr(method, '__func__', method)) != expected:
                return False
        if _fingerprint(pipeline_class._forward) != _PIPELINE_HASH:
            return False
        original = model.generate_with_fallback.__func__
    except (AttributeError, OSError, TypeError, importlib.metadata.PackageNotFoundError):
        return False
    model._ispeak_retention_original = original
    model.generate_with_fallback = MethodType(_generate_with_fallback, model)
    return True
