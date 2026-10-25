"""Instance-scoped, version-guarded reduction of Whisper timestamp copies.

Keep upstream normalization, filtering, beam handling and alignment unchanged.
Only feed the requested alignment heads to its concatenation step. Original
generation outputs remain untouched, including attentions retained by callers.
"""
import hashlib
import importlib.metadata
import inspect
import operator
import textwrap
from types import MethodType, SimpleNamespace

UPSTREAM_SHA256 = '2233bde8d26c6ed3089f04e43f007c734d9c657a77f1edb86f677ec8cfad0c46'


class _Inputs(dict):
    def __getattr__(self, name):
        return self[name]


def _extract_selected(self, generate_outputs, alignment_heads, time_precision=0.02,
                      num_frames=None, num_input_ids=None):
    original = self._ispeak_timestamp_function
    heads = tuple((operator.index(layer), operator.index(head))
                  for layer, head in alignment_heads)
    # Invalid/unusual head indices retain upstream behavior rather than changing
    # Python's negative indexing into slice semantics.
    if not heads or any(layer < 0 or head < 0 for layer, head in heads):
        return original(self, generate_outputs, alignment_heads, time_precision,
                        num_frames, num_input_ids)
    compact = _Inputs(
        sequences=generate_outputs.sequences,
        cross_attentions=tuple(
            tuple(step[layer][:, head:head + 1] for layer, head in heads)
            for step in generate_outputs.cross_attentions
        ),
    )
    if 'beam_indices' in generate_outputs:
        compact['beam_indices'] = generate_outputs.beam_indices
    proxy = SimpleNamespace(config=SimpleNamespace(
        decoder_layers=len(heads), median_filter_width=self.config.median_filter_width,
    ))
    return original(proxy, compact, [(i, 0) for i in range(len(heads))],
                    time_precision, num_frames, num_input_ids)


def install_selected_head_timestamps(model):
    """Install only on the reviewed 5.10.1 method; otherwise leave it unchanged.

    No global/library monkey patch, model copies, decoding changes or state from
    individual requests are retained. Unsupported installations use upstream.
    """
    try:
        if importlib.metadata.version('transformers') != '5.10.1':
            return False
        method = model._extract_token_timestamps
        source = textwrap.dedent(inspect.getsource(method.__func__)).replace('\r\n', '\n')
        if hashlib.sha256(source.encode()).hexdigest() != UPSTREAM_SHA256:
            return False
    except (AttributeError, OSError, TypeError, importlib.metadata.PackageNotFoundError):
        return False
    model._ispeak_timestamp_function = method.__func__
    model._extract_token_timestamps = MethodType(_extract_selected, model)
    return True
