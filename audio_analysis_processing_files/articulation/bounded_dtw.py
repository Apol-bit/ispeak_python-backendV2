"""Exact default cosine DTW statistics without quadratic matrix storage.

Only the terminal cost and selected path length are needed by the caller.
Time remains quadratic; workspace is linear in the second sequence length.
"""
import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


@njit(cache=False)
def _advance(costs, previous, lengths, current, current_lengths):
    current[0] = np.inf
    current_lengths[0] = 0
    for j in range(1, len(current)):
        cost = costs[j - 1]
        # Match librosa's strict-less-than comparison and diagonal, horizontal,
        # vertical step order. Do not swap sequence axes: ties can change length.
        best = previous[j - 1] + cost
        length = lengths[j - 1] + 1
        candidate = current[j - 1] + cost
        if candidate < best:
            best = candidate
            length = current_lengths[j - 1] + 1
        candidate = previous[j] + cost
        if candidate < best:
            best = candidate
            length = lengths[j] + 1
        current[j] = best
        current_lengths[j] = length


def cosine_dtw_statistics(reference, user):
    """Return (terminal cost, path length), matching default librosa DTW.

Inputs have shape (features, frames). No band, downsampling, approximation,
dtype reduction, or path-length substitution is used.
"""
    reference = np.asarray(reference)
    user = np.asarray(user)
    if (reference.ndim != 2 or user.ndim != 2 or
            reference.shape[0] != user.shape[0] or
            not reference.shape[0] or not reference.shape[1] or not user.shape[1]):
        raise ValueError('DTW requires nonempty matching feature dimensions')
    width = user.shape[1] + 1
    previous = np.full(width, np.inf, dtype=np.float64)
    current = np.empty(width, dtype=np.float64)
    lengths = np.zeros(width, dtype=np.int64)
    current_lengths = np.empty(width, dtype=np.int64)
    previous[0] = 0.0
    # scipy's cosine cdist converts to contiguous float64 internally. Do that
    # once for the reused axis rather than copying the entire sequence per row.
    user_frames = np.ascontiguousarray(user.T, dtype=np.float64)
    for i in range(reference.shape[1]):
        costs = cdist(reference[:, i:i + 1].T, user_frames, metric='cosine')[0]
        if not np.all(np.isfinite(costs)):
            raise ValueError('DTW cosine costs must be finite')
        _advance(costs, previous, lengths, current, current_lengths)
        previous, current = current, previous
        lengths, current_lengths = current_lengths, lengths
    return float(previous[-1]), int(lengths[-1])
