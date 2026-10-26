"""Conservative complementary text rules; timing is ASR evidence, not audio proof."""
import math
import re

from audio_analysis_processing_files.word_alignment import WORD

_CLEAR_HESITATION = re.compile(r'u(?:m+|h+m*)\Z', re.IGNORECASE)
_RESTART_MARKERS = frozenset({
    'ano', 'anoh', 'parang', 'like', 'so', 'you know', 'alam mo', 'alam nyo',
    'ah', 'ahh', 'ahhh', 'eh',
})
_MENTION = frozenset({
    'word', 'words', 'term', 'terms', 'spelling', 'spell', 'spelled',
    'pronounce', 'pronunciation', 'salita', 'salitang', 'kataga', 'katagang',
    'token', 'tokens', 'label', 'labelled',
})
_COMPARISON = frozenset({
    'feel', 'feels', 'felt', 'look', 'looks', 'looked', 'sound', 'sounds',
    'sounded', 'seem', 'seems', 'seemed', 'taste', 'tastes', 'smell', 'smells',
    'mukha', 'mukhang', 'tila',
})
_COMPARISON_TIME = frozenset({'before', 'previously', 'yesterday', 'dati', 'noon', 'kahapon'})


def clear_hesitation(text):
    """True for a single non-lexical um/uh form, which has no grammatical reading."""
    tokens = WORD.findall(text)
    return len(tokens) == 1 and bool(_CLEAR_HESITATION.fullmatch(tokens[0]))


def valid_timing(segments):
    """Reject missing, zero-length, non-finite, reversed or overlapping word times."""
    previous_end = 0.0
    found = False
    for segment in segments:
        if not str(segment.get('text', '')).strip():
            continue
        start, end = segment.get('start'), segment.get('end')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in (start, end)):
            return False
        if start < 0 or end <= start or start + 1e-6 < previous_end:
            return False
        previous_end, found = end, True
    return found


def explicit_mention(text, start, end):
    """A quoted/defined token is not evidence of the speaker's hesitation."""
    prefix = text[:start]
    if prefix.count('"') % 2 or prefix.count('\u201c') > prefix.count('\u201d'):
        return True
    # Apostrophes internal to words (I'm, di'ba) are not quote delimiters.
    if len(re.findall(r"(?<!\w)'|'(?!\w)", prefix)) % 2:
        return True
    curly = re.sub(r"(?<=\w)\u2019(?=\w)", '', prefix)
    if curly.count('\u2018') > curly.count('\u2019'):
        return True
    body = text[start:end].strip()
    if len(body) > 1 and body[0] in '\'"\u201c\u2018' and body[-1] in '\'"\u201d\u2019':
        return True
    clause = re.split(r'[.!?;\n]', prefix)[-1]
    preceding = [m.group().casefold() for m in WORD.finditer(clause)][-4:]
    return bool(_MENTION.intersection(preceding))


def confirmation_reason(text, match, aligned_words, timing_available):
    """Confirm only narrow lexical patterns with valid timing; return None to abstain.

    Pause gaps are not required: Whisper word times are contiguous within a
    decoding segment, so a hesitation's silence is absorbed into a word's
    duration and is not observable as a gap. Clear um/uh forms are confirmed.
    Ambiguous markers require a punctuated, two-token identical restart and
    pass the mention, question and comparison vetoes; a marker alone does not.
    'as in' and the remaining recovered vocabulary stay review-only.
    """
    if not timing_available or explicit_mention(text, match.start(), match.end()):
        return None
    if clear_hesitation(match.group()):
        return 'non-lexical hesitation form (um/uh) with structurally valid word timing'
    term = ' '.join(match.group().casefold().split())
    if term not in _RESTART_MARKERS:
        return None
    tokens = list(WORD.finditer(text))
    left = [m for m in tokens if m.end() <= match.start()]
    right = [m for m in tokens if m.start() >= match.end()]
    if len(left) < 2 or len(right) < 2:
        return None
    if not re.search(r'[,;:\-\u2013\u2014]', text[left[-1].end():match.start()]):
        return None
    if not re.search(r'[,;:\-\u2013\u2014]', text[match.end():right[0].start()]):
        return None
    left_words = [m.group().casefold() for m in left]
    right_words = [m.group().casefold() for m in right]
    if left_words[-2:] != right_words[:2]:
        return None
    # Question and comparison controls have the same timing/restart shape.
    stop = re.search(r'[.!?]', text[match.end():])
    if stop and stop.group() == '?':
        return None
    if {'tanong', 'question', 'asking', 'tinatanong'}.intersection(left_words[-4:]):
        return None
    if term in {'like', 'parang'} and (
            _COMPARISON.intersection(left_words[-3:]) or _COMPARISON_TIME.intersection(right_words[:8])):
        return None
    return 'bilingual punctuated two-token restart with structurally valid word timing'
