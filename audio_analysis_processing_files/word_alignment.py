"""Shared lexical counting and conservative timestamp-fragment reconstruction."""
import difflib
import re

WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def word_tokens(text):
    """Count internal hyphens/apostrophes as part of a word, never punctuation alone."""
    return [match.group().casefold().replace('’', "'") for match in WORD.finditer(text)]


def reconstruct_words(segments, full_transcript=None):
    """Join only continuation boundaries verified against aligned ASR text tokens.

    Original segment dictionaries are never modified. Returned character ranges
    refer to the reconstructed text; times retain each original fragment's evidence.
    Missing/inconsistent content remains visible to the caller's coverage check.
    """
    entries = [(str(segment.get('text', '')).strip(), segment) for segment in segments]
    entries = [(word, segment) for word, segment in entries if word]

    def build(joins):
        text = ''
        aligned = []
        for index, (word, segment) in enumerate(entries):
            if index and index not in joins:
                text += ' '
            start = len(text)
            text += word
            timestamp = lambda value: float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
            aligned.append(dict(word=word, char_start=start, char_end=len(text),
                                start=timestamp(segment.get('start')), end=timestamp(segment.get('end'))))
        return text, aligned

    candidates = {index for index in range(1, len(entries))
                  if re.search(r'\w$', entries[index-1][0]) and re.match(r'^-\w', entries[index][0])}
    verified = set()
    if full_transcript is not None and candidates:
        proposed, aligned = build(candidates)
        matches = list(WORD.finditer(proposed))
        matcher = difflib.SequenceMatcher(a=word_tokens(proposed), b=word_tokens(full_transcript), autojunk=False)
        for block in matcher.get_matching_blocks():
            for match in matches[block.a:block.a + block.size]:
                for index in candidates:
                    boundary = aligned[index]['char_start']
                    if match.start() < boundary < match.end():
                        verified.add(index)
    return build(verified)
