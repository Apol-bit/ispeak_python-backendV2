"""Transcript vocabulary for review, never sufficient evidence for a penalty.

The legacy detector treated ordinary Filipino words as always-fillers. Keep
recognition separate from the contextual model's accepted classifications.
Taglish uses both vocabularies without a language-code gate.
"""
import re

ENGLISH = {
    'um', 'umm', 'ummm', 'uhm', 'uhmm', 'uh', 'uhh', 'uhhh',
    'er', 'err', 'ah', 'ahh', 'hm', 'hmm', 'mm', 'mmm',
    'like', 'so', 'well', 'right', 'okay', 'basically', 'actually',
    'you know', 'i mean', 'you see', 'kind of', 'sort of', 'i guess',
    'you know what i mean', 'to be honest',
}
FILIPINO = {
    'ah', 'eh', 'ano', 'anoh', 'parang', 'kumbaga', 'ganun', 'ganon',
    'yung', 'yun', 'basta', 'diba', 'kasi', 'tapos', 'naman', 'nga',
    'alam mo', 'alam nyo', 'alam mo ba', 'di ba', 'parang ganun',
    'parang ganon', 'kumbaga parang', 'ano ba', 'ano yun', 'ano yung',
    'yung ano', 'so ayun', 'ibig sabihin',
}
# Longest phrases win, with no substring matches inside words or hyphenated
# lexical units. Whitespace may vary; punctuation does not join a phrase.
_TERMS = sorted(ENGLISH | FILIPINO, key=lambda term: (-len(term.split()), -len(term), term))
VOCABULARY = re.compile(
    r"(?<![\w'’\-])(?:" + '|'.join(r'\s+'.join(map(re.escape, term.split())) for term in _TERMS)
    + r")(?![\w'’\-])", re.IGNORECASE,
)
