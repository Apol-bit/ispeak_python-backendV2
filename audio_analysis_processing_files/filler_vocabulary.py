"""Recovered pre-e80f08a vocabulary; membership alone never confirms a filler.

Source: e80f08a^, clarity_analysis_module/voice_fillerwords_detection.py.
The legacy always/repeat categories are provenance, not restored penalties.
English and Filipino are matched together for Taglish, without a language gate.
"""
import re

LEGACY_ENGLISH_ALWAYS_FILLER = frozenset(['actually', 'ah', 'aha', 'ahh', 'ahhh', 'anyway', 'anyways', 'basically', 'em', 'emm',
 'er', 'err', 'errr', 'hm', 'hmm', 'hmmm', 'honestly', 'literally', 'mhm', 'mm', 'mmm',
 'mmmm', 'uh', 'uhh', 'uhhh', 'uhm', 'uhmm', 'um', 'umm', 'ummm', 'whatever'])
LEGACY_ENGLISH_MULTI_WORD_FILLERS = frozenset(['and all that', 'and everything', 'and stuff', 'as i said', 'at the end of the day',
 'for the most part', 'i feel like', 'i guess', 'i mean', 'in a sense', 'kind of',
 'like i said', 'or something', 'or whatever', 'sort of', 'the thing is',
 'to be honest', 'what i mean is', 'you know', 'you know what i mean', 'you see'])
LEGACY_ENGLISH_REPEAT_ONLY_FILLERS = frozenset(['just', 'like', 'ok', 'okay', 'really', 'right', 'so', 'very', 'well', 'yeah', 'yes'])
LEGACY_FILIPINO_ALWAYS_FILLER = frozenset([' eh', 'aba', 'ah', 'alam mo', 'alam nyo', 'ano', 'anoh', 'ay', 'ba', 'basta', 'di ba',
 'diba', 'eh', 'ewan', 'ewan ko', 'ganon', 'ganun', 'grabe', 'halimbawa', 'hay',
 'hay nako', 'hays', 'jusko', 'jusmio', 'kase', 'kasi', 'kumbaga', 'nako', 'naman',
 'nga', 'no', 'noh', 'oh', 'oo', 'pala', 'parang', 'saka', 'sige', 'siguro', 'sus',
 'talaga', 'tapos', 'totoo', 'tsaka', 'yon', 'yun', 'yung'])
LEGACY_FILIPINO_MULTI_WORD_FILLERS = frozenset(['alam mo', 'alam mo ba', 'alam nyo', 'ang ibig sabihin', 'ano ba', 'ano yun',
 'ano yung', 'basta ganon', 'basta ganun', 'di ba', 'diba', 'ewan ko', 'ewan ko ba',
 'ganon ganon', 'ganun ganun', 'hay nako', 'hay naku', 'ibig sabihin', 'kasi ano',
 'kasi naman', 'kumbaga ganun', 'kumbaga parang', 'naku naman', 'oo nga',
 'parang ganon', 'parang ganun', 'saka tapos', 'so ayun', 'tapos yung', 'yung ano',
 'yung parang'])
LEGACY_FILIPINO_REPEAT_ONLY_FILLERS = frozenset(['lang', 'naman', 'nga', 'pala', 'saka', 'tapos', 'tsaka'])

ENGLISH = frozenset(term.strip().casefold() for term in LEGACY_ENGLISH_ALWAYS_FILLER | LEGACY_ENGLISH_MULTI_WORD_FILLERS | LEGACY_ENGLISH_REPEAT_ONLY_FILLERS)
FILIPINO = frozenset(term.strip().casefold() for term in LEGACY_FILIPINO_ALWAYS_FILLER | LEGACY_FILIPINO_MULTI_WORD_FILLERS | LEGACY_FILIPINO_REPEAT_ONLY_FILLERS)
# Explicitly reported mixed-language phrase absent from the historical lists.
ADDITIONAL_REVIEW_PHRASES = frozenset({'as in'})
_TERMS = sorted(ENGLISH | FILIPINO | ADDITIONAL_REVIEW_PHRASES, key=lambda term: (-len(term.split()), -len(term), term))
# Apostrophes inside a lexical unit block substring matches; surrounding quotes
# allow recognition for review (the evidence rules exclude quoted mentions).
VOCABULARY = re.compile(
    r"(?<![\w\-])(?<!\w['\u2019])(?:"
    + '|'.join(r'\s+'.join(map(re.escape, term.split())) for term in _TERMS)
    + r")(?![\w\-]|['\u2019]\w)", re.IGNORECASE,
)
