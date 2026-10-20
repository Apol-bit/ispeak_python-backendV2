# Timestamp continuation alignment and lexical counting

This local change preserves ASR full text, original segment words/times/confidences,
returned word timestamps, and all API/assessment field names. No database migration
or existing-session update is performed. Classification thresholds, weights and
models are unchanged. No inference accuracy improvement has been measured.

## Reconstruction

The shared `audio_analysis_processing_files/word_alignment.py` helper proposes
removing only the inserted separator before a `-word` continuation when the
preceding fragment ends in a word character. It verifies resulting lexical tokens
against the full ASR transcript using ordered sequence matching. Only boundaries
inside matching tokens are joined. Without full ASR evidence, no joins are made.
Unverified fragments remain separate; no characters are deleted from recognized
words. Original case and punctuation remain in classifier input.

Every original fragment retains its own timestamp and receives rebuilt character
offsets in the new input. Classifier predictions map through character overlap to
those same fragment times. A prediction covering a complete joined word receives
the union of its fragment times; a prediction covering a fragment retains that
fragment's evidence. Reported grouped span text is sliced from reconstructed input.

Coverage compares ordered normalized tokens, not just counts. Real omissions,
substitutions and reorderings continue to warn. Ordinary punctuation differences
remain ignored as before; coverage is lexical equivalence, not proof of acoustic
accuracy or complete classifier prediction coverage. Repeated identical words can
make sequence alignment ambiguous; this is conservative text verification, not a
new forced acoustic aligner. Independent timestamp accuracy remains unverified.

## Shared counting policy (behavioral change)

Filler-rate denominator, speaking WPM, active articulation rate, and coverage now
use the same reconstructed lexical tokens: Unicode `\w+` groups (including numbers
and underscores), with internal ASCII hyphens or straight/curly apostrophes retained
within a word. Comparison is case-insensitive and treats straight/curly apostrophes
equally. Standalone punctuation, hyphens and apostrophes count as zero. For example,
`i-share`, `don't`, and `don’t` each count once; `ano,` counts once. This normalization
does not rewrite the actual transcript or classifier input. Other dash characters
are punctuation boundaries rather than automatically repaired continuations.

The denominator counts reconstructed observed content, never extra words present
only in the ASR full transcript. Grouped filler spans remain the numerator; uncertain
candidates remain excluded. Formula/rounding rules and scoring weights are unchanged.
Reference analyses use the same policy through the existing analysis path.

## UAT arithmetic replay, not new inference

The three authorized text fixtures reproduce all identified continuation boundaries:
English 41/41 remains matched; Filipino 53/50 becomes 50/50; Taglish 56/53 becomes
53/53. Original stored UAT results are not rewritten. With the same two detected
spans and original articulation/energy components, using previously inspected M4A
durations as a proxy for decoded duration gives:

| Sample | WPM before → illustration | Filler score | Pace | Clarity | Overall |
| --- | --- | --- | --- | --- | --- |
| Filipino | 122 → 115.1 | 85 → 84 | 100 → 95.9 | 89 → 89 | 95.6 → 94.2 |
| Taglish | 119.1 → 112.8 | 86 → 85 | 99.2 → 94.0 | 89 → 88 | 95.3 → 93.1 |

These are explicit behavioral changes from removing fragmentation-dependent
denominators. They are not replacement results for those recordings. Corrected
ModernBERT input can change actual classifications, and decoded durations may
differ from container metadata; a separately approved future real-audio test is
required. Enunciation still uses original timestamp segments, so this change does
not revise its underlying timing heuristic. Pronunciation may remain unavailable.

## Verification and deployment boundary

Model-free tests cover the three UAT text fixtures, offsets and time unions,
punctuation/apostrophes, code switching, repeated and multi-hyphen continuations,
missing/replaced content, unavailable full text, zero lexical words, legacy call
signatures and scoring arithmetic. The new runtime module is already included by
the Docker folder COPY and `.dockerignore` folder allowlist; neither file changes.
No public response fields or Node/Flutter contracts change. Running services have
not been restarted and may still hold old imported code. Do not restart, submit
audio or deploy without the appropriate separate authorization.
