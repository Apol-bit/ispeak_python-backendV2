# Local contextual filler classifier

Filler detection never calls an API and never downloads a model at runtime.

Place a fine-tuned Hugging Face token-classification model in:

```text
models/filler_classifier/
```

Alternatively, set `ISPEAK_FILLER_MODEL_PATH` to another local directory.
Loading always uses `local_files_only=True`.

The model must classify the complete English/Filipino transcript with these
labels:

- `B-FILLER` / `I-FILLER` (or `FILLER`)
- `O` (or `CONTEXT_WORD`)
- `UNCERTAIN` is optional

Train the model on complete sentences, not isolated words. The same expression
must appear as both a filler and a meaningful context word in the training data.
Without such a locally trained model, the backend returns
`analysis_available: false` and does not change the speaker's clarity score.
