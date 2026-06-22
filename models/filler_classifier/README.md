# Contextual filler classifier goes here

Copy a locally fine-tuned Hugging Face token-classification model here. It must
contain `config.json`, tokenizer files, and either `model.safetensors` or
`pytorch_model.bin`.

Required labels are `B-FILLER` / `I-FILLER` (or `FILLER`) and `O` (or
`CONTEXT_WORD`). `UNCERTAIN` is optional. Loading uses `local_files_only=True`.
