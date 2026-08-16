"""Download and adapt a disfluency-detection model for local filler-word classification.

Downloads the `arielcerdap/modernbert-base-multiclass-disfluency-v2` model from
Hugging Face and remaps its labels so the filler_words.py module can consume it:

    FP (Filled Pause) -> FILLER
    O  (Outside)      -> O
    RP, RV, PW        -> CONTEXT_WORD   (non-filler disfluencies treated as context)

The model is saved to  models/filler_classifier/  with only the files needed for
inference (no optimizer, scheduler, or checkpoint files).
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TARGET_DIR = PROJECT_ROOT / "models" / "filler_classifier"
SOURCE_MODEL = "arielcerdap/modernbert-base-multiclass-disfluency-v2"

# Map source labels to the schema expected by filler_words.py
LABEL_REMAP = {
    "O":  "O",
    "FP": "FILLER",          # Filled Pause -> FILLER
    "RP": "CONTEXT_WORD",    # Repair
    "RV": "CONTEXT_WORD",    # Revision
    "PW": "CONTEXT_WORD",    # Partial Word
}

# Only download files needed for inference
ALLOW_PATTERNS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def main() -> int:
    if TARGET_DIR.is_dir() and (TARGET_DIR / "config.json").is_file():
        # Verify labels are correct
        config = json.loads((TARGET_DIR / "config.json").read_text(encoding="utf-8"))
        labels = set(config.get("id2label", {}).values())
        if "FILLER" in labels:
            print(f"Filler classifier already present and configured: {TARGET_DIR}")
            return 0
        print("Filler classifier directory exists but labels need remapping...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is unavailable. Install it with:\n"
            "  pip install huggingface-hub"
        ) from exc

    # ---- Download model files ----
    print(f"Downloading {SOURCE_MODEL} ...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=SOURCE_MODEL,
        local_dir=str(TARGET_DIR),
        allow_patterns=ALLOW_PATTERNS,
    )
    print("Download complete.")

    # ---- Fix tokenizer_config.json ----
    # ModernBERT uses "TokenizersBackend" which doesn't exist in transformers <5.
    # Replace with the standard PreTrainedTokenizerFast class.
    tok_cfg_path = TARGET_DIR / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
        if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
            tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tok_cfg_path.write_text(
                json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("Fixed tokenizer_class: TokenizersBackend -> PreTrainedTokenizerFast")

    # ---- Remap labels in config.json ----
    config_path = TARGET_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    old_id2label = config.get("id2label", {})
    new_id2label = {}
    new_label2id: dict[str, int] = {}

    for idx_str, old_label in old_id2label.items():
        new_label = LABEL_REMAP.get(old_label, "O")
        new_id2label[idx_str] = new_label
        if new_label not in new_label2id:
            new_label2id[new_label] = int(idx_str)

    config["id2label"] = new_id2label
    config["label2id"] = new_label2id

    print(f"Remapped labels: {dict(new_id2label)}")

    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ---- Clean up unnecessary files ----
    for pattern in ("optimizer.*", "scheduler.*", "rng_state.*",
                    "scaler.*", "training_args.*", "trainer_state.*"):
        for f in TARGET_DIR.glob(pattern):
            f.unlink()
    # Remove checkpoint subdirectories
    for d in TARGET_DIR.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            import shutil
            shutil.rmtree(d)
            print(f"  Removed checkpoint: {d.name}")

    print(f"Filler classifier saved to: {TARGET_DIR}")

    # ---- Final validation ----
    labels = set(new_id2label.values())
    if "FILLER" not in labels:
        print("WARNING: FILLER label not found in saved config!")
        return 1
    if not ({"O", "CONTEXT_WORD"} & labels):
        print("WARNING: No O or CONTEXT_WORD label found!")
        return 1

    has_weights = any(TARGET_DIR.glob("*.safetensors")) or (TARGET_DIR / "pytorch_model.bin").is_file()
    if not has_weights:
        print("WARNING: No model weights found!")
        return 1

    print("Validation passed -- filler classifier is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
