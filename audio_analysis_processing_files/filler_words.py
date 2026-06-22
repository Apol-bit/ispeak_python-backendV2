"""Offline contextual filler-word classification.

This module deliberately contains no filler-word dictionary and never calls an
API. It expects a local Hugging Face token-classification model trained with
``FILLER`` and ``CONTEXT_WORD`` labels (``UNCERTAIN`` is optional).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Protocol

MODEL_PATH_ENV = "ISPEAK_FILLER_MODEL_PATH"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "filler_classifier"
MINIMUM_MODEL_CONFIDENCE = 0.60


class ClassifierUnavailable(RuntimeError):
    """Raised when the local classifier cannot be loaded or used."""


class TokenClassifier(Protocol):
    def __call__(self, text: str) -> List[Dict[str, Any]]: ...


class LocalTokenFillerClassifier:
    """Load a token-classification model from a local directory only."""

    def __init__(self, model_path: str | os.PathLike[str]):
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            self.model_path = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ClassifierUnavailable(f"Local filler model not found: {path}") from exc
        if not self.model_path.is_dir():
            raise ClassifierUnavailable(
                f"Local filler model path is not a directory: {self.model_path}"
            )

        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            # local_files_only=True is the hard network boundary. Transformers
            # must not download a model, tokenizer, or configuration.
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                str(self.model_path),
                local_files_only=True,
            )
            self._validate_labels(model.config.id2label)
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1,
            )
        except ClassifierUnavailable:
            raise
        except Exception as exc:
            raise ClassifierUnavailable(
                f"Could not load local filler classifier: {exc}"
            ) from exc

    @staticmethod
    def _validate_labels(id2label: Dict[int, str]) -> None:
        labels = {_normalize_label(label) for label in id2label.values()}
        if "FILLER" not in labels or not ({"CONTEXT_WORD", "O"} & labels):
            raise ClassifierUnavailable(
                "The local model must define FILLER and CONTEXT_WORD (or O) labels"
            )

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        return list(self._pipeline(text))


_classifier_lock = threading.Lock()
_cached_classifier: LocalTokenFillerClassifier | None = None
_cached_model_path: str | None = None


def _normalize_label(label: Any) -> str:
    normalized = str(label or "").strip().upper().replace("-", "_")
    if normalized.startswith("B_") or normalized.startswith("I_"):
        normalized = normalized[2:]
    return normalized


def _default_classifier() -> LocalTokenFillerClassifier:
    global _cached_classifier, _cached_model_path

    configured_path = os.getenv(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)).strip()

    with _classifier_lock:
        if _cached_classifier is None or _cached_model_path != configured_path:
            _cached_classifier = LocalTokenFillerClassifier(configured_path)
            _cached_model_path = configured_path
        return _cached_classifier


def filler_classifier_status() -> Dict[str, Any]:
    """Report local classifier availability without loading model weights."""
    configured = os.getenv(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH)).strip()
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=False)
    has_config = (path / "config.json").is_file()
    has_weights = any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").is_file()
    return {
        "available": path.is_dir() and has_config and has_weights,
        "path": str(path),
        "network_allowed": False,
    }


def _build_transcript(
    word_segments: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Build text while retaining character-to-audio timestamp alignment."""
    text_parts: List[str] = []
    aligned_words: List[Dict[str, Any]] = []
    cursor = 0

    for segment in word_segments:
        word = str(segment.get("text", "")).strip()
        if not word:
            continue
        if text_parts:
            text_parts.append(" ")
            cursor += 1
        char_start = cursor
        text_parts.append(word)
        cursor += len(word)
        aligned_words.append({
            "word": word,
            "char_start": char_start,
            "char_end": cursor,
            "start": _timestamp(segment.get("start")),
            "end": _timestamp(segment.get("end")),
        })

    return "".join(text_parts), aligned_words


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _words_overlapping(
    aligned_words: List[Dict[str, Any]],
    char_start: int,
    char_end: int,
) -> List[Dict[str, Any]]:
    return [
        word for word in aligned_words
        if word["char_start"] < char_end and word["char_end"] > char_start
    ]


def _classification_item(
    prediction: Dict[str, Any],
    aligned_words: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    char_start = int(prediction.get("start", 0))
    char_end = int(prediction.get("end", char_start))
    words = _words_overlapping(aligned_words, char_start, char_end)
    if not words:
        return None
    return {
        "word": " ".join(word["word"] for word in words),
        "start": round(min(word["start"] for word in words), 3),
        "end": round(max(word["end"] for word in words), 3),
        "confidence": round(float(prediction.get("score", 0.0)), 4),
        "reason": "local contextual token-classification model",
    }


def _unavailable_result(message: str, total_words: int) -> Dict[str, Any]:
    return {
        "analysis_available": False,
        "filler_count": 0,
        "filler_rate": 0.0,
        "filler_words": [],
        "filler_candidates": [],
        "filler_score": None,
        "total_words": total_words,
        "message": message,
        "method": "local token-classification model only",
    }


def analyze_fillers(
    word_segments: List[Dict[str, Any]],
    classifier: TokenClassifier | Callable[[str], List[Dict[str, Any]]] | None = None,
    minimum_confidence: float = MINIMUM_MODEL_CONFIDENCE,
    **_: Any,
) -> Dict[str, Any]:
    """Classify words from their full context using a local model.

    No hard-coded word list or contextual rule is used. Predictions below the
    confidence threshold are returned as uncertain candidates and do not lower
    the filler score.
    """
    transcript, aligned_words = _build_transcript(word_segments)
    if not aligned_words:
        return {
            "analysis_available": True,
            "filler_count": 0,
            "filler_rate": 0.0,
            "filler_words": [],
            "filler_candidates": [],
            "filler_score": 100,
            "total_words": 0,
            "message": "No speech detected",
            "method": "local token-classification model only",
        }

    try:
        active_classifier = classifier or _default_classifier()
        predictions = active_classifier(transcript)
    except ClassifierUnavailable as exc:
        return _unavailable_result(str(exc), len(aligned_words))
    except Exception as exc:
        return _unavailable_result(
            f"Local filler classification failed: {exc}",
            len(aligned_words),
        )

    fillers: List[Dict[str, Any]] = []
    uncertain: List[Dict[str, Any]] = []
    seen_ranges = set()
    for prediction in predictions:
        label = _normalize_label(
            prediction.get("entity_group", prediction.get("entity", prediction.get("label")))
        )
        if label not in {"FILLER", "UNCERTAIN"}:
            continue
        item = _classification_item(prediction, aligned_words)
        if item is None:
            continue
        key = (item["start"], item["end"], item["word"].lower())
        if key in seen_ranges:
            continue
        seen_ranges.add(key)
        if label == "FILLER" and item["confidence"] >= minimum_confidence:
            fillers.append(item)
        else:
            item["reason"] = "local model prediction is uncertain"
            uncertain.append(item)

    fillers.sort(key=lambda item: item["start"])
    uncertain.sort(key=lambda item: item["start"])
    filler_count = len(fillers)
    filler_rate = round(filler_count / len(aligned_words) * 100, 1)
    filler_score = round(max(0.0, 100.0 - filler_rate * 4.0))

    return {
        "analysis_available": True,
        "filler_count": filler_count,
        "filler_rate": filler_rate,
        "filler_words": fillers,
        "filler_candidates": uncertain,
        "filler_score": filler_score,
        "total_words": len(aligned_words),
        "message": "No filler words detected" if filler_count == 0 else (
            "Minimal filler-word usage" if filler_rate < 5 else
            "Moderate filler-word usage" if filler_rate < 15 else
            "High filler-word usage"
        ),
        "method": "local contextual token-classification model",
    }
