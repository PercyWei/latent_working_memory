from __future__ import annotations

from typing import Any, Mapping


def document_rejection_reason(record: Mapping[str, Any], min_chars: int) -> str | None:
    """Check source fields and explicit length only; the model judges sample content."""
    for name in ("id", "url"):
        if not isinstance(record[name], str) or not record[name].strip():
            raise ValueError(f"FineWeb {name} must be a non-empty string")
    if not isinstance(record["text"], str):
        raise ValueError("FineWeb text must be a string")
    if len(record["text"].strip()) < min_chars:
        return "too_short"
    return None
