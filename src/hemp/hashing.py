"""Stable hashing helpers shared by datasets, prompts, and plans."""

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no extra whitespace, so equal data gives equal text."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
