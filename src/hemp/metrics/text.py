"""Text helpers shared by the built-in metrics."""

import json
import re
from typing import Any

from hemp.hashing import canonical_json

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def strip_code_fence(text: str) -> str:
    """'```json\\n{...}\\n```' -> '{...}'. Other text is returned stripped."""
    s = text.strip()
    match = _FENCE_RE.match(s)
    return match.group(1).strip() if match else s


def parse_json(text: str, allow_code_fence: bool = True) -> Any:
    """Parse JSON from model output. Raises ValueError with a short reason."""
    s = strip_code_fence(text) if allow_code_fence else text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"not valid JSON ({e.msg} at line {e.lineno}, column {e.colno})") from None


def normalize(text: str, case_sensitive: bool = False) -> str:
    """Trim whitespace, wrapping quotes, and a trailing period; collapse inner whitespace."""
    s = " ".join(text.split())
    s = s.strip("\"'`").rstrip(".").strip()
    return s if case_sensitive else s.lower()


def as_text(value: Any) -> str:
    """Gold values may be non-strings (numbers, dicts) in JSONL; compare them as text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict | list):
        return canonical_json(value)
    return str(value)


def first_number(text: str) -> float | None:
    match = _NUMBER_RE.search(text.replace(",", ""))
    return float(match.group()) if match else None


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())
