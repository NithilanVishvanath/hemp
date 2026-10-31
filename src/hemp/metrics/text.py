"""Text helpers shared by the built-in metrics."""

import json
import math
import re
import unicodedata
from typing import Any

from hemp.hashing import canonical_json

# A whole-output code block: ```lang\n...``` or ```...```.
_FENCE_RE = re.compile(r"^```(?:[A-Za-z0-9_+-]*[ \t]*\n)?(.*?)\n?[ \t]*```$", re.DOTALL)
# Numbers, with optional thousands separators ("1,234.5") and exponents.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
# Characters LLMs commonly wrap a short answer in: quotes (straight and curly),
# backticks, and markdown emphasis.
_WRAPPERS = " \t\r\n\"'`*_" + "“”‘’"  # noqa: RUF001 - curly quotes on purpose
_ARTICLES = frozenset({"a", "an", "the"})


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
    """Canonical form of a short answer for comparison.

    Unicode-normalizes, removes a surrounding code block, wrapping quotes/markdown,
    and a trailing period, and collapses whitespace. '**Positive.**' -> 'positive'.
    """
    s = unicodedata.normalize("NFKC", strip_code_fence(text))
    s = s.strip(_WRAPPERS).rstrip(".").strip(_WRAPPERS)
    s = " ".join(s.split())
    return s if case_sensitive else s.casefold()


def as_text(value: Any) -> str:
    """Gold values may be non-strings (numbers, dicts) in JSONL; compare them as text."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, dict | list):
        return canonical_json(value)
    return str(value)


def is_missing(value: Any) -> bool:
    """No usable value: None, NaN, or a blank string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def find_numbers(text: str) -> list[float]:
    return [float(m.group().replace(",", "")) for m in _NUMBER_RE.finditer(text)]


def tokens(text: str) -> list[str]:
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def answer_tokens(text: str) -> list[str]:
    """Tokens for answer overlap, without articles (as in SQuAD scoring)."""
    return [t for t in tokens(text) if t not in _ARTICLES]


_MISSING = object()


def get_path(obj: Any, path: str) -> Any:
    """Look up 'a.b.c' in nested dicts. Returns a sentinel when absent (see has_path)."""
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def has_path(obj: Any, path: str) -> bool:
    return get_path(obj, path) is not _MISSING


def same_value(a: Any, b: Any, case_sensitive: bool = False) -> bool:
    """Loose equality for JSON values: strings normalized, numbers by value, recursing."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k], case_sensitive) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            same_value(x, y, case_sensitive) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, str) or isinstance(b, str):
        return normalize(as_text(a), case_sensitive) == normalize(as_text(b), case_sensitive)
    numeric = (int, float)
    if (
        isinstance(a, numeric)
        and isinstance(b, numeric)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return math.isclose(a, b)
    return canonical_json(a) == canonical_json(b)
