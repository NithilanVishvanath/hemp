"""Value classification and column type inference."""

import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from hemp.hashing import canonical_json
from hemp.profile.schema import ColumnType

__all__ = [
    "Value",
    "ValueKind",
    "canonical_json",
    "classify",
    "infer_column_type",
    "is_categorical",
]

_INT_RE = re.compile(r"[+-]?\d+")
_FLOAT_RE = re.compile(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?")

# A string column is categorical when it repeats a small set of short values.
MAX_CATEGORIES = 50
MAX_CATEGORICAL_UNIQUE_RATIO = 0.5
MAX_CATEGORY_LENGTH = 100


class ValueKind(StrEnum):
    MISSING = "missing"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    OBJECT = "object"
    ARRAY = "array"


@dataclass(frozen=True)
class Value:
    kind: ValueKind
    value: Any  # parsed value: "3" -> 3, '{"a": 1}' -> {"a": 1}
    text: str  # the value as text; used for length stats
    from_json_string: bool = False


def classify(raw: Any) -> Value:
    """Classify one cell. Strings are parsed too, since CSV cells are always strings."""
    if raw is None:
        return Value(ValueKind.MISSING, None, "")
    if isinstance(raw, bool):
        return Value(ValueKind.BOOLEAN, raw, str(raw).lower())
    if isinstance(raw, int):
        return Value(ValueKind.INTEGER, raw, str(raw))
    if isinstance(raw, float):
        if math.isnan(raw):
            return Value(ValueKind.MISSING, None, "")
        return Value(ValueKind.FLOAT, raw, str(raw))
    if isinstance(raw, dict):
        return Value(ValueKind.OBJECT, raw, canonical_json(raw))
    if isinstance(raw, list):
        return Value(ValueKind.ARRAY, raw, canonical_json(raw))

    text = str(raw)
    s = text.strip()
    if not s:
        return Value(ValueKind.MISSING, None, "")
    if s.lower() in ("true", "false"):
        return Value(ValueKind.BOOLEAN, s.lower() == "true", text)
    if _INT_RE.fullmatch(s):
        return Value(ValueKind.INTEGER, int(s), text)
    if _FLOAT_RE.fullmatch(s):
        return Value(ValueKind.FLOAT, float(s), text)
    if s[0] in "{[":
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            pass
        else:
            kind = ValueKind.OBJECT if isinstance(parsed, dict) else ValueKind.ARRAY
            return Value(kind, parsed, text, from_json_string=True)
    return Value(ValueKind.STRING, text, text)


def infer_column_type(values: Iterable[Value]) -> ColumnType:
    """Type from the kinds present. Scalars mixed with plain strings read as text."""
    kinds = {v.kind for v in values} - {ValueKind.MISSING}
    structured = {ValueKind.OBJECT, ValueKind.ARRAY}
    if not kinds:
        return ColumnType.EMPTY
    if kinds == {ValueKind.BOOLEAN}:
        return ColumnType.BOOLEAN
    if kinds == {ValueKind.INTEGER}:
        return ColumnType.INTEGER
    if kinds <= {ValueKind.INTEGER, ValueKind.FLOAT}:
        return ColumnType.FLOAT
    if kinds <= structured:
        return ColumnType.JSON
    if kinds & structured:
        return ColumnType.MIXED
    return ColumnType.TEXT  # the profiler refines this to CATEGORICAL via is_categorical()


def is_categorical(n_unique: int, n_present: int, max_length: int) -> bool:
    return (
        n_present > 0
        and n_unique <= MAX_CATEGORIES
        and n_unique / n_present <= MAX_CATEGORICAL_UNIQUE_RATIO
        and max_length <= MAX_CATEGORY_LENGTH
    )
