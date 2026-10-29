"""Build a DatasetProfile from rows. Deterministic: no model calls."""

import hashlib
import statistics
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hemp.dataset import Row, load_dataset
from hemp.profile.infer import (
    Value,
    ValueKind,
    canonical_json,
    classify,
    infer_column_type,
    is_categorical,
)
from hemp.profile.roles import assign_roles
from hemp.profile.schema import (
    ColumnProfile,
    ColumnType,
    DatasetProfile,
    LengthStats,
    NumericStats,
    ValueCount,
)
from hemp.prompt import template_fields

MAX_TOP_VALUES = 10
MAX_UNIQUE_FOR_TOP_VALUES = 20
SAMPLE_VALUE_MAX_CHARS = 500


def profile_dataset(
    path: str | Path, *, prompt: str | None = None, n_samples: int = 5
) -> DatasetProfile:
    """Load a dataset file and profile it. `prompt` is the prompt template text, if known."""
    rows, fmt = load_dataset(path)
    return profile_rows(rows, prompt=prompt, source=str(path), fmt=fmt.value, n_samples=n_samples)


def profile_rows(
    rows: Sequence[Row],
    *,
    prompt: str | None = None,
    source: str | None = None,
    fmt: str | None = None,
    n_samples: int = 5,
) -> DatasetProfile:
    names = list(dict.fromkeys(key for row in rows for key in row))
    columns = [_profile_column(name, [classify(row.get(name)) for row in rows]) for name in names]

    fields = template_fields(prompt) if prompt is not None else None
    columns = assign_roles(columns, fields)

    return DatasetProfile(
        source=source,
        format=fmt,
        n_rows=len(rows),
        dataset_hash=dataset_hash(rows),
        columns=columns,
        sample_rows=[_truncate_row(row) for row in rows[:n_samples]],
        prompt_fields=fields,
        missing_prompt_fields=[f for f in fields or [] if f not in names],
    )


def dataset_hash(rows: Sequence[Row]) -> str:
    """SHA-256 of the row contents. Key order and file details (BOM, line endings) don't matter."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _profile_column(name: str, values: list[Value]) -> ColumnProfile:
    present = [v for v in values if v.kind is not ValueKind.MISSING]
    counts = Counter(canonical_json(v.value) for v in present)
    n_unique = len(counts)

    ctype = infer_column_type(present)
    if ctype is ColumnType.TEXT and is_categorical(
        n_unique, len(present), max(len(v.text) for v in present)
    ):
        ctype = ColumnType.CATEGORICAL

    return ColumnProfile(
        name=name,
        type=ctype,
        n_missing=len(values) - len(present),
        n_unique=n_unique,
        unique_ratio=round(n_unique / len(present), 4) if present else 0.0,
        top_values=_top_values(ctype, present, counts),
        lengths=_length_stats(ctype, present),
        numeric=_numeric_stats(ctype, present),
        json_keys=_json_keys(ctype, present),
        json_in_strings=any(v.from_json_string for v in present),
    )


def _top_values(
    ctype: ColumnType, present: list[Value], counts: Counter[str]
) -> list[ValueCount] | None:
    if ctype in (ColumnType.TEXT, ColumnType.JSON, ColumnType.MIXED, ColumnType.EMPTY):
        return None
    if len(counts) > MAX_UNIQUE_FOR_TOP_VALUES:
        return None
    if len(present) > 1 and len(counts) == len(present):
        return None  # every value is different, so counts say nothing
    first_seen: dict[str, Any] = {}
    for v in present:
        first_seen.setdefault(canonical_json(v.value), v.value)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:MAX_TOP_VALUES]
    return [ValueCount(value=first_seen[key], count=n) for key, n in ranked]


def _length_stats(ctype: ColumnType, present: list[Value]) -> LengthStats | None:
    text_like = (ColumnType.TEXT, ColumnType.CATEGORICAL, ColumnType.JSON, ColumnType.MIXED)
    if ctype not in text_like:
        return None
    chars = [len(v.text) for v in present]
    words = [len(v.text.split()) for v in present]
    return LengthStats(
        min=min(chars),
        max=max(chars),
        mean=round(statistics.fmean(chars), 2),
        median=statistics.median(chars),
        mean_words=round(statistics.fmean(words), 2),
    )


def _numeric_stats(ctype: ColumnType, present: list[Value]) -> NumericStats | None:
    if ctype not in (ColumnType.INTEGER, ColumnType.FLOAT):
        return None
    nums = [float(v.value) for v in present]
    return NumericStats(min=min(nums), max=max(nums), mean=round(statistics.fmean(nums), 4))


def _json_keys(ctype: ColumnType, present: list[Value]) -> dict[str, float] | None:
    if ctype not in (ColumnType.JSON, ColumnType.MIXED):
        return None
    objects = [v.value for v in present if v.kind is ValueKind.OBJECT]
    if not objects:
        return None
    key_counts = Counter(k for obj in objects for k in obj)
    ranked = sorted(key_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {k: round(n / len(objects), 4) for k, n in ranked}


def _truncate_row(row: Row) -> Row:
    """Shorten long strings so sample rows stay small enough for the planner's context."""
    out: Row = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > SAMPLE_VALUE_MAX_CHARS:
            value = value[:SAMPLE_VALUE_MAX_CHARS] + "…"
        out[key] = value
    return out
