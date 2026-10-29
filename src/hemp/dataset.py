"""Load datasets (JSONL, JSON, CSV, TSV) into a list of row dicts."""

import csv
import io
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

Row = dict[str, Any]


class DatasetFormat(StrEnum):
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"


_EXTENSIONS = {
    ".jsonl": DatasetFormat.JSONL,
    ".ndjson": DatasetFormat.JSONL,
    ".json": DatasetFormat.JSON,
    ".csv": DatasetFormat.CSV,
    ".tsv": DatasetFormat.TSV,
}


class DatasetError(ValueError):
    """The dataset file can't be read as a table of rows."""


def detect_format(path: Path) -> DatasetFormat:
    try:
        return _EXTENSIONS[path.suffix.lower()]
    except KeyError:
        supported = ", ".join(sorted(_EXTENSIONS))
        raise DatasetError(
            f"Don't know how to read '{path.name}'. Use one of these extensions: {supported}."
        ) from None


def load_dataset(path: str | Path) -> tuple[list[Row], DatasetFormat]:
    """Read a dataset file. Each row must be a JSON object or a table row."""
    path = Path(path)
    fmt = detect_format(path)
    if not path.is_file():
        raise DatasetError(f"Dataset file not found: {path}")
    text = path.read_text(encoding="utf-8-sig")
    if fmt is DatasetFormat.JSONL:
        rows = _parse_jsonl(text, path)
    elif fmt is DatasetFormat.JSON:
        rows = _parse_json(text, path)
    else:
        rows = _parse_delimited(text, "\t" if fmt is DatasetFormat.TSV else ",")
    if not rows:
        raise DatasetError(f"'{path.name}' has no rows. Add at least one example.")
    return rows, fmt


def _parse_jsonl(text: str, path: Path) -> list[Row]:
    rows: list[Row] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as e:
            raise DatasetError(f"{path.name}, line {lineno}: invalid JSON ({e.msg}).") from None
        if not isinstance(value, dict):
            raise DatasetError(
                f"{path.name}, line {lineno}: each line must be a JSON object "
                f"(one example per line), got {type(value).__name__}."
            )
        rows.append(value)
    return rows


def _parse_json(text: str, path: Path) -> list[Row]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise DatasetError(f"{path.name}: invalid JSON ({e.msg}, line {e.lineno}).") from None
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise DatasetError(f"{path.name}: expected a JSON array of objects, one per example.")
    return value


def _parse_delimited(text: str, delimiter: str) -> list[Row]:
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    # csv puts extra cells from over-long rows under the key None; drop them.
    return [{k: v for k, v in row.items() if k is not None} for row in reader]
