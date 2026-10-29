from pathlib import Path

import pytest

from hemp.dataset import DatasetError, DatasetFormat, load_dataset


def write(tmp_path: Path, name: str, content: str, encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding=encoding)
    return path


def test_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = write(tmp_path, "d.jsonl", '{"a": 1}\n\n{"a": 2, "b": "x"}\n')
    rows, fmt = load_dataset(path)
    assert fmt is DatasetFormat.JSONL
    assert rows == [{"a": 1}, {"a": 2, "b": "x"}]


def test_jsonl_invalid_line_reports_line_number(tmp_path: Path) -> None:
    path = write(tmp_path, "d.jsonl", '{"a": 1}\n{oops\n')
    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(path)


def test_jsonl_non_object_line(tmp_path: Path) -> None:
    path = write(tmp_path, "d.jsonl", "[1, 2]\n")
    with pytest.raises(DatasetError, match="JSON object"):
        load_dataset(path)


def test_json_array(tmp_path: Path) -> None:
    path = write(tmp_path, "d.json", '[{"a": 1}, {"a": 2}]')
    rows, fmt = load_dataset(path)
    assert fmt is DatasetFormat.JSON
    assert len(rows) == 2


def test_json_must_be_array_of_objects(tmp_path: Path) -> None:
    path = write(tmp_path, "d.json", '{"a": 1}')
    with pytest.raises(DatasetError, match="array of objects"):
        load_dataset(path)


def test_csv_with_bom_and_quoted_newlines(tmp_path: Path) -> None:
    path = write(tmp_path, "d.csv", 'text,label\n"hello\nworld",pos\nbye,neg\n', "utf-8-sig")
    rows, fmt = load_dataset(path)
    assert fmt is DatasetFormat.CSV
    assert rows == [{"text": "hello\nworld", "label": "pos"}, {"text": "bye", "label": "neg"}]


def test_tsv(tmp_path: Path) -> None:
    path = write(tmp_path, "d.tsv", "a\tb\n1\t2\n")
    rows, fmt = load_dataset(path)
    assert fmt is DatasetFormat.TSV
    assert rows == [{"a": "1", "b": "2"}]


def test_unknown_extension(tmp_path: Path) -> None:
    path = write(tmp_path, "d.parquet", "")
    with pytest.raises(DatasetError, match=r"\.jsonl"):
        load_dataset(path)


def test_empty_dataset(tmp_path: Path) -> None:
    path = write(tmp_path, "d.jsonl", "\n\n")
    with pytest.raises(DatasetError, match="no rows"):
        load_dataset(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.csv")
