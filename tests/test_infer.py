import math

import pytest

from hemp.profile.infer import ValueKind, classify, infer_column_type, is_categorical
from hemp.profile.schema import ColumnType


@pytest.mark.parametrize(
    ("raw", "kind", "value"),
    [
        (None, ValueKind.MISSING, None),
        ("", ValueKind.MISSING, None),
        ("   ", ValueKind.MISSING, None),
        (math.nan, ValueKind.MISSING, None),
        (True, ValueKind.BOOLEAN, True),
        ("FALSE", ValueKind.BOOLEAN, False),
        (3, ValueKind.INTEGER, 3),
        (" -12 ", ValueKind.INTEGER, -12),
        (2.5, ValueKind.FLOAT, 2.5),
        ("1e3", ValueKind.FLOAT, 1000.0),
        (".5", ValueKind.FLOAT, 0.5),
        ({"a": 1}, ValueKind.OBJECT, {"a": 1}),
        ([1, 2], ValueKind.ARRAY, [1, 2]),
        ('{"a": 1}', ValueKind.OBJECT, {"a": 1}),
        ("[1, 2]", ValueKind.ARRAY, [1, 2]),
        ("{not json", ValueKind.STRING, "{not json"),
        ("hello", ValueKind.STRING, "hello"),
    ],
)
def test_classify(raw: object, kind: ValueKind, value: object) -> None:
    v = classify(raw)
    assert v.kind is kind
    assert v.value == value


def test_json_string_flag() -> None:
    assert classify('{"a": 1}').from_json_string
    assert not classify({"a": 1}).from_json_string


def types(*raws: object) -> ColumnType:
    return infer_column_type(classify(r) for r in raws)


def test_column_types() -> None:
    assert types(None, "") is ColumnType.EMPTY
    assert types(True, "false", None) is ColumnType.BOOLEAN
    assert types(1, "2") is ColumnType.INTEGER
    assert types(1, 2.5) is ColumnType.FLOAT
    assert types({"a": 1}, "[1]") is ColumnType.JSON
    assert types({"a": 1}, "text") is ColumnType.MIXED
    assert types("a", 1, True) is ColumnType.TEXT


def test_is_categorical() -> None:
    assert is_categorical(n_unique=3, n_present=100, max_length=10)
    assert not is_categorical(n_unique=60, n_present=1000, max_length=10)  # too many values
    assert not is_categorical(n_unique=8, n_present=10, max_length=10)  # mostly unique
    assert not is_categorical(n_unique=2, n_present=100, max_length=500)  # too long
    assert not is_categorical(n_unique=0, n_present=0, max_length=0)
