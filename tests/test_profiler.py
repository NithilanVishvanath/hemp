from pathlib import Path

from hemp.dataset import Row
from hemp.profile import ColumnRole, ColumnType, dataset_hash, profile_dataset, profile_rows
from hemp.profile.profiler import SAMPLE_VALUE_MAX_CHARS
from hemp.profile.roles import name_tokens

REVIEWS = [
    "Great product, works exactly as described and arrived early.",
    "Terrible. Broke after two days and support never replied.",
    "It's fine. Nothing special but does the job.",
    "Absolutely love it, would buy again without hesitation.",
    "Not worth the money at all, very disappointed.",
    "Decent quality for the price, a bit noisy though.",
]
LABELS = ["positive", "negative", "neutral", "positive", "negative", "neutral"]


def classification_rows() -> list[Row]:
    return [
        {"id": i, "review": text, "label": label}
        for i, (text, label) in enumerate(zip(REVIEWS, LABELS, strict=True))
    ]


def test_types_and_stats() -> None:
    p = profile_rows(classification_rows())
    assert p.n_rows == 6
    assert [c.name for c in p.columns] == ["id", "review", "label"]

    review = p.column("review")
    assert review.type is ColumnType.TEXT
    assert review.top_values is None
    assert review.lengths is not None
    assert review.lengths.min == min(len(r) for r in REVIEWS)
    assert review.lengths.max == max(len(r) for r in REVIEWS)

    label = p.column("label")
    assert label.type is ColumnType.CATEGORICAL
    assert label.n_unique == 3
    assert label.top_values is not None
    assert {v.value: v.count for v in label.top_values} == {
        "positive": 2,
        "negative": 2,
        "neutral": 2,
    }

    ident = p.column("id")
    assert ident.type is ColumnType.INTEGER
    assert ident.numeric is not None
    assert (ident.numeric.min, ident.numeric.max) == (0, 5)


def test_roles_without_prompt_use_names() -> None:
    p = profile_rows(classification_rows())
    assert p.column("id").role is ColumnRole.ID
    assert p.column("review").role is ColumnRole.INPUT
    assert p.gold_columns == ["label"]
    assert p.prompt_fields is None


def test_roles_with_prompt() -> None:
    p = profile_rows(classification_rows(), prompt="Classify the sentiment: {review}")
    assert p.input_columns == ["review"]
    assert p.gold_columns == ["label"]
    assert p.prompt_fields == ["review"]
    assert p.missing_prompt_fields == []


def test_single_unused_column_becomes_gold() -> None:
    rows = [{"q": f"question {i}", "a": f"some answer number {i}"} for i in range(5)]
    p = profile_rows(rows, prompt="Answer: {q}")
    assert p.gold_columns == ["a"]
    assert "only column" in p.column("a").role_reason


def test_prompt_placeholder_wins_over_gold_name() -> None:
    rows = [{"answer": "x", "question": "y"}]
    p = profile_rows(rows, prompt="{question} draft answer: {answer}")
    assert p.gold_columns == []
    assert set(p.input_columns) == {"question", "answer"}


def test_missing_prompt_fields() -> None:
    p = profile_rows([{"text": "hi"}], prompt="{text} {context}")
    assert p.missing_prompt_fields == ["context"]


def test_no_gold_when_nothing_matches() -> None:
    rows = [{"text": f"t{i}", "source": "web" if i % 2 else "app"} for i in range(4)]
    p = profile_rows(rows)
    assert p.gold_columns == []
    assert p.column("source").role is ColumnRole.UNKNOWN


def test_id_must_be_unique() -> None:
    rows = [{"user_id": 1}, {"user_id": 1}]
    assert profile_rows(rows).column("user_id").role is not ColumnRole.ID


def test_json_column_native_and_string() -> None:
    rows = [
        {"native": {"name": "a", "age": 1}, "encoded": '{"name": "a"}'},
        {"native": {"name": "b"}, "encoded": '{"name": "b", "tags": []}'},
    ]
    p = profile_rows(rows)
    native, encoded = p.column("native"), p.column("encoded")
    assert native.type is ColumnType.JSON
    assert native.json_keys == {"name": 1.0, "age": 0.5}
    assert not native.json_in_strings
    assert encoded.type is ColumnType.JSON
    assert encoded.json_in_strings
    assert encoded.json_keys == {"name": 1.0, "tags": 0.5}


def test_missing_values_and_ragged_rows() -> None:
    rows = [{"a": "x", "b": 1}, {"a": ""}, {"a": None, "b": 2, "c": "late"}]
    p = profile_rows(rows)
    assert [c.name for c in p.columns] == ["a", "b", "c"]
    assert p.column("a").n_missing == 2
    assert p.column("b").n_missing == 1
    assert p.column("c").n_missing == 2


def test_empty_column() -> None:
    p = profile_rows([{"x": None, "label": ""}, {"x": "", "label": None}])
    assert p.column("x").type is ColumnType.EMPTY
    assert p.column("label").role is ColumnRole.METADATA  # an empty column can't be gold


def test_sample_rows_truncated_and_limited() -> None:
    rows = [{"text": "x" * 1000}] * 10
    p = profile_rows(rows, n_samples=3)
    assert len(p.sample_rows) == 3
    assert len(p.sample_rows[0]["text"]) == SAMPLE_VALUE_MAX_CHARS + 1


def test_hash_ignores_key_order_but_not_values() -> None:
    assert dataset_hash([{"a": 1, "b": 2}]) == dataset_hash([{"b": 2, "a": 1}])
    assert dataset_hash([{"a": 1}]) != dataset_hash([{"a": 2}])


def test_profile_is_deterministic() -> None:
    rows = classification_rows()
    assert profile_rows(rows) == profile_rows(rows)


def test_csv_file_matches_jsonl_types(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text(
        "id,score,flag,label\n1,0.5,true,yes\n2,1.5,false,no\n3,2,true,yes\n4,3,false,no\n"
    )
    p = profile_dataset(csv_path)
    assert p.format == "csv"
    assert p.source == str(csv_path)
    assert p.column("id").type is ColumnType.INTEGER
    assert p.column("score").type is ColumnType.FLOAT
    assert p.column("flag").type is ColumnType.BOOLEAN
    assert p.column("label").type is ColumnType.CATEGORICAL


def test_profile_round_trips_through_json() -> None:
    p = profile_rows(classification_rows(), prompt="{review}")
    assert type(p).model_validate_json(p.model_dump_json()) == p


def test_name_tokens() -> None:
    assert name_tokens("expectedOutput") == ["expected", "output"]
    assert name_tokens("Ground Truth") == ["ground", "truth"]
    assert name_tokens("gold_label_v2") == ["gold", "label", "v2"]
