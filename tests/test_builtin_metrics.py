"""Edge cases for the built-in metrics. Outputs mimic how LLMs actually answer."""

import math
from typing import Any

import pytest

from hemp.metrics import MetricContext, MetricError, MetricResult, get_metric
from hemp.metrics.embeddings import Embedder, embedding_backend
from hemp.metrics.text import find_numbers, normalize, strip_code_fence


def result(name: str, output: str, gold: Any = None, **params: Any) -> MetricResult:
    return get_metric(name)(MetricContext(output=output, gold=gold), params)


def passed(name: str, output: str, gold: Any = None, **params: Any) -> bool | None:
    return result(name, output, gold, **params).passed


# --- text helpers -------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ("```JSON\n{}\n```", "{}"),
        ('```{"a": 1}```', '{"a": 1}'),
        ("```\npositive\n```", "positive"),
        ("  plain  ", "plain"),
        ("text before ```json\n{}\n```", "text before ```json\n{}\n```"),  # not a whole block
    ],
)
def test_strip_code_fence(raw: str, expected: str) -> None:
    assert strip_code_fence(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Positive", "positive"),
        ("**Positive**", "positive"),
        ("_positive_", "positive"),
        ("“positive”", "positive"),
        ("'positive'.", "positive"),
        ("`positive`", "positive"),
        ("positive.\n", "positive"),
        ("very   positive", "very positive"),
        ("Ｐositive", "positive"),  # noqa: RUF001 - full-width letter, NFKC-normalized
        ("Straße", "strasse"),  # casefold, not just lower
    ],
)
def test_normalize(raw: str, expected: str) -> None:
    assert normalize(raw) == expected


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("1,234.5 and -7", [1234.5, -7.0]),
        ("values 1,2", [1.0, 2.0]),  # a list, not "12"
        ("12,34", [12.0, 34.0]),  # not a thousands separator
        ("1e3 or .5 or +2", [1000.0, 0.5, 2.0]),
        ("v2 x3", []),  # digits glued to words aren't numbers
        ("none", []),
    ],
)
def test_find_numbers(text: str, numbers: list[float]) -> None:
    assert find_numbers(text) == numbers


# --- format checks ------------------------------------------------------------


def test_non_empty() -> None:
    assert passed("non_empty", "x")
    assert not passed("non_empty", "")
    assert not passed("non_empty", " \n\t")


def test_valid_json() -> None:
    assert passed("valid_json", '{"a": [1, 2]}')
    assert passed("valid_json", "[]")
    assert passed("valid_json", "```json\n{}\n```")
    assert not passed("valid_json", "```json\n{}\n```", allow_code_fence=False)
    assert not passed("valid_json", "{'a': 1}")  # single quotes aren't JSON
    assert not passed("valid_json", 'Here you go: {"a": 1}')
    assert not passed("valid_json", "")
    assert "line 1" in result("valid_json", "{a").detail


def test_json_keys() -> None:
    assert passed("json_keys", '{"a": 1, "b": 2}', required=["a"])
    assert not passed("json_keys", '{"b": 2}', required=["a"])
    assert passed("json_keys", '{"a": null}', required=["a"])  # present, even if null
    assert not passed("json_keys", '{"a": 1, "b": 2}', required=["a"], allow_extra=False)
    assert "expected a JSON object" in result("json_keys", "[1]", required=["a"]).detail


def test_json_keys_nested() -> None:
    out = '{"address": {"city": "Pune", "zip": "411001"}, "name": "x"}'
    assert passed("json_keys", out, required=["address.city", "name"])
    assert not passed("json_keys", out, required=["address.country"])
    assert not passed("json_keys", '{"address": "Pune"}', required=["address.city"])
    assert passed("json_keys", out, required=["address.city", "name"], allow_extra=False)


@pytest.mark.parametrize(
    "output", ["positive", " Positive.", '"positive"', "**Positive**", "“positive”"]
)
def test_allowed_values_accepts_common_wrapping(output: str) -> None:
    assert passed("allowed_values", output, values=["positive", "negative"])


@pytest.mark.parametrize(
    "output",
    [
        "The sentiment is positive",
        "positive\n\nThe review praises the product.",
        "positive, negative",
        "",
    ],
)
def test_allowed_values_rejects_extra_text(output: str) -> None:
    assert not passed("allowed_values", output, values=["positive", "negative"])


def test_allowed_values_case_sensitive() -> None:
    assert not passed("allowed_values", "Positive", values=["positive"], case_sensitive=True)
    assert passed("allowed_values", "Positive", values=["Positive"], case_sensitive=True)


def test_length() -> None:
    assert passed("length", "one two three", max=3)
    assert not passed("length", "one two three four", max=3)
    assert passed("length", "  one   two  ", min=2, max=2)  # whitespace-insensitive
    assert not passed("length", "ab", unit="chars", min=3)
    assert passed("length", "a\nb", unit="lines", min=2, max=2)
    assert not passed("length", "", min=1)
    assert result("length", "one two", max=5).detail == "2 words"


def test_length_param_validation() -> None:
    with pytest.raises(MetricError, match="at least one"):
        get_metric("length").validate_params({"unit": "words"})
    with pytest.raises(MetricError, match="must not exceed"):
        get_metric("length").validate_params({"min": 5, "max": 2})


def test_regex() -> None:
    assert passed("regex", "Order #123", pattern=r"#\d+")
    assert not passed("regex", "Order 123", pattern=r"#\d+")
    assert passed("regex", "fine", pattern="as an ai", should_match=False)
    assert not passed(
        "regex", "As an AI model", pattern="as an ai", ignore_case=True, should_match=False
    )
    with pytest.raises(MetricError, match="invalid regular expression"):
        get_metric("regex").validate_params({"pattern": "(unclosed"})


# --- reference metrics ------------------------------------------------------------


@pytest.mark.parametrize("gold", [None, "", "   ", math.nan])
def test_missing_gold_is_skipped_not_failed(gold: Any) -> None:
    for name in ("exact_match", "token_f1", "numeric_match", "json_field_match"):
        r = result(name, "anything", gold=gold)
        assert r.skipped, name
        assert r.score is None


def test_skipped_and_scored_results_are_consistent() -> None:
    assert MetricResult.skip("n/a").skipped
    with pytest.raises(ValueError):
        MetricResult(passed=True, score=None)
    with pytest.raises(ValueError):
        MetricResult(passed=True, score=1.5)


def test_exact_match() -> None:
    assert passed("exact_match", "Positive.", gold="positive")
    assert passed("exact_match", "**positive**", gold="positive")
    assert passed("exact_match", "3", gold=3)
    assert passed("exact_match", "true", gold=True)
    assert not passed("exact_match", "negative", gold="positive")
    assert not passed("exact_match", "Positive", gold="positive", case_sensitive=True)
    assert "expected 'positive'" in result("exact_match", "no", gold="positive").detail


def test_token_f1() -> None:
    r = result("token_f1", "the cat sat", gold="the cat sat down")
    assert r.passed and r.score == pytest.approx(0.8)  # articles ignored: P=1, R=2/3
    assert result("token_f1", "a cat", gold="the cat").score == 1.0
    assert result("token_f1", "dog", gold="cat").score == 0.0
    assert result("token_f1", "Cat!", gold="cat").score == 1.0  # case and punctuation
    assert result("token_f1", "", gold="cat").score == 0.0
    assert not passed("token_f1", "cat", gold="the cat sat on the mat", threshold=0.9)


def test_numeric_match() -> None:
    assert passed("numeric_match", "1,234.0", gold=1234)
    assert passed("numeric_match", "42", gold="42.0")
    assert passed("numeric_match", "-3.5", gold=-3.5)
    assert not passed("numeric_match", "about 105", gold="100")
    assert passed("numeric_match", "about 105", gold="100", rel_tol=0.05)
    assert passed("numeric_match", "101", gold=100, abs_tol=1)
    assert "no number" in result("numeric_match", "none", gold=1).detail
    assert "not a number" in result("numeric_match", "1", gold="n/a").detail


def test_numeric_match_picks_last_number_by_default() -> None:
    out = "In 2023 we sold 1,500 units, so the answer is 42."
    assert passed("numeric_match", out, gold=42)
    assert not passed("numeric_match", out, gold=42, pick="first")
    assert passed("numeric_match", out, gold=2023, pick="first")


def test_json_field_match() -> None:
    gold = {"name": "Ada", "age": 36}
    r = result("json_field_match", '{"name": "ada", "age": 35}', gold=gold)
    assert not r.passed and r.score == 0.5 and "age" in r.detail
    assert passed("json_field_match", '{"name": "Ada", "age": 36.0, "x": 1}', gold=gold)
    assert passed(
        "json_field_match", "```json\n" + '{"name": "Ada", "age": 36}' + "\n```", gold=gold
    )
    assert not passed("json_field_match", '{"name": "Ada"}', gold=gold)  # missing field
    assert not passed("json_field_match", "not json", gold=gold)
    assert not passed("json_field_match", "[1]", gold=gold)
    assert "gold value is not" in result("json_field_match", "{}", gold="oops").detail


def test_json_field_match_selected_and_nested_fields() -> None:
    gold = '{"person": {"name": "Ada", "tags": ["a", "B"]}, "id": 1}'
    out = '{"person": {"name": "ADA", "tags": ["A", "b"]}, "id": 2}'
    assert passed("json_field_match", out, gold=gold, fields=["person"])  # nested, case-insensitive
    assert passed("json_field_match", out, gold=gold, fields=["person.name"])
    assert not passed("json_field_match", out, gold=gold, fields=["id"])
    assert not passed("json_field_match", out, gold=gold, fields=["person"], case_sensitive=True)
    assert not passed("json_field_match", out, gold=gold, fields=["person.missing"])


def test_json_field_match_list_order_matters() -> None:
    assert not passed("json_field_match", '{"t": ["b", "a"]}', gold={"t": ["a", "b"]})


# --- semantic similarity (fake embeddings; real model in integration tests) -------


def fake_backend(vectors: dict[str, list[float]], calls: list[list[str]]) -> Any:
    def backend(model: str) -> Embedder:
        def embed(texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return [vectors[t] for t in texts]

        return embed

    return backend


def test_semantic_similarity_scores_cosine() -> None:
    vectors = {"Paris": [1.0, 0.0], "The capital is Paris": [0.8, 0.6], "London": [0.0, 1.0]}
    with embedding_backend(fake_backend(vectors, [])):
        close = result("semantic_similarity", "The capital is Paris", gold="Paris")
        far = result("semantic_similarity", "London", gold="Paris")
    assert close.passed and close.score == pytest.approx(0.8)
    assert not far.passed and far.score == 0.0


def test_semantic_similarity_clamps_negative_and_rejects_empty() -> None:
    vectors = {"yes": [1.0, 0.0], "no": [-1.0, 0.0]}
    with embedding_backend(fake_backend(vectors, [])):
        assert result("semantic_similarity", "no", gold="yes").score == 0.0
        assert result("semantic_similarity", "  ", gold="yes").detail == "empty output"


def test_semantic_similarity_caches_embeddings() -> None:
    calls: list[list[str]] = []
    vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    with embedding_backend(fake_backend(vectors, calls)):
        result("semantic_similarity", "a", gold="b")
        result("semantic_similarity", "b", gold="a")
    assert calls == [["a", "b"]]


def test_semantic_similarity_threshold_param() -> None:
    vectors = {"x": [1.0, 0.0], "y": [0.8, 0.6]}
    with embedding_backend(fake_backend(vectors, [])):
        assert not passed("semantic_similarity", "x", gold="y", threshold=0.9)
