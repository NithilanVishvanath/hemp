from typing import Any

import pytest

from hemp.metrics import MetricContext, get_metric


def run(name: str, output: str, gold: Any = None, **params: Any) -> tuple[bool, float, str]:
    r = get_metric(name)(MetricContext(output=output, gold=gold), params)
    return r.passed, r.score, r.detail


def passed(name: str, output: str, gold: Any = None, **params: Any) -> bool:
    return run(name, output, gold, **params)[0]


def test_non_empty() -> None:
    assert passed("non_empty", "x")
    assert not passed("non_empty", "  \n")


def test_valid_json_with_and_without_fence() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert passed("valid_json", fenced)
    assert not passed("valid_json", fenced, allow_code_fence=False)
    assert not passed("valid_json", "{a: 1}")


def test_json_keys() -> None:
    assert passed("json_keys", '{"a": 1, "b": 2}', required=["a"])
    assert not passed("json_keys", '{"b": 2}', required=["a"])
    assert not passed("json_keys", '{"a": 1, "b": 2}', required=["a"], allow_extra=False)
    assert "expected a JSON object" in run("json_keys", "[1]", required=["a"])[2]


@pytest.mark.parametrize("output", ["positive", " Positive.", '"positive"'])
def test_allowed_values_tolerates_case_quotes_period(output: str) -> None:
    assert passed("allowed_values", output, values=["positive", "negative"])


def test_allowed_values_rejects_extra_text() -> None:
    assert not passed("allowed_values", "The sentiment is positive", values=["positive"])
    assert not passed("allowed_values", "Positive", values=["positive"], case_sensitive=True)


def test_length() -> None:
    assert passed("length", "one two three", max=3)
    assert not passed("length", "one two three four", max=3)
    assert not passed("length", "ab", unit="chars", min=3)
    assert passed("length", "a\nb", unit="lines", min=2, max=2)


def test_regex() -> None:
    assert passed("regex", "Order #123", pattern=r"#\d+")
    assert not passed(
        "regex", "As an AI model", pattern="as an ai", ignore_case=True, should_match=False
    )


def test_exact_match() -> None:
    assert passed("exact_match", "Positive.", gold="positive")
    assert passed("exact_match", "3", gold=3)
    assert passed("exact_match", "true", gold=True)
    assert not passed("exact_match", "negative", gold="positive")


def test_token_f1() -> None:
    ok, score, _ = run("token_f1", "the cat sat", gold="the cat sat down")
    assert ok and score == pytest.approx(2 * 1 * 0.75 / 1.75)
    assert run("token_f1", "dog", gold="cat")[1] == 0.0
    assert not passed("token_f1", "the cat", gold="the cat sat on the mat", threshold=0.9)


def test_numeric_match() -> None:
    assert passed("numeric_match", "The answer is 1,234.0", gold=1234)
    assert not passed("numeric_match", "about 105", gold="100")
    assert passed("numeric_match", "about 105", gold="100", rel_tol=0.05)
    assert "no number" in run("numeric_match", "none", gold=1)[2]


def test_json_field_match() -> None:
    gold = {"name": "Ada", "age": 36}
    ok, score, detail = run("json_field_match", '{"name": "ada", "age": 35}', gold=gold)
    assert not ok and score == 0.5 and "age" in detail
    assert passed("json_field_match", '{"name": "Ada", "age": 36.0, "x": 1}', gold=gold)
    assert passed(
        "json_field_match", '{"name": "Ada"}', gold='{"name": "Ada", "age": 1}', fields=["name"]
    )
    assert not passed("json_field_match", "not json", gold=gold)
