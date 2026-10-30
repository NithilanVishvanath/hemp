import pytest
from pydantic import Field

import hemp
from hemp.metrics import (
    MetricContext,
    MetricError,
    MetricLayer,
    MetricParams,
    MetricResult,
    catalog,
    get_metric,
    list_metrics,
)

BUILTINS = {
    "non_empty",
    "valid_json",
    "json_keys",
    "allowed_values",
    "length",
    "regex",
    "exact_match",
    "token_f1",
    "numeric_match",
    "json_field_match",
}


class MentionsParams(MetricParams):
    word: str = Field(description="word that must appear")


def test_builtins_registered() -> None:
    assert {m.name for m in list_metrics()} >= BUILTINS
    assert get_metric("exact_match").requires_gold
    assert get_metric("regex").layer is MetricLayer.FORMAT


def test_custom_metric_via_public_decorator() -> None:
    @hemp.metric("mentions", params=MentionsParams)
    def mentions(ctx: MetricContext, params: MentionsParams) -> MetricResult:
        """The output mentions a word."""
        return MetricResult.check(params.word in ctx.output)

    info = get_metric("mentions")
    assert info.layer is MetricLayer.CUSTOM
    assert info.description == "The output mentions a word."
    assert info(MetricContext(output="hello world"), {"word": "world"}).passed
    assert not info(MetricContext(output="hello"), {"word": "world"}).passed


def test_duplicate_name_rejected() -> None:
    with pytest.raises(MetricError, match="already registered"):

        @hemp.metric("exact_match")
        def other(ctx: MetricContext, params: MetricParams) -> MetricResult:
            return MetricResult.check(True)


def test_unknown_metric_lists_available() -> None:
    with pytest.raises(MetricError, match=r"Available metrics: .*exact_match"):
        get_metric("nope")


def test_param_validation_messages() -> None:
    info = get_metric("length")
    with pytest.raises(MetricError, match="at least one of"):
        info.validate_params({})
    with pytest.raises(MetricError, match="Extra inputs"):
        info.validate_params({"max": 5, "maxx": 3})
    with pytest.raises(MetricError, match="Expected params: unit"):
        info.validate_params({"unit": "pages", "max": 1})


def test_catalog_shape() -> None:
    entries = {e["metric"]: e for e in catalog(MetricLayer.FORMAT)}
    assert "exact_match" not in entries
    regex = entries["regex"]
    assert regex["layer"] == "format"
    assert "required" in regex["params"]["pattern"]
    assert regex["params"]["should_match"].startswith("bool = True")
    assert entries["length"]["params"]["max"].startswith("int | null = None")
