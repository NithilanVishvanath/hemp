"""Deterministic format checks on the output alone."""

import re
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from hemp.metrics.registry import MetricContext, MetricLayer, MetricParams, MetricResult, metric
from hemp.metrics.text import normalize, parse_json


@metric("non_empty", layer=MetricLayer.FORMAT)
def non_empty(ctx: MetricContext, params: MetricParams) -> MetricResult:
    """The output is not blank."""
    return MetricResult.check(
        bool(ctx.output.strip()), "" if ctx.output.strip() else "empty output"
    )


class JsonParams(MetricParams):
    allow_code_fence: bool = Field(True, description="accept JSON wrapped in a ``` code block")


@metric("valid_json", layer=MetricLayer.FORMAT, params=JsonParams)
def valid_json(ctx: MetricContext, params: JsonParams) -> MetricResult:
    """The output parses as JSON."""
    try:
        parse_json(ctx.output, params.allow_code_fence)
    except ValueError as e:
        return MetricResult.check(False, str(e))
    return MetricResult.check(True)


class JsonKeysParams(JsonParams):
    required: list[str] = Field(min_length=1, description="keys the JSON object must contain")
    allow_extra: bool = Field(True, description="allow keys not listed in `required`")


@metric("json_keys", layer=MetricLayer.FORMAT, params=JsonKeysParams)
def json_keys(ctx: MetricContext, params: JsonKeysParams) -> MetricResult:
    """The output is a JSON object with the required keys."""
    try:
        value = parse_json(ctx.output, params.allow_code_fence)
    except ValueError as e:
        return MetricResult.check(False, str(e))
    if not isinstance(value, dict):
        return MetricResult.check(False, f"expected a JSON object, got {type(value).__name__}")
    problems = []
    missing = [k for k in params.required if k not in value]
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")
    if not params.allow_extra:
        extra = [k for k in value if k not in params.required]
        if extra:
            problems.append(f"unexpected keys: {', '.join(extra)}")
    return MetricResult.check(not problems, "; ".join(problems))


class AllowedValuesParams(MetricParams):
    values: list[str] = Field(min_length=1, description="the allowed outputs, e.g. a label set")
    case_sensitive: bool = False


@metric("allowed_values", layer=MetricLayer.FORMAT, params=AllowedValuesParams)
def allowed_values(ctx: MetricContext, params: AllowedValuesParams) -> MetricResult:
    """The whole output is one of a fixed set of values (ignoring quotes, trailing period)."""
    got = normalize(ctx.output, params.case_sensitive)
    allowed = {normalize(v, params.case_sensitive) for v in params.values}
    if got in allowed:
        return MetricResult.check(True)
    shown = ctx.output.strip()[:60]
    return MetricResult.check(False, f"'{shown}' is not one of: {', '.join(params.values)}")


class LengthParams(MetricParams):
    unit: Literal["words", "chars", "lines"] = "words"
    min: int | None = Field(None, ge=0, description="inclusive lower bound")
    max: int | None = Field(None, ge=0, description="inclusive upper bound")

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.min is None and self.max is None:
            raise ValueError("set at least one of `min` or `max`")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("`min` must not exceed `max`")
        return self


@metric("length", layer=MetricLayer.FORMAT, params=LengthParams)
def length(ctx: MetricContext, params: LengthParams) -> MetricResult:
    """The output's length in words, characters, or lines is within bounds."""
    text = ctx.output.strip()
    if params.unit == "words":
        n = len(text.split())
    elif params.unit == "chars":
        n = len(text)
    else:
        n = len(text.splitlines())
    if params.min is not None and n < params.min:
        return MetricResult.check(False, f"{n} {params.unit}, below the minimum of {params.min}")
    if params.max is not None and n > params.max:
        return MetricResult.check(False, f"{n} {params.unit}, above the maximum of {params.max}")
    return MetricResult.check(True, f"{n} {params.unit}")


class RegexParams(MetricParams):
    pattern: str = Field(description="Python regular expression, searched anywhere in the output")
    should_match: bool = Field(True, description="false to require that the pattern is absent")
    ignore_case: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from None
        return value


@metric("regex", layer=MetricLayer.FORMAT, params=RegexParams)
def regex(ctx: MetricContext, params: RegexParams) -> MetricResult:
    """A regular expression is found (or, with should_match=false, not found) in the output."""
    flags = re.IGNORECASE if params.ignore_case else 0
    found = re.search(params.pattern, ctx.output, flags) is not None
    if found == params.should_match:
        return MetricResult.check(True)
    detail = "pattern not found" if params.should_match else "forbidden pattern found"
    return MetricResult.check(False, f"{detail}: {params.pattern}")
