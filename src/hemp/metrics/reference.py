"""Reference metrics: compare the output with a gold column."""

import math
from collections import Counter
from typing import Any

from pydantic import Field

from hemp.hashing import canonical_json
from hemp.metrics.registry import MetricContext, MetricLayer, MetricParams, MetricResult, metric
from hemp.metrics.text import as_text, first_number, normalize, parse_json, tokens


class ExactMatchParams(MetricParams):
    case_sensitive: bool = False


@metric("exact_match", layer=MetricLayer.REFERENCE, params=ExactMatchParams, requires_gold=True)
def exact_match(ctx: MetricContext, params: ExactMatchParams) -> MetricResult:
    """The output equals the gold value (ignoring whitespace, quotes, a trailing period)."""
    got = normalize(ctx.output, params.case_sensitive)
    want = normalize(as_text(ctx.gold), params.case_sensitive)
    if got == want:
        return MetricResult.check(True)
    return MetricResult.check(False, f"expected '{as_text(ctx.gold)[:60]}'")


class TokenF1Params(MetricParams):
    threshold: float = Field(0.5, ge=0, le=1, description="minimum F1 to count as passed")


@metric("token_f1", layer=MetricLayer.REFERENCE, params=TokenF1Params, requires_gold=True)
def token_f1(ctx: MetricContext, params: TokenF1Params) -> MetricResult:
    """Word-overlap F1 between the output and the gold text (SQuAD style)."""
    got, want = tokens(ctx.output), tokens(as_text(ctx.gold))
    if not got or not want:
        score = 1.0 if got == want else 0.0
    else:
        overlap = sum((Counter(got) & Counter(want)).values())
        if overlap == 0:
            score = 0.0
        else:
            precision, recall = overlap / len(got), overlap / len(want)
            score = 2 * precision * recall / (precision + recall)
    return MetricResult(passed=score >= params.threshold, score=score, detail=f"F1 {score:.2f}")


class NumericMatchParams(MetricParams):
    abs_tol: float = Field(0.0, ge=0, description="allowed absolute difference")
    rel_tol: float = Field(0.0, ge=0, description="allowed relative difference, e.g. 0.05 = 5%")


@metric("numeric_match", layer=MetricLayer.REFERENCE, params=NumericMatchParams, requires_gold=True)
def numeric_match(ctx: MetricContext, params: NumericMatchParams) -> MetricResult:
    """The first number in the output equals the gold number, within tolerance."""
    got = first_number(ctx.output)
    want = first_number(as_text(ctx.gold))
    if want is None:
        return MetricResult.check(False, f"gold value '{as_text(ctx.gold)[:60]}' is not a number")
    if got is None:
        return MetricResult.check(False, "no number in the output")
    ok = math.isclose(got, want, rel_tol=params.rel_tol, abs_tol=params.abs_tol)
    return MetricResult.check(ok, "" if ok else f"got {got:g}, expected {want:g}")


class JsonFieldMatchParams(MetricParams):
    fields: list[str] | None = Field(
        None, description="fields to compare; null compares every field in the gold object"
    )
    case_sensitive: bool = False
    allow_code_fence: bool = True


@metric(
    "json_field_match", layer=MetricLayer.REFERENCE, params=JsonFieldMatchParams, requires_gold=True
)
def json_field_match(ctx: MetricContext, params: JsonFieldMatchParams) -> MetricResult:
    """Fraction of fields in the output JSON object that equal the gold object's fields."""
    gold = ctx.gold
    if isinstance(gold, str):
        try:
            gold = parse_json(gold)
        except ValueError:
            return MetricResult.check(False, "gold value is not valid JSON")
    if not isinstance(gold, dict):
        return MetricResult.check(False, "gold value is not a JSON object")
    try:
        output = parse_json(ctx.output, params.allow_code_fence)
    except ValueError as e:
        return MetricResult.check(False, f"output is {e}")
    if not isinstance(output, dict):
        return MetricResult.check(False, "output is not a JSON object")

    fields = params.fields if params.fields is not None else list(gold)
    if not fields:
        return MetricResult.check(True, "no fields to compare")
    wrong = [
        f
        for f in fields
        if f not in output or not _same(output[f], gold.get(f), params.case_sensitive)
    ]
    score = 1 - len(wrong) / len(fields)
    detail = f"mismatched: {', '.join(wrong)}" if wrong else ""
    return MetricResult(passed=not wrong, score=score, detail=detail)


def _same(a: Any, b: Any, case_sensitive: bool) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return normalize(as_text(a), case_sensitive) == normalize(as_text(b), case_sensitive)
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return math.isclose(a, b)
    return canonical_json(a) == canonical_json(b)
