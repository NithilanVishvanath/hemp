"""Reference metrics: compare the output with a gold column.

Rows whose gold value is missing are skipped by the registry before these run.
"""

import math
from collections import Counter
from typing import Literal

from pydantic import Field

from hemp.metrics.embeddings import DEFAULT_EMBED_MODEL, cosine, embed
from hemp.metrics.registry import MetricContext, MetricLayer, MetricParams, MetricResult, metric
from hemp.metrics.text import (
    answer_tokens,
    as_text,
    find_numbers,
    get_path,
    has_path,
    normalize,
    parse_json,
    same_value,
)


class ExactMatchParams(MetricParams):
    case_sensitive: bool = False


@metric("exact_match", layer=MetricLayer.REFERENCE, params=ExactMatchParams, requires_gold=True)
def exact_match(ctx: MetricContext, params: ExactMatchParams) -> MetricResult:
    """The output equals the gold value (ignoring whitespace, quotes, markdown, a final period)."""
    got = normalize(ctx.output, params.case_sensitive)
    want = normalize(as_text(ctx.gold), params.case_sensitive)
    if got == want:
        return MetricResult.check(True)
    return MetricResult.check(False, f"expected '{as_text(ctx.gold)[:60]}'")


class TokenF1Params(MetricParams):
    threshold: float = Field(0.5, ge=0, le=1, description="minimum F1 to count as passed")


@metric("token_f1", layer=MetricLayer.REFERENCE, params=TokenF1Params, requires_gold=True)
def token_f1(ctx: MetricContext, params: TokenF1Params) -> MetricResult:
    """Word-overlap F1 between the output and the gold text (SQuAD style, articles ignored)."""
    got, want = answer_tokens(ctx.output), answer_tokens(as_text(ctx.gold))
    if not got or not want:
        score = 1.0 if got == want else 0.0
    else:
        overlap = sum((Counter(got) & Counter(want)).values())
        precision, recall = overlap / len(got), overlap / len(want)
        score = 2 * precision * recall / (precision + recall) if overlap else 0.0
    return MetricResult(passed=score >= params.threshold, score=score, detail=f"F1 {score:.2f}")


class SemanticSimilarityParams(MetricParams):
    model: str = Field(DEFAULT_EMBED_MODEL, description="LiteLLM embedding model")
    threshold: float = Field(0.75, ge=0, le=1, description="minimum cosine similarity to pass")


@metric(
    "semantic_similarity",
    layer=MetricLayer.REFERENCE,
    params=SemanticSimilarityParams,
    requires_gold=True,
)
def semantic_similarity(ctx: MetricContext, params: SemanticSimilarityParams) -> MetricResult:
    """Embedding cosine similarity between the output and the gold text.

    Measures whether they say similar things, not whether facts match: "the capital
    is Paris" and "the capital is London" score high. Use a rubric criterion for
    correctness.
    """
    output, gold = ctx.output.strip(), as_text(ctx.gold).strip()
    if not output:
        return MetricResult.check(False, "empty output")
    a, b = embed(params.model, [output, gold])
    similarity = min(1.0, max(0.0, cosine(a, b)))
    return MetricResult(
        passed=similarity >= params.threshold,
        score=similarity,
        detail=f"similarity {similarity:.2f}",
    )


class NumericMatchParams(MetricParams):
    abs_tol: float = Field(0.0, ge=0, description="allowed absolute difference")
    rel_tol: float = Field(0.0, ge=0, description="allowed relative difference, e.g. 0.05 = 5%")
    pick: Literal["last", "first"] = Field(
        "last",
        description="which number in the output is the answer; 'last' suits explanations "
        "that end with the result",
    )


@metric("numeric_match", layer=MetricLayer.REFERENCE, params=NumericMatchParams, requires_gold=True)
def numeric_match(ctx: MetricContext, params: NumericMatchParams) -> MetricResult:
    """A number in the output (by default the last) equals the gold number, within tolerance."""
    gold_numbers = find_numbers(as_text(ctx.gold))
    if not gold_numbers:
        return MetricResult.check(False, f"gold value '{as_text(ctx.gold)[:60]}' is not a number")
    found = find_numbers(ctx.output)
    if not found:
        return MetricResult.check(False, "no number in the output")
    want = gold_numbers[0]
    got = found[-1] if params.pick == "last" else found[0]
    ok = math.isclose(got, want, rel_tol=params.rel_tol, abs_tol=params.abs_tol)
    return MetricResult.check(ok, "" if ok else f"got {got:g}, expected {want:g}")


class JsonFieldMatchParams(MetricParams):
    fields: list[str] | None = Field(
        None,
        description="fields to compare (a.b for nested); null compares every top-level gold field",
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
        if not has_path(output, f)
        or not has_path(gold, f)
        or not same_value(get_path(output, f), get_path(gold, f), params.case_sensitive)
    ]
    score = 1 - len(wrong) / len(fields)
    detail = f"mismatched: {', '.join(wrong)}" if wrong else ""
    return MetricResult(passed=not wrong, score=score, detail=detail)
