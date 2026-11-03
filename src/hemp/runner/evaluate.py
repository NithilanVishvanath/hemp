"""Apply a plan's checks to outputs. Deterministic: no model calls (except embeddings)."""

from hemp.dataset import Row
from hemp.metrics import MetricContext, get_metric
from hemp.metrics.text import is_missing
from hemp.plan.schema import CheckSpec, EvalPlan
from hemp.runner.parallel import short_error
from hemp.runner.results import CheckOutcome


def evaluate_output(
    plan: EvalPlan, row: Row, output: str, *, error: str | None = None
) -> list[CheckOutcome]:
    """Run every check in the plan on one output.

    When the model call failed (`error`), every applicable check fails. A check that
    crashes is recorded as a failed outcome with `error=True` rather than stopping the run.
    """
    return [_one(spec, row, output, error) for spec in plan.checks()]


def _one(spec: CheckSpec, row: Row, output: str, model_error: str | None) -> CheckOutcome:
    info = get_metric(spec.metric)
    gold = row.get(spec.gold_column) if spec.gold_column else None
    if info.requires_gold and is_missing(gold):
        return CheckOutcome(
            check_id=spec.id, passed=None, score=None, detail="no answer value for this row"
        )
    if model_error is not None:
        return CheckOutcome(
            check_id=spec.id, passed=False, score=0.0, detail="no output: the model call failed"
        )
    try:
        result = info(MetricContext(output=output, row=row, gold=gold), spec.params)
    except Exception as e:
        return CheckOutcome(
            check_id=spec.id,
            passed=False,
            score=0.0,
            detail=short_error(e),
            error=True,
        )
    return CheckOutcome(
        check_id=spec.id, passed=result.passed, score=result.score, detail=result.detail
    )
