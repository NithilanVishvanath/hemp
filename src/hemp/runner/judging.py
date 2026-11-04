"""Ask the judge every plan question about every output."""

from collections.abc import Sequence
from typing import Any

from hemp.dataset import Row
from hemp.metrics.judge import NO_REFERENCE, CalibratedJudge, Judge, ask, make_judge
from hemp.metrics.text import is_missing
from hemp.plan.schema import Criterion, EvalPlan
from hemp.runner.parallel import run_all, short_error
from hemp.runner.results import CheckOutcome, ExampleResult
from hemp.runner.target import (
    ProgressFn,
    PromptContent,
    explain_failure,
    prompt_images,
    prompt_text,
    value_text,
)


class JudgeError(RuntimeError):
    """The judge model can't be reached at all."""


def judge_examples(
    plan: EvalPlan,
    rows: Sequence[Row],
    tasks: Sequence[PromptContent],
    examples: Sequence[ExampleResult],
    judge_lm: Any,
    *,
    workers: int = 4,
    on_done: ProgressFn | None = None,
    calibrated: CalibratedJudge | None = None,
) -> list[list[CheckOutcome]]:
    """Outcomes for each example (in order), one per question (in plan order).

    `tasks` are the filled-in user prompts; the judge sees them as the task, even for
    baselines, since the questions are about the user's task. A failed judge call
    (other than the first) is recorded as an error and left out of scores.
    """
    judge = make_judge(calibrated)
    judge.set_lm(judge_lm)
    jobs = [(ex, criterion) for ex in examples for criterion in plan.criteria]

    def one(job: tuple[ExampleResult, Criterion]) -> CheckOutcome:
        ex, criterion = job
        outcome = _judge_one(judge, criterion, rows[ex.index], tasks[ex.index], ex)
        if on_done is not None:
            on_done()
        return outcome

    def check_first(outcome: CheckOutcome) -> None:
        if outcome.error:
            model = str(getattr(judge_lm, "model", "the judge model"))
            raise JudgeError(
                explain_failure(
                    outcome.detail.removeprefix("judge failed: "), model, "--judge-model"
                )
                + " Or run without the judge using --no-judge."
            )

    outcomes = run_all(one, jobs, workers=workers, check_first=check_first)
    n = len(plan.criteria)
    return [outcomes[i * n : (i + 1) * n] for i in range(len(examples))]


def reference_for(criterion: Criterion, row: Row) -> str | None:
    """What the judge sees as the reference; None when the row has no answer to compare to."""
    if criterion.gold_column is None:
        return NO_REFERENCE
    gold = row.get(criterion.gold_column)
    return None if is_missing(gold) else value_text(gold)


def _judge_one(
    judge: Judge, criterion: Criterion, row: Row, task: PromptContent, ex: ExampleResult
) -> CheckOutcome:
    reference = reference_for(criterion, row)
    if reference is None:
        return CheckOutcome(
            check_id=criterion.id,
            passed=None,
            score=None,
            detail="no answer value for this row",
        )
    if ex.error is not None:
        return CheckOutcome(
            check_id=criterion.id,
            passed=False,
            score=0.0,
            detail="no output: the model call failed",
        )
    try:
        verdict = ask(
            judge,
            task=prompt_text(task),
            output=ex.output,
            question=criterion.question,
            reference=reference,
            images=prompt_images(task),
        )
    except Exception as e:
        return CheckOutcome(
            check_id=criterion.id,
            passed=None,
            score=None,
            detail=f"judge failed: {short_error(e)}",
            error=True,
        )
    return CheckOutcome(
        check_id=criterion.id,
        passed=verdict.passed,
        score=1.0 if verdict.passed else 0.0,
        detail=verdict.reason,
    )
