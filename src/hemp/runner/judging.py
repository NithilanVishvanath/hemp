"""Ask the judge every plan question about every output."""

from collections.abc import Sequence
from typing import Any

from hemp.dataset import Row
from hemp.metrics.judge import NO_REFERENCE, Judge, ask
from hemp.metrics.text import is_missing
from hemp.plan.schema import Criterion, EvalPlan
from hemp.runner.parallel import run_all, short_error
from hemp.runner.results import CheckOutcome, ExampleResult
from hemp.runner.target import ProgressFn, explain_failure, value_text


class JudgeError(RuntimeError):
    """The judge model can't be reached at all."""


def judge_examples(
    plan: EvalPlan,
    rows: Sequence[Row],
    tasks: Sequence[str],
    examples: Sequence[ExampleResult],
    judge_lm: Any,
    *,
    workers: int = 4,
    on_done: ProgressFn | None = None,
) -> list[list[CheckOutcome]]:
    """Outcomes for each example (in order), one per question (in plan order).

    `tasks` are the filled-in user prompts; the judge sees them as the task, even for
    baselines, since the questions are about the user's task. A failed judge call
    (other than the first) is recorded as an error and left out of scores.
    """
    judge = Judge()
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


def _judge_one(
    judge: Judge, criterion: Criterion, row: Row, task: str, ex: ExampleResult
) -> CheckOutcome:
    reference = NO_REFERENCE
    if criterion.gold_column is not None:
        gold = row.get(criterion.gold_column)
        if is_missing(gold):
            return CheckOutcome(
                check_id=criterion.id,
                passed=None,
                score=None,
                detail="no answer value for this row",
            )
        reference = value_text(gold)
    if ex.error is not None:
        return CheckOutcome(
            check_id=criterion.id,
            passed=False,
            score=0.0,
            detail="no output: the model call failed",
        )
    try:
        verdict = ask(
            judge, task=task, output=ex.output, question=criterion.question, reference=reference
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
