"""Execute a frozen plan: run the prompt and baselines, then score every output."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from hemp._version import __version__
from hemp.dataset import Row
from hemp.plan.schema import BaselineKind, EvalPlan
from hemp.plan.validate import plan_problems
from hemp.profile.profiler import dataset_hash
from hemp.prompt import prompt_hash
from hemp.runner.baselines import majority_answer, zero_shot_prompt
from hemp.runner.evaluate import evaluate_output
from hemp.runner.judging import judge_examples
from hemp.runner.results import (
    PROMPT_VARIANT,
    ExampleResult,
    RunInfo,
    RunResult,
    VariantResult,
)
from hemp.runner.target import (
    Generation,
    ProgressFn,
    PromptError,
    TextLM,
    check_prompt_fields,
    fill_prompt,
    generate,
)

# Called once per step with (step name, number of items); returns a function called
# after each item is done. Steps are variant names, or JUDGE_PREFIX + variant name.
ProgressFactory = Callable[[str, int], ProgressFn]
JUDGE_PREFIX = "judge:"


class RunError(ValueError):
    """The run can't start: the plan, prompt and dataset don't fit together."""


def run_eval(
    prompt: str,
    rows: Sequence[Row],
    plan: EvalPlan,
    *,
    lm: TextLM,
    judge_lm: Any | None = None,
    baselines: bool = True,
    workers: int = 4,
    prompt_source: str | None = None,
    dataset_source: str | None = None,
    plan_source: str | None = None,
    progress: ProgressFactory | None = None,
    reuse: RunResult | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run `prompt` on every row with the target `lm` and score it with `plan`.

    The plan is used as-is; it's never changed or re-planned here. With `baselines`,
    the plan's baselines are run and scored with the same checks. With `judge_lm`
    (a dspy.LM), the plan's questions are answered by the judge; without it they're
    left out.

    `reuse` is an earlier run with the same plan, dataset, target and judge models
    (see `Store.find_reusable_run`). Its baseline results are copied instead of
    recomputed, since they can't differ.
    """
    _check_inputs(prompt, rows, plan)
    judging = judge_lm is not None and bool(plan.criteria)
    target_model = str(getattr(lm, "model", "unknown"))
    judge_model = str(getattr(judge_lm, "model", "unknown")) if judging else None
    if reuse is not None:
        _check_reusable(reuse, plan, rows, target_model, judge_model)
    tasks = [fill_prompt(prompt, row) for row in rows]

    fresh = [_run_prompt(tasks, rows, plan, lm, workers, progress)]
    copied: list[VariantResult] = []
    baselines_from: dict[str, int] = {}
    if baselines:
        for kind in plan.baselines:
            earlier = reuse.variant(kind.value) if reuse is not None else None
            if reuse is not None and reuse.id is not None and earlier is not None:
                copied.append(earlier)
                baselines_from[kind.value] = reuse.baselines_from.get(kind.value, reuse.id)
            else:
                fresh.append(_run_baseline(kind, rows, plan, lm, workers, progress))
    if judging:
        fresh = [_judge(v, tasks, rows, plan, judge_lm, workers, progress) for v in fresh]
    order = [PROMPT_VARIANT, *(k.value for k in plan.baselines)]
    variants = sorted([*fresh, *copied], key=lambda v: order.index(v.name))
    info = RunInfo(
        created_at=now or datetime.now(UTC),
        hemp_version=__version__,
        prompt_hash=prompt_hash(prompt),
        dataset_hash=dataset_hash(rows),
        plan_hash=plan.plan_hash(),
        target_model=target_model,
        judge_model=judge_model,
        prompt_source=prompt_source,
        dataset_source=dataset_source,
        plan_source=plan_source,
        n_examples=len(rows),
    )
    questions = {c.id: c.question for c in plan.criteria} if judging else {}
    return RunResult(
        info=info, variants=variants, questions=questions, baselines_from=baselines_from
    )


def _check_reusable(
    earlier: RunResult,
    plan: EvalPlan,
    rows: Sequence[Row],
    target_model: str,
    judge_model: str | None,
) -> None:
    mismatches = [
        name
        for name, before, now in (
            ("plan", earlier.info.plan_hash, plan.plan_hash()),
            ("dataset", earlier.info.dataset_hash, dataset_hash(rows)),
            ("target model", earlier.info.target_model, target_model),
            ("judge model", earlier.info.judge_model, judge_model),
        )
        if before != now
    ]
    if mismatches:
        raise RunError(
            f"Can't reuse baselines from run #{earlier.id}: its {', '.join(mismatches)} "
            "differ from this run's."
        )


def _check_inputs(prompt: str, rows: Sequence[Row], plan: EvalPlan) -> None:
    if not rows:
        raise RunError("The dataset has no rows. Add at least one example.")
    problems = plan_problems(plan)
    if problems:
        raise RunError("The plan has problems:\n" + "\n".join(f"  - {p}" for p in problems))
    try:
        check_prompt_fields(prompt, rows)
    except PromptError as e:
        raise RunError(str(e)) from None
    columns = {key for row in rows for key in row}
    for kind, names in (("input", plan.input_columns), ("answer", plan.gold_columns)):
        missing = [c for c in names if c not in columns]
        if missing:
            raise RunError(
                f"The plan's {kind} column(s) {', '.join(missing)} aren't in the dataset. "
                f"Fix `columns:` in the plan. Dataset columns: {', '.join(sorted(columns))}."
            )


def _run_prompt(
    tasks: Sequence[str],
    rows: Sequence[Row],
    plan: EvalPlan,
    lm: TextLM,
    workers: int,
    progress: ProgressFactory | None,
) -> VariantResult:
    on_done = progress(PROMPT_VARIANT, len(rows)) if progress else None
    generations = generate(lm, tasks, workers=workers, on_done=on_done)
    return _score(PROMPT_VARIANT, rows, plan, generations)


def _run_baseline(
    kind: BaselineKind,
    rows: Sequence[Row],
    plan: EvalPlan,
    lm: TextLM,
    workers: int,
    progress: ProgressFactory | None,
) -> VariantResult:
    on_done = progress(kind.value, len(rows)) if progress else None
    if kind is BaselineKind.MAJORITY_CLASS:
        answer = majority_answer(plan, rows)
        generations = []
        for _ in rows:
            generations.append(Generation(output=answer, error=None, seconds=0.0))
            if on_done:
                on_done()
    else:
        texts = [zero_shot_prompt(plan, row) for row in rows]
        generations = generate(lm, texts, workers=workers, on_done=on_done)
    return _score(kind.value, rows, plan, generations)


def _score(
    name: str, rows: Sequence[Row], plan: EvalPlan, generations: Sequence[Generation]
) -> VariantResult:
    examples = [
        ExampleResult(
            index=i,
            output=gen.output,
            error=gen.error,
            seconds=round(gen.seconds, 3),
            checks=evaluate_output(plan, row, gen.output, error=gen.error),
        )
        for i, (row, gen) in enumerate(zip(rows, generations, strict=True))
    ]
    return VariantResult(name=name, examples=examples)


def _judge(
    variant: VariantResult,
    tasks: Sequence[str],
    rows: Sequence[Row],
    plan: EvalPlan,
    judge_lm: Any,
    workers: int,
    progress: ProgressFactory | None,
) -> VariantResult:
    total = len(variant.examples) * len(plan.criteria)
    on_done = progress(f"{JUDGE_PREFIX}{variant.name}", total) if progress else None
    verdicts = judge_examples(
        plan, rows, tasks, variant.examples, judge_lm, workers=workers, on_done=on_done
    )
    examples = [
        ex.model_copy(update={"checks": [*ex.checks, *found]})
        for ex, found in zip(variant.examples, verdicts, strict=True)
    ]
    return variant.model_copy(update={"examples": examples})
