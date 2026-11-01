"""Execute a frozen plan: run the prompt and baselines, then score every output."""

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from hemp._version import __version__
from hemp.dataset import Row
from hemp.plan.schema import BaselineKind, EvalPlan
from hemp.plan.validate import plan_problems
from hemp.profile.profiler import dataset_hash
from hemp.prompt import prompt_hash
from hemp.runner.baselines import majority_answer, zero_shot_prompt
from hemp.runner.evaluate import evaluate_output
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

# Called once per variant with (variant name, number of examples); returns a
# function called after each example is done.
ProgressFactory = Callable[[str, int], ProgressFn]


class RunError(ValueError):
    """The run can't start: the plan, prompt and dataset don't fit together."""


def run_eval(
    prompt: str,
    rows: Sequence[Row],
    plan: EvalPlan,
    *,
    lm: TextLM,
    judge_model: str | None = None,
    baselines: bool = True,
    workers: int = 4,
    prompt_source: str | None = None,
    dataset_source: str | None = None,
    plan_source: str | None = None,
    progress: ProgressFactory | None = None,
    now: datetime | None = None,
) -> RunResult:
    """Run `prompt` on every row with the target `lm` and score it with `plan`.

    The plan is used as-is; it's never changed or re-planned here. With `baselines`,
    the plan's baselines are run and scored with the same checks.
    """
    _check_inputs(prompt, rows, plan)
    variants = [_run_prompt(prompt, rows, plan, lm, workers, progress)]
    if baselines:
        for kind in plan.baselines:
            variants.append(_run_baseline(kind, rows, plan, lm, workers, progress))
    info = RunInfo(
        created_at=now or datetime.now(UTC),
        hemp_version=__version__,
        prompt_hash=prompt_hash(prompt),
        dataset_hash=dataset_hash(rows),
        plan_hash=plan.plan_hash(),
        target_model=str(getattr(lm, "model", "unknown")),
        judge_model=judge_model,
        prompt_source=prompt_source,
        dataset_source=dataset_source,
        plan_source=plan_source,
        n_examples=len(rows),
    )
    return RunResult(info=info, variants=variants)


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
    prompt: str,
    rows: Sequence[Row],
    plan: EvalPlan,
    lm: TextLM,
    workers: int,
    progress: ProgressFactory | None,
) -> VariantResult:
    texts = [fill_prompt(prompt, row) for row in rows]
    on_done = progress(PROMPT_VARIANT, len(rows)) if progress else None
    generations = generate(lm, texts, workers=workers, on_done=on_done)
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
