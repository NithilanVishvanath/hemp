"""Human labels for judge questions, and choosing which outputs to label.

A label is keyed by exactly what the judge sees (question, task, output, reference),
so it stays valid across runs. Rewording a question means its old labels no longer
apply, which is intended: the criterion changed.
"""

import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import dspy
from pydantic import BaseModel, ConfigDict

from hemp.dataset import Row
from hemp.hashing import canonical_json, sha256_text
from hemp.images import image_columns
from hemp.metrics.judge import to_dspy_image
from hemp.plan.schema import EvalPlan
from hemp.prompt import template_fields
from hemp.runner.judging import reference_for
from hemp.runner.results import RunResult
from hemp.runner.target import build_prompt, prompt_images, prompt_text

JUDGE_INPUTS = ("task", "output", "reference", "question", "images")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def label_key(
    question: str, task: str, output: str, reference: str, images: Sequence[str] = ()
) -> str:
    parts: list[object] = [question, task, output, reference]
    if images:
        parts.append(list(images))
    return sha256_text(canonical_json(parts))


class LabelItem(_Model):
    """One judge question about one stored output, waiting for a human answer."""

    question_id: str
    question: str
    task: str
    output: str
    reference: str
    images: tuple[str, ...] = ()  # image files or URLs in the task, in order
    judge_passed: bool  # what the judge said in the run (hidden while labeling)
    run_id: int | None
    variant: str
    example: int

    @property
    def key(self) -> str:
        return label_key(self.question, self.task, self.output, self.reference, self.images)


class Label(_Model):
    question: str
    task: str
    output: str
    reference: str
    passed: bool  # the human's answer
    images: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        return label_key(self.question, self.task, self.output, self.reference, self.images)

    def as_example(self) -> dspy.Example:
        return dspy.Example(
            task=self.task,
            output=self.output,
            reference=self.reference,
            question=self.question,
            images=[to_dspy_image(i) for i in self.images],
            answer=self.passed,
        ).with_inputs(*JUDGE_INPUTS)


def label_candidates(
    run: RunResult,
    rows: Sequence[Row],
    prompt: str,
    plan: EvalPlan,
    *,
    skip: set[str] | None = None,
    n: int = 20,
    seed: int = 0,
) -> list[LabelItem]:
    """Up to `n` outputs to label, spread across questions and across the judge's yes/no.

    Outputs from baselines are included: they're often bad, which gives the judge
    examples of failures to learn from. Items whose key is in `skip` are left out.
    """
    skip = skip or set()
    images = image_columns(rows, {*template_fields(prompt), *plan.input_columns})
    base_dir = Path(run.info.data_dir) if run.info.data_dir else None
    groups: dict[tuple[str, bool], list[LabelItem]] = defaultdict(list)
    seen: set[str] = set()
    for criterion in plan.criteria:
        if run.questions.get(criterion.id) != criterion.question:
            continue  # this run judged a different version of the question
        for variant in run.variants:
            for ex in variant.examples:
                outcome = next((o for o in ex.checks if o.check_id == criterion.id), None)
                if outcome is None or outcome.passed is None or ex.error is not None:
                    continue
                row = rows[ex.index]
                reference = reference_for(criterion, row)
                if reference is None:
                    continue
                content = build_prompt(prompt, row, images, base_dir)
                item = LabelItem(
                    question_id=criterion.id,
                    question=criterion.question,
                    task=prompt_text(content),
                    output=ex.output,
                    reference=reference,
                    images=tuple(i.location for i in prompt_images(content)),
                    judge_passed=outcome.passed,
                    run_id=run.id,
                    variant=variant.name,
                    example=ex.index,
                )
                if item.key in skip or item.key in seen:
                    continue
                seen.add(item.key)
                groups[(criterion.id, outcome.passed)].append(item)

    rng = random.Random(seed)
    queues = [groups[k] for k in sorted(groups)]
    for queue in queues:
        rng.shuffle(queue)
    picked: list[LabelItem] = []
    while len(picked) < n and any(queues):
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked
