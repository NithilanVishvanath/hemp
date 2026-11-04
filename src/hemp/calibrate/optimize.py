"""Measure the judge against human labels and tune it with DSPy.

With enough labels, they're split in two: the judge is tuned on one half (DSPy
picks worked examples to show it) and both the default and the tuned judge are
tested on the other half, which tuning never saw. The tuned judge is only kept if
it agrees with the human more on that held-out half.
"""

import contextlib
import io
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import dspy
from pydantic import BaseModel, ConfigDict

from hemp.calibrate.agreement import Agreement, agreement
from hemp.calibrate.labels import Label
from hemp.metrics.judge import (
    CalibratedJudge,
    Judge,
    _as_bool,
    make_judge,
    module_state,
    to_dspy_image,
)
from hemp.runner.parallel import run_all

MIN_LABELS_TO_TUNE = 8
MAX_EXAMPLES_SHOWN = 4  # worked examples the tuned judge sees in its prompt


class CalibrationFailed(RuntimeError):
    """The judge couldn't be measured, e.g. its model is unreachable."""


class QuestionAgreement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    n_labels: int
    before: Agreement  # the default judge
    after: Agreement | None = None  # the tuned judge, on the same held-out labels


class CalibrationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_labels: int
    n_train: int
    n_test: int  # labels the agreement numbers are measured on
    before: Agreement
    after: Agreement | None
    by_question: list[QuestionAgreement]
    n_failed: int  # judge calls that failed and were left out
    tuned: CalibratedJudge | None  # set only when tuning improved agreement

    @property
    def improved(self) -> bool:
        return self.tuned is not None


def calibrate(
    labels: Sequence[Label],
    judge_lm: Any,
    *,
    judge_model: str,
    workers: int = 4,
    seed: int = 0,
    now: datetime | None = None,
) -> CalibrationReport:
    """Measure the default judge on the labels and, with enough of them, tune it."""
    if not labels:
        raise CalibrationFailed("There are no labels yet. Label some outputs first.")
    base = make_judge()
    base.set_lm(judge_lm)

    if len(labels) < MIN_LABELS_TO_TUNE:
        verdicts = _predict(base, labels, workers)
        return CalibrationReport(
            n_labels=len(labels),
            n_train=0,
            n_test=len(labels),
            before=_agree(labels, verdicts),
            after=None,
            by_question=_by_question(labels, verdicts, None),
            n_failed=sum(1 for v in verdicts if v is None),
            tuned=None,
        )

    train, test = split(labels, seed)
    before_verdicts = _predict(base, test, workers)
    tuned_judge = _tune(train, judge_lm, seed)
    after_verdicts = _predict(tuned_judge, test, workers)
    before, after = _agree(test, before_verdicts), _agree(test, after_verdicts)
    improved = (
        after.accuracy is not None
        and before.accuracy is not None
        and after.accuracy > before.accuracy
    )
    tuned = (
        CalibratedJudge(
            judge_model=judge_model,
            state=module_state(tuned_judge),
            created_at=now or datetime.now(UTC).replace(microsecond=0),
            n_labels=len(labels),
            n_test=len(test),
            agreement=after.accuracy,
            kappa=after.kappa,
        )
        if improved
        else None
    )
    return CalibrationReport(
        n_labels=len(labels),
        n_train=len(train),
        n_test=len(test),
        before=before,
        after=after,
        by_question=_by_question(test, before_verdicts, after_verdicts),
        n_failed=sum(1 for v in [*before_verdicts, *after_verdicts] if v is None),
        tuned=tuned,
    )


def split(labels: Sequence[Label], seed: int = 0) -> tuple[list[Label], list[Label]]:
    """Half for tuning, half for testing, keeping yes and no labels in both halves."""
    rng = random.Random(seed)
    train: list[Label] = []
    test: list[Label] = []
    for answer in (True, False):
        group = sorted((lb for lb in labels if lb.passed is answer), key=lambda lb: lb.key)
        rng.shuffle(group)
        half = len(group) // 2
        train += group[half:]
        test += group[:half]
    return train, test


def _tune(train: Sequence[Label], judge_lm: Any, seed: int) -> Judge:
    def metric(example: dspy.Example, pred: dspy.Prediction, trace: object = None) -> bool:
        try:
            return _as_bool(pred.answer) is bool(example.answer)
        except ValueError:
            return False

    student = make_judge()
    student.set_lm(judge_lm)
    optimizer = dspy.BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=MAX_EXAMPLES_SHOWN,
        max_labeled_demos=MAX_EXAMPLES_SHOWN,
    )
    trainset = [lb.as_example() for lb in train]
    random.Random(seed).shuffle(trainset)
    # redirect_stdout: BootstrapFewShot prints a progress line we don't want in HEMP's output.
    with dspy.context(lm=judge_lm), contextlib.redirect_stdout(io.StringIO()):
        tuned: Judge = optimizer.compile(student, trainset=trainset)
    tuned.set_lm(judge_lm)
    return tuned


def _predict(judge: Judge, labels: Sequence[Label], workers: int) -> list[bool | None]:
    def one(label: Label) -> bool | None:
        try:
            pred = judge(
                task=label.task,
                output=label.output,
                question=label.question,
                reference=label.reference,
                images=[to_dspy_image(i) for i in label.images],
            )
            return _as_bool(pred.answer)
        except Exception:
            return None

    def check_first(verdict: bool | None) -> None:
        if verdict is None:
            raise CalibrationFailed(
                "The judge couldn't answer. Is its model running (`ollama serve`)? "
                "You can pick another one with --judge-model."
            )

    return run_all(one, labels, workers=workers, check_first=check_first)


def _agree(labels: Sequence[Label], verdicts: Sequence[bool | None]) -> Agreement:
    return agreement(
        [(lb.passed, v) for lb, v in zip(labels, verdicts, strict=True) if v is not None]
    )


def _by_question(
    labels: Sequence[Label],
    before: Sequence[bool | None],
    after: Sequence[bool | None] | None,
) -> list[QuestionAgreement]:
    questions = list(dict.fromkeys(lb.question for lb in labels))
    out = []
    for question in questions:
        idx = [i for i, lb in enumerate(labels) if lb.question == question]
        subset = [labels[i] for i in idx]
        out.append(
            QuestionAgreement(
                question=question,
                n_labels=len(idx),
                before=_agree(subset, [before[i] for i in idx]),
                after=_agree(subset, [after[i] for i in idx]) if after is not None else None,
            )
        )
    return out
