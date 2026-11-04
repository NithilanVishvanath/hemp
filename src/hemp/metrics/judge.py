"""The rubric judge: an LLM answers one yes/no question about one output.

It's a DSPy module so `hemp calibrate` can optimize it against human labels.
Unlike the target runner, it's fine for DSPy adapters to format its prompt.

A calibrated judge is saved next to its plan as `<plan>.judge.json` and used by
`hemp run` whenever the judge model matches.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel, ConfigDict, ValidationError

from hemp.hashing import canonical_json, sha256_text
from hemp.images import ImageRef

NO_REFERENCE = "(none)"


class JudgeQuestion(dspy.Signature):
    """Check one output of an LLM prompt by answering a yes/no question about it.

    The task is the exact prompt the model received. Judge only the output, using
    the task and the reference answer (when given) as context. Answer yes only if the
    output clearly satisfies the question; when in doubt, answer no. Give a short
    reason before the answer.
    """

    task: str = dspy.InputField(desc="the prompt the model received")
    output: str = dspy.InputField(desc="the model's output to judge")
    reference: str = dspy.InputField(
        desc=f"reference answer(s) from the dataset, or {NO_REFERENCE}"
    )
    question: str = dspy.InputField(desc="yes/no question; yes means the output is good")
    images: list[dspy.Image] = dspy.InputField(
        desc="images that are part of the task, in the order of its [image N] markers"
    )
    reason: str = dspy.OutputField(desc="one or two sentences")
    answer: bool = dspy.OutputField(desc="true for yes, false for no")


class Judge(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.Predict(JudgeQuestion)

    def forward(
        self,
        task: str,
        output: str,
        question: str,
        reference: str = NO_REFERENCE,
        images: list[dspy.Image] | None = None,
    ) -> dspy.Prediction:
        return self.judge(
            task=task, output=output, reference=reference, question=question, images=images or []
        )


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def ask(
    judge: Judge,
    *,
    task: str,
    output: str,
    question: str,
    reference: str,
    images: Sequence[ImageRef] = (),
) -> Verdict:
    """One judgment. Raises whatever the model or the parser raises."""
    pred = judge(
        task=task,
        output=output,
        question=question,
        reference=reference,
        images=[to_dspy_image(i) for i in images],
    )
    return Verdict(passed=_as_bool(pred.answer), reason=" ".join(str(pred.reason).split()))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("yes", "true", "y", "1"):
        return True
    if text in ("no", "false", "n", "0"):
        return False
    raise ValueError(f"The judge answered {value!r} instead of yes or no.")


class JudgeFileError(ValueError):
    """A calibrated judge file is missing or unreadable."""


class CalibratedJudge(BaseModel):
    """A judge tuned against human labels, plus how well it agrees with them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge_model: str  # the model it was tuned and tested with
    state: dict[str, Any]  # dspy module state: the chosen examples
    created_at: datetime
    n_labels: int
    n_test: int  # held-out labels the agreement numbers come from
    agreement: float | None
    kappa: float | None

    @property
    def fingerprint(self) -> str:
        return sha256_text(canonical_json(self.state))[:8]


def judge_file_for(plan_path: str | Path) -> Path:
    """`prompt.plan.yaml` -> `prompt.plan.judge.json`."""
    path = Path(plan_path)
    return path.with_name(f"{path.stem}.judge.json")


def read_calibrated(path: str | Path) -> CalibratedJudge:
    path = Path(path)
    try:
        return CalibratedJudge.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as e:
        raise JudgeFileError(
            f"Couldn't read the calibrated judge {path}: {str(e).splitlines()[0]}. "
            "Delete it to use the default judge, or re-create it with `hemp calibrate`."
        ) from None


def write_calibrated(calibrated: CalibratedJudge, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(calibrated.model_dump_json(indent=2), encoding="utf-8")
    return path


def module_state(judge: Judge) -> dict[str, Any]:
    """The judge's state without its LM, so the file doesn't pin a model or API settings."""
    state: dict[str, Any] = judge.dump_state()
    for predictor in state.values():
        if isinstance(predictor, dict):
            predictor["lm"] = None
    return state


def make_judge(calibrated: CalibratedJudge | None = None) -> Judge:
    judge = Judge()
    if calibrated is not None:
        judge.load_state(calibrated.state)
        # Calibration only picks worked examples; the instructions and fields
        # always come from this version of HEMP.
        judge.judge.signature = JudgeQuestion
    return judge


def to_dspy_image(image: ImageRef | str) -> dspy.Image:
    """A dspy.Image from an ImageRef or a stored image location."""
    ref = image if isinstance(image, ImageRef) else ImageRef(source=image, location=image)
    if ref.is_file or ref.location.startswith("data:"):
        return dspy.Image(url=f"data:{ref.media_type};base64,{ref.base64_data()}")
    return dspy.Image(url=ref.location)


def judge_label(model: str, calibrated: CalibratedJudge | None) -> str:
    """How a run records its judge: the model, plus which calibration if any."""
    return model if calibrated is None else f"{model} (calibrated {calibrated.fingerprint})"
