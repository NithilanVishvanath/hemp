"""The rubric judge: an LLM answers one yes/no question about one output.

It's a DSPy module so calibration (milestone 8) can optimize it against human
labels. Unlike the target runner, it's fine for DSPy adapters to format its prompt.
"""

from dataclasses import dataclass

import dspy

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
    reason: str = dspy.OutputField(desc="one or two sentences")
    answer: bool = dspy.OutputField(desc="true for yes, false for no")


class Judge(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.Predict(JudgeQuestion)

    def forward(
        self, task: str, output: str, question: str, reference: str = NO_REFERENCE
    ) -> dspy.Prediction:
        return self.judge(task=task, output=output, reference=reference, question=question)


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def ask(judge: Judge, *, task: str, output: str, question: str, reference: str) -> Verdict:
    """One judgment. Raises whatever the model or the parser raises."""
    pred = judge(task=task, output=output, question=question, reference=reference)
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
