from typing import Any

import pytest
from dspy.utils import DummyLM
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan

from hemp.metrics.judge import NO_REFERENCE, Judge, _as_bool, ask
from hemp.plan.schema import Criterion, EvalPlan
from hemp.runner import JudgeError, run_eval
from hemp.runner.judging import judge_examples
from hemp.runner.results import ExampleResult

YES = {"reason": "It does.", "answer": "True"}
NO = {"reason": "It doesn't.", "answer": "False"}


def judge_lm(answers: dict[str, dict[str, Any]], model: str = "fake/judge") -> DummyLM:
    """Answers by the first key found in the judge's prompt (DummyLM dict mode)."""
    lm = DummyLM(answers)
    lm.model = model
    return lm


def with_questions(*criteria: Criterion) -> EvalPlan:
    return make_plan().model_copy(update={"criteria": list(criteria)})


def test_ask_returns_a_verdict() -> None:
    judge = Judge()
    judge.set_lm(judge_lm({"polite": YES}))
    verdict = ask(judge, task="t", output="o", question="Is it polite?", reference=NO_REFERENCE)
    assert verdict.passed is True
    assert verdict.reason == "It does."


@pytest.mark.parametrize(
    ("value", "expected"), [(True, True), ("yes", True), ("No", False), ("false", False)]
)
def test_as_bool(value: object, expected: bool) -> None:
    assert _as_bool(value) is expected


def test_as_bool_rejects_other_answers() -> None:
    with pytest.raises(ValueError, match="instead of yes or no"):
        _as_bool("maybe")


def test_judge_examples_outcomes() -> None:
    plan = with_questions(
        Criterion(id="q1", question="Is it short?"),
        Criterion(id="matches_label", question="Same as the reference?", gold_column="label"),
    )
    rows = [{"review": "a", "label": "positive"}, {"review": "b", "label": None}]
    rows.append({"review": "c", "label": "negative"})
    examples = [
        ExampleResult(index=0, output="positive"),
        ExampleResult(index=1, output="negative"),
        ExampleResult(index=2, output="", error="ValueError: boom"),
    ]
    lm = judge_lm({"short": YES, "reference": NO})
    found = judge_examples(plan, rows, ["task"] * 3, examples, lm, workers=1)

    assert [[o.check_id for o in ex] for ex in found] == [["q1", "matches_label"]] * 3
    (q1, ref), (_, ref_b), (q1_c, ref_c) = found
    assert (q1.passed, q1.detail) == (True, "It does.")
    assert ref.passed is False
    assert ref_b.skipped and ref_b.detail == "no answer value for this row"
    assert q1_c.passed is False and "model call failed" in q1_c.detail  # no judge call
    assert ref_c.passed is False


def test_judge_sees_task_output_and_reference() -> None:
    plan = with_questions(Criterion(id="m", question="Same answer?", gold_column="label"))
    lm = judge_lm({"Same answer?": YES})
    judge_examples(
        plan,
        [{"review": "r", "label": "positive"}],
        ["THE TASK"],
        [ExampleResult(index=0, output="THE OUTPUT")],
        lm,
    )
    sent = str(lm.history[-1]["messages"])
    assert "THE TASK" in sent
    assert "THE OUTPUT" in sent
    assert "positive" in sent


def test_first_judge_failure_stops_the_run() -> None:
    lm = judge_lm({})  # DummyLM answers nothing parseable
    lm.model = "ollama_chat/qwen2.5:7b"
    plan = with_questions(Criterion(id="q1", question="Is it short?"))
    examples = [ExampleResult(index=i, output="x") for i in range(3)]
    with pytest.raises(JudgeError, match="--no-judge"):
        judge_examples(plan, SENTIMENT_ROWS, ["t"] * 3, examples, lm)


def test_later_judge_failures_are_left_out_of_scores() -> None:
    plan = with_questions(Criterion(id="q1", question="Is it short?"))
    lm = judge_lm({"OK OUTPUT": YES})  # anything else can't be parsed
    examples = [ExampleResult(index=0, output="OK OUTPUT"), ExampleResult(index=1, output="?")]
    _, (failed,) = judge_examples(plan, SENTIMENT_ROWS, ["t", "t"], examples, lm, workers=1)
    assert failed.error and failed.skipped
    assert failed.detail.startswith("judge failed: ")


def test_run_eval_with_judge() -> None:
    plan = with_questions(Criterion(id="q1", question="Is the label negative?"))
    # The judge says yes when the output it sees is "negative".
    lm = judge_lm({"negative\n": YES, "positive\n": NO})
    result = run_eval(
        SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(keyword_sentiment), judge_lm=lm
    )
    assert result.info.judge_model == "fake/judge"
    assert result.questions == {"q1": "Is the label negative?"}
    for variant in result.variants:  # baselines are judged too
        assert variant.examples[0].checks[-1].check_id == "q1"
    q1 = next(s for s in result.prompt.summary() if s.check_id == "q1")
    assert (q1.n_passed, q1.n_scored) == (3, 12)  # only the "Hate it" rows


def test_run_eval_without_judge_leaves_questions_out() -> None:
    plan = with_questions(Criterion(id="q1", question="Is it short?"))
    result = run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), baselines=False)
    assert result.info.judge_model is None
    assert result.questions == {}
    assert all(o.check_id != "q1" for ex in result.prompt.examples for o in ex.checks)
