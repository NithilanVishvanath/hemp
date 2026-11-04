import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from dspy.utils import DummyLM
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan
from typer.testing import CliRunner

import hemp.calibrate.optimize as optimize
from hemp.calibrate import (
    MIN_LABELS_TO_TUNE,
    CalibrationFailed,
    Label,
    agreement,
    calibrate,
    label_candidates,
)
from hemp.calibrate.optimize import split
from hemp.cli import app
from hemp.metrics.judge import (
    CalibratedJudge,
    Judge,
    judge_file_for,
    judge_label,
    make_judge,
    module_state,
    read_calibrated,
    write_calibrated,
)
from hemp.plan.schema import Criterion, EvalPlan
from hemp.runner import RunResult, run_eval
from hemp.store import Store

QUESTION = "Is the label negative?"
YES = {"reason": "Yes.", "answer": "True"}
NO = {"reason": "No.", "answer": "False"}


def dummy(answers: dict[str, dict[str, Any]], model: str = "fake/judge") -> DummyLM:
    lm = DummyLM(answers)
    lm.model = model
    return lm


def honest_judge() -> DummyLM:
    """Says yes exactly when the output it sees is 'negative'."""
    return dummy({"negative\n": YES, "positive\n": NO, "neutral\n": NO})


def yes_judge() -> DummyLM:
    return dummy({"## question ##": YES})


def judged_plan() -> EvalPlan:
    return make_plan().model_copy(update={"criteria": [Criterion(id="q1", question=QUESTION)]})


def judged_run(plan: EvalPlan | None = None) -> RunResult:
    return run_eval(
        SENTIMENT_PROMPT,
        SENTIMENT_ROWS,
        plan or judged_plan(),
        lm=FakeLM(keyword_sentiment),
        judge_lm=honest_judge(),
    ).model_copy(update={"id": 1})


def label(output: str, passed: bool, n: int = 0) -> Label:
    return Label(
        question=QUESTION, task=f"task {n}", output=output, reference="(none)", passed=passed
    )


# --- agreement ----------------------------------------------------------------------


def test_agreement_perfect_and_chance() -> None:
    perfect = agreement([(True, True), (False, False), (True, True), (False, False)])
    assert (perfect.accuracy, perfect.kappa) == (1.0, 1.0)
    chance = agreement([(True, True), (True, False), (False, True), (False, False)])
    assert chance.accuracy == 0.5
    assert chance.kappa == pytest.approx(0.0)


def test_agreement_kappa_undefined_when_one_answer_only() -> None:
    a = agreement([(True, True), (True, True)])
    assert a.accuracy == 1.0 and a.kappa is None
    assert agreement([]).accuracy is None


def test_always_yes_judge_has_high_accuracy_but_low_kappa() -> None:
    pairs = [(True, True)] * 9 + [(False, True)]
    a = agreement(pairs)
    assert a.accuracy == pytest.approx(0.9)
    assert a.kappa == pytest.approx(0.0)


# --- choosing what to label ---------------------------------------------------------


def test_label_candidates_cover_both_answers_and_skip_known() -> None:
    run = judged_run()
    items = label_candidates(run, SENTIMENT_ROWS, SENTIMENT_PROMPT, judged_plan(), n=4)
    assert len(items) == 4
    assert {i.judge_passed for i in items} == {True, False}
    assert all(i.task.startswith("Classify the sentiment") for i in items)

    rest = label_candidates(
        run, SENTIMENT_ROWS, SENTIMENT_PROMPT, judged_plan(), skip={i.key for i in items}, n=100
    )
    assert not {i.key for i in items} & {i.key for i in rest}


def test_label_candidates_ignore_reworded_questions() -> None:
    reworded = judged_plan().model_copy(
        update={"criteria": [Criterion(id="q1", question="Is it negative?")]}
    )
    assert label_candidates(judged_run(), SENTIMENT_ROWS, SENTIMENT_PROMPT, reworded) == []


def test_split_keeps_both_answers_in_each_half() -> None:
    labels = [label("x", i % 3 == 0, i) for i in range(12)]
    train, test = split(labels)
    assert {lb.passed for lb in train} == {lb.passed for lb in test} == {True, False}
    assert not {lb.key for lb in train} & {lb.key for lb in test}
    assert split(labels) == (train, test)  # same split every time


# --- labels in the store ------------------------------------------------------------


def test_labels_are_stored_and_relabeling_replaces(tmp_path: Path) -> None:
    items = label_candidates(judged_run(), SENTIMENT_ROWS, SENTIMENT_PROMPT, judged_plan(), n=2)
    with Store(tmp_path / "runs.db") as store:
        store.save_label(items[0], True)
        store.save_label(items[1], False)
        store.save_label(items[0], False)  # changed my mind
        labels = store.load_labels([QUESTION])
        assert sorted(lb.passed for lb in labels) == [False, False]
        assert store.load_labels(["Some other question?"]) == []


# --- measuring and tuning -----------------------------------------------------------


def test_few_labels_measure_without_tuning() -> None:
    labels = [label("negative", True, 0), label("positive", False, 1), label("neutral", True, 2)]
    report = calibrate(labels, honest_judge(), judge_model="fake/judge")
    assert report.after is None and report.tuned is None
    assert report.before.n == 3
    assert report.before.n_agree == 2  # the human said yes to "neutral", the judge said no


def test_tuning_is_kept_only_when_it_helps(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = [label("negative", True, i) for i in range(5)]
    labels += [label("positive", False, i) for i in range(5, 10)]
    assert len(labels) >= MIN_LABELS_TO_TUNE

    def tuned_with(lm: DummyLM) -> Any:
        def fake_tune(train: Any, judge_lm: Any, seed: int) -> Judge:
            judge = make_judge()
            judge.set_lm(lm)
            return judge

        return fake_tune

    # The default judge says yes to everything; the "tuned" one is right every time.
    monkeypatch.setattr(optimize, "_tune", tuned_with(honest_judge()))
    better = calibrate(labels, yes_judge(), judge_model="fake/judge")
    assert better.before.accuracy == 0.5
    assert better.after is not None and better.after.accuracy == 1.0
    assert better.tuned is not None
    assert better.tuned.agreement == 1.0 and better.tuned.n_test == better.n_test

    monkeypatch.setattr(optimize, "_tune", tuned_with(yes_judge()))
    same = calibrate(labels, yes_judge(), judge_model="fake/judge")
    assert same.after is not None and same.tuned is None


def test_real_tuning_runs_and_picks_examples() -> None:
    labels = [label("negative", True, i) for i in range(5)]
    labels += [label("positive", False, i) for i in range(5, 10)]
    judge = optimize._tune(labels, honest_judge(), seed=0)
    assert judge.judge.demos  # DSPy chose worked examples to show the judge


def test_unreachable_judge_fails_clearly() -> None:
    with pytest.raises(CalibrationFailed, match="--judge-model"):
        calibrate([label("x", True)], dummy({}), judge_model="fake/judge")


# --- the calibrated judge file ------------------------------------------------------


def calibrated(model: str = "fake/judge") -> CalibratedJudge:
    judge = make_judge()
    judge.judge.demos = [label("negative", True).as_example()]
    return CalibratedJudge(
        judge_model=model,
        state=module_state(judge),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        n_labels=10,
        n_test=5,
        agreement=0.8,
        kappa=0.6,
    )


def test_calibrated_file_round_trip(tmp_path: Path) -> None:
    path = write_calibrated(calibrated(), judge_file_for(tmp_path / "prompt.plan.yaml"))
    assert path.name == "prompt.plan.judge.json"
    loaded = read_calibrated(path)
    assert loaded == calibrated()
    assert len(make_judge(loaded).judge.demos) == 1
    assert judge_label("m", loaded) == f"m (calibrated {loaded.fingerprint})"
    assert judge_label("m", None) == "m"


def test_run_records_the_calibrated_judge() -> None:
    result = run_eval(
        SENTIMENT_PROMPT,
        SENTIMENT_ROWS,
        judged_plan(),
        lm=FakeLM(keyword_sentiment),
        judge_lm=honest_judge(),
        calibrated=calibrated(),
        baselines=False,
    )
    assert result.info.judge_model == f"fake/judge (calibrated {calibrated().fingerprint})"


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hemp.cli

    def make_lm(model: str) -> Any:
        return honest_judge() if model == "fake/judge" else FakeLM(keyword_sentiment, model)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEMP_DB", raising=False)
    monkeypatch.setenv("HEMP_JUDGE_MODEL", "fake/judge")
    monkeypatch.setattr(hemp.cli, "make_lm", make_lm)
    (tmp_path / "prompt.txt").write_text(SENTIMENT_PROMPT, encoding="utf-8")
    data = "".join(json.dumps(r) + "\n" for r in SENTIMENT_ROWS)
    (tmp_path / "data.jsonl").write_text(data, encoding="utf-8")
    cli("plan", "prompt.txt", "data.jsonl", "--no-llm")
    plan = tmp_path / "prompt.plan.yaml"
    text = plan.read_text(encoding="utf-8")
    plan.write_text(text.replace("questions: []", f"questions:\n  - id: q1\n    ask: {QUESTION}"))
    return tmp_path


def cli(*args: str, input: str | None = None) -> Any:
    return CliRunner().invoke(app, list(args), input=input)


def test_cli_calibrate_needs_a_judged_run(project: Path) -> None:
    result = cli("calibrate", "prompt.plan.yaml")
    assert result.exit_code == 1
    assert "no run of this plan has judged outputs" in result.output
    assert "No labels yet" in result.output


def test_cli_label_and_measure(project: Path) -> None:
    assert cli("run", "prompt.txt", "data.jsonl").exit_code == 0
    result = cli("calibrate", "prompt.plan.yaml", "-n", "4", input="y\nn\ns\nq\n")
    assert result.exit_code == 0, result.output
    assert "Label 1 of 4" in result.output
    assert QUESTION in result.output
    assert "Label 4 of 4" in result.output  # the q came after the skip
    assert "How often the judge agrees with your 2 label(s)" in result.output
    assert f"Label at least {MIN_LABELS_TO_TUNE} outputs" in result.output

    again = cli("calibrate", "prompt.plan.yaml", "--label", "0")  # just re-measure
    assert again.exit_code == 0, again.output
    assert "2 label(s)" in again.output


def test_cli_saves_a_better_judge_and_run_uses_it(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert cli("run", "prompt.txt", "data.jsonl").exit_code == 0
    monkeypatch.setattr(optimize, "MIN_LABELS_TO_TUNE", 4)

    # The human says yes to everything. The default (honest) judge often says no;
    # the "tuned" judge says yes too, so it agrees more and gets saved.
    def fake_tune(train: Any, judge_lm: Any, seed: int) -> Judge:
        judge = make_judge()
        judge.judge.demos = [lb.as_example() for lb in train[:1]]
        judge.set_lm(yes_judge())
        return judge

    monkeypatch.setattr(optimize, "_tune", fake_tune)
    result = cli("calibrate", "prompt.plan.yaml", "-n", "10", input="y\n" * 10)
    assert result.exit_code == 0, result.output
    assert "Tuned judge" in result.output
    assert "saved to prompt.plan.judge.json" in result.output
    saved = read_calibrated(project / "prompt.plan.judge.json")
    assert saved.judge_model == "fake/judge"

    rerun = cli("run", "prompt.txt", "data.jsonl")
    assert rerun.exit_code == 0, rerun.output
    assert "Using the calibrated judge" in rerun.output
    assert f"(calibrated {saved.fingerprint})" in rerun.output

    other = cli("run", "prompt.txt", "data.jsonl", "--judge-model", "other/judge")
    assert "was tuned for fake/judge" in other.output
