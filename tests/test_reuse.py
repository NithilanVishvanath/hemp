"""Baselines are copied from an earlier run when nothing they depend on changed."""

import json
from pathlib import Path

import pytest
from dspy.utils import DummyLM
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan
from typer.testing import CliRunner

from hemp.cli import app
from hemp.plan.schema import Criterion
from hemp.runner import RunError, RunResult, run_eval
from hemp.store import Store

V2_PROMPT = SENTIMENT_PROMPT + "\nBe careful."


def stored_run(store: Store, result: RunResult) -> RunResult:
    run_id = store.save_run(result, prompt_text=SENTIMENT_PROMPT, rows=SENTIMENT_ROWS)
    return store.load_run(run_id)


def test_reused_baselines_make_no_model_calls(tmp_path: Path) -> None:
    plan = make_plan()
    with Store(tmp_path / "runs.db") as store:
        first = stored_run(store, run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM()))
        lm = FakeLM(keyword_sentiment)
        second = run_eval(V2_PROMPT, SENTIMENT_ROWS, plan, lm=lm, reuse=first)

    assert len(lm.prompts) == len(SENTIMENT_ROWS)  # only the prompt itself ran
    assert [v.name for v in second.variants] == ["prompt", "majority_class", "zero_shot"]
    assert second.baselines_from == {"majority_class": first.id, "zero_shot": first.id}
    assert second.variant("zero_shot") == first.variant("zero_shot")


def test_reuse_points_at_the_run_that_computed_them(tmp_path: Path) -> None:
    plan = make_plan()
    with Store(tmp_path / "runs.db") as store:
        first = stored_run(store, run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM()))
        second = stored_run(
            store, run_eval(V2_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), reuse=first)
        )
        third = run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), reuse=second)
    assert set(third.baselines_from.values()) == {first.id}


def test_reused_baselines_keep_their_judge_verdicts(tmp_path: Path) -> None:
    plan = make_plan().model_copy(
        update={"criteria": [Criterion(id="q1", question="Is the label right?")]}
    )

    def judge() -> DummyLM:
        lm = DummyLM({"Is the label right?": {"reason": "ok", "answer": "True"}})
        lm.model = "fake/judge"
        return lm

    with Store(tmp_path / "runs.db") as store:
        first = stored_run(
            store,
            run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), judge_lm=judge()),
        )
        second_judge = judge()
        second = run_eval(
            V2_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), judge_lm=second_judge, reuse=first
        )
    assert len(second_judge.history) == len(SENTIMENT_ROWS)  # only the prompt was judged
    zero_shot = second.variant("zero_shot")
    assert zero_shot is not None
    assert zero_shot.examples[0].checks[-1].check_id == "q1"


def test_refuses_to_reuse_a_different_setup(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        first = stored_run(
            store, run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, make_plan(), lm=FakeLM())
        )
    with pytest.raises(RunError, match="target model differ"):
        run_eval(
            V2_PROMPT, SENTIMENT_ROWS, make_plan(), lm=FakeLM(model="other/model"), reuse=first
        )


def test_find_reusable_run(tmp_path: Path) -> None:
    plan = make_plan()
    with Store(tmp_path / "runs.db") as store:

        def find(target: str = "fake/target", judge: str | None = None) -> int | None:
            return store.find_reusable_run(
                plan_hash=plan.plan_hash(),
                dataset_hash=first.info.dataset_hash,
                target_model=target,
                judge_model=judge,
            )

        first = stored_run(store, run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM()))
        no_baselines = run_eval(
            SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=FakeLM(), baselines=False
        )
        stored_run(store, no_baselines)

        assert find() == first.id  # the later run has no baselines to copy
        assert find(target="other/model") is None
        assert find(judge="fake/judge") is None


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture
def lms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[FakeLM]:
    import hemp.cli

    made: list[FakeLM] = []

    def make(model: str) -> FakeLM:
        made.append(FakeLM(keyword_sentiment, model))
        return made[-1]

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEMP_DB", raising=False)
    monkeypatch.setattr(hemp.cli, "make_lm", make)
    (tmp_path / "v1.txt").write_text(SENTIMENT_PROMPT, encoding="utf-8")
    (tmp_path / "v2.txt").write_text(V2_PROMPT, encoding="utf-8")
    data = "".join(json.dumps(r) + "\n" for r in SENTIMENT_ROWS)
    (tmp_path / "data.jsonl").write_text(data, encoding="utf-8")
    CliRunner().invoke(app, ["plan", "v1.txt", "data.jsonl", "--no-llm", "-o", "plan.yaml"])
    return made


def run_cli(*args: str) -> str:
    result = CliRunner().invoke(app, ["run", *args, "data.jsonl", "--plan", "plan.yaml"])
    assert result.exit_code == 0, result.output
    return result.output


def test_cli_reuses_baselines(lms: list[FakeLM]) -> None:
    first = run_cli("v1.txt")
    assert "copied from" not in first
    assert len(lms[-1].prompts) == 2 * len(SENTIMENT_ROWS)  # prompt + zero_shot

    second = run_cli("v2.txt")
    assert "Baselines copied from run #1" in second
    assert "50% (6/12)" in second  # majority_class still shown
    assert len(lms[-1].prompts) == len(SENTIMENT_ROWS)

    fresh = run_cli("v2.txt", "--fresh-baselines")
    assert "copied from" not in fresh
    assert len(lms[-1].prompts) == 2 * len(SENTIMENT_ROWS)
