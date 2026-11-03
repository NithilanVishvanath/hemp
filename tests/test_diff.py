import json
from pathlib import Path

import pytest
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan
from typer.testing import CliRunner

from hemp.cli import app
from hemp.dataset import Row
from hemp.plan.schema import Criterion, EvalPlan
from hemp.runner import RunResult, run_eval
from hemp.store import diff_runs

LABELS = {r["review"]: r["label"] for r in SENTIMENT_ROWS}


def perfect(prompt: str) -> str:
    return next(label for review, label in LABELS.items() if review in prompt)


def run(
    answer: object,
    rows: list[Row] = SENTIMENT_ROWS,
    prompt: str = SENTIMENT_PROMPT,
    plan: EvalPlan | None = None,
    run_id: int = 1,
) -> RunResult:
    result = run_eval(
        prompt,
        rows,
        plan or make_plan(),
        lm=FakeLM(answer),
        baselines=False,  # type: ignore[arg-type]
    )
    return result.model_copy(update={"id": run_id})


def check(diff: object, check_id: str) -> object:
    return next(c for c in diff.checks if c.check_id == check_id)  # type: ignore[attr-defined]


def test_fixed_examples() -> None:
    a = run(keyword_sentiment, run_id=1)
    b = run(perfect, prompt=SENTIMENT_PROMPT + "\nThink carefully.", run_id=2)
    diff = diff_runs(a, b, SENTIMENT_ROWS, SENTIMENT_ROWS)

    assert diff.changed == ["prompt"]
    assert diff.same == ["plan", "dataset", "target model", "judge model"]
    exact = check(diff, "exact_match_label")
    assert exact.change == pytest.approx(25.0)  # 75% -> 100%
    assert len(exact.fixed) == 3 and exact.broke == []
    fixed = exact.fixed[0]
    assert fixed.row["review"] == "It's ok"
    assert (fixed.output_a, fixed.output_b) == ("positive", "neutral")
    assert diff.n_fixed == 3 and diff.n_broke == 0
    assert not diff.regressed


def test_broke_examples_and_regression() -> None:
    diff = diff_runs(run(perfect), run(keyword_sentiment), SENTIMENT_ROWS, SENTIMENT_ROWS)
    assert diff.regressed
    assert len(check(diff, "exact_match_label").broke) == 3


def test_same_setup_has_no_changes() -> None:
    diff = diff_runs(run(perfect), run(perfect), SENTIMENT_ROWS, SENTIMENT_ROWS)
    assert diff.changed == []
    assert diff.n_fixed == diff.n_broke == 0


def test_examples_pair_by_row_contents() -> None:
    rows_b = [*reversed(SENTIMENT_ROWS), {"review": "Meh", "label": "neutral"}]
    a = run(keyword_sentiment)
    b = run(perfect_or_neutral, rows=rows_b)
    diff = diff_runs(a, b, SENTIMENT_ROWS, rows_b)
    assert "dataset" in diff.changed
    assert (diff.n_paired, diff.n_only_a, diff.n_only_b) == (12, 0, 1)
    assert len(check(diff, "exact_match_label").fixed) == 3  # still the "It's ok" rows


def perfect_or_neutral(prompt: str) -> str:
    return "neutral" if "Meh" in prompt else perfect(prompt)


def test_plan_and_question_changes() -> None:
    plan_a = make_plan().model_copy(
        update={"criteria": [Criterion(id="q1", question="Is it right?")]}
    )
    plan_b = make_plan().model_copy(
        update={"criteria": [Criterion(id="q1", question="Is it exactly right?")]}
    )
    a = run(perfect, plan=plan_a).model_copy(update={"questions": {"q1": "Is it right?"}})
    b = run(perfect, plan=plan_b).model_copy(update={"questions": {"q1": "Is it exactly right?"}})
    diff = diff_runs(a, b, SENTIMENT_ROWS, SENTIMENT_ROWS)
    assert "plan" in diff.changed
    assert diff.question_edits == ["q1"]


def test_check_only_in_one_run() -> None:
    plan_b = make_plan()
    plan_b = plan_b.model_copy(update={"format_checks": plan_b.format_checks[:1]})
    diff = diff_runs(run(perfect), run(perfect, plan=plan_b), SENTIMENT_ROWS, SENTIMENT_ROWS)
    allowed = check(diff, "allowed_values")
    assert allowed.a is not None and allowed.b is None
    assert allowed.change is None


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import hemp.cli

    def answer(prompt: str) -> str:
        # The "improved" prompt gets every label right; the first one misses neutral.
        return perfect(prompt) if "carefully" in prompt else keyword_sentiment(prompt)

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEMP_DB", raising=False)
    monkeypatch.setattr(hemp.cli, "make_lm", lambda model: FakeLM(answer, model))
    (tmp_path / "v1.txt").write_text(SENTIMENT_PROMPT, encoding="utf-8")
    (tmp_path / "v2.txt").write_text(SENTIMENT_PROMPT + "\nThink carefully.", encoding="utf-8")
    data = "".join(json.dumps(r) + "\n" for r in SENTIMENT_ROWS)
    (tmp_path / "data.jsonl").write_text(data, encoding="utf-8")
    return tmp_path


def cli(*args: str) -> object:
    return CliRunner().invoke(app, list(args))


def test_cli_diff(project: Path) -> None:
    assert cli("plan", "v1.txt", "data.jsonl", "--no-llm", "-o", "plan.yaml").exit_code == 0
    first = cli("run", "v1.txt", "data.jsonl", "--plan", "plan.yaml")
    assert first.exit_code == 0, first.output
    too_few = cli("diff")
    assert too_few.exit_code == 1
    assert "at least two runs" in too_few.output

    assert cli("run", "v2.txt", "data.jsonl", "--plan", "plan.yaml").exit_code == 0
    result = cli("diff")  # the two latest runs
    assert result.exit_code == 0, result.output
    assert "Changed: prompt" in result.output
    assert "+Think carefully." in result.output
    assert "+25 pts" in result.output
    assert "Fixed (3)" in result.output
    assert "Broke (" not in result.output

    backwards = cli("diff", "2", "1", "--fail-on-regression")
    assert backwards.exit_code == 1
    assert "Broke (3)" in backwards.output
    assert "-25 pts" in backwards.output

    assert cli("diff", "1", "--fail-on-regression").exit_code == 0  # 1 vs latest


def test_cli_diff_errors(project: Path) -> None:
    cli("plan", "v1.txt", "data.jsonl", "--no-llm", "-o", "plan.yaml")
    cli("run", "v1.txt", "data.jsonl", "--plan", "plan.yaml")
    same = cli("diff", "1", "1")
    assert same.exit_code == 1
    assert "two different runs" in same.output
    missing = cli("diff", "1", "9")
    assert missing.exit_code == 1
    assert "no run #9" in missing.output
