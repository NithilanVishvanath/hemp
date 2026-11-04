from pathlib import Path

import pytest
from typer.testing import CliRunner

from hemp import __version__
from hemp.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("profile", "plan", "validate", "metrics", "run", "diff", "calibrate", "show"):
        assert command in result.output


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Classify: {text}")
    data = tmp_path / "data.jsonl"
    data.write_text('{"text": "hi"}\n')
    return prompt, data


def test_plan_rejects_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(tmp_path / "nope.txt"), str(tmp_path / "x.jsonl")])
    assert result.exit_code == 2


def test_profile_command_table(files: tuple[Path, Path]) -> None:
    prompt, data = files
    result = runner.invoke(app, ["profile", str(data), "--prompt", str(prompt)])
    assert result.exit_code == 0, result.output
    assert "text" in result.output
    assert "input" in result.output


def test_profile_command_json(files: tuple[Path, Path]) -> None:
    import json

    _, data = files
    result = runner.invoke(app, ["profile", str(data), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_rows"] == 1
    assert payload["columns"][0]["name"] == "text"


def test_profile_command_bad_dataset(tmp_path: Path) -> None:
    data = tmp_path / "bad.jsonl"
    data.write_text("not json\n")
    result = runner.invoke(app, ["profile", str(data)])
    assert result.exit_code == 1
    assert "line 1" in result.output


def test_plan_no_llm_writes_valid_plan(files: tuple[Path, Path]) -> None:
    prompt, data = files
    result = runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    assert result.exit_code == 0, result.output
    plan_path = prompt.with_name("prompt.plan.yaml")
    assert plan_path.exists()
    assert "without an LLM" in result.output

    again = runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    assert again.exit_code == 1
    assert "--force" in again.output

    check = runner.invoke(app, ["validate", str(plan_path)])
    assert check.exit_code == 0, check.output
    assert "Plan is valid" in check.output


def test_plan_with_fake_model(files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from dspy.utils import DummyLM

    import hemp.cli

    answer = {
        "reasoning": "r",
        "format_checks": json.dumps([{"metric": "length", "params": {"max": 3}}]),
        "criteria": json.dumps([{"question": "Is the label justified by the text?"}]),
    }
    monkeypatch.setattr(hemp.cli, "make_lm", lambda model: DummyLM([answer]))
    prompt, data = files
    out = prompt.with_name("custom.yaml")
    result = runner.invoke(
        app, ["plan", str(prompt), str(data), "--out", str(out), "--planner-model", "fake/m"]
    )
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "Is the label justified by the text?" in text
    assert "planner: fake/m" in text


def test_validate_reports_problems(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "columns: Field required" in result.output


def test_metrics_lists_builtins() -> None:
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0
    assert "exact_match" in result.output
    assert "allowed_values" in result.output


# --- run / show ---------------------------------------------------------------------


@pytest.fixture
def sentiment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    import json

    from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment

    import hemp.cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEMP_DB", raising=False)
    monkeypatch.setattr(hemp.cli, "make_lm", lambda model: FakeLM(keyword_sentiment, model))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(SENTIMENT_PROMPT, encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in SENTIMENT_ROWS), encoding="utf-8")
    return prompt, data


def test_run_needs_a_plan(sentiment: tuple[Path, Path]) -> None:
    prompt, data = sentiment
    result = runner.invoke(app, ["run", str(prompt), str(data)])
    assert result.exit_code == 1
    assert "No plan found" in result.output
    assert "hemp plan" in result.output


def test_run_then_show(sentiment: tuple[Path, Path], tmp_path: Path) -> None:
    prompt, data = sentiment
    planned = runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    assert planned.exit_code == 0, planned.output

    result = runner.invoke(app, ["run", str(prompt), str(data), "--target-model", "fake/m"])
    assert result.exit_code == 0, result.output
    assert "exact_match_label" in result.output
    assert "75% (9/12)" in result.output  # the prompt
    assert "50% (6/12)" in result.output  # majority_class baseline
    assert "Saved as run #1" in result.output
    assert (tmp_path / ".hemp" / "hemp.db").exists()

    shown = runner.invoke(app, ["show", "1"])
    assert shown.exit_code == 0, shown.output
    assert "75% (9/12)" in shown.output
    assert "Failed examples (3" in shown.output
    assert "It's ok" in shown.output

    listed = runner.invoke(app, ["show"])
    assert listed.exit_code == 0, listed.output
    assert "#1" in listed.output
    assert "fake/m" in listed.output


def test_run_without_baselines_and_custom_db(sentiment: tuple[Path, Path], tmp_path: Path) -> None:
    prompt, data = sentiment
    runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    db = tmp_path / "custom.db"
    result = runner.invoke(app, ["run", str(prompt), str(data), "--no-baselines", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "(baseline)" not in result.output
    assert db.exists()


def test_run_warns_when_data_changed(sentiment: tuple[Path, Path]) -> None:
    prompt, data = sentiment
    runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    with data.open("a", encoding="utf-8") as f:
        f.write('{"review": "Meh", "label": "neutral"}\n')
    result = runner.invoke(app, ["run", str(prompt), str(data)])
    assert result.exit_code == 0, result.output
    assert "has changed since the plan was made" in result.output


def test_run_reports_unreachable_model(
    sentiment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import FakeLM

    import hemp.cli

    prompt, data = sentiment
    runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    monkeypatch.setattr(
        hemp.cli, "make_lm", lambda model: FakeLM(lambda p: ConnectionError("refused"), model)
    )
    result = runner.invoke(app, ["run", str(prompt), str(data)])
    assert result.exit_code == 1
    assert "ollama serve" in result.output
    assert "Saved" not in result.output


def test_show_unknown_run(sentiment: tuple[Path, Path]) -> None:
    result = runner.invoke(app, ["show", "42"])
    assert result.exit_code == 1
    assert "no run #42" in result.output


def test_show_with_no_runs(sentiment: tuple[Path, Path]) -> None:
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 0
    assert "No runs yet" in result.output


def _add_question(prompt: Path) -> None:
    plan = prompt.with_name("prompt.plan.yaml")
    text = plan.read_text(encoding="utf-8")
    text = text.replace("questions: []", "questions:\n  - id: q1\n    ask: Negative?")
    plan.write_text(text, encoding="utf-8")


def test_run_with_judge(sentiment: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    from dspy.utils import DummyLM
    from fakes import FakeLM, keyword_sentiment

    import hemp.cli

    def fake_make_lm(model: str) -> object:
        if model == "fake/judge":
            lm = DummyLM(
                {
                    "negative\n": {"reason": "Negative.", "answer": "True"},
                    "positive\n": {"reason": "Not negative.", "answer": "False"},
                }
            )
            lm.model = model
            return lm
        return FakeLM(keyword_sentiment, model)

    monkeypatch.setattr(hemp.cli, "make_lm", fake_make_lm)
    prompt, data = sentiment
    runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    _add_question(prompt)

    result = runner.invoke(app, ["run", str(prompt), str(data), "--judge-model", "fake/judge"])
    assert result.exit_code == 0, result.output
    assert "judge: fake/judge" in result.output
    assert "Judge questions" in result.output
    assert "Negative?" in result.output
    assert "25% (3/12)" in result.output
    assert "q1: Not negative." in result.output  # judge's reason on a failed example

    shown = runner.invoke(app, ["show", "1"])
    assert "Negative?" in shown.output


def test_run_no_judge(sentiment: tuple[Path, Path]) -> None:
    prompt, data = sentiment
    runner.invoke(app, ["plan", str(prompt), str(data), "--no-llm"])
    _add_question(prompt)
    result = runner.invoke(app, ["run", str(prompt), str(data), "--no-judge"])
    assert result.exit_code == 0, result.output
    assert "Skipped 1 judge question(s)" in result.output
    assert "Judge questions" not in result.output
