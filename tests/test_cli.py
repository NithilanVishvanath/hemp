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


def test_stub_command_exits_with_message(files: tuple[Path, Path]) -> None:
    prompt, data = files
    result = runner.invoke(app, ["run", str(prompt), str(data)])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


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
    assert "planner_model: fake/m" in text


def test_validate_reports_problems(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1
    assert "meta" in result.output


def test_metrics_lists_builtins() -> None:
    result = runner.invoke(app, ["metrics"])
    assert result.exit_code == 0
    assert "exact_match" in result.output
    assert "allowed_values" in result.output
