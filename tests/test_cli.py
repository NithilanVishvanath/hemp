from pathlib import Path

import pytest
from typer.testing import CliRunner

from hemp import __version__
from hemp.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("plan", "run", "diff", "calibrate", "show"):
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


@pytest.mark.parametrize("command", ["plan", "run"])
def test_stub_commands_exit_with_message(command: str, files: tuple[Path, Path]) -> None:
    prompt, data = files
    result = runner.invoke(app, [command, str(prompt), str(data)])
    assert result.exit_code == 1
    assert "not implemented yet" in result.output


def test_plan_rejects_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plan", str(tmp_path / "nope.txt"), str(tmp_path / "x.jsonl")])
    assert result.exit_code == 2
