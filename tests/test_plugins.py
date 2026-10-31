import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from dspy.utils import DummyLM
from typer.testing import CliRunner

from hemp.cli import app
from hemp.metrics import MetricContext, MetricError, get_metric, load_plugin
from hemp.metrics.plugins import plugin_ref
from hemp.plan import PlanError, create_plan, read_plan, write_plan
from hemp.profile import profile_rows

PLUGIN = '''
import hemp
from pydantic import Field


class PoliteParams(hemp.MetricParams):
    word: str = Field("please", description="word a polite reply contains")


@hemp.metric("polite", params=PoliteParams)
def polite(ctx: hemp.MetricContext, params: PoliteParams) -> hemp.MetricResult:
    """The output contains a polite word."""
    return hemp.MetricResult.check(params.word in ctx.output.lower())
'''


def write_plugin(directory: Path, name: str = "my_metrics.py", body: str = PLUGIN) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_load_file_plugin(tmp_path: Path) -> None:
    write_plugin(tmp_path)
    assert load_plugin("my_metrics.py", tmp_path) == ["polite"]
    info = get_metric("polite")
    assert info.description == "The output contains a polite word."
    assert info(MetricContext(output="Please wait"), {}).passed
    assert load_plugin("my_metrics.py", tmp_path) == []  # loading again is a no-op


def test_load_module_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "my_hemp_metrics"
    write_plugin(package, "__init__.py")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_plugin("my_hemp_metrics") == ["polite"]


def test_missing_plugin(tmp_path: Path) -> None:
    with pytest.raises(MetricError, match="not found"):
        load_plugin("nope.py", tmp_path)
    with pytest.raises(MetricError, match="Couldn't load metric plugin no_such_module"):
        load_plugin("no_such_module")


def test_broken_plugin_names_the_error(tmp_path: Path) -> None:
    write_plugin(tmp_path, "broken.py", "def broken(:\n")
    with pytest.raises(MetricError, match=r"broken\.py: SyntaxError"):
        load_plugin("broken.py", tmp_path)


def test_plugin_cannot_replace_builtin(tmp_path: Path) -> None:
    body = PLUGIN.replace('"polite"', '"exact_match"')
    write_plugin(tmp_path, "clash.py", body)
    with pytest.raises(MetricError, match="already registered"):
        load_plugin("clash.py", tmp_path)


def test_plugin_ref(tmp_path: Path) -> None:
    plugin = write_plugin(tmp_path / "metrics")
    assert plugin_ref(str(plugin), tmp_path) == "metrics/my_metrics.py"
    assert plugin_ref(str(plugin), tmp_path / "plans") == "../metrics/my_metrics.py"
    assert plugin_ref("mypkg.metrics", tmp_path) == "mypkg.metrics"


def custom_plan(tmp_path: Path) -> Path:
    """A plan in tmp_path/plans that uses the 'polite' metric from tmp_path/metrics."""
    write_plugin(tmp_path / "metrics")
    load_plugin("metrics/my_metrics.py", tmp_path)
    profile = profile_rows([{"q": "hi", "a": "hello"}] * 3, prompt="{q}")
    lm = DummyLM(
        [
            {
                "reasoning": "r",
                "format_checks": json.dumps([{"metric": "polite", "params": {"word": "thanks"}}]),
                "criteria": "[]",
            }
        ]
    )
    plan = create_plan(
        "{q}",
        profile,
        lm=lm,
        plugins=["../metrics/my_metrics.py"],
        now=datetime(2026, 1, 1, tzinfo=UTC),
    ).plan
    assert [c.metric for c in plan.custom_metrics] == ["polite"]
    assert plan.custom_metrics[0].params == {"word": "thanks"}
    return write_plan(plan, tmp_path / "plans" / "p.plan.yaml")


def test_planner_can_pick_plugin_metric_and_plan_reloads_it(tmp_path: Path) -> None:
    path = custom_plan(tmp_path)
    # Simulate a fresh process: the metric isn't registered until the plan loads it.
    from hemp.metrics import registry

    del registry._REGISTRY["polite"]
    import sys

    for name in [m for m in sys.modules if m.startswith("hemp_plugin_")]:
        del sys.modules[name]

    plan = read_plan(path)
    assert plan.plugins == ["../metrics/my_metrics.py"]
    assert get_metric("polite").layer.value == "custom"


def test_plan_with_missing_plugin_fails_clearly(tmp_path: Path) -> None:
    path = custom_plan(tmp_path)
    (tmp_path / "metrics" / "my_metrics.py").unlink()
    from hemp.metrics import registry

    del registry._REGISTRY["polite"]
    import sys

    for name in [m for m in sys.modules if m.startswith("hemp_plugin_")]:
        del sys.modules[name]
    with pytest.raises(PlanError, match="Metric plugin not found"):
        read_plan(path)


def test_cli_plugin_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    write_plugin(tmp_path)
    (tmp_path / "prompt.txt").write_text("Reply to {q}")
    (tmp_path / "data.jsonl").write_text('{"q": "hi"}\n')
    runner = CliRunner()

    listed = runner.invoke(app, ["metrics", "--plugin", "my_metrics.py"])
    assert listed.exit_code == 0, listed.output
    assert "polite" in listed.output

    out = runner.invoke(
        app,
        [
            "plan",
            "prompt.txt",
            "data.jsonl",
            "--no-llm",
            "--plugin",
            "my_metrics.py",
            "--out",
            "plans/p.yaml",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "plugins: [../my_metrics.py]" in (tmp_path / "plans" / "p.yaml").read_text()

    checked = runner.invoke(app, ["validate", "plans/p.yaml"])
    assert checked.exit_code == 0, checked.output


def test_cli_bad_plugin(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["metrics", "--plugin", str(tmp_path / "missing.py")])
    assert result.exit_code == 1
    assert "not found" in result.output
