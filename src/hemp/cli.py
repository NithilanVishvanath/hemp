"""Command-line interface for HEMP."""

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from hemp._version import __version__
from hemp.config import ModelConfig, make_lm
from hemp.dataset import DatasetError
from hemp.metrics import list_metrics
from hemp.metrics.registry import param_docs
from hemp.plan import PlanError, PlannerError, create_plan, read_plan, write_plan
from hemp.plan.render import render_plan
from hemp.profile import profile_dataset
from hemp.profile.render import print_profile

app = typer.Typer(
    name="hemp",
    help="HEMP (Help Evaluate My Prompt): plan, run, and compare evaluations of a prompt.",
    no_args_is_help=True,
    add_completion=False,
)
# emoji=False: model names like 'minimax-m3:cloud' must not turn into emoji.
console = Console(emoji=False)
err_console = Console(stderr=True, emoji=False)

PromptArg = Annotated[
    Path, typer.Argument(help="Prompt template file.", exists=True, dir_okay=False)
]
DataArg = Annotated[
    Path,
    typer.Argument(help="Dataset file (JSONL, JSON, CSV, or TSV).", exists=True, dir_okay=False),
]


def _fail(message: str) -> NoReturn:
    # markup=False: messages can contain [brackets] from user data or model errors.
    err_console.print(message, style="red", markup=False)
    raise typer.Exit(code=1)


def _warn(message: str) -> None:
    err_console.print(f"Warning: {message}", style="yellow", markup=False)


def _not_implemented(command: str, milestone: int) -> NoReturn:
    err_console.print(
        f"[yellow]`hemp {command}` is not implemented yet[/yellow] "
        f"(planned for milestone {milestone}; see CLAUDE.md)."
    )
    raise typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"hemp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the HEMP version and exit.",
        ),
    ] = False,
) -> None:
    """HEMP (Help Evaluate My Prompt)."""
    # Don't crash on characters the terminal can't encode (e.g. Windows cp1252 consoles).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.command()
def profile(
    data: DataArg,
    prompt: Annotated[
        Path | None,
        typer.Option(
            "--prompt",
            "-p",
            help="Prompt template file. Its {placeholders} tell inputs apart from gold columns.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    samples: Annotated[int, typer.Option(help="Number of sample rows to keep.", min=0)] = 5,
    as_json: Annotated[bool, typer.Option("--json", help="Print the profile as JSON.")] = False,
) -> None:
    """Profile a dataset: column types, roles, value stats. No model calls."""
    prompt_text = prompt.read_text(encoding="utf-8-sig") if prompt else None
    try:
        result = profile_dataset(data, prompt=prompt_text, n_samples=samples)
    except DatasetError as e:
        _fail(str(e))
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        print_profile(result)


@app.command()
def plan(
    prompt: PromptArg,
    data: DataArg,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where to write the plan. Default: <prompt>.plan.yaml"),
    ] = None,
    planner_model: Annotated[
        str | None,
        typer.Option(
            help="LiteLLM model for planning, e.g. ollama_chat/qwen2.5:14b or openai/gpt-4o-mini. "
            "Default: $HEMP_PLANNER_MODEL, $HEMP_MODEL, or a local Ollama model."
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Skip the LLM; only rule-based checks and baselines."),
    ] = False,
    force: Annotated[bool, typer.Option(help="Overwrite an existing plan file.")] = False,
) -> None:
    """Profile the dataset and write an editable evaluation plan (YAML)."""
    out_path = out or prompt.with_name(f"{prompt.stem}.plan.yaml")
    if out_path.exists() and not force:
        _fail(
            f"{out_path} already exists and may contain your edits. "
            "Pass --force to overwrite it, or --out to write somewhere else."
        )
    prompt_text = prompt.read_text(encoding="utf-8-sig")
    try:
        data_profile = profile_dataset(data, prompt=prompt_text)
    except DatasetError as e:
        _fail(str(e))
    for field in data_profile.missing_prompt_fields:
        _warn(
            f"the prompt uses {{{field}}} but the dataset has no '{field}' column. "
            "Runs will fail until they match."
        )

    model = planner_model or ModelConfig.from_env().planner
    try:
        if no_llm:
            result = create_plan(prompt_text, data_profile, lm=None)
        else:
            with err_console.status(f"Planning with {model}..."):
                result = create_plan(
                    prompt_text, data_profile, lm=make_lm(model), planner_model=model
                )
    except PlannerError as e:
        _fail(str(e))

    write_plan(result.plan, out_path, overwrite=True)
    console.print(render_plan(result.plan))
    for warning in result.warnings:
        _warn(warning)
    console.print(
        f"\nWrote [bold]{out_path}[/bold]. Review and edit it, then check it with "
        f"[bold]hemp validate {out_path}[/bold]."
    )


@app.command()
def validate(
    plan_file: Annotated[
        Path, typer.Argument(help="Evaluation plan (YAML).", exists=True, dir_okay=False)
    ],
) -> None:
    """Check an edited plan file and show its hash."""
    try:
        loaded = read_plan(plan_file)
    except PlanError as e:
        _fail(str(e))
    console.print(render_plan(loaded))
    console.print(f"\n[green]Plan is valid.[/green] Hash: {loaded.plan_hash()}")


@app.command()
def metrics() -> None:
    """List the metrics a plan can use, with their params."""
    table = Table(header_style="bold")
    table.add_column("Metric")
    table.add_column("Layer")
    table.add_column("What it checks", overflow="fold")
    table.add_column("Params", overflow="fold")
    for info in list_metrics():
        params = "\n".join(f"{k}: {v}" for k, v in param_docs(info.params_model).items())
        table.add_row(info.name, info.layer.value, info.description, params or "-")
    console.print(table)


@app.command()
def run(prompt: PromptArg, data: DataArg) -> None:
    """Execute the frozen evaluation plan and store results."""
    _not_implemented("run", 5)


@app.command()
def diff(
    run_a: Annotated[str, typer.Argument(help="Baseline run ID.")],
    run_b: Annotated[str, typer.Argument(help="Run ID to compare against the baseline.")],
) -> None:
    """Show criterion-level regressions and improvements between two runs."""
    _not_implemented("diff", 7)


@app.command()
def calibrate(
    plan_file: Annotated[
        Path, typer.Argument(help="Evaluation plan (YAML).", exists=True, dir_okay=False)
    ],
) -> None:
    """Label a few outputs, optimize the judge, and report judge-human agreement."""
    _not_implemented("calibrate", 8)


@app.command()
def show(run_id: Annotated[str, typer.Argument(help="Run ID to display.")]) -> None:
    """Show the results of a stored run."""
    _not_implemented("show", 5)
