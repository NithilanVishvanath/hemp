"""Command-line interface for HEMP."""

import sys
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from hemp import __version__
from hemp.dataset import DatasetError
from hemp.profile import profile_dataset
from hemp.profile.render import print_profile

app = typer.Typer(
    name="hemp",
    help="HEMP (Help Evaluate My Prompt): plan, run, and compare evaluations of a prompt.",
    no_args_is_help=True,
    add_completion=False,
)
err_console = Console(stderr=True)

PromptArg = Annotated[
    Path, typer.Argument(help="Prompt template file.", exists=True, dir_okay=False)
]
DataArg = Annotated[
    Path,
    typer.Argument(help="Dataset file (JSONL, JSON, CSV, or TSV).", exists=True, dir_okay=False),
]


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
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1) from None
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        print_profile(result)


@app.command()
def plan(prompt: PromptArg, data: DataArg) -> None:
    """Profile the dataset and generate an editable evaluation plan."""
    _not_implemented("plan", 3)


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
