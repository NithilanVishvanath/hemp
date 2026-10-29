"""Command-line interface for HEMP."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from hemp import __version__

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
    Path, typer.Argument(help="Dataset file (JSONL or CSV).", exists=True, dir_okay=False)
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
