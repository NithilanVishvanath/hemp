"""Command-line interface for HEMP."""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from hemp._version import __version__
from hemp.config import ModelConfig, make_lm
from hemp.dataset import DatasetError, load_dataset
from hemp.metrics import MetricError, list_metrics, load_plugin
from hemp.metrics.plugins import plugin_ref
from hemp.metrics.registry import param_docs
from hemp.plan import PlanError, PlannerError, create_plan, read_plan, write_plan
from hemp.plan.render import render_plan
from hemp.profile import profile_dataset
from hemp.profile.profiler import dataset_hash
from hemp.profile.render import print_profile
from hemp.runner import PROMPT_VARIANT, RunError, RunResult, TargetError, run_eval
from hemp.runner.render import render_failures, render_run, render_runs
from hemp.store import Store, StoreError

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
PluginOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--plugin",
        help="Python file or module with your own @hemp.metric metrics. Repeatable.",
    ),
]

DbOpt = Annotated[
    Path | None,
    typer.Option("--db", help="Results database. Default: $HEMP_DB or .hemp/hemp.db"),
]


def _load_plugins(specs: list[str]) -> None:
    for spec in specs:
        try:
            names = load_plugin(spec)
        except MetricError as e:
            _fail(str(e))
        if names:
            err_console.print(f"Loaded {spec}: {', '.join(names)}", style="dim", markup=False)


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
            help="LiteLLM model for planning, e.g. ollama_chat/qwen2.5:7b or openai/gpt-4o-mini. "
            "Default: $HEMP_PLANNER_MODEL, $HEMP_MODEL, or a local Ollama model."
        ),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Skip the LLM; only rule-based checks and baselines."),
    ] = False,
    force: Annotated[bool, typer.Option(help="Overwrite an existing plan file.")] = False,
    plugin: PluginOpt = None,
) -> None:
    """Profile the dataset and write an editable evaluation plan (YAML)."""
    out_path = out or prompt.with_name(f"{prompt.stem}.plan.yaml")
    plugins = plugin or []
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

    _load_plugins(plugins)
    plugin_refs = [plugin_ref(spec, out_path.parent) for spec in plugins]
    models = ModelConfig.from_env()
    model = planner_model or models.planner
    try:
        if no_llm:
            result = create_plan(
                prompt_text, data_profile, lm=None, embed_model=models.embed, plugins=plugin_refs
            )
        else:
            with err_console.status(f"Planning with {model}..."):
                result = create_plan(
                    prompt_text,
                    data_profile,
                    lm=make_lm(model),
                    planner_model=model,
                    embed_model=models.embed,
                    plugins=plugin_refs,
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
def metrics(plugin: PluginOpt = None) -> None:
    """List the metrics a plan can use, with their params."""
    _load_plugins(plugin or [])
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
def run(
    prompt: PromptArg,
    data: DataArg,
    plan_file: Annotated[
        Path | None,
        typer.Option("--plan", help="Plan to run. Default: <prompt>.plan.yaml"),
    ] = None,
    target_model: Annotated[
        str | None,
        typer.Option(
            help="LiteLLM model that runs your prompt. "
            "Default: $HEMP_TARGET_MODEL, $HEMP_MODEL, or a local Ollama model."
        ),
    ] = None,
    workers: Annotated[int, typer.Option(help="How many model calls to make at once.", min=1)] = 4,
    no_baselines: Annotated[
        bool, typer.Option("--no-baselines", help="Skip the plan's baselines.")
    ] = False,
    db: DbOpt = None,
) -> None:
    """Run your prompt on the dataset, score it with the plan, and store the results."""
    plan_path = plan_file or prompt.with_name(f"{prompt.stem}.plan.yaml")
    if not plan_path.is_file():
        _fail(
            f"No plan found at {plan_path}. Create one with `hemp plan {prompt} {data}`, "
            "or point to one with --plan."
        )
    try:
        loaded = read_plan(plan_path)
        rows, _ = load_dataset(data)
    except (PlanError, DatasetError) as e:
        _fail(str(e))
    if dataset_hash(rows) != loaded.meta.dataset_hash:
        _warn(
            f"{data} has changed since the plan was made. That's fine if you changed it on purpose."
        )
    prompt_text = prompt.read_text(encoding="utf-8-sig")
    model = target_model or ModelConfig.from_env().target

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=err_console,
        transient=True,
    ) as bar:

        def track(variant: str, total: int) -> Callable[[], None]:
            label = "your prompt" if variant == PROMPT_VARIANT else f"baseline {variant}"
            task = bar.add_task(f"Running {label}", total=total)
            return lambda: bar.advance(task)

        try:
            result = run_eval(
                prompt_text,
                rows,
                loaded,
                lm=make_lm(model),
                baselines=not no_baselines,
                workers=workers,
                prompt_source=_display_path(prompt),
                dataset_source=_display_path(data),
                plan_source=_display_path(plan_path),
                progress=track,
            )
        except (RunError, TargetError) as e:
            _fail(str(e))

    try:
        with Store(db) as store:
            run_id = store.save_run(
                result,
                prompt_text=prompt_text,
                rows=rows,
                plan_text=plan_path.read_text(encoding="utf-8-sig"),
            )
            stored = store.path
    except StoreError as e:
        _fail(str(e))
    result = result.model_copy(update={"id": run_id})

    console.print(render_run(result))
    _report_check_errors(result)
    console.print(render_failures(result, rows, limit=3))
    if loaded.criteria:
        console.print(
            f"\n{len(loaded.criteria)} question(s) for the LLM judge aren't scored yet; "
            "that comes in a later HEMP version.",
            style="yellow",
            markup=False,
        )
    console.print(
        f"\nSaved as run #{run_id} in {_display_path(stored)}. See it again with "
        f"`hemp show {run_id}{f' --db {db}' if db else ''}`.",
        markup=False,
    )


@app.command()
def show(
    run_id: Annotated[
        int | None, typer.Argument(help="Run number to show. Leave out to list recent runs.")
    ] = None,
    failures: Annotated[int, typer.Option(help="How many failed examples to show.", min=0)] = 5,
    db: DbOpt = None,
) -> None:
    """Show a stored run, or list recent runs."""
    try:
        with Store(db) as store:
            if run_id is None:
                console.print(render_runs(store.list_runs()))
                return
            result = store.load_run(run_id)
            rows = store.load_rows(run_id)
    except StoreError as e:
        _fail(str(e))
    console.print(render_run(result))
    _report_check_errors(result)
    if failures:
        console.print(render_failures(result, rows, limit=failures))


def _report_check_errors(result: RunResult) -> None:
    for variant in result.variants:
        for summary in variant.summary():
            if summary.n_errors:
                example = next(
                    o.detail
                    for ex in variant.examples
                    for o in ex.checks
                    if o.check_id == summary.check_id and o.error
                )
                _warn(
                    f"check {summary.check_id} crashed on {summary.n_errors} example(s) "
                    f"({variant.name}), counted as fails. First error: {example}"
                )


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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
