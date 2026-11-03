"""Terminal tables for run results."""

from collections.abc import Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from hemp.dataset import Row
from hemp.runner.results import PROMPT_VARIANT, CheckSummary, RunInfo, RunResult
from hemp.runner.target import value_text


def render_run(result: RunResult) -> RenderableType:
    info = result.info
    title = f"Run #{result.id}" if result.id is not None else "Run"
    judge = f", judge: {info.judge_model}" if info.judge_model else ""
    header = Text.assemble(
        (title, "bold"),
        f"  {info.n_examples} examples, target: {info.target_model}{judge}, "
        f"plan {info.plan_hash[:12]}",
    )

    table = Table(header_style="bold")
    table.add_column("Check", overflow="fold")
    for variant in result.variants:
        name = "Your prompt" if variant.name == PROMPT_VARIANT else f"{variant.name} (baseline)"
        table.add_column(name, justify="right")

    summaries = [{s.check_id: s for s in v.summary()} for v in result.variants]
    check_ids = list(dict.fromkeys(cid for s in summaries for cid in s))
    checks = [c for c in check_ids if c not in result.questions]
    questions = [c for c in check_ids if c in result.questions]
    for check_id in checks:
        table.add_row(check_id, *(_cell(s.get(check_id)) for s in summaries))
    if questions:
        table.add_section()
        table.add_row(Text("Judge questions (yes = pass)", style="bold"))
        for qid in questions:
            label = Text.assemble((qid, "cyan"), f" {result.questions[qid]}")
            table.add_row(label, *(_cell(s.get(qid)) for s in summaries))

    errors = [v.n_model_errors for v in result.variants]
    if any(errors):
        table.add_row(
            Text("model errors", style="red"), *(Text(str(n), style="red") for n in errors)
        )
    parts: list[RenderableType] = [header, table]
    if not check_ids:
        parts.append(Text("The plan has no checks, so nothing was scored.", style="yellow"))
    return Group(*parts)


def _cell(summary: CheckSummary | None) -> Text:
    if summary is None:
        return Text("-", style="dim")
    rate = summary.pass_rate
    if rate is None:
        return Text("skipped", style="dim")
    text = Text(f"{rate:.0%} ({summary.n_passed}/{summary.n_scored})")
    if summary.mean_score is not None and abs(summary.mean_score - rate) > 1e-9:
        text.append(f" avg {summary.mean_score:.2f}", style="dim")
    if summary.n_skipped:
        text.append(f" {summary.n_skipped} skipped", style="dim")
    if summary.n_errors:
        text.append(f" {summary.n_errors} errors", style="red")
    return text


def render_failures(result: RunResult, rows: Sequence[Row], limit: int = 5) -> RenderableType:
    """Up to `limit` examples where the prompt failed a check, with its output."""
    failed = [
        ex
        for ex in result.prompt.examples
        if ex.error is not None or any(o.passed is False for o in ex.checks)
    ]
    if not failed:
        return Text("Your prompt passed every check on every example.", style="green")
    table = Table(
        title=f"Failed examples ({len(failed)}, showing {min(limit, len(failed))})",
        title_justify="left",
        header_style="bold",
        show_lines=True,
    )
    table.add_column("#", style="dim")
    table.add_column("Row", overflow="fold", ratio=2)
    table.add_column("Output", overflow="fold", ratio=2)
    table.add_column("Failed checks", overflow="fold", ratio=2)
    for ex in failed[:limit]:
        row = rows[ex.index] if ex.index < len(rows) else {}
        shown_row = "\n".join(f"{k}: {_short(v)}" for k, v in row.items())
        reasons = [
            f"{o.check_id}" + (f": {o.detail}" if o.detail else "")
            for o in ex.checks
            if o.passed is False
        ]
        output = Text(ex.error, style="red") if ex.error else Text(_short(ex.output, 400))
        table.add_row(str(ex.index), shown_row, output, "\n".join(reasons))
    return table


def render_runs(runs: Sequence[tuple[int, RunInfo]]) -> RenderableType:
    if not runs:
        return Text("No runs yet. Start one with `hemp run prompt.txt data.jsonl`.")
    table = Table(header_style="bold")
    table.add_column("Run", no_wrap=True)
    table.add_column("When", no_wrap=True)
    for name in ("Prompt", "Dataset", "Model"):
        table.add_column(name, overflow="fold")
    table.add_column("Plan", no_wrap=True)
    for run_id, info in runs:
        table.add_row(
            f"#{run_id}",
            info.created_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            info.prompt_source or info.prompt_hash[:8],
            info.dataset_source or info.dataset_hash[:8],
            info.target_model,
            info.plan_hash[:8],
        )
    return table


def _short(value: object, limit: int = 200) -> str:
    text = value_text(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."
