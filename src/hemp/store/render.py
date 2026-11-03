"""Terminal output for `hemp diff`."""

import difflib

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from hemp.runner.results import CheckSummary
from hemp.runner.target import value_text
from hemp.store.diff import CheckDiff, ExampleChange, RunDiff


def _name(run_id: int | None) -> str:
    return f"#{run_id}" if run_id is not None else "?"


def render_diff(
    diff: RunDiff, prompt_a: str, prompt_b: str, *, examples: int = 5
) -> RenderableType:
    a, b = _name(diff.id_a), _name(diff.id_b)
    parts: list[RenderableType] = [
        Text.assemble(
            ("Run ", "bold"), (a, "bold cyan"), " -> ", ("run ", "bold"), (b, "bold cyan")
        ),
        Text.assemble(
            ("Changed: ", "bold"),
            ", ".join(diff.changed) or "nothing (a re-run of the same setup)",
            ("   Same: ", "bold"),
            ", ".join(diff.same) or "nothing",
        ),
    ]
    parts += [Text(f"Note: {n}", style="yellow") for n in _notes(diff, a, b)]
    if prompt_a != prompt_b:
        parts.append(_prompt_changes(prompt_a, prompt_b))

    table = Table(header_style="bold")
    table.add_column("Check", overflow="fold")
    table.add_column(f"Run {a}", justify="right")
    table.add_column(f"Run {b}", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Fixed", justify="right")
    table.add_column("Broke", justify="right")
    checks = [c for c in diff.checks if c.question is None]
    questions = [c for c in diff.checks if c.question is not None]
    for c in checks:
        table.add_row(c.check_id, *_cells(c))
    if questions:
        table.add_section()
        table.add_row(Text("Judge questions (yes = pass)", style="bold"))
        for c in questions:
            table.add_row(Text.assemble((c.check_id, "cyan"), f" {c.question}"), *_cells(c))
    parts.append(table)

    broke = [e for c in diff.checks for e in c.broke]
    fixed = [e for c in diff.checks for e in c.fixed]
    if broke:
        parts.append(_examples(f"Broke ({len(broke)})", broke, examples, a, b, style="red"))
    if fixed:
        parts.append(_examples(f"Fixed ({len(fixed)})", fixed, examples, a, b, style="green"))
    if not broke and not fixed:
        parts.append(Text("No example changed between pass and fail.", style="dim"))
    return Group(*parts)


def _notes(diff: RunDiff, a: str, b: str) -> list[str]:
    notes = []
    if "plan" in diff.changed:
        notes.append(
            "the plan changed too, so some differences may come from changed checks, "
            "not the prompt."
        )
    if diff.question_edits:
        notes.append(
            f"these questions were reworded between the runs: {', '.join(diff.question_edits)}."
        )
    if "judge model" in diff.changed:
        notes.append("the judge model changed, so question scores may shift because of it.")
    if diff.n_only_a or diff.n_only_b:
        notes.append(
            f"the datasets differ: {diff.n_paired} examples are in both runs, "
            f"{diff.n_only_a} only in {a}, {diff.n_only_b} only in {b}. Fixed/broke count "
            "shared examples only; percentages use each run's own examples."
        )
    return notes


def _prompt_changes(prompt_a: str, prompt_b: str) -> RenderableType:
    text = Text()
    lines = difflib.unified_diff(prompt_a.splitlines(), prompt_b.splitlines(), lineterm="", n=0)
    for line in lines:
        if line.startswith(("---", "+++", "@@")):
            continue
        style = "red" if line.startswith("-") else "green"
        text.append(line + "\n", style=style)
    return Group(Text("Prompt changes:", style="bold"), text)


def _cells(c: CheckDiff) -> list[Text]:
    if c.a is None:
        change = Text("new check", style="dim")
    elif c.b is None:
        change = Text("removed", style="dim")
    elif c.change is None:
        change = Text("-", style="dim")
    elif abs(c.change) < 0.5:
        change = Text("same", style="dim")
    else:
        style = "green" if c.change > 0 else "red"
        change = Text(f"{c.change:+.0f} pts", style=style)
    return [
        _rate(c.a),
        _rate(c.b),
        change,
        Text(str(len(c.fixed)), style="green" if c.fixed else "dim"),
        Text(str(len(c.broke)), style="red" if c.broke else "dim"),
    ]


def _rate(summary: CheckSummary | None) -> Text:
    if summary is None or summary.pass_rate is None:
        return Text("-", style="dim")
    return Text(f"{summary.pass_rate:.0%} ({summary.n_passed}/{summary.n_scored})")


def _examples(
    title: str, changes: list[ExampleChange], limit: int, a: str, b: str, *, style: str
) -> RenderableType:
    shown = changes[:limit]
    more = f", showing {len(shown)}" if len(shown) < len(changes) else ""
    table = Table(
        title=Text(f"{title}{more}", style=f"bold {style}"),
        title_justify="left",
        header_style="bold",
        show_lines=True,
    )
    table.add_column("Check", overflow="fold")
    table.add_column("Row", overflow="fold")
    table.add_column(f"Run {a}", overflow="fold")
    table.add_column(f"Run {b}", overflow="fold")
    for e in shown:
        row = "\n".join(f"{k}: {_short(value_text(v))}" for k, v in e.row.items())
        table.add_row(
            e.check_id,
            row,
            _output(e.output_a, e.detail_a),
            _output(e.output_b, e.detail_b),
        )
    return table


def _output(output: str, detail: str) -> Text:
    text = Text(_short(output, 300))
    if detail:
        text.append(f"\n{detail}", style="dim")
    return text


def _short(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."
