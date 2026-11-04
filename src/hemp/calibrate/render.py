"""Terminal output for `hemp calibrate`: labeling screens and the agreement report."""

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from hemp.calibrate.agreement import Agreement
from hemp.calibrate.labels import LabelItem
from hemp.calibrate.optimize import MIN_LABELS_TO_TUNE, CalibrationReport
from hemp.metrics.judge import NO_REFERENCE

TASK_PREVIEW_CHARS = 1500


def render_item(item: LabelItem, number: int, total: int) -> RenderableType:
    task = item.task
    if len(task) > TASK_PREVIEW_CHARS:
        task = task[:TASK_PREVIEW_CHARS] + "\n[...]"
    source = "your prompt" if item.variant == "prompt" else f"baseline {item.variant}"
    parts: list[RenderableType] = [
        Text.assemble(("Question: ", "bold"), (item.question, "bold cyan")),
        Text(""),
        Text("The task the model got:", style="bold"),
        Text(task, style="dim"),
    ]
    if item.images:
        parts += [Text(""), Text("Images in the task (open them to look):", style="bold")]
        parts += [Text(f"  [image {n}] {path}") for n, path in enumerate(item.images, start=1)]
    if item.reference != NO_REFERENCE:
        parts += [Text(""), Text.assemble(("Reference answer: ", "bold"), item.reference)]
    parts += [Text(""), Text(f"Output ({source}):", style="bold"), Text(item.output)]
    return Panel(Group(*parts), title=f"Label {number} of {total}", title_align="left")


def _cell(a: Agreement | None) -> Text:
    if a is None:
        return Text("-", style="dim")
    if a.accuracy is None:
        return Text("no verdicts", style="dim")
    kappa = "n/a" if a.kappa is None else f"{a.kappa:.2f}"
    return Text.assemble(f"{a.accuracy:.0%} ({a.n_agree}/{a.n})", (f"  kappa {kappa}", "dim"))


def render_report(report: CalibrationReport) -> RenderableType:
    tuned = report.after is not None
    table = Table(header_style="bold", title_justify="left")
    table.add_column("Question", overflow="fold")
    table.add_column("Labels", justify="right")
    table.add_column("Default judge" if tuned else "Judge agrees with you", justify="right")
    if tuned:
        table.add_column("Tuned judge", justify="right")
    for q in report.by_question:
        cells = [_cell(q.before)] + ([_cell(q.after)] if tuned else [])
        table.add_row(q.question, str(q.n_labels), *cells)
    table.add_section()
    overall = [_cell(report.before)] + ([_cell(report.after)] if tuned else [])
    table.add_row(Text("All questions", style="bold"), str(report.n_test), *overall)

    if tuned:
        intro = (
            f"Tuned the judge on {report.n_train} of your {report.n_labels} labels and tested "
            f"both judges on the other {report.n_test}, which tuning never saw."
        )
    else:
        intro = f"How often the judge agrees with your {report.n_labels} label(s):"
    parts: list[RenderableType] = [Text(intro), table]
    parts.append(
        Text(
            "Kappa corrects for agreement by chance: 1 is perfect, 0 is no better than "
            "guessing, above 0.6 is good.",
            style="dim",
        )
    )
    if report.n_failed:
        parts.append(
            Text(f"{report.n_failed} judge call(s) failed and were left out.", style="yellow")
        )
    if not tuned:
        parts.append(
            Text(
                f"Label at least {MIN_LABELS_TO_TUNE} outputs to tune the judge "
                f"(you have {report.n_labels}).",
                style="yellow",
            )
        )
    return Group(*parts)
