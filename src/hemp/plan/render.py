"""Terminal summary of an EvalPlan."""

import json

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from hemp.plan.schema import CheckSpec, EvalPlan


def render_plan(plan: EvalPlan) -> RenderableType:
    parts: list[RenderableType] = [
        Text.assemble(
            ("Plan ", "bold"),
            (plan.plan_hash()[:12], "bold cyan"),
            f"  inputs: {', '.join(plan.input_columns) or 'none'}",
            f"  gold: {', '.join(plan.gold_columns) or 'none'}",
        )
    ]
    sections = [
        ("Format checks", plan.format_checks),
        ("Reference metrics", plan.reference_metrics),
        ("Custom metrics", plan.custom_metrics),
    ]
    for title, specs in sections:
        if specs:
            parts.append(_checks_table(title, specs))

    if plan.criteria:
        table = Table(title="Rubric criteria (LLM judge, yes = pass)", title_justify="left")
        table.add_column("ID", style="dim")
        table.add_column("Question", overflow="fold")
        for c in plan.criteria:
            table.add_row(c.id, c.question)
        parts.append(table)
    else:
        parts.append(Text("No rubric criteria.", style="yellow"))

    if plan.baselines:
        table = Table(title="Baselines", title_justify="left")
        table.add_column("ID", style="dim")
        table.add_column("Kind")
        table.add_column("Why", overflow="fold")
        for b in plan.baselines:
            table.add_row(b.id, b.kind.value, b.rationale)
        parts.append(table)
    return Group(*parts)


def _checks_table(title: str, specs: list[CheckSpec]) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("ID", style="dim")
    table.add_column("Metric")
    table.add_column("Params", overflow="fold")
    table.add_column("Why", overflow="fold")
    for s in specs:
        params = dict(s.params)
        if s.gold_column:
            params = {"gold_column": s.gold_column, **params}
        shown = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in params.items())
        table.add_row(s.id, s.metric, shown, s.rationale)
    return table
