"""Terminal summary of an EvalPlan, using the same words as the plan file."""

import json

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from hemp.plan.io import non_default_params
from hemp.plan.schema import EvalPlan


def render_plan(plan: EvalPlan) -> RenderableType:
    answer = ", ".join(plan.gold_columns) or "none"
    parts: list[RenderableType] = [
        Text.assemble(
            ("Plan ", "bold"),
            (plan.plan_hash()[:12], "bold cyan"),
            f"  input: {', '.join(plan.input_columns)}  answer: {answer}",
        )
    ]

    checks = Table(title="Checks (automatic)", title_justify="left")
    checks.add_column("Check")
    checks.add_column("Settings", overflow="fold")
    checks.add_column("Why", overflow="fold")
    for spec in plan.checks():
        settings = {"compare_to": spec.gold_column} if spec.gold_column else {}
        settings.update(non_default_params(spec))
        shown = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in settings.items())
        checks.add_row(spec.metric, shown, spec.rationale)
    parts.append(checks)

    if plan.criteria:
        questions = Table(title='Questions (LLM judge, "yes" = pass)', title_justify="left")
        questions.add_column("ID", style="dim")
        questions.add_column("Question", overflow="fold")
        for c in plan.criteria:
            questions.add_row(c.id, c.question)
        parts.append(questions)
    else:
        parts.append(Text("No questions for the LLM judge.", style="yellow"))

    if plan.baselines:
        listed = ", ".join(f"{b.value} ({b.description})" for b in plan.baselines)
        parts.append(Text.assemble(("Baselines to beat: ", "bold"), listed))
    return Group(*parts)
