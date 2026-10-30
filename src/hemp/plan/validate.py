"""Check a plan against the metric registry (names, sections, params, gold columns)."""

from hemp.metrics import MetricError, MetricLayer, get_metric
from hemp.plan.schema import CheckSpec, EvalPlan

_SECTION_LAYERS = {
    "format_checks": {MetricLayer.FORMAT},
    "reference_metrics": {MetricLayer.REFERENCE},
    "custom_metrics": {MetricLayer.CUSTOM, MetricLayer.FORMAT, MetricLayer.REFERENCE},
}


def plan_problems(plan: EvalPlan) -> list[str]:
    """Every problem found, as messages naming the item's id. Empty when the plan is valid."""
    problems: list[str] = []
    sections: dict[str, list[CheckSpec]] = {
        "format_checks": plan.format_checks,
        "reference_metrics": plan.reference_metrics,
        "custom_metrics": plan.custom_metrics,
    }
    for section, specs in sections.items():
        for spec in specs:
            try:
                info = get_metric(spec.metric)
                info.validate_params(spec.params)
            except MetricError as e:
                problems.append(f"{section} / {spec.id}: {e}")
                continue
            if info.layer not in _SECTION_LAYERS[section]:
                problems.append(
                    f"{section} / {spec.id}: '{spec.metric}' is a {info.layer.value} metric "
                    f"and doesn't belong in {section}."
                )
            if info.requires_gold and spec.gold_column is None:
                problems.append(
                    f"{section} / {spec.id}: '{spec.metric}' compares against a gold column; "
                    "set gold_column."
                )
    return problems
