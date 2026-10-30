"""Plan parts computed from the profile with fixed rules (no LLM).

Reference metrics follow the gold column's type; baselines follow from the data.
"""

from hemp.metrics import get_metric
from hemp.plan.schema import Baseline, BaselineKind, CheckSpec
from hemp.profile import ColumnProfile, ColumnType, DatasetProfile

MAX_LABELS_FOR_ALLOWED_VALUES = 20


def default_format_checks() -> list[CheckSpec]:
    return [
        CheckSpec(
            id="non_empty",
            metric="non_empty",
            rationale="every prompt should produce some output",
        )
    ]


def label_set_check(profile: DatasetProfile) -> CheckSpec | None:
    """An allowed_values check from a categorical gold column's values."""
    for name in profile.gold_columns:
        col = profile.column(name)
        if (
            col.type is ColumnType.CATEGORICAL
            and col.top_values
            and col.n_unique <= len(col.top_values)
        ):
            return CheckSpec(
                id="allowed_values",
                metric="allowed_values",
                params={"values": [str(v.value) for v in col.top_values], "case_sensitive": False},
                rationale=f"outputs should be one of the labels seen in {name}",
            )
    return None


def default_reference_metrics(profile: DatasetProfile) -> list[CheckSpec]:
    specs = []
    for name in profile.gold_columns:
        spec = _reference_for(profile.column(name))
        if spec is not None:
            specs.append(spec)
    return specs


def _reference_for(col: ColumnProfile) -> CheckSpec | None:
    def spec(metric: str, rationale: str, **params: object) -> CheckSpec:
        return CheckSpec(
            id=f"{metric}_{col.name}",
            metric=metric,
            params=get_metric(metric).validate_params(params).model_dump(mode="json"),
            gold_column=col.name,
            rationale=rationale,
        )

    low_cardinality = col.top_values is not None
    if col.type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN) or (
        col.type is ColumnType.INTEGER and low_cardinality
    ):
        return spec("exact_match", f"{col.name} holds labels, so outputs should match exactly")
    if col.type in (ColumnType.INTEGER, ColumnType.FLOAT):
        return spec(
            "numeric_match",
            f"{col.name} is numeric; loosen abs_tol/rel_tol if close answers should count",
            abs_tol=0.0,
            rel_tol=0.0,
        )
    if col.type is ColumnType.JSON:
        return spec(
            "json_field_match",
            f"{col.name} holds JSON, so compare field by field",
            fields=None,
        )
    if col.type is ColumnType.TEXT:
        return spec(
            "token_f1",
            f"{col.name} is free text, so measure word overlap with the reference",
            threshold=0.5,
        )
    return None


def default_baselines(profile: DatasetProfile) -> list[Baseline]:
    baselines: list[Baseline] = []
    for name in profile.gold_columns:
        col = profile.column(name)
        if col.top_values is not None and col.type in (
            ColumnType.CATEGORICAL,
            ColumnType.BOOLEAN,
            ColumnType.INTEGER,
        ):
            baselines.append(
                Baseline(
                    id=f"majority_{name}",
                    kind=BaselineKind.MAJORITY_CLASS,
                    gold_column=name,
                    rationale="always answering the most common label; the prompt should beat it",
                )
            )
    if profile.input_columns:
        bare = "\n".join(f"{c}: {{{c}}}" for c in profile.input_columns)
        baselines.append(
            Baseline(
                id="zero_shot",
                kind=BaselineKind.ZERO_SHOT,
                prompt=bare,
                rationale="the input with no instructions, to show what the prompt adds",
            )
        )
    return baselines
