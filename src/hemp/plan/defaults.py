"""Plan parts computed from the profile with fixed rules (no LLM).

Reference metrics follow the gold column's type; baselines follow from the data.
"""

from hemp.metrics import get_metric
from hemp.metrics.embeddings import DEFAULT_EMBED_MODEL
from hemp.plan.schema import BaselineKind, CheckSpec
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
                rationale=f'outputs should be one of the labels in "{name}"',
            )
    return None


def default_reference_metrics(
    profile: DatasetProfile, embed_model: str = DEFAULT_EMBED_MODEL
) -> list[CheckSpec]:
    specs = []
    for name in profile.gold_columns:
        spec = _reference_for(profile.column(name), embed_model)
        if spec is not None:
            specs.append(spec)
    return specs


def _reference_for(col: ColumnProfile, embed_model: str) -> CheckSpec | None:
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
        return spec("exact_match", f'outputs should match "{col.name}" exactly')
    if col.type in (ColumnType.INTEGER, ColumnType.FLOAT):
        return spec(
            "numeric_match",
            "answers are numbers; loosen abs_tol/rel_tol if close answers should count",
            abs_tol=0.0,
            rel_tol=0.0,
        )
    if col.type is ColumnType.JSON:
        return spec(
            "json_field_match",
            "answers are JSON, so compare field by field",
            fields=None,
        )
    if col.type is ColumnType.TEXT:
        return spec(
            "semantic_similarity",
            "answers are free text, so compare meaning. This checks the output says "
            "something similar, not that its facts are right",
            model=embed_model,
            threshold=0.75,
        )
    return None


def default_baselines(profile: DatasetProfile) -> list[BaselineKind]:
    """majority_class when an answer column holds labels; zero_shot whenever there's input."""
    baselines: list[BaselineKind] = []
    has_labels = any(
        profile.column(name).top_values is not None
        and profile.column(name).type
        in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.INTEGER)
        for name in profile.gold_columns
    )
    if has_labels:
        baselines.append(BaselineKind.MAJORITY_CLASS)
    if profile.input_columns:
        baselines.append(BaselineKind.ZERO_SHOT)
    return baselines
