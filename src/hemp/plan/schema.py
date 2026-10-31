"""EvalPlan: the hashable description of how a prompt is evaluated.

This is HEMP's internal shape. The YAML file people edit uses a simpler layout;
see `hemp.plan.io` for the translation.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hemp.hashing import canonical_json, sha256_text

PLAN_SCHEMA_VERSION = 1


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CheckSpec(_Model):
    """One registered metric with its params. Used for format, reference and custom checks."""

    id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    params: dict[str, Any] = {}
    gold_column: str | None = None  # set for metrics that compare against a gold column
    rationale: str = ""


class Criterion(_Model):
    """A yes/no question an LLM judge answers about one output. Yes means it passes."""

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)


class BaselineKind(StrEnum):
    MAJORITY_CLASS = "majority_class"  # always answer the most common gold value
    ZERO_SHOT = "zero_shot"  # the bare input columns with no instructions

    @property
    def description(self) -> str:
        return _BASELINE_DESCRIPTIONS[self]


_BASELINE_DESCRIPTIONS = {
    BaselineKind.MAJORITY_CLASS: "always answer the most common label",
    BaselineKind.ZERO_SHOT: "the input with no instructions",
}


class PlanMeta(_Model):
    """Where the plan came from. Not part of the plan hash."""

    hemp_version: str
    created_at: datetime
    planner_model: str | None = None
    prompt_hash: str
    dataset_hash: str
    dataset_source: str | None = None


class EvalPlan(_Model):
    schema_version: int = PLAN_SCHEMA_VERSION
    meta: PlanMeta
    # Python files (relative to the plan file) or modules defining custom metrics.
    plugins: list[str] = []
    input_columns: list[str]
    gold_columns: list[str] = []
    format_checks: list[CheckSpec] = []
    reference_metrics: list[CheckSpec] = []
    criteria: list[Criterion] = []
    custom_metrics: list[CheckSpec] = []
    baselines: list[BaselineKind] = []

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} isn't supported by this HEMP "
                f"(expected {PLAN_SCHEMA_VERSION}). Re-create the plan with `hemp plan`."
            )
        ids = [c.id for c in self.checks()] + [q.id for q in self.criteria]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"ids must be unique; duplicated: {', '.join(dupes)}")
        for spec in self.checks():
            if spec.gold_column is not None and spec.gold_column not in self.gold_columns:
                raise ValueError(
                    f"'{spec.id}' compares to '{spec.gold_column}', which isn't an answer "
                    "column. Add it under columns: answer, or fix the name."
                )
        if BaselineKind.MAJORITY_CLASS in self.baselines and not self.gold_columns:
            raise ValueError("majority_class needs an answer column; remove it or add one.")
        return self

    def checks(self) -> list[CheckSpec]:
        """All metric checks, in file order: format, reference, then custom."""
        return [*self.format_checks, *self.reference_metrics, *self.custom_metrics]

    def content(self) -> dict[str, Any]:
        """Everything that defines the evaluation, i.e. the plan minus `meta`."""
        return self.model_dump(mode="json", exclude={"meta"})

    def plan_hash(self) -> str:
        return sha256_text(canonical_json(self.content()))
