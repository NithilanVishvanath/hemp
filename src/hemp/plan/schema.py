"""EvalPlan: the reviewable, hashable description of how a prompt is evaluated."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hemp.hashing import canonical_json, sha256_text

PLAN_SCHEMA_VERSION = 1


class _Model(BaseModel):
    # extra="forbid" turns typos in hand-edited plans into errors instead of silent no-ops.
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
    source: str = ""  # the prompt instruction this criterion checks


class BaselineKind(StrEnum):
    MAJORITY_CLASS = "majority_class"  # always predict the most common gold value
    ZERO_SHOT = "zero_shot"  # the bare input with no instructions


class Baseline(_Model):
    id: str = Field(min_length=1)
    kind: BaselineKind
    gold_column: str | None = None
    prompt: str | None = None  # template used by zero_shot
    rationale: str = ""


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
    input_columns: list[str]
    gold_columns: list[str] = []
    format_checks: list[CheckSpec] = []
    reference_metrics: list[CheckSpec] = []
    criteria: list[Criterion] = []
    custom_metrics: list[CheckSpec] = []
    baselines: list[Baseline] = []

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} isn't supported by this HEMP "
                f"(expected {PLAN_SCHEMA_VERSION}). Re-create the plan with `hemp plan`."
            )
        ids = [item.id for item in self.all_items()]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"ids must be unique; duplicated: {', '.join(dupes)}")
        gold_users: list[CheckSpec | Baseline] = [
            *self.reference_metrics,
            *self.custom_metrics,
            *self.baselines,
        ]
        for item in gold_users:
            if item.gold_column is not None and item.gold_column not in self.gold_columns:
                raise ValueError(
                    f"'{item.id}' uses gold column '{item.gold_column}', which isn't listed "
                    "in gold_columns. Add it there or fix the name."
                )
        return self

    def all_items(self) -> list[CheckSpec | Criterion | Baseline]:
        return [
            *self.format_checks,
            *self.reference_metrics,
            *self.criteria,
            *self.custom_metrics,
            *self.baselines,
        ]

    def content(self) -> dict[str, Any]:
        """Everything that defines the evaluation, i.e. the plan minus `meta`."""
        return self.model_dump(mode="json", exclude={"meta"})

    def plan_hash(self) -> str:
        return sha256_text(canonical_json(self.content()))
