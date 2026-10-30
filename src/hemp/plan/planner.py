"""The planner: reads a prompt and a dataset profile and proposes an EvalPlan.

The LLM only proposes format checks (picked from the metric registry) and yes/no
rubric criteria. Every proposal is validated; anything invalid is dropped with a
warning. Reference metrics and baselines come from fixed rules in `defaults`.
"""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import dspy
from pydantic import BaseModel, Field

from hemp._version import __version__
from hemp.hashing import canonical_json
from hemp.metrics import MetricError, MetricLayer, catalog, get_metric
from hemp.plan.defaults import (
    default_baselines,
    default_format_checks,
    default_reference_metrics,
    label_set_check,
)
from hemp.plan.schema import CheckSpec, Criterion, EvalPlan, PlanMeta
from hemp.profile import DatasetProfile
from hemp.prompt import prompt_hash

MAX_CRITERIA = 12
SAMPLE_ROWS_FOR_PLANNER = 3
SAMPLE_VALUE_MAX_CHARS = 200


class PlannerError(RuntimeError):
    """The planner model couldn't be reached or returned something unusable."""


class ProposedCheck(BaseModel):
    metric: str = Field(description="name of a metric from available_checks")
    params: dict[str, Any] = Field(default_factory=dict)
    instruction: str = Field("", description="the prompt text this check enforces")


class ProposedCriterion(BaseModel):
    question: str = Field(description="yes/no question; 'yes' means the output is good")
    instruction: str = Field("", description="the prompt text this criterion checks")


class ProposeEvaluation(dspy.Signature):
    """Design an evaluation for an LLM prompt, given the prompt and a profile of its dataset.

    1. format_checks: for each instruction in the prompt that code can verify (output
       format, allowed labels, length limits, required JSON keys, required or forbidden
       patterns), select a metric from available_checks and set its params. Use only
       metric names and params listed there. Skip checks the prompt doesn't ask for.
    2. criteria: for each remaining instruction that needs judgment (tone, faithfulness
       to the input, completeness, following a constraint), write one yes/no question
       about a single output, answerable from the input row and the output. "Yes" must
       mean the output follows the instruction. One instruction per question. Never use
       rating scales. Don't repeat what format checks or reference answers already cover.

    Only evaluate what the prompt actually asks for. Usually 0-5 format checks and
    2-8 criteria.
    """

    prompt_template: str = dspy.InputField(
        desc="the prompt being evaluated; {name} placeholders are filled from dataset columns"
    )
    dataset_profile: str = dspy.InputField(desc="columns, their types and roles, sample rows")
    available_checks: str = dspy.InputField(desc="JSON list of metrics you may select")
    format_checks: list[ProposedCheck] = dspy.OutputField()
    criteria: list[ProposedCriterion] = dspy.OutputField()


class Planner(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.propose = dspy.ChainOfThought(ProposeEvaluation)

    def forward(
        self, prompt_template: str, dataset_profile: str, available_checks: str
    ) -> dspy.Prediction:
        return self.propose(
            prompt_template=prompt_template,
            dataset_profile=dataset_profile,
            available_checks=available_checks,
        )


@dataclass
class PlanResult:
    plan: EvalPlan
    warnings: list[str] = field(default_factory=list)


def create_plan(
    prompt: str,
    profile: DatasetProfile,
    *,
    lm: dspy.LM | None,
    planner_model: str | None = None,
    now: datetime | None = None,
) -> PlanResult:
    """Build an EvalPlan. With `lm=None`, only the rule-based parts are included."""
    warnings: list[str] = []
    format_checks = default_format_checks()
    custom: list[CheckSpec] = []
    criteria: list[Criterion] = []

    if lm is None:
        warnings.append(
            "Planned without an LLM, so there are no rubric criteria or prompt-specific "
            "format checks. Add them to the plan by hand, or re-run without --no-llm."
        )
    else:
        proposal = _propose(prompt, profile, lm)
        checks, check_warnings = _accept_checks(proposal.format_checks)
        warnings += check_warnings
        for spec in checks:
            layer = get_metric(spec.metric).layer
            (custom if layer is MetricLayer.CUSTOM else format_checks).append(spec)
        criteria, criteria_warnings = _accept_criteria(proposal.criteria)
        warnings += criteria_warnings

    if not any(c.metric == "allowed_values" for c in format_checks):
        label_check = label_set_check(profile)
        if label_check is not None:
            format_checks.append(label_check)

    plan = EvalPlan(
        meta=PlanMeta(
            hemp_version=__version__,
            created_at=now or datetime.now(UTC).replace(microsecond=0),
            planner_model=planner_model if lm is not None else None,
            prompt_hash=prompt_hash(prompt),
            dataset_hash=profile.dataset_hash,
            dataset_source=profile.source,
        ),
        input_columns=profile.input_columns,
        gold_columns=profile.gold_columns,
        format_checks=_unique_ids(format_checks),
        reference_metrics=default_reference_metrics(profile),
        criteria=criteria,
        custom_metrics=_unique_ids(custom),
        baselines=default_baselines(profile),
    )
    return PlanResult(plan=plan, warnings=warnings)


def describe_profile(profile: DatasetProfile) -> str:
    """Compact text version of the profile for the planner's prompt."""
    lines = [f"rows: {profile.n_rows}", "columns:"]
    for col in profile.columns:
        bits = [col.type.value, col.role.value]
        if col.top_values:
            bits.append("values: " + ", ".join(f"{v.value} ({v.count})" for v in col.top_values))
        if col.lengths:
            bits.append(f"{col.lengths.min}-{col.lengths.max} chars")
        if col.json_keys:
            bits.append("keys: " + ", ".join(col.json_keys))
        if col.n_missing:
            bits.append(f"{col.n_missing} missing")
        lines.append(f"- {col.name}: " + "; ".join(bits))
    lines.append("sample rows:")
    for row in profile.sample_rows[:SAMPLE_ROWS_FOR_PLANNER]:
        short = {
            k: (
                v[:SAMPLE_VALUE_MAX_CHARS] + "…"
                if isinstance(v, str) and len(v) > SAMPLE_VALUE_MAX_CHARS
                else v
            )
            for k, v in row.items()
        }
        lines.append(json.dumps(short, ensure_ascii=False))
    return "\n".join(lines)


def _propose(prompt: str, profile: DatasetProfile, lm: dspy.LM) -> dspy.Prediction:
    available = json.dumps(catalog(MetricLayer.FORMAT, MetricLayer.CUSTOM), indent=1)
    try:
        with dspy.context(lm=lm):
            return Planner()(
                prompt_template=prompt,
                dataset_profile=describe_profile(profile),
                available_checks=available,
            )
    except Exception as e:
        raise PlannerError(_explain_failure(e, getattr(lm, "model", "the planner model"))) from e


def _explain_failure(error: Exception, model: str) -> str:
    text = f"{type(error).__name__}: {error}"
    if "AdapterParseError" in text or "parse" in text.lower():
        return (
            f"The planner model ({model}) returned output HEMP couldn't parse. "
            "Try again, try a larger model with --planner-model, or use --no-llm."
        )
    hint = (
        "Is Ollama running (`ollama serve`) and the model pulled (`ollama pull <model>`)? "
        if model.startswith("ollama")
        else "Check the model name and API key. "
    )
    return (
        f"Couldn't get a plan from {model}: {text.splitlines()[0][:300]}\n{hint}"
        "You can pick another model with --planner-model, or plan without an LLM using --no-llm."
    )


def _accept_checks(proposed: list[ProposedCheck]) -> tuple[list[CheckSpec], list[str]]:
    accepted: list[CheckSpec] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for p in proposed:
        try:
            info = get_metric(p.metric)
            if info.layer not in (MetricLayer.FORMAT, MetricLayer.CUSTOM) or info.requires_gold:
                raise MetricError(f"'{p.metric}' can't be used as a format check")
            params = info.validate_params(p.params).model_dump(mode="json")
        except MetricError as e:
            warnings.append(f"Dropped a proposed check: {e}")
            continue
        key = canonical_json([p.metric, params])
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            CheckSpec(id=p.metric, metric=p.metric, params=params, rationale=p.instruction.strip())
        )
    return accepted, warnings


def _accept_criteria(
    proposed: list[ProposedCriterion],
) -> tuple[list[Criterion], list[str]]:
    criteria: list[Criterion] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for p in proposed:
        question = " ".join(p.question.split())
        if not question or question.lower() in seen:
            continue
        seen.add(question.lower())
        criteria.append(
            Criterion(id=f"c{len(criteria) + 1}", question=question, source=p.instruction.strip())
        )
    if len(criteria) > MAX_CRITERIA:
        warnings.append(
            f"The planner proposed {len(criteria)} criteria; kept the first {MAX_CRITERIA}."
        )
        criteria = criteria[:MAX_CRITERIA]
    return criteria, warnings


def _unique_ids(specs: list[CheckSpec]) -> list[CheckSpec]:
    """Suffix repeated ids: length, length_2, ..."""
    counts: dict[str, int] = {}
    out = []
    for spec in specs:
        counts[spec.id] = counts.get(spec.id, 0) + 1
        n = counts[spec.id]
        out.append(spec if n == 1 else spec.model_copy(update={"id": f"{spec.id}_{n}"}))
    return out
