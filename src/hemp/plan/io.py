"""Read and write plans as simple, editable YAML.

The file layout is designed for people, and differs from EvalPlan:

    columns:      input columns and the answer (gold) column
    checks:       one flat list of metrics; settings sit next to `check:`,
                  defaults are hidden, `compare_to` names the answer column
    questions:    yes/no questions for the LLM judge
    baselines:    names of simple strategies to beat
    about:        HEMP's bookkeeping (hashes, model), at the bottom

Reading a file and writing it back gives the same plan and the same hash.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hemp.metrics import MetricError, MetricLayer, get_metric, load_plugin
from hemp.metrics.registry import RESERVED_PARAM_NAMES
from hemp.plan.schema import (
    PLAN_SCHEMA_VERSION,
    BaselineKind,
    CheckSpec,
    Criterion,
    EvalPlan,
    PlanMeta,
)
from hemp.plan.validate import plan_problems

# Keys a check entry uses itself; metric params can't have these names.
CHECK_KEYS = RESERVED_PARAM_NAMES

HEADER = """\
# HEMP evaluation plan. Edit it freely, then check your edits with:
#   hemp validate {name}
#
# `hemp metrics` lists every check and its settings.
"""


class PlanError(ValueError):
    """A plan file is missing, malformed, or invalid."""


# --- writing -------------------------------------------------------------------


def dump_plan(plan: EvalPlan, name: str = "<this file>") -> str:
    parts = [HEADER.format(name=name)]
    if plan.plugins:
        parts.append(
            _section("plugins", plan.plugins, "your own metric files, loaded automatically")
        )
    columns: dict[str, Any] = {"input": plan.input_columns}
    if plan.gold_columns:
        columns["answer"] = (
            plan.gold_columns[0] if len(plan.gold_columns) == 1 else plan.gold_columns
        )
    parts.append(_section("columns", columns))
    parts.append(_section("checks", _checks_doc(plan), "automatic, no LLM needed"))
    if plan.criteria:
        questions = [{"id": c.id, "ask": c.question} for c in plan.criteria]
        parts.append(_section("questions", questions, 'an LLM judge answers each; "yes" = pass'))
    else:
        parts.append(
            "questions: []  # yes/no questions for an LLM judge, e.g.\n"
            "#  - id: q1\n#    ask: Is the answer polite?\n"
        )
    if plan.baselines:
        lines = ["baselines:  # simple strategies your prompt should beat"]
        lines += [f"  - {b.value}  # {b.description}" for b in plan.baselines]
        parts.append("\n".join(lines) + "\n")
    parts.append(
        "# --- Written by HEMP. Don't edit below this line. ---\n"
        + _yaml({"about": _about_doc(plan)})
    )
    return "\n".join(parts)


def write_plan(plan: EvalPlan, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise PlanError(
            f"{path} already exists and may contain your edits. "
            "Pass --force to overwrite it, or --out to write somewhere else."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_plan(plan, path.name), encoding="utf-8")
    return path


def _checks_doc(plan: EvalPlan) -> list[dict[str, Any]]:
    checks = plan.checks()
    derived = derive_ids([(c.metric, c.gold_column) for c in checks])
    docs = []
    for spec, auto_id in zip(checks, derived, strict=True):
        doc: dict[str, Any] = {}
        if spec.id != auto_id:
            doc["id"] = spec.id
        doc["check"] = spec.metric
        if spec.gold_column:
            doc["compare_to"] = spec.gold_column
        doc.update(non_default_params(spec))
        if spec.rationale:
            doc["why"] = spec.rationale
        docs.append(doc)
    return docs


def non_default_params(spec: CheckSpec) -> dict[str, Any]:
    """The check's settings that differ from the metric's defaults."""
    try:
        model = get_metric(spec.metric).params_model
    except MetricError:
        return dict(spec.params)
    defaults = {
        name: field.get_default(call_default_factory=True)
        for name, field in model.model_fields.items()
        if not field.is_required()
    }
    return {k: v for k, v in spec.params.items() if k not in defaults or defaults[k] != v}


def _about_doc(plan: EvalPlan) -> dict[str, Any]:
    meta = plan.meta
    about: dict[str, Any] = {
        "created": meta.created_at.isoformat().replace("+00:00", "Z"),
        "planner": meta.planner_model,
        "dataset": Path(meta.dataset_source).as_posix() if meta.dataset_source else None,
        "prompt_hash": meta.prompt_hash,
        "dataset_hash": meta.dataset_hash,
        "hemp_version": meta.hemp_version,
        "schema_version": plan.schema_version,
    }
    return {k: v for k, v in about.items() if v is not None}


class _Dumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        # Indent list items under their key, like most hand-written YAML.
        super().increase_indent(flow, False)


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    )


def _represent_list(dumper: yaml.SafeDumper, value: list[Any]) -> yaml.SequenceNode:
    # Short lists of plain values read best inline: values: [a, b, c]
    flow = all(isinstance(v, str | int | float | bool) for v in value) and len(str(value)) < 70
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=flow)


_Dumper.add_representer(str, _represent_str)
_Dumper.add_representer(list, _represent_list)


def _yaml(data: Any) -> str:
    return yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)


def _section(key: str, value: Any, comment: str | None = None) -> str:
    text = _yaml({key: value})
    if comment:
        first, _, rest = text.partition("\n")
        text = f"{first}  # {comment}\n{rest}"
    return text


# --- reading -------------------------------------------------------------------


class _Doc(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Columns(_Doc):
    input: list[str] = Field(min_length=1)
    answer: str | list[str] | None = None


class _Question(_Doc):
    id: str = Field(min_length=1)
    ask: str = Field(min_length=1)


class _About(_Doc):
    created: str
    planner: str | None = None
    dataset: str | None = None
    prompt_hash: str
    dataset_hash: str
    hemp_version: str
    schema_version: int = PLAN_SCHEMA_VERSION


class _PlanDoc(_Doc):
    plugins: list[str] = []
    columns: _Columns
    checks: list[dict[str, Any]] = []
    questions: list[_Question] = []
    baselines: list[BaselineKind] = []
    about: _About


def parse_plan(text: str, source: str = "plan", plugin_dir: Path | None = None) -> EvalPlan:
    """Parse and fully validate plan YAML, including check names and settings.

    The plan's plugins are loaded first (file paths relative to `plugin_dir`, default
    the current directory) so their checks can be validated too.
    """
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PlanError(f"{source} isn't valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise PlanError(f"{source} should be a YAML mapping. Re-create it with `hemp plan`.")
    if "format_checks" in data or "meta" in data:
        raise PlanError(
            f"{source} uses an older plan layout. Re-create it with `hemp plan ... --force` "
            "and copy over any edits you made."
        )
    try:
        doc = _PlanDoc.model_validate(data)
    except ValidationError as e:
        raise PlanError(f"{source} has problems:\n{_list_errors(e)}") from None

    for spec in doc.plugins:
        try:
            load_plugin(spec, plugin_dir)
        except MetricError as e:
            raise PlanError(f"{source}: {e}") from None

    sections, problems = _read_checks(doc.checks)
    if problems:
        raise PlanError(f"{source} has problems:\n" + "\n".join(f"  - {p}" for p in problems))
    answer = doc.columns.answer
    try:
        plan = EvalPlan(
            schema_version=doc.about.schema_version,
            meta=PlanMeta(
                hemp_version=doc.about.hemp_version,
                created_at=doc.about.created,
                planner_model=doc.about.planner,
                prompt_hash=doc.about.prompt_hash,
                dataset_hash=doc.about.dataset_hash,
                dataset_source=doc.about.dataset,
            ),
            plugins=doc.plugins,
            input_columns=doc.columns.input,
            gold_columns=[answer] if isinstance(answer, str) else (answer or []),
            format_checks=sections[MetricLayer.FORMAT],
            reference_metrics=sections[MetricLayer.REFERENCE],
            custom_metrics=sections[MetricLayer.CUSTOM],
            criteria=[Criterion(id=q.id, question=q.ask) for q in doc.questions],
            baselines=doc.baselines,
        )
    except ValidationError as e:
        raise PlanError(f"{source} has problems:\n{_list_errors(e)}") from None

    remaining = plan_problems(plan)
    if remaining:
        raise PlanError(f"{source} has problems:\n" + "\n".join(f"  - {p}" for p in remaining))
    return plan


def read_plan(path: str | Path) -> EvalPlan:
    path = Path(path)
    if not path.is_file():
        raise PlanError(f"Plan file not found: {path}. Create one with `hemp plan`.")
    text = path.read_text(encoding="utf-8-sig")
    return parse_plan(text, source=str(path), plugin_dir=path.parent)


def derive_ids(checks: list[tuple[str, str | None]]) -> list[str]:
    """Default check ids: the metric name (plus the answer column), with _2, _3 for repeats."""
    seen: dict[str, int] = {}
    ids = []
    for metric, gold in checks:
        base = f"{metric}_{gold}" if gold else metric
        seen[base] = seen.get(base, 0) + 1
        ids.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return ids


def _read_checks(
    entries: list[dict[str, Any]],
) -> tuple[dict[MetricLayer, list[CheckSpec]], list[str]]:
    sections: dict[MetricLayer, list[CheckSpec]] = {layer: [] for layer in MetricLayer}
    problems: list[str] = []
    parsed: list[tuple[int, dict[str, Any]]] = []
    for i, entry in enumerate(entries, start=1):
        where = f"checks #{i}"
        metric = entry.get("check")
        if not isinstance(metric, str) or not metric:
            problems.append(f"{where}: every check needs a `check:` name, e.g. `check: non_empty`.")
            continue
        try:
            info = get_metric(metric)
        except MetricError as e:
            problems.append(f"{where} ({metric}): {e}")
            continue
        params = {k: v for k, v in entry.items() if k not in CHECK_KEYS}
        try:
            full = info.validate_params(params).model_dump(mode="json")
        except MetricError as e:
            problems.append(f"{where} ({metric}): {e}")
            continue
        parsed.append((i, {**entry, "_params": full, "_layer": info.layer}))

    derived = derive_ids([(e["check"], e.get("compare_to")) for _, e in parsed])
    for (_, entry), auto_id in zip(parsed, derived, strict=True):
        sections[entry["_layer"]].append(
            CheckSpec(
                id=str(entry.get("id") or auto_id),
                metric=entry["check"],
                params=entry["_params"],
                gold_column=entry.get("compare_to"),
                rationale=str(entry.get("why") or ""),
            )
        )
    return sections, problems


def _list_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        loc = " / ".join(str(p) for p in err["loc"]) or "(top level)"
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "not a known setting (typo?)"
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)
