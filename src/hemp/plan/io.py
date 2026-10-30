"""Read and write plans as editable YAML."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from hemp.plan.schema import EvalPlan
from hemp.plan.validate import plan_problems

HEADER = """\
# HEMP evaluation plan. Review and edit it before running.
#
# format_checks      deterministic checks on each output
# reference_metrics  compare each output with a gold column
# criteria           yes/no questions an LLM judge answers per output ("yes" = pass)
# custom_metrics     your own metrics registered with @hemp.metric
# baselines          simple strategies the prompt should beat
#
# `metric` must be a registered metric; `hemp metrics` lists them and their params.
# Editing anything except `meta` gives the plan a new hash, i.e. a new plan version.
# Check your edits with: hemp validate <this file>
"""


class PlanError(ValueError):
    """A plan file is missing, malformed, or invalid."""


class _Dumper(yaml.SafeDumper):
    pass


def _str_representer(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_representer)


def dump_plan(plan: EvalPlan) -> str:
    data = plan.model_dump(mode="json", exclude_none=True)
    body = yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=100)
    return HEADER + "\n" + body


def write_plan(plan: EvalPlan, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise PlanError(
            f"{path} already exists and may contain your edits. "
            "Pass --force to overwrite it, or --out to write somewhere else."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_plan(plan), encoding="utf-8")
    return path


def parse_plan(text: str, source: str = "plan") -> EvalPlan:
    """Parse and fully validate plan YAML, including metric names and params."""
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise PlanError(f"{source} isn't valid YAML: {e}") from None
    if not isinstance(data, dict):
        raise PlanError(f"{source} should be a YAML mapping. Re-create it with `hemp plan`.")
    try:
        plan = EvalPlan.model_validate(data)
    except ValidationError as e:
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
            for err in e.errors()
        )
        raise PlanError(f"{source} has problems:\n{problems}") from None
    problems_found = plan_problems(plan)
    if problems_found:
        listed = "\n".join(f"  - {p}" for p in problems_found)
        raise PlanError(f"{source} has problems:\n{listed}")
    return plan


def read_plan(path: str | Path) -> EvalPlan:
    path = Path(path)
    if not path.is_file():
        raise PlanError(f"Plan file not found: {path}. Create one with `hemp plan`.")
    return parse_plan(path.read_text(encoding="utf-8-sig"), source=str(path))
