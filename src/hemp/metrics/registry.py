"""Metric registry: the only metrics a plan can use.

The planner picks metrics from here and fills in their params; it never writes
metric code. Users add their own with `@hemp.metric`.
"""

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Self, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hemp.dataset import Row
from hemp.metrics.text import is_missing


class MetricLayer(StrEnum):
    FORMAT = "format"  # deterministic checks on the output alone
    REFERENCE = "reference"  # compare the output to a gold column
    CUSTOM = "custom"  # user plugins


class MetricParams(BaseModel):
    """Base class for metric params. Unknown params are rejected."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MetricResult(BaseModel):
    """One metric on one output. `passed`/`score` are None when the metric was skipped."""

    model_config = ConfigDict(frozen=True)

    passed: bool | None
    score: float | None = Field(ge=0, le=1)  # equals 1/0 for pass/fail metrics
    detail: str = ""

    @model_validator(mode="after")
    def _skip_is_consistent(self) -> Self:
        if (self.passed is None) != (self.score is None):
            raise ValueError("passed and score must both be None (skipped) or both set")
        return self

    @property
    def skipped(self) -> bool:
        return self.passed is None

    @classmethod
    def check(cls, passed: bool, detail: str = "") -> "MetricResult":
        return cls(passed=passed, score=1.0 if passed else 0.0, detail=detail)

    @classmethod
    def skip(cls, detail: str) -> "MetricResult":
        """Not applicable to this row, e.g. its gold value is missing. Excluded from scores."""
        return cls(passed=None, score=None, detail=detail)


@dataclass(frozen=True)
class MetricContext:
    output: str  # raw model output
    row: Row = field(default_factory=dict)  # the dataset row the output was produced for
    gold: Any = None  # value of the gold column, for reference metrics


class MetricError(ValueError):
    """A metric name or its params are invalid."""


P = TypeVar("P", bound=MetricParams)
MetricFn = Callable[[MetricContext, P], MetricResult]


@dataclass(frozen=True)
class MetricInfo:
    name: str
    layer: MetricLayer
    description: str
    params_model: type[MetricParams]
    fn: Callable[[MetricContext, Any], MetricResult]
    requires_gold: bool

    def validate_params(self, params: Mapping[str, Any]) -> MetricParams:
        try:
            return self.params_model.model_validate(dict(params))
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'params'}: {err['msg']}"
                for err in e.errors()
            )
            raise MetricError(
                f"Invalid params for metric '{self.name}': {problems}. "
                f"Expected params: {describe_params(self.params_model) or 'none'}."
            ) from None

    def __call__(self, ctx: MetricContext, params: Mapping[str, Any]) -> MetricResult:
        validated = self.validate_params(params)
        if self.requires_gold and is_missing(ctx.gold):
            return MetricResult.skip("no gold value for this row")
        return self.fn(ctx, validated)


_REGISTRY: dict[str, MetricInfo] = {}

# Keys a check entry in a plan file uses itself (see hemp.plan.io).
RESERVED_PARAM_NAMES = frozenset({"id", "check", "compare_to", "why"})


def metric(
    name: str | None = None,
    *,
    layer: MetricLayer = MetricLayer.CUSTOM,
    params: type[MetricParams] = MetricParams,
    requires_gold: bool = False,
    description: str | None = None,
) -> Callable[[MetricFn[P]], MetricFn[P]]:
    """Register a metric function `fn(ctx: MetricContext, params) -> MetricResult`.

    The description (default: the docstring's first line) is shown to the planner,
    so say plainly what the metric checks.
    """

    def register(fn: MetricFn[P]) -> MetricFn[P]:
        metric_name = name or fn.__name__
        clashes = sorted(set(params.model_fields) & RESERVED_PARAM_NAMES)
        if clashes:
            raise MetricError(
                f"Metric '{metric_name}' has params named {', '.join(clashes)}, which plan "
                "files use for other things. Rename them."
            )
        doc = description or (inspect.getdoc(fn) or "").split("\n")[0]
        existing = _REGISTRY.get(metric_name)
        if existing is not None and existing.fn is not fn:
            raise MetricError(
                f"A metric named '{metric_name}' is already registered. Pick another name."
            )
        _REGISTRY[metric_name] = MetricInfo(
            name=metric_name,
            layer=layer,
            description=doc,
            params_model=params,
            fn=fn,
            requires_gold=requires_gold,
        )
        return fn

    return register


def get_metric(name: str) -> MetricInfo:
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise MetricError(f"Unknown metric '{name}'. Available metrics: {available}.") from None


def list_metrics(*layers: MetricLayer) -> list[MetricInfo]:
    return sorted(
        (m for m in _REGISTRY.values() if not layers or m.layer in layers),
        key=lambda m: (list(MetricLayer).index(m.layer), m.name),
    )


def catalog(*layers: MetricLayer) -> list[dict[str, Any]]:
    """Compact description of metrics, for the planner's prompt."""
    return [
        {
            "metric": m.name,
            "layer": m.layer.value,
            "description": m.description,
            "params": param_docs(m.params_model),
        }
        for m in list_metrics(*layers)
    ]


def describe_params(model: type[MetricParams]) -> str:
    return ", ".join(f"{k} ({v})" for k, v in param_docs(model).items())


def param_docs(model: type[MetricParams]) -> dict[str, str]:
    docs: dict[str, str] = {}
    for fname, finfo in model.model_fields.items():
        text = _type_name(finfo.annotation)
        text += " (required)" if finfo.is_required() else f" = {finfo.default!r}"
        if finfo.description:
            text += f": {finfo.description}"
        docs[fname] = text
    return docs


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        return " | ".join(_type_name(a) for a in get_args(annotation))
    if origin is not None:
        args = ", ".join(_type_name(a) for a in get_args(annotation))
        origin_name = getattr(origin, "__name__", None) or str(origin).replace("typing.", "")
        return f"{origin_name}[{args}]"
    if annotation is type(None):
        return "null"
    return getattr(annotation, "__name__", None) or str(annotation).replace("typing.", "")
