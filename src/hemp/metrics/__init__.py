"""Metric registry and built-in metrics."""

# Importing the built-in modules registers their metrics.
from hemp.metrics import format, reference  # noqa: F401
from hemp.metrics.plugins import load_plugin
from hemp.metrics.registry import (
    MetricContext,
    MetricError,
    MetricInfo,
    MetricLayer,
    MetricParams,
    MetricResult,
    catalog,
    get_metric,
    list_metrics,
    metric,
)

__all__ = [
    "MetricContext",
    "MetricError",
    "MetricInfo",
    "MetricLayer",
    "MetricParams",
    "MetricResult",
    "catalog",
    "get_metric",
    "list_metrics",
    "load_plugin",
    "metric",
]
