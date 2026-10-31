"""Load user metric plugins: Python files or importable modules that use @hemp.metric."""

import hashlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path

from hemp.metrics.registry import _REGISTRY, MetricError


def is_file_plugin(spec: str) -> bool:
    return spec.endswith(".py") or "/" in spec or "\\" in spec


def load_plugin(spec: str, base_dir: Path | None = None) -> list[str]:
    """Import a plugin and return the names of metrics it registered.

    `spec` is a .py file path (relative to `base_dir`, default the current directory)
    or a module name like `mypkg.metrics`. Loading the same plugin twice is a no-op.
    """
    before = set(_REGISTRY)
    try:
        if is_file_plugin(spec):
            _import_file((base_dir or Path.cwd()) / spec)
        else:
            importlib.import_module(spec)
    except MetricError:
        raise
    except FileNotFoundError:
        where = f" (looked in {base_dir})" if base_dir else ""
        raise MetricError(f"Metric plugin not found: {spec}{where}.") from None
    except Exception as e:
        raise MetricError(
            f"Couldn't load metric plugin {spec}: {type(e).__name__}: {e}. Fix the error "
            "in the plugin, or remove it from the plan's `plugins` list."
        ) from e
    return sorted(set(_REGISTRY) - before)


def _import_file(path: Path) -> None:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    module_name = f"hemp_plugin_{resolved.stem}_{digest}"
    if module_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise MetricError(f"Couldn't import {resolved} as a Python module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise


def plugin_ref(spec: str, plan_dir: Path) -> str:
    """How a plan should refer to a plugin: file paths relative to the plan, with '/'."""
    if not is_file_plugin(spec):
        return spec
    resolved = Path(spec).resolve()
    try:
        return resolved.relative_to(plan_dir.resolve()).as_posix()
    except ValueError:
        try:
            return Path(os.path.relpath(resolved, plan_dir.resolve())).as_posix()
        except ValueError:  # different drive on Windows
            return resolved.as_posix()
