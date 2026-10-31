import sys
from collections.abc import Iterator

import pytest

from hemp.metrics import registry


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Metrics and plugins loaded inside a test don't leak into other tests."""
    saved = dict(registry._REGISTRY)
    modules = set(sys.modules)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
    for name in set(sys.modules) - modules:
        if name.startswith("hemp_plugin_") or name.startswith("my_hemp_metrics"):
            del sys.modules[name]
