import os
import sys
from collections.abc import Iterator

# A very wide terminal for CLI output, so messages never wrap. Otherwise where a
# line wraps depends on the temp folder's path length, which differs per OS, and
# phrases the tests look for can get split across lines.
os.environ["COLUMNS"] = "1000"

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
