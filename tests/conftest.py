from collections.abc import Iterator

import pytest

from hemp.metrics import registry


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Metrics registered inside a test don't leak into other tests."""
    saved = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)
