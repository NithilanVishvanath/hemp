"""Real embedding model. Run with: uv run pytest -m integration (needs nomic-embed-text)."""

import pytest

from hemp.metrics import MetricContext, get_metric


def similarity(output: str, gold: str) -> float:
    score = get_metric("semantic_similarity")(MetricContext(output=output, gold=gold), {}).score
    assert score is not None
    return score


@pytest.mark.integration
def test_default_threshold_separates_related_from_unrelated() -> None:
    assert similarity("The capital of France is Paris.", "Paris") >= 0.75
    assert similarity("At sea level water boils at 100°C.", "Water boils at 100 C") >= 0.75
    assert similarity("London", "Paris") < 0.75
    assert similarity("Photosynthesis makes sugar from light.", "Water boils at 100 C") < 0.75
