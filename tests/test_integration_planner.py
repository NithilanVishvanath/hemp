"""Real-model planner test. Run with: uv run pytest -m integration

Uses $HEMP_PLANNER_MODEL / $HEMP_MODEL, or the default local Ollama model.
"""

from pathlib import Path

import pytest

from hemp.config import ModelConfig, make_lm
from hemp.plan import create_plan, dump_plan, parse_plan
from hemp.profile import profile_dataset

EXAMPLE = Path(__file__).parent.parent / "examples" / "sentiment"


@pytest.mark.integration
def test_planner_on_sentiment_example() -> None:
    prompt = (EXAMPLE / "prompt.txt").read_text(encoding="utf-8")
    profile = profile_dataset(EXAMPLE / "data.jsonl", prompt=prompt)
    model = ModelConfig.from_env().planner

    result = create_plan(prompt, profile, lm=make_lm(model), planner_model=model)

    plan = result.plan
    assert plan.gold_columns == ["label"]
    assert any(c.metric == "allowed_values" for c in plan.format_checks)
    assert all(c.question.strip() for c in plan.criteria)
    assert parse_plan(dump_plan(plan)) == plan  # whatever the model said, the plan is valid
