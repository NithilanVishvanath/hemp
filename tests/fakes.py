"""Test doubles shared by runner, store and CLI tests."""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from hemp.dataset import Row
from hemp.plan import EvalPlan, create_plan
from hemp.profile import profile_rows

SENTIMENT_PROMPT = "Classify the sentiment as positive, negative or neutral.\nReview: {review}"

SENTIMENT_ROWS: list[Row] = [
    {"review": "Love it", "label": "positive"},
    {"review": "Hate it", "label": "negative"},
    {"review": "It's ok", "label": "neutral"},
    {"review": "Great stuff", "label": "positive"},
] * 3


class FakeLM:
    """Answers like dspy.LM: `lm(prompt) -> [text]`. Records every prompt it gets."""

    def __init__(
        self, answer: Callable[[str], Any] | None = None, model: str = "fake/target"
    ) -> None:
        self.model = model
        self.answer = answer or (lambda prompt: "positive")
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> list[Any]:
        with self._lock:
            self.prompts.append(prompt)
        result = self.answer(prompt)
        if isinstance(result, Exception):
            raise result
        return [result]


def keyword_sentiment(prompt: str) -> str:
    """A 'model' that spots 'Hate' as negative and calls everything else positive."""
    if "Hate" in prompt:
        return "negative"
    return "positive"


def make_plan(rows: list[Row] = SENTIMENT_ROWS, prompt: str = SENTIMENT_PROMPT) -> EvalPlan:
    profile = profile_rows(rows, prompt=prompt)
    return create_plan(prompt, profile, lm=None, now=datetime(2026, 1, 1, tzinfo=UTC)).plan
