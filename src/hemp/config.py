"""Model configuration. Planner, judge and target models are set separately.

Any LiteLLM model string works (e.g. "openai/gpt-4o-mini"). Defaults to a local
Ollama model; override with HEMP_PLANNER_MODEL / HEMP_JUDGE_MODEL / HEMP_TARGET_MODEL,
or HEMP_MODEL for all three.
"""

import os

import dspy
from pydantic import BaseModel

DEFAULT_MODEL = "ollama_chat/qwen2.5:14b"


class ModelConfig(BaseModel):
    planner: str = DEFAULT_MODEL
    judge: str = DEFAULT_MODEL
    target: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "ModelConfig":
        fallback = os.environ.get("HEMP_MODEL") or DEFAULT_MODEL
        return cls(
            planner=os.environ.get("HEMP_PLANNER_MODEL") or fallback,
            judge=os.environ.get("HEMP_JUDGE_MODEL") or fallback,
            target=os.environ.get("HEMP_TARGET_MODEL") or fallback,
        )


def make_lm(model: str, *, temperature: float = 0.0) -> dspy.LM:
    """A dspy.LM for HEMP's own modules. Temperature 0 keeps planning as repeatable as possible."""
    return dspy.LM(model, temperature=temperature)
