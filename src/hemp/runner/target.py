"""Run the user's prompt exactly as written.

DSPy adapters rewrite prompts, so the target model is called through `dspy.LM`
directly: the filled-in template goes out as a single user message, unchanged.
"""

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import dspy

from hemp.dataset import Row
from hemp.prompt import FIELD_RE, template_fields


class PromptError(ValueError):
    """The prompt can't be filled in from the dataset."""


class TargetError(RuntimeError):
    """The target model can't be reached at all."""


class TextLM(Protocol):
    """Anything called like `dspy.LM`: `lm("prompt") -> [output, ...]`."""

    model: str

    def __call__(self, prompt: str) -> list[Any]: ...


def value_text(value: Any) -> str:
    """How a cell value appears in a prompt: strings as-is, other values as JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def fill_prompt(template: str, row: Row) -> str:
    """Replace each `{column}` with the row's value. Nothing else in the prompt changes."""
    missing = [f for f in template_fields(template) if f not in row]
    if missing:
        raise PromptError(
            f"The prompt uses {', '.join('{' + f + '}' for f in missing)} but the row has no "
            f"such column. Columns: {', '.join(row) or 'none'}."
        )
    return FIELD_RE.sub(lambda m: value_text(row[m.group(1)]), template)


def check_prompt_fields(template: str, rows: Sequence[Row]) -> None:
    """Fail early, before any model calls, if a placeholder has no matching column."""
    columns = {key for row in rows for key in row}
    missing = [f for f in template_fields(template) if f not in columns]
    if missing:
        raise PromptError(
            f"The prompt uses {', '.join('{' + f + '}' for f in missing)} but the dataset has "
            f"no such column. Columns: {', '.join(sorted(columns)) or 'none'}. "
            "Rename the placeholder or the column so they match."
        )


class RawPrompt(dspy.Module):
    """Sends a prompt string to the LM as-is and returns the text it answers."""

    def __init__(self, lm: TextLM) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, prompt: str) -> str:
        return output_text(self.lm(prompt))


def output_text(outputs: list[Any]) -> str:
    """The first completion's text. dspy returns strings, or dicts when there's extra data."""
    if not outputs:
        return ""
    first = outputs[0]
    if isinstance(first, dict):
        first = first.get("text")
    return "" if first is None else str(first)


@dataclass(frozen=True)
class Generation:
    output: str
    error: str | None
    seconds: float


ProgressFn = Callable[[], None]


def generate(
    lm: TextLM,
    prompts: Sequence[str],
    *,
    workers: int = 4,
    on_done: ProgressFn | None = None,
) -> list[Generation]:
    """Run every prompt through the LM, keeping input order.

    The first prompt runs alone: if it fails, the model is probably unreachable,
    so this raises TargetError instead of failing every row. Later failures are
    recorded on their row.
    """
    program = RawPrompt(lm)

    def one(prompt: str) -> Generation:
        start = time.perf_counter()
        try:
            output = program(prompt=prompt)
        except Exception as e:
            error = f"{type(e).__name__}: {e}".splitlines()[0][:500]
            result = Generation(output="", error=error, seconds=time.perf_counter() - start)
        else:
            result = Generation(output=output, error=None, seconds=time.perf_counter() - start)
        if on_done is not None:
            on_done()
        return result

    if not prompts:
        return []
    first = one(prompts[0])
    if first.error is not None:
        raise TargetError(_explain_failure(first.error, getattr(lm, "model", "the target model")))
    if workers <= 1:
        return [first, *(one(p) for p in prompts[1:])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [first, *pool.map(one, prompts[1:])]


def _explain_failure(error: str, model: str) -> str:
    hint = (
        "Is Ollama running (`ollama serve`) and the model pulled (`ollama pull <model>`)?"
        if model.startswith("ollama")
        else "Check the model name and API key."
    )
    return (
        f"Couldn't get an answer from {model}: {error}\n{hint} "
        "You can pick another model with --target-model."
    )
