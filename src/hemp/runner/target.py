"""Run the user's prompt exactly as written.

DSPy adapters rewrite prompts, so the target model is called through `dspy.LM`
directly: the filled-in template goes out as a single user message, unchanged.

When a placeholder is an image column, the message is split into parts and the
image itself goes where the placeholder was; the text around it is unchanged.
"""

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import dspy

from hemp.dataset import Row
from hemp.images import ImageRef, resolve
from hemp.prompt import FIELD_RE, template_fields
from hemp.runner.parallel import run_all, short_error


class PromptError(ValueError):
    """The prompt can't be filled in from the dataset."""


class TargetError(RuntimeError):
    """The target model can't be reached at all."""


class TextLM(Protocol):
    """Anything called like `dspy.LM`: `lm("prompt") -> [output, ...]`, or
    `lm(lm15.Request) -> lm15.Response` for prompts with images."""

    model: str

    def __call__(self, prompt: Any) -> Any: ...


# A filled-in prompt: plain text, or text and images in order.
Part = str | ImageRef
PromptContent = str | tuple[Part, ...]


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


def build_prompt(
    template: str, row: Row, images: set[str] | frozenset[str] = frozenset(), base_dir: Any = None
) -> PromptContent:
    """Fill the template. Placeholders for image columns become the images themselves.

    Without image placeholders this is exactly `fill_prompt`.
    """
    fields = set(template_fields(template))
    if not fields & set(images):
        return fill_prompt(template, row)
    fill_prompt(template, row)  # raises the usual error for missing columns
    parts: list[Part] = []
    pos = 0
    for match in FIELD_RE.finditer(template):
        name = match.group(1)
        parts.append(template[pos : match.start()])
        value = row[name]
        if name in images and isinstance(value, str) and value.strip():
            parts.append(resolve(value, base_dir))
        else:
            parts.append(value_text(value))
        pos = match.end()
    parts.append(template[pos:])
    return merge_parts(parts)


def merge_parts(parts: list[Part]) -> tuple[Part, ...]:
    """Join neighbouring text and drop empty text."""
    merged: list[Part] = []
    for part in parts:
        if isinstance(part, str) and merged and isinstance(merged[-1], str):
            merged[-1] += part
        else:
            merged.append(part)
    return tuple(p for p in merged if not (isinstance(p, str) and p == ""))


def prompt_text(content: PromptContent) -> str:
    """The prompt as text, with `[image N]` where each image goes."""
    if isinstance(content, str):
        return content
    out = []
    n = 0
    for part in content:
        if isinstance(part, str):
            out.append(part)
        else:
            n += 1
            out.append(f"[image {n}]")
    return "".join(out)


def prompt_images(content: PromptContent) -> list[ImageRef]:
    return [] if isinstance(content, str) else [p for p in content if isinstance(p, ImageRef)]


def to_request(model: str, content: tuple[Part, ...]) -> Any:
    """A dspy lm15 request with one user message holding the text and images in order."""
    from dspy import lm15

    parts: list[Any] = []
    for part in content:
        if isinstance(part, str):
            parts.append(lm15.TextPart(text=part))
        elif part.is_file or part.location.startswith("data:"):
            parts.append(lm15.ImagePart(media_type=part.media_type, data=part.base64_data()))
        else:
            parts.append(lm15.ImagePart(media_type=part.media_type, url=part.location))
    message = lm15.Message(role="user", parts=tuple(parts))
    return lm15.Request(model=model, messages=(message,))


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

    def forward(self, prompt: PromptContent) -> str:
        if isinstance(prompt, str):
            return output_text(self.lm(prompt))
        reply = self.lm(to_request(self.lm.model, prompt))
        if isinstance(reply, list):  # test doubles answer like a plain call
            return output_text(reply)
        return str(getattr(reply, "text", None) or "")


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
    prompts: Sequence[PromptContent],
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

    def one(prompt: PromptContent) -> Generation:
        start = time.perf_counter()
        output, error = "", None
        try:
            output = program(prompt=prompt)
        except Exception as e:
            error = short_error(e)
        result = Generation(output=output, error=error, seconds=time.perf_counter() - start)
        if on_done is not None:
            on_done()
        return result

    def check_first(gen: Generation) -> None:
        if gen.error is not None:
            model = getattr(lm, "model", "the target model")
            raise TargetError(explain_failure(gen.error, model, "--target-model"))

    return run_all(one, prompts, workers=workers, check_first=check_first)


def explain_failure(error: str, model: str, option: str) -> str:
    hint = (
        "Is Ollama running (`ollama serve`) and the model pulled (`ollama pull <model>`)?"
        if model.startswith("ollama")
        else "Check the model name and API key."
    )
    return (
        f"Couldn't get an answer from {model}: {error}\n{hint} "
        f"You can pick another model with {option}."
    )
