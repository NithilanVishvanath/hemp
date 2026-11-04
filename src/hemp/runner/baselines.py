"""Trivial baselines, so a prompt's scores have something to be compared with.

- majority_class: answer the most common value of the answer column for every row.
  No model calls.
- zero_shot: send the target model only the input values, with no instructions.
"""

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from hemp.dataset import Row
from hemp.hashing import canonical_json
from hemp.images import resolve
from hemp.metrics.text import is_missing
from hemp.plan.schema import EvalPlan
from hemp.runner.target import Part, PromptContent, merge_parts, value_text


def majority_answer(plan: EvalPlan, rows: Sequence[Row]) -> str:
    """The most common answer in the first answer column (ties: the one seen first)."""
    if not plan.gold_columns:
        raise ValueError("majority_class needs an answer column.")
    column = plan.gold_columns[0]
    values = [row.get(column) for row in rows]
    present = [v for v in values if not is_missing(v)]
    if not present:
        return ""
    counts = Counter(canonical_json(v) for v in present)
    best = max(counts.values())
    winner = next(v for v in present if counts[canonical_json(v)] == best)
    return value_text(winner)


def zero_shot_prompt(
    plan: EvalPlan,
    row: Row,
    images: set[str] | frozenset[str] = frozenset(),
    base_dir: Path | None = None,
) -> PromptContent:
    """Just the input: the bare value for one input column, `name: value` lines for several.

    Image columns are sent as the images themselves.
    """
    columns = plan.input_columns

    def value(name: str) -> Part:
        raw = row.get(name)
        if name in images and isinstance(raw, str) and raw.strip():
            return resolve(raw, base_dir)
        return value_text(raw)

    if not images & set(columns):
        if len(columns) == 1:
            return value_text(row.get(columns[0]))
        return "\n".join(f"{c}: {value_text(row.get(c))}" for c in columns)
    if len(columns) == 1:
        return (value(columns[0]),)
    parts: list[Part] = []
    for i, name in enumerate(columns):
        parts += [("\n" if i else "") + f"{name}: ", value(name)]
    return merge_parts(parts)
