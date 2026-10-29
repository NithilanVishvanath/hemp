"""Guess each column's role (input, gold, id, ...) from its name and the prompt."""

import re

from hemp.profile.schema import ColumnProfile, ColumnRole, ColumnType

GOLD_HINTS = frozenset(
    {
        "label",
        "labels",
        "target",
        "targets",
        "gold",
        "answer",
        "answers",
        "expected",
        "reference",
        "references",
        "truth",
        "y",
        "output",
        "class",
        "category",
        "solution",
        "ideal",
    }
)
INPUT_HINTS = frozenset(
    {
        "input",
        "text",
        "question",
        "query",
        "prompt",
        "context",
        "document",
        "passage",
        "sentence",
        "review",
        "message",
        "content",
        "x",
    }
)
ID_NAMES = frozenset({"id", "uuid", "guid", "uid", "idx", "index", "key"})


def name_tokens(name: str) -> list[str]:
    """'expectedOutput', 'expected_output', 'Expected Output' -> ['expected', 'output']."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t]


def _is_id(col: ColumnProfile) -> bool:
    tokens = name_tokens(col.name)
    named_like_id = bool(tokens) and tokens[-1] in ID_NAMES
    return named_like_id and col.n_missing == 0 and col.unique_ratio == 1.0


def assign_roles(
    columns: list[ColumnProfile], prompt_fields: list[str] | None
) -> list[ColumnProfile]:
    """Return copies of the columns with role and role_reason set.

    With a prompt, its placeholders mark the inputs and the other columns are gold
    candidates. Without one, only column names are used.
    """
    used = set(prompt_fields) if prompt_fields is not None else None
    result: list[ColumnProfile] = []
    for col in columns:
        role, reason = _role_for(col, used)
        result.append(col.model_copy(update={"role": role, "role_reason": reason}))

    # With a prompt, a single unused column is most likely the answer column.
    if used is not None and not any(c.role is ColumnRole.GOLD for c in result):
        leftovers = [i for i, c in enumerate(result) if c.role is ColumnRole.METADATA]
        if len(leftovers) == 1:
            i = leftovers[0]
            result[i] = result[i].model_copy(
                update={
                    "role": ColumnRole.GOLD,
                    "role_reason": "the only column the prompt doesn't use",
                }
            )
    return result


def _role_for(col: ColumnProfile, used: set[str] | None) -> tuple[ColumnRole, str]:
    if used is not None and col.name in used:
        return ColumnRole.INPUT, f"used in the prompt as {{{col.name}}}"
    if _is_id(col):
        return ColumnRole.ID, "named like an identifier and unique per row"
    if col.type is ColumnType.EMPTY:
        return ColumnRole.METADATA, "every value is missing"
    tokens = name_tokens(col.name)
    gold_hit = next((t for t in tokens if t in GOLD_HINTS), None)
    if gold_hit:
        reason = f"name suggests a reference answer ('{gold_hit}')"
        if used is not None:
            reason += " and the prompt doesn't use it"
        return ColumnRole.GOLD, reason
    if used is not None:
        return ColumnRole.METADATA, "the prompt doesn't use it"
    input_hit = next((t for t in tokens if t in INPUT_HINTS), None)
    if input_hit:
        return ColumnRole.INPUT, f"name suggests model input ('{input_hit}')"
    return ColumnRole.UNKNOWN, "no prompt given and the name doesn't suggest a role"
