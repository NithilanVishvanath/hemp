"""Prompt templates: `{column}` placeholders filled from dataset rows.

Only `{name}` with a simple identifier counts as a placeholder, so JSON examples
in a prompt (`{"a": 1}`) and doubled braces (`{{literal}}`) are left alone.
"""

import re

_FIELD_RE = re.compile(r"(?<!\{)\{([A-Za-z_][\w\-]*)\}(?!\})")


def template_fields(template: str) -> list[str]:
    """Placeholder names in the order they first appear."""
    return list(dict.fromkeys(_FIELD_RE.findall(template)))
