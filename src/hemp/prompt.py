"""Prompt templates: `{column}` placeholders filled from dataset rows.

Only `{name}` with a simple identifier counts as a placeholder, so JSON examples
in a prompt (`{"a": 1}`) and doubled braces (`{{literal}}`) are left alone.
"""

import re

from hemp.hashing import sha256_text

_FIELD_RE = re.compile(r"(?<!\{)\{([A-Za-z_][\w\-]*)\}(?!\})")


def template_fields(template: str) -> list[str]:
    """Placeholder names in the order they first appear."""
    return list(dict.fromkeys(_FIELD_RE.findall(template)))


def prompt_hash(template: str) -> str:
    """Hash of the prompt text. Only line endings are normalized; everything else counts."""
    return sha256_text(template.replace("\r\n", "\n"))
