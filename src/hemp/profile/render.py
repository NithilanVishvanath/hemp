"""Terminal rendering of a DatasetProfile."""

from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text

from hemp.profile.schema import ColumnProfile, ColumnRole, ColumnType, DatasetProfile

_ROLE_STYLES = {
    ColumnRole.INPUT: "cyan",
    ColumnRole.GOLD: "bold green",
    ColumnRole.ID: "dim",
    ColumnRole.METADATA: "dim",
    ColumnRole.UNKNOWN: "yellow",
}


def render_profile(profile: DatasetProfile) -> RenderableType:
    header = Text.assemble(
        (profile.source or "dataset", "bold"),
        f"  {profile.n_rows} rows",
        f", {profile.format.upper()}" if profile.format else "",
        (f"  hash {profile.dataset_hash[:12]}", "dim"),
    )

    table = Table(show_lines=False, header_style="bold")
    table.add_column("Column")
    table.add_column("Type")
    table.add_column("Role")
    table.add_column("Missing", justify="right")
    table.add_column("Unique", justify="right")
    table.add_column("Details", overflow="fold")
    for col in profile.columns:
        table.add_row(
            col.name,
            col.type.value,
            Text(col.role.value, style=_ROLE_STYLES[col.role]),
            str(col.n_missing),
            str(col.n_unique),
            _details(col),
        )

    reasons = Text()
    for col in profile.columns:
        reasons.append(f"  {col.name}", style="bold")
        reasons.append(f" -> {col.role.value}: {col.role_reason}\n", style="dim")

    parts: list[RenderableType] = [header, table, Text("Roles", style="bold"), reasons]
    parts.extend(_warnings(profile))
    return Group(*parts)


def print_profile(profile: DatasetProfile, console: Console | None = None) -> None:
    (console or Console()).print(render_profile(profile))


def _details(col: ColumnProfile) -> str:
    bits: list[str] = []
    if col.top_values:
        shown = ", ".join(f"{v.value} ({v.count})" for v in col.top_values[:5])
        more = len(col.top_values) - 5
        bits.append(shown + (f", +{more} more" if more > 0 else ""))
    if col.numeric and not col.top_values:
        n = col.numeric
        bits.append(f"range {n.min:g} to {n.max:g}, mean {n.mean:g}")
    if col.lengths and col.type is not ColumnType.CATEGORICAL:
        ln = col.lengths
        bits.append(f"{ln.min} to {ln.max} chars (mean {ln.mean:g}, ~{ln.mean_words:g} words)")
    if col.json_keys:
        bits.append("keys: " + ", ".join(list(col.json_keys)[:8]))
    if col.json_in_strings:
        bits.append("JSON stored as strings")
    return "; ".join(bits)


def _warnings(profile: DatasetProfile) -> list[RenderableType]:
    out: list[RenderableType] = []
    for field in profile.missing_prompt_fields:
        out.append(
            Text(
                f"! The prompt uses {{{field}}} but the dataset has no '{field}' column. "
                "Rename the placeholder or the column so they match.",
                style="red",
            )
        )
    if not profile.gold_columns:
        out.append(
            Text(
                "! No gold (reference answer) column found, so only format and rubric checks "
                "will apply. If one exists, rename it to e.g. 'label' or 'expected_output'.",
                style="yellow",
            )
        )
    return out
