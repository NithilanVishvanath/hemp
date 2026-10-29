"""DatasetProfile: what the planner knows about a dataset."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict


class ColumnType(StrEnum):
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"  # short strings from a small set of values
    TEXT = "text"  # free-form strings
    JSON = "json"  # objects/arrays, natively or as JSON-encoded strings
    MIXED = "mixed"  # structured and scalar values in the same column
    EMPTY = "empty"  # every value missing


class ColumnRole(StrEnum):
    INPUT = "input"  # fed to the prompt
    GOLD = "gold"  # reference answer to compare outputs against
    ID = "id"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LengthStats(_Model):
    """Character lengths of the values (JSON values measured as compact JSON)."""

    min: int
    max: int
    mean: float
    median: float
    mean_words: float


class NumericStats(_Model):
    min: float
    max: float
    mean: float


class ValueCount(_Model):
    value: Any
    count: int


class ColumnProfile(_Model):
    name: str
    type: ColumnType
    role: ColumnRole = ColumnRole.UNKNOWN
    role_reason: str = ""
    n_missing: int
    n_unique: int
    unique_ratio: float
    top_values: list[ValueCount] | None = None
    lengths: LengthStats | None = None
    numeric: NumericStats | None = None
    json_keys: dict[str, float] | None = None  # key -> fraction of objects containing it
    json_in_strings: bool = False  # values are JSON encoded inside strings


class DatasetProfile(_Model):
    source: str | None = None
    format: str | None = None
    n_rows: int
    dataset_hash: str
    columns: list[ColumnProfile]
    sample_rows: list[dict[str, Any]]
    prompt_fields: list[str] | None = None  # placeholders in the prompt, if one was given
    missing_prompt_fields: list[str] = []  # placeholders with no matching column

    def column(self, name: str) -> ColumnProfile:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(name)

    def columns_with_role(self, role: ColumnRole) -> list[str]:
        return [c.name for c in self.columns if c.role is role]

    @property
    def gold_columns(self) -> list[str]:
        return self.columns_with_role(ColumnRole.GOLD)

    @property
    def input_columns(self) -> list[str]:
        return self.columns_with_role(ColumnRole.INPUT)
