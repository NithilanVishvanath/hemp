"""SQLite store for runs. One row per example per check, so any summary can be re-derived.

Default location: `.hemp/hemp.db` in the current directory, or $HEMP_DB.
"""

import json
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from hemp.calibrate.labels import Label, LabelItem
from hemp.dataset import Row
from hemp.runner.results import (
    PROMPT_VARIANT,
    CheckOutcome,
    ExampleResult,
    RunInfo,
    RunResult,
    VariantResult,
)

DEFAULT_DB_PATH = Path(".hemp") / "hemp.db"
SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    hemp_version   TEXT NOT NULL,
    prompt_hash    TEXT NOT NULL,
    dataset_hash   TEXT NOT NULL,
    plan_hash      TEXT NOT NULL,
    target_model   TEXT NOT NULL,
    judge_model    TEXT,
    prompt_source  TEXT,
    dataset_source TEXT,
    plan_source    TEXT,
    n_examples     INTEGER NOT NULL,
    prompt_text    TEXT NOT NULL,
    plan_text      TEXT,
    questions_json TEXT NOT NULL DEFAULT '{}',
    baselines_from_json TEXT NOT NULL DEFAULT '{}',
    data_dir       TEXT
);
CREATE TABLE IF NOT EXISTS outputs (
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    variant  TEXT NOT NULL,
    example  INTEGER NOT NULL,
    row_json TEXT NOT NULL,
    output   TEXT NOT NULL,
    error    TEXT,
    seconds  REAL NOT NULL,
    PRIMARY KEY (run_id, variant, example)
);
CREATE TABLE IF NOT EXISTS results (
    run_id   INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    variant  TEXT NOT NULL,
    example  INTEGER NOT NULL,
    check_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    passed   INTEGER,
    score    REAL,
    detail   TEXT NOT NULL,
    error    INTEGER NOT NULL,
    PRIMARY KEY (run_id, variant, example, check_id)
);
CREATE INDEX IF NOT EXISTS runs_by_plan ON runs(plan_hash);
CREATE TABLE IF NOT EXISTS labels (
    key        TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    question   TEXT NOT NULL,
    task       TEXT NOT NULL,
    output     TEXT NOT NULL,
    reference  TEXT NOT NULL,
    passed     INTEGER NOT NULL,
    run_id     INTEGER,
    variant    TEXT,
    example    INTEGER,
    images_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS labels_by_question ON labels(question);
"""

# Columns added after their table was first released: (table, name, definition).
_ADDED_COLUMNS = [
    ("runs", "questions_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("runs", "baselines_from_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("runs", "data_dir", "TEXT"),
    ("labels", "images_json", "TEXT NOT NULL DEFAULT '[]'"),
]


class StoreError(RuntimeError):
    """The results database can't be opened or doesn't have the asked-for run."""


def default_db_path() -> Path:
    return Path(os.environ.get("HEMP_DB") or DEFAULT_DB_PATH)


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_db_path()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path)
        except (OSError, sqlite3.Error) as e:
            raise StoreError(f"Couldn't open the results database {self.path}: {e}") from None
        try:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._migrate()
        except BaseException:
            self._db.close()
            raise

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StoreError(f"{self.path} was written by a newer HEMP. Upgrade HEMP to read it.")
        with self._db:
            self._db.executescript(_SCHEMA)
            for table, name, definition in _ADDED_COLUMNS:  # columns older databases lack
                columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
                if name not in columns:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # --- writing ---------------------------------------------------------------

    def save_run(
        self,
        result: RunResult,
        *,
        prompt_text: str,
        rows: Sequence[Row],
        plan_text: str | None = None,
    ) -> int:
        """Store a run and return its id."""
        info = result.info
        with self._transaction() as db:
            cursor = db.execute(
                "INSERT INTO runs (created_at, hemp_version, prompt_hash, dataset_hash, "
                "plan_hash, target_model, judge_model, prompt_source, dataset_source, "
                "plan_source, n_examples, prompt_text, plan_text, questions_json, "
                "baselines_from_json, data_dir) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    info.created_at.isoformat(),
                    info.hemp_version,
                    info.prompt_hash,
                    info.dataset_hash,
                    info.plan_hash,
                    info.target_model,
                    info.judge_model,
                    info.prompt_source,
                    info.dataset_source,
                    info.plan_source,
                    info.n_examples,
                    prompt_text,
                    plan_text,
                    json.dumps(result.questions, ensure_ascii=False),
                    json.dumps(result.baselines_from),
                    info.data_dir,
                ),
            )
            run_id = cursor.lastrowid
            assert run_id is not None
            for variant in result.variants:
                db.executemany(
                    "INSERT INTO outputs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            run_id,
                            variant.name,
                            ex.index,
                            json.dumps(rows[ex.index], ensure_ascii=False),
                            ex.output,
                            ex.error,
                            ex.seconds,
                        )
                        for ex in variant.examples
                    ],
                )
                db.executemany(
                    "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            run_id,
                            variant.name,
                            ex.index,
                            o.check_id,
                            position,
                            None if o.passed is None else int(o.passed),
                            o.score,
                            o.detail,
                            int(o.error),
                        )
                        for ex in variant.examples
                        for position, o in enumerate(ex.checks)
                    ],
                )
        return run_id

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._db:
            yield self._db

    # --- reading ---------------------------------------------------------------

    def list_runs(self, limit: int = 20) -> list[tuple[int, RunInfo]]:
        """Most recent runs first."""
        rows = self._db.execute(
            f"SELECT {_INFO_COLUMNS}, id FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [(row[-1], _info(row)) for row in rows]

    def load_run(self, run_id: int) -> RunResult:
        found = self._db.execute(
            f"SELECT {_INFO_COLUMNS}, questions_json, baselines_from_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if found is None:
            raise StoreError(
                f"There's no run #{run_id} in {self.path}. `hemp show` lists the stored runs."
            )
        outcomes: dict[tuple[str, int], list[CheckOutcome]] = {}
        for variant, example, check_id, passed, score, detail, error in self._db.execute(
            "SELECT variant, example, check_id, passed, score, detail, error FROM results "
            "WHERE run_id = ? ORDER BY variant, example, position",
            (run_id,),
        ):
            outcomes.setdefault((variant, example), []).append(
                CheckOutcome(
                    check_id=check_id,
                    passed=None if passed is None else bool(passed),
                    score=score,
                    detail=detail,
                    error=bool(error),
                )
            )
        variants: dict[str, list[ExampleResult]] = {}
        for variant, example, output, error, seconds in self._db.execute(
            "SELECT variant, example, output, error, seconds FROM outputs "
            "WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ):
            variants.setdefault(variant, []).append(
                ExampleResult(
                    index=example,
                    output=output,
                    error=error,
                    seconds=seconds,
                    checks=outcomes.get((variant, example), []),
                )
            )
        return RunResult(
            id=run_id,
            info=_info(found[:-2]),
            variants=[VariantResult(name=n, examples=ex) for n, ex in variants.items()],
            questions=json.loads(found[-2]),
            baselines_from=json.loads(found[-1]),
        )

    def load_rows(self, run_id: int) -> list[Row]:
        """The dataset rows a run used, in order."""
        rows = self._db.execute(
            "SELECT row_json FROM outputs WHERE run_id = ? AND variant = "
            "(SELECT variant FROM outputs WHERE run_id = ? LIMIT 1) ORDER BY example",
            (run_id, run_id),
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def latest_run_id(self) -> int | None:
        found = self._db.execute("SELECT MAX(id) FROM runs").fetchone()
        return found[0] if found else None

    def find_reusable_run(
        self, *, plan_hash: str, dataset_hash: str, target_model: str, judge_model: str | None
    ) -> int | None:
        """The latest run with baselines whose plan, dataset and models all match."""
        found = self._db.execute(
            "SELECT id FROM runs WHERE plan_hash = ? AND dataset_hash = ? AND target_model = ? "
            "AND judge_model IS ? AND EXISTS (SELECT 1 FROM outputs "
            "WHERE outputs.run_id = runs.id AND outputs.variant != ?) "
            "ORDER BY id DESC LIMIT 1",
            (plan_hash, dataset_hash, target_model, judge_model, PROMPT_VARIANT),
        ).fetchone()
        return found[0] if found else None

    def latest_judged_run(self, plan_hash: str) -> int | None:
        """The latest run of this plan whose judge questions were answered."""
        found = self._db.execute(
            "SELECT id FROM runs WHERE plan_hash = ? AND questions_json != '{}' "
            "ORDER BY id DESC LIMIT 1",
            (plan_hash,),
        ).fetchone()
        return found[0] if found else None

    # --- labels ------------------------------------------------------------------

    def save_label(self, item: LabelItem, passed: bool) -> None:
        """Store a human answer. Labeling the same thing again replaces the old answer."""
        with self._transaction() as db:
            db.execute(
                "INSERT OR REPLACE INTO labels (key, created_at, question, task, output, "
                "reference, passed, run_id, variant, example, images_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.key,
                    datetime.now(UTC).isoformat(),
                    item.question,
                    item.task,
                    item.output,
                    item.reference,
                    int(passed),
                    item.run_id,
                    item.variant,
                    item.example,
                    json.dumps(list(item.images)),
                ),
            )

    def load_labels(self, questions: Sequence[str]) -> list[Label]:
        """Labels for these question texts, oldest first."""
        if not questions:
            return []
        marks = ", ".join("?" for _ in questions)
        rows = self._db.execute(
            "SELECT question, task, output, reference, passed, images_json FROM labels "
            f"WHERE question IN ({marks}) ORDER BY created_at, key",
            list(questions),
        ).fetchall()
        return [
            Label(
                question=q,
                task=t,
                output=o,
                reference=r,
                passed=bool(p),
                images=tuple(json.loads(i)),
            )
            for q, t, o, r, p, i in rows
        ]

    def latest_run_ids(self, n: int) -> list[int]:
        """The ids of the `n` most recent runs, oldest first."""
        rows = self._db.execute("SELECT id FROM runs ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [r[0] for r in reversed(rows)]

    def load_texts(self, run_id: int) -> tuple[str, str | None]:
        """The prompt and plan file text a run used."""
        found = self._db.execute(
            "SELECT prompt_text, plan_text FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if found is None:
            raise StoreError(
                f"There's no run #{run_id} in {self.path}. `hemp show` lists the stored runs."
            )
        return found[0], found[1]


_INFO_COLUMNS = (
    "created_at, hemp_version, prompt_hash, dataset_hash, plan_hash, target_model, "
    "judge_model, prompt_source, dataset_source, plan_source, data_dir, n_examples"
)


def _info(row: Sequence[object]) -> RunInfo:
    names = [c.strip() for c in _INFO_COLUMNS.split(",")]
    values = dict(zip(names, row, strict=False))
    values["created_at"] = datetime.fromisoformat(str(values["created_at"]))
    return RunInfo.model_validate(values)
