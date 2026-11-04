import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan

from hemp.runner import RunResult, run_eval
from hemp.store import Store, StoreError


def make_run(hour: int = 0) -> RunResult:
    return run_eval(
        SENTIMENT_PROMPT,
        SENTIMENT_ROWS,
        make_plan(),
        lm=FakeLM(keyword_sentiment),
        prompt_source="prompt.txt",
        dataset_source="data.jsonl",
        now=datetime(2026, 2, 1, hour, tzinfo=UTC),
    )


def save(store: Store, result: RunResult) -> int:
    return store.save_run(result, prompt_text=SENTIMENT_PROMPT, rows=SENTIMENT_ROWS)


def test_round_trip(tmp_path: Path) -> None:
    result = make_run()
    with Store(tmp_path / "runs.db") as store:
        run_id = save(store, result)
        loaded = store.load_run(run_id)
        assert loaded == result.model_copy(update={"id": run_id})
        assert store.load_rows(run_id) == SENTIMENT_ROWS


def test_one_row_per_example_per_check(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    with Store(path) as store:
        save(store, make_run())
    db = sqlite3.connect(path)
    (n,) = db.execute("SELECT COUNT(*) FROM results WHERE variant = 'prompt'").fetchone()
    db.close()
    assert n == len(SENTIMENT_ROWS) * 3  # three checks in the plan


def test_list_runs_newest_first(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store:
        first = save(store, make_run(1))
        second = save(store, make_run(2))
        assert [run_id for run_id, _ in store.list_runs()] == [second, first]
        assert store.latest_run_id() == second
        info = store.list_runs()[0][1]
        assert info.prompt_source == "prompt.txt"
        assert info.created_at == datetime(2026, 2, 1, 2, tzinfo=UTC)


def test_reopening_keeps_runs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "runs.db"
    with Store(path) as store:
        run_id = save(store, make_run())
    with Store(path) as store:
        assert store.load_run(run_id).id == run_id


def test_missing_run(tmp_path: Path) -> None:
    with Store(tmp_path / "runs.db") as store, pytest.raises(StoreError, match="no run #7"):
        store.load_run(7)


def test_newer_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "runs.db"
    db = sqlite3.connect(path)
    db.execute("PRAGMA user_version = 99")
    db.close()
    with pytest.raises(StoreError, match="newer HEMP"):
        Store(path)


def test_default_path_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEMP_DB", str(tmp_path / "env.db"))
    with Store() as store:
        assert store.path == tmp_path / "env.db"


def test_questions_round_trip(tmp_path: Path) -> None:
    result = make_run().model_copy(update={"questions": {"q1": "Is it short?"}})
    with Store(tmp_path / "runs.db") as store:
        run_id = save(store, result)
        assert store.load_run(run_id).questions == {"q1": "Is it short?"}


def test_upgrades_a_version_1_database(tmp_path: Path) -> None:
    from hemp.store import db as db_module

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    v1_schema = db_module._SCHEMA
    for _, name, definition in db_module._ADDED_COLUMNS:
        v1_schema = re.sub(rf",\n\s+{name}\s+{re.escape(definition)}", "", v1_schema)
    old.executescript(v1_schema)
    old.execute("PRAGMA user_version = 1")
    columns = {row[1] for row in old.execute("PRAGMA table_info(runs)")}
    columns |= {row[1] for row in old.execute("PRAGMA table_info(labels)")}
    assert not {name for _, name, _ in db_module._ADDED_COLUMNS} & columns
    old.close()
    with Store(path) as store:
        run_id = save(store, make_run())
        loaded = store.load_run(run_id)
        assert loaded.questions == {}
        assert loaded.baselines_from == {}
