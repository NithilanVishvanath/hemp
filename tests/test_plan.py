import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from dspy.utils import DummyLM

from hemp.plan import (
    CheckSpec,
    Criterion,
    EvalPlan,
    PlanError,
    PlannerError,
    create_plan,
    describe_profile,
    dump_plan,
    parse_plan,
    read_plan,
    write_plan,
)
from hemp.plan.planner import MAX_CRITERIA
from hemp.profile import DatasetProfile, profile_rows

PROMPT = (
    "Classify the review as positive, negative or neutral. Reply with the label only.\n"
    "Be fair to the reviewer.\n\nReview: {review}"
)
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def sentiment_profile() -> DatasetProfile:
    rows = [
        {"id": i, "review": f"review text number {i}", "label": label}
        for i, label in enumerate(["positive", "negative", "neutral"] * 3)
    ]
    return profile_rows(rows, prompt=PROMPT)


def fake_lm(checks: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> DummyLM:
    return DummyLM(
        [
            {
                "reasoning": "thinking",
                "format_checks": json.dumps(checks),
                "criteria": json.dumps(criteria),
            }
        ]
    )


def plan_with(checks: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> Any:
    return create_plan(
        PROMPT, sentiment_profile(), lm=fake_lm(checks, criteria), planner_model="dummy", now=NOW
    )


# --- planner -----------------------------------------------------------------


def test_planner_builds_full_plan() -> None:
    result = plan_with(
        checks=[
            {
                "metric": "allowed_values",
                "params": {"values": ["positive", "negative", "neutral"]},
                "instruction": "Reply with the label only.",
            }
        ],
        criteria=[{"question": "Is the label fair to the reviewer?", "instruction": "Be fair"}],
    )
    plan = result.plan
    assert result.warnings == []
    assert plan.input_columns == ["review"]
    assert plan.gold_columns == ["label"]
    assert [c.metric for c in plan.format_checks] == ["non_empty", "allowed_values"]
    allowed = plan.format_checks[1]
    assert allowed.params == {
        "values": ["positive", "negative", "neutral"],
        "case_sensitive": False,
    }  # defaults filled in explicitly
    assert allowed.rationale == "Reply with the label only."
    assert [(m.metric, m.gold_column) for m in plan.reference_metrics] == [("exact_match", "label")]
    assert plan.criteria == [
        Criterion(id="c1", question="Is the label fair to the reviewer?", source="Be fair")
    ]
    assert [b.kind.value for b in plan.baselines] == ["majority_class", "zero_shot"]
    assert plan.meta.planner_model == "dummy"
    assert plan.meta.dataset_hash == sentiment_profile().dataset_hash


def test_planner_drops_invalid_proposals_with_warnings() -> None:
    result = plan_with(
        checks=[
            {"metric": "made_up", "params": {}},
            {"metric": "length", "params": {"max": "lots"}},
            {"metric": "exact_match", "params": {}},  # needs gold; not a format check
            {"metric": "length", "params": {"max": 5}},
            {"metric": "length", "params": {"max": 5}},  # duplicate
        ],
        criteria=[],
    )
    assert len(result.warnings) == 3
    assert any("made_up" in w for w in result.warnings)
    lengths = [c for c in result.plan.format_checks if c.metric == "length"]
    assert len(lengths) == 1


def test_label_check_added_from_data_when_planner_skips_it() -> None:
    plan = plan_with(checks=[], criteria=[]).plan
    allowed = [c for c in plan.format_checks if c.metric == "allowed_values"]
    assert len(allowed) == 1
    assert set(allowed[0].params["values"]) == {"positive", "negative", "neutral"}


def test_criteria_deduped_numbered_and_capped() -> None:
    questions = [{"question": f"Is it ok {i}?"} for i in range(MAX_CRITERIA + 3)]
    questions.insert(1, {"question": "  is it OK 0? "})  # duplicate after normalizing
    result = plan_with(checks=[], criteria=questions)
    ids = [c.id for c in result.plan.criteria]
    assert ids == [f"c{i}" for i in range(1, MAX_CRITERIA + 1)]
    assert any("kept the first" in w for w in result.warnings)


def test_repeated_check_ids_get_suffixes() -> None:
    plan = plan_with(
        checks=[
            {"metric": "regex", "params": {"pattern": "a"}},
            {"metric": "regex", "params": {"pattern": "b"}},
        ],
        criteria=[],
    ).plan
    assert [c.id for c in plan.format_checks if c.metric == "regex"] == ["regex", "regex_2"]


def test_no_llm_plan() -> None:
    result = create_plan(PROMPT, sentiment_profile(), lm=None, now=NOW)
    assert result.plan.criteria == []
    assert result.plan.meta.planner_model is None
    assert "without an LLM" in result.warnings[0]


def test_lm_failure_becomes_planner_error() -> None:
    class BrokenLM(DummyLM):  # type: ignore[misc]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("connection refused")

    lm = BrokenLM([])
    lm.model = "ollama_chat/missing"
    with pytest.raises(PlannerError, match="ollama serve"):
        create_plan(PROMPT, sentiment_profile(), lm=lm, now=NOW)


def test_describe_profile_mentions_columns_and_samples() -> None:
    text = describe_profile(sentiment_profile())
    assert "- review: text; input" in text
    assert "- label: categorical; gold; values: " in text
    assert '"review": "review text number 0"' in text


# --- schema & hashing -------------------------------------------------------


def base_plan() -> EvalPlan:
    return create_plan(PROMPT, sentiment_profile(), lm=None, now=NOW).plan


def test_hash_ignores_meta_but_tracks_content() -> None:
    plan = base_plan()
    later = plan.model_copy(
        update={"meta": plan.meta.model_copy(update={"created_at": datetime.now(UTC)})}
    )
    assert later.plan_hash() == plan.plan_hash()
    edited = plan.model_copy(update={"criteria": [Criterion(id="c1", question="Is it polite?")]})
    assert edited.plan_hash() != plan.plan_hash()


def test_duplicate_ids_rejected() -> None:
    data = base_plan().model_dump(mode="json")
    data["format_checks"].append(data["format_checks"][0])
    with pytest.raises(ValueError, match="duplicated: non_empty"):
        EvalPlan.model_validate(data)


def test_gold_column_must_be_listed() -> None:
    data = base_plan().model_dump(mode="json")
    data["reference_metrics"][0]["gold_column"] = "nope"
    with pytest.raises(ValueError, match="isn't listed in gold_columns"):
        EvalPlan.model_validate(data)


# --- YAML I/O ---------------------------------------------------------------


def test_yaml_round_trip_preserves_hash(tmp_path: Path) -> None:
    plan = plan_with(
        checks=[{"metric": "regex", "params": {"pattern": "^(positive|negative|neutral)$"}}],
        criteria=[{"question": "Multi\nline?"}],
    ).plan
    path = write_plan(plan, tmp_path / "p.plan.yaml")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# HEMP evaluation plan")
    assert "gold_column: null" not in text  # empty fields are left out to keep it readable
    loaded = read_plan(path)
    assert loaded == plan
    assert loaded.plan_hash() == plan.plan_hash()


def test_write_refuses_to_overwrite(tmp_path: Path) -> None:
    path = write_plan(base_plan(), tmp_path / "p.yaml")
    with pytest.raises(PlanError, match="--force"):
        write_plan(base_plan(), path)
    write_plan(base_plan(), path, overwrite=True)


def edited(transform: Any) -> str:
    import yaml

    data = yaml.safe_load(dump_plan(base_plan()))
    transform(data)
    return yaml.safe_dump(data)


def test_parse_rejects_unknown_metric() -> None:
    text = edited(lambda d: d["format_checks"][0].update(metric="nope"))
    with pytest.raises(PlanError, match="format_checks / non_empty: Unknown metric 'nope'"):
        parse_plan(text)


def test_parse_rejects_bad_params() -> None:
    text = edited(lambda d: d["format_checks"][1]["params"].update(values=[]))
    with pytest.raises(PlanError, match="allowed_values"):
        parse_plan(text)


def test_parse_rejects_typo_field() -> None:
    text = edited(lambda d: d["criteria"].append({"id": "c1", "questoin": "Is it ok?"}))
    with pytest.raises(PlanError, match="questoin"):
        parse_plan(text)


def test_parse_rejects_metric_in_wrong_section() -> None:
    def move(d: dict[str, Any]) -> None:
        d["format_checks"].append(
            {"id": "em", "metric": "exact_match", "params": {}, "gold_column": "label"}
        )

    with pytest.raises(PlanError, match="reference metric and doesn't belong in format_checks"):
        parse_plan(edited(move))


def test_parse_rejects_bad_yaml_and_non_mapping() -> None:
    with pytest.raises(PlanError, match="isn't valid YAML"):
        parse_plan("a: [unclosed")
    with pytest.raises(PlanError, match="YAML mapping"):
        parse_plan("- just\n- a list\n")


def test_read_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PlanError, match="hemp plan"):
        read_plan(tmp_path / "missing.yaml")


def test_checkspec_requires_id() -> None:
    with pytest.raises(ValueError):
        CheckSpec(id="", metric="non_empty")


def test_planner_repeats_of_rule_checks_are_dropped() -> None:
    plan = plan_with(checks=[{"metric": "non_empty", "params": {}}], criteria=[]).plan
    assert [c.id for c in plan.format_checks].count("non_empty") == 1
    assert "non_empty_2" not in [c.id for c in plan.format_checks]


def test_prompt_labels_replace_data_labels() -> None:
    plan = plan_with(
        checks=[{"metric": "allowed_values", "params": {"values": ["Positive", "Negative"]}}],
        criteria=[],
    ).plan
    allowed = [c for c in plan.format_checks if c.metric == "allowed_values"]
    assert len(allowed) == 1
    assert allowed[0].params["values"] == ["Positive", "Negative"]


def test_planner_check_gets_default_rationale() -> None:
    plan = plan_with(checks=[{"metric": "length", "params": {"max": 3}}], criteria=[]).plan
    length = next(c for c in plan.format_checks if c.metric == "length")
    assert length.rationale == "proposed by the planner from the prompt"


def test_planner_is_told_about_existing_checks() -> None:
    lm = fake_lm(checks=[], criteria=[])
    create_plan(PROMPT, sentiment_profile(), lm=lm, now=NOW)
    sent = str(lm.history[-1]["messages"])
    assert "already_included" in sent
    assert "exact_match" in sent
    assert "allowed_values" in sent
