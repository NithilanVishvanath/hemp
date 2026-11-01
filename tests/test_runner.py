from datetime import UTC, datetime

import pytest
from fakes import SENTIMENT_PROMPT, SENTIMENT_ROWS, FakeLM, keyword_sentiment, make_plan

import hemp
from hemp.metrics import MetricContext, MetricResult
from hemp.plan.schema import CheckSpec
from hemp.runner import PROMPT_VARIANT, RunError, TargetError, fill_prompt, run_eval
from hemp.runner.baselines import majority_answer, zero_shot_prompt
from hemp.runner.evaluate import evaluate_output
from hemp.runner.results import CheckOutcome, ExampleResult, VariantResult
from hemp.runner.target import PromptError, check_prompt_fields, generate, output_text

# --- filling the prompt -------------------------------------------------------------


def test_fill_prompt_only_touches_placeholders() -> None:
    template = 'Say {{literal}} and use JSON like {"a": 1}.\nText: {text}\nAgain: {text}'
    filled = fill_prompt(template, {"text": "hi", "other": "x"})
    assert filled == 'Say {{literal}} and use JSON like {"a": 1}.\nText: hi\nAgain: hi'


def test_fill_prompt_formats_values() -> None:
    template = "{a}|{b}|{c}|{d}"
    row = {"a": {"k": "é"}, "b": [1, 2], "c": None, "d": 3}
    assert fill_prompt(template, row) == '{"k": "é"}|[1, 2]||3'


def test_fill_prompt_missing_column() -> None:
    with pytest.raises(PromptError, match=r"\{question\}"):
        fill_prompt("Q: {question}", {"text": "hi"})


def test_check_prompt_fields_lists_columns() -> None:
    with pytest.raises(PromptError, match="Columns: label, review"):
        check_prompt_fields("{text}", [{"review": "a", "label": "b"}])


def test_output_text_shapes() -> None:
    assert output_text(["hello"]) == "hello"
    assert output_text([{"text": "hi", "reasoning_content": "..."}]) == "hi"
    assert output_text([{"text": None}]) == ""
    assert output_text([]) == ""


# --- calling the model --------------------------------------------------------------


def test_generate_keeps_order_with_workers() -> None:
    lm = FakeLM(lambda p: p.upper())
    prompts = [f"p{i}" for i in range(20)]
    gens = generate(lm, prompts, workers=4)
    assert [g.output for g in gens] == [p.upper() for p in prompts]
    assert all(g.error is None for g in gens)


def test_generate_first_failure_stops_the_run() -> None:
    lm = FakeLM(lambda p: ConnectionError("refused"), model="ollama_chat/qwen2.5:7b")
    with pytest.raises(TargetError, match="ollama serve"):
        generate(lm, ["a", "b", "c"])
    assert lm.prompts == ["a"]  # gave up after the first call


def test_generate_records_later_failures() -> None:
    lm = FakeLM(lambda p: ValueError("too long") if p == "b" else "ok")
    gens = generate(lm, ["a", "b", "c"], workers=1)
    assert [g.output for g in gens] == ["ok", "", "ok"]
    assert gens[1].error == "ValueError: too long"


def test_generate_counts_progress() -> None:
    done: list[int] = []
    generate(FakeLM(), ["a", "b", "c"], on_done=lambda: done.append(1))
    assert len(done) == 3


# --- baselines ----------------------------------------------------------------------


def test_majority_answer() -> None:
    plan = make_plan()
    assert majority_answer(plan, SENTIMENT_ROWS) == "positive"
    # Ties go to the value seen first; missing values don't count.
    rows = [{"review": "x", "label": None}, {"review": "x", "label": "b"}]
    rows += [{"review": "x", "label": "a"}, {"review": "x", "label": "a"}]
    rows += [{"review": "x", "label": "b"}]
    assert majority_answer(plan, rows) == "b"


def test_zero_shot_prompt() -> None:
    plan = make_plan()
    assert zero_shot_prompt(plan, {"review": "Love it", "label": "x"}) == "Love it"
    two = plan.model_copy(update={"input_columns": ["title", "review"]})
    assert zero_shot_prompt(two, {"title": "T", "review": "R"}) == "title: T\nreview: R"


# --- scoring ------------------------------------------------------------------------


def test_evaluate_output() -> None:
    plan = make_plan()
    outcomes = {o.check_id: o for o in evaluate_output(plan, SENTIMENT_ROWS[0], "Positive")}
    assert outcomes["non_empty"].passed
    assert outcomes["allowed_values"].passed
    assert outcomes["exact_match_label"].passed


def test_failed_model_call_fails_checks_but_still_skips_missing_gold() -> None:
    plan = make_plan()
    row = {"review": "x", "label": None}
    outcomes = {o.check_id: o for o in evaluate_output(plan, row, "", error="boom")}
    assert outcomes["non_empty"].passed is False
    assert "model call failed" in outcomes["allowed_values"].detail
    assert outcomes["exact_match_label"].skipped


def test_crashing_check_is_recorded_not_raised() -> None:
    @hemp.metric("explodes")
    def explodes(ctx: MetricContext, params: hemp.MetricParams) -> MetricResult:
        raise RuntimeError("bug in my metric")

    plan = make_plan()
    plan = plan.model_copy(update={"custom_metrics": [CheckSpec(id="explodes", metric="explodes")]})
    (outcome,) = [o for o in evaluate_output(plan, SENTIMENT_ROWS[0], "x") if o.error]
    assert outcome.passed is False
    assert outcome.detail == "RuntimeError: bug in my metric"


def test_summary_counts() -> None:
    def outcome(passed: bool | None, score: float | None = None) -> CheckOutcome:
        if score is None and passed is not None:
            score = float(passed)
        return CheckOutcome(check_id="c", passed=passed, score=score)

    variant = VariantResult(
        name="prompt",
        examples=[
            ExampleResult(index=0, output="", checks=[outcome(True, 0.9)]),
            ExampleResult(index=1, output="", checks=[outcome(False, 0.5)]),
            ExampleResult(index=2, output="", checks=[outcome(None)]),
            ExampleResult(index=3, output="", error="x", checks=[outcome(False)]),
        ],
    )
    (summary,) = variant.summary()
    assert (summary.n_passed, summary.n_scored, summary.n_skipped) == (1, 3, 1)
    assert summary.pass_rate == pytest.approx(1 / 3)
    assert summary.mean_score == pytest.approx((0.9 + 0.5 + 0.0) / 3)
    assert variant.n_model_errors == 1


# --- a whole run --------------------------------------------------------------------


def test_run_eval_end_to_end() -> None:
    plan = make_plan()
    lm = FakeLM(keyword_sentiment)
    now = datetime(2026, 2, 1, tzinfo=UTC)
    result = run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, plan, lm=lm, workers=2, now=now)

    assert [v.name for v in result.variants] == [PROMPT_VARIANT, "majority_class", "zero_shot"]
    # The prompt goes out exactly as written, filled in; zero_shot sends the bare input.
    assert lm.prompts[0] == SENTIMENT_PROMPT.replace("{review}", "Love it")
    assert "Hate it" in lm.prompts  # a zero-shot prompt
    assert len(lm.prompts) == 2 * len(SENTIMENT_ROWS)  # majority_class makes no calls

    def rate(variant: str, check: str) -> float | None:
        found = result.variant(variant)
        assert found is not None
        return next(s for s in found.summary() if s.check_id == check).pass_rate

    assert rate("prompt", "exact_match_label") == pytest.approx(0.75)
    assert rate("majority_class", "exact_match_label") == pytest.approx(0.5)
    assert rate("zero_shot", "exact_match_label") == pytest.approx(0.75)

    info = result.info
    assert info.plan_hash == plan.plan_hash()
    assert info.target_model == "fake/target"
    assert info.n_examples == len(SENTIMENT_ROWS)
    assert info.created_at == now
    assert info.hemp_version == hemp.__version__


def test_run_eval_without_baselines() -> None:
    lm = FakeLM()
    result = run_eval(SENTIMENT_PROMPT, SENTIMENT_ROWS, make_plan(), lm=lm, baselines=False)
    assert [v.name for v in result.variants] == [PROMPT_VARIANT]
    assert len(lm.prompts) == len(SENTIMENT_ROWS)


def test_run_eval_checks_columns_before_calling_the_model() -> None:
    lm = FakeLM()
    with pytest.raises(RunError, match=r"\{text\}"):
        run_eval("{text}", SENTIMENT_ROWS, make_plan(), lm=lm)
    rows = [{"review": r["review"]} for r in SENTIMENT_ROWS]  # answer column gone
    with pytest.raises(RunError, match="answer column"):
        run_eval(SENTIMENT_PROMPT, rows, make_plan(), lm=lm)
    with pytest.raises(RunError, match="no rows"):
        run_eval(SENTIMENT_PROMPT, [], make_plan(), lm=lm)
    assert lm.prompts == []
