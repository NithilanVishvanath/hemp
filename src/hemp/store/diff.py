"""Compare two runs check by check and example by example.

Examples are paired by their row contents, not their position, so a diff still
works when rows were added, removed, or reordered. Only the user's prompt is
compared; baselines are context, not something being changed.
"""

from collections import defaultdict
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from hemp.dataset import Row
from hemp.hashing import canonical_json
from hemp.runner.results import CheckOutcome, CheckSummary, ExampleResult, RunResult


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExampleChange(_Model):
    """One check on one example that went from pass to fail, or fail to pass."""

    check_id: str
    row: Row
    index_a: int
    index_b: int
    output_a: str
    output_b: str
    detail_a: str
    detail_b: str


class CheckDiff(_Model):
    check_id: str
    question: str | None = None  # the judge question's text, for questions
    a: CheckSummary | None  # None when the check isn't in that run
    b: CheckSummary | None
    fixed: list[ExampleChange] = []  # failed in run A, passes in run B
    broke: list[ExampleChange] = []  # passed in run A, fails in run B

    @property
    def change(self) -> float | None:
        """Change in pass rate, in percentage points."""
        if self.a is None or self.b is None or self.a.pass_rate is None or self.b.pass_rate is None:
            return None
        return (self.b.pass_rate - self.a.pass_rate) * 100


class RunDiff(_Model):
    id_a: int | None
    id_b: int | None
    changed: list[str]  # what differs between the runs: "prompt", "plan", ...
    same: list[str]
    checks: list[CheckDiff]
    n_paired: int  # examples present in both runs
    n_only_a: int
    n_only_b: int
    question_edits: list[str] = []  # question ids whose text changed

    @property
    def regressed(self) -> bool:
        """True if any check's pass rate went down."""
        return any(c.change is not None and c.change < -1e-9 for c in self.checks)

    @property
    def n_fixed(self) -> int:
        return sum(len(c.fixed) for c in self.checks)

    @property
    def n_broke(self) -> int:
        return sum(len(c.broke) for c in self.checks)


def diff_runs(a: RunResult, b: RunResult, rows_a: Sequence[Row], rows_b: Sequence[Row]) -> RunDiff:
    """Compare run B against run A (the "before")."""
    pairs, only_a, only_b = _pair_examples(a.prompt.examples, b.prompt.examples, rows_a, rows_b)

    summaries_a = {s.check_id: s for s in a.prompt.summary()}
    summaries_b = {s.check_id: s for s in b.prompt.summary()}
    check_ids = list(dict.fromkeys([*summaries_a, *summaries_b]))
    questions = {**a.questions, **b.questions}

    checks = []
    for check_id in check_ids:
        fixed: list[ExampleChange] = []
        broke: list[ExampleChange] = []
        for ex_a, ex_b, row in pairs:
            out_a, out_b = _outcome(ex_a, check_id), _outcome(ex_b, check_id)
            if out_a is None or out_b is None or out_a.skipped or out_b.skipped:
                continue
            if out_a.passed == out_b.passed:
                continue
            change = ExampleChange(
                check_id=check_id,
                row=row,
                index_a=ex_a.index,
                index_b=ex_b.index,
                output_a=ex_a.error or ex_a.output,
                output_b=ex_b.error or ex_b.output,
                detail_a=out_a.detail,
                detail_b=out_b.detail,
            )
            (fixed if out_b.passed else broke).append(change)
        checks.append(
            CheckDiff(
                check_id=check_id,
                question=questions.get(check_id),
                a=summaries_a.get(check_id),
                b=summaries_b.get(check_id),
                fixed=fixed,
                broke=broke,
            )
        )

    changed, same = _what_changed(a, b)
    return RunDiff(
        id_a=a.id,
        id_b=b.id,
        changed=changed,
        same=same,
        checks=checks,
        n_paired=len(pairs),
        n_only_a=only_a,
        n_only_b=only_b,
        question_edits=[
            qid
            for qid in a.questions
            if qid in b.questions and a.questions[qid] != b.questions[qid]
        ],
    )


def _what_changed(a: RunResult, b: RunResult) -> tuple[list[str], list[str]]:
    fields = {
        "prompt": (a.info.prompt_hash, b.info.prompt_hash),
        "plan": (a.info.plan_hash, b.info.plan_hash),
        "dataset": (a.info.dataset_hash, b.info.dataset_hash),
        "target model": (a.info.target_model, b.info.target_model),
        "judge model": (a.info.judge_model, b.info.judge_model),
    }
    changed = [name for name, (x, y) in fields.items() if x != y]
    same = [name for name in fields if name not in changed]
    return changed, same


def _pair_examples(
    examples_a: Sequence[ExampleResult],
    examples_b: Sequence[ExampleResult],
    rows_a: Sequence[Row],
    rows_b: Sequence[Row],
) -> tuple[list[tuple[ExampleResult, ExampleResult, Row]], int, int]:
    """Pair examples whose rows are identical. Repeated rows pair up in order."""
    waiting: dict[str, list[ExampleResult]] = defaultdict(list)
    for ex in examples_a:
        waiting[canonical_json(rows_a[ex.index])].append(ex)
    pairs = []
    only_b = 0
    for ex in examples_b:
        row = rows_b[ex.index]
        queue = waiting.get(canonical_json(row))
        if queue:
            pairs.append((queue.pop(0), ex, row))
        else:
            only_b += 1
    only_a = sum(len(q) for q in waiting.values())
    return pairs, only_a, only_b


def _outcome(example: ExampleResult, check_id: str) -> CheckOutcome | None:
    return next((o for o in example.checks if o.check_id == check_id), None)
