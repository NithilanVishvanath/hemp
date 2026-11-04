"""Results of a run: one outcome per example per check, plus derived summaries.

A run has several variants: the user's prompt ("prompt") and each baseline
("majority_class", "zero_shot"). All of them are scored with the same checks.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict

PROMPT_VARIANT = "prompt"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CheckOutcome(_Model):
    """One check on one output. `passed`/`score` are None when the check was skipped."""

    check_id: str
    passed: bool | None
    score: float | None
    detail: str = ""
    # The check or the judge crashed. A crashed check counts as a fail; a failed
    # judge call has no verdict (passed is None) and is left out of scores.
    error: bool = False

    @property
    def skipped(self) -> bool:
        return self.passed is None


class ExampleResult(_Model):
    index: int  # position of the row in the dataset, from 0
    output: str
    error: str | None = None  # the model call failed; `output` is empty
    seconds: float = 0.0
    checks: list[CheckOutcome] = []


class CheckSummary(_Model):
    check_id: str
    n_passed: int
    n_scored: int  # examples with a verdict
    n_skipped: int  # not applicable, e.g. no answer value
    n_errors: int  # the check or judge crashed
    mean_score: float | None

    @property
    def pass_rate(self) -> float | None:
        return self.n_passed / self.n_scored if self.n_scored else None


class VariantResult(_Model):
    name: str
    examples: list[ExampleResult]

    @property
    def n_model_errors(self) -> int:
        return sum(1 for e in self.examples if e.error is not None)

    def summary(self) -> list[CheckSummary]:
        """Per-check totals, in the order checks appear in the results."""
        by_check: dict[str, list[CheckOutcome]] = {}
        for example in self.examples:
            for outcome in example.checks:
                by_check.setdefault(outcome.check_id, []).append(outcome)
        summaries = []
        for check_id, outcomes in by_check.items():
            scored = [o for o in outcomes if not o.skipped]
            scores = [o.score for o in scored if o.score is not None]
            summaries.append(
                CheckSummary(
                    check_id=check_id,
                    n_passed=sum(1 for o in scored if o.passed),
                    n_scored=len(scored),
                    n_skipped=sum(1 for o in outcomes if o.skipped and not o.error),
                    n_errors=sum(1 for o in outcomes if o.error),
                    mean_score=sum(scores) / len(scores) if scores else None,
                )
            )
        return summaries


class RunInfo(_Model):
    """What a run was made from. Every stored run records all of this."""

    created_at: datetime
    hemp_version: str
    prompt_hash: str
    dataset_hash: str
    plan_hash: str
    target_model: str
    judge_model: str | None = None
    prompt_source: str | None = None
    dataset_source: str | None = None
    plan_source: str | None = None
    data_dir: str | None = None  # where relative image paths were looked up
    n_examples: int


class RunResult(_Model):
    info: RunInfo
    variants: list[VariantResult]
    questions: dict[str, str] = {}  # judge question ids -> text, for questions that were judged
    # Baselines copied from an earlier run instead of recomputed: name -> that run's id.
    baselines_from: dict[str, int] = {}
    id: int | None = None  # set once the run is stored

    def variant(self, name: str) -> VariantResult | None:
        return next((v for v in self.variants if v.name == name), None)

    @property
    def prompt(self) -> VariantResult:
        found = self.variant(PROMPT_VARIANT)
        if found is None:
            raise ValueError("This run has no results for the prompt itself.")
        return found
