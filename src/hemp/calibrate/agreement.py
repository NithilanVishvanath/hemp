"""How well the judge agrees with human labels: accuracy and Cohen's kappa.

Accuracy alone flatters a judge on lopsided labels (if 90% of outputs are good, a
judge that always says yes agrees 90% of the time). Kappa corrects for agreement
expected by chance: 1 is perfect, 0 is no better than chance.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict


class Agreement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n: int
    n_agree: int
    accuracy: float | None
    kappa: float | None  # None when chance agreement is total (one answer only)


def agreement(pairs: Sequence[tuple[bool, bool]]) -> Agreement:
    """Agreement over (human, judge) verdict pairs."""
    n = len(pairs)
    if n == 0:
        return Agreement(n=0, n_agree=0, accuracy=None, kappa=None)
    n_agree = sum(1 for human, judge in pairs if human == judge)
    observed = n_agree / n
    human_yes = sum(1 for human, _ in pairs if human) / n
    judge_yes = sum(1 for _, judge in pairs if judge) / n
    chance = human_yes * judge_yes + (1 - human_yes) * (1 - judge_yes)
    kappa = None if chance >= 1 else (observed - chance) / (1 - chance)
    return Agreement(n=n, n_agree=n_agree, accuracy=observed, kappa=kappa)
