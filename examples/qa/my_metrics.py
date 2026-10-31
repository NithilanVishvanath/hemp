"""Example custom metric. Use with: hemp plan ... --plugin examples/qa/my_metrics.py"""

from pydantic import Field

import hemp


class NoHedgingParams(hemp.MetricParams):
    phrases: list[str] = Field(
        default=["as an ai", "i think", "it is possible that"],
        description="phrases that count as hedging",
    )


@hemp.metric("no_hedging", params=NoHedgingParams)
def no_hedging(ctx: hemp.MetricContext, params: NoHedgingParams) -> hemp.MetricResult:
    """The answer doesn't hedge with phrases like 'as an AI' or 'I think'."""
    text = ctx.output.lower()
    found = [p for p in params.phrases if p in text]
    return hemp.MetricResult.check(not found, f"hedges: {', '.join(found)}" if found else "")
