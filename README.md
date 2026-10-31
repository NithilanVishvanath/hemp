# hemp

HEMP (Help Evaluate My Prompt) takes a prompt and a dataset, works out how the prompt should be evaluated, then runs that evaluation and tracks results across prompt versions.

> Early development. Working so far: `hemp profile`, `hemp plan`, `hemp validate`, `hemp metrics`. The other commands are stubs.

## Profile a dataset

```
uv run hemp profile examples/sentiment/data.jsonl --prompt examples/sentiment/prompt.txt
```

Shows each column's type, its role (input, gold answer, id, metadata) and value stats. Add `--json` for machine-readable output. No model calls.

## Create an evaluation plan

```
uv run hemp plan examples/sentiment/prompt.txt examples/sentiment/data.jsonl
```

Writes `examples/sentiment/prompt.plan.yaml`: which columns are input and answer, automatic checks, yes/no questions for an LLM judge, and baselines to beat. Review and edit it, then check your edits with `uv run hemp validate <plan>`. `uv run hemp metrics` lists the metrics a plan can use.

The planner uses a local Ollama model by default (`ollama_chat/qwen2.5:7b`). Pick another with `--planner-model` or `HEMP_PLANNER_MODEL` (any LiteLLM model string), or skip the LLM with `--no-llm`.

## Your own metrics

Write a metric in a Python file:

```python
# my_metrics.py
import hemp


@hemp.metric("polite")
def polite(ctx: hemp.MetricContext, params: hemp.MetricParams) -> hemp.MetricResult:
    """The output says please or thanks."""
    text = ctx.output.lower()
    return hemp.MetricResult.check("please" in text or "thank" in text)
```

Pass it with `--plugin my_metrics.py` (to `hemp plan` or `hemp metrics`). The planner can then choose it, and the plan remembers the file, so `hemp validate` loads it automatically.

## Models

| What | Default | Override |
|---|---|---|
| Planner, judge, target | `ollama_chat/qwen2.5:7b` | `HEMP_PLANNER_MODEL`, `HEMP_JUDGE_MODEL`, `HEMP_TARGET_MODEL`, or `HEMP_MODEL` for all |
| Embeddings (free-text answers) | `ollama/nomic-embed-text` | `HEMP_EMBED_MODEL` |

Any LiteLLM model string works. For the defaults: `ollama pull qwen2.5:7b` and `ollama pull nomic-embed-text`.

## Install (development)

```
uv sync
uv run hemp --help
```

## Development

```
uv run pytest                   # unit tests
uv run pytest -m integration    # tests that hit a real model
uv run ruff check . && uv run ruff format .
uv run mypy
```
