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

Writes `examples/sentiment/prompt.plan.yaml`: format checks, reference metrics, yes/no rubric criteria and baselines. Review and edit it, then check your edits with `uv run hemp validate <plan>`. `uv run hemp metrics` lists the metrics a plan can use.

The planner uses a local Ollama model by default (`ollama_chat/qwen2.5:7b`). Pick another with `--planner-model` or `HEMP_PLANNER_MODEL` (any LiteLLM model string), or skip the LLM with `--no-llm`.

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
