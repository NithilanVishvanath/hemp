# hemp

HEMP (Help Evaluate My Prompt) takes a prompt and a dataset, works out how the prompt should be evaluated, then runs that evaluation and tracks results across prompt versions.

> Early development. Working so far: `hemp profile`. The other commands are stubs.

## Profile a dataset

```
uv run hemp profile examples/sentiment/data.jsonl --prompt examples/sentiment/prompt.txt
```

Shows each column's type, its role (input, gold answer, id, metadata) and value stats. Add `--json` for machine-readable output. No model calls.

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
