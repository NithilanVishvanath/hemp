# hemp

HEMP (Help Evaluate My Prompt) takes a prompt and a dataset, works out how the prompt should be evaluated, then runs that evaluation and tracks results across prompt versions.

> Early development: the CLI commands exist but are stubs for now.

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
