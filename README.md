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

## Run the evaluation

```
uv run hemp run examples/sentiment/prompt.txt examples/sentiment/data.jsonl
```

Runs your prompt on every row exactly as written, scores the outputs with the plan's checks, and does the same for the baselines so you can see whether your prompt beats them. Baselines don't depend on your prompt, so when the plan, data and models match an earlier run, their results are copied from it instead of recomputed (`--fresh-baselines` forces a recompute, `--no-baselines` skips them). Results are saved to `.hemp/hemp.db` (change it with `--db` or `$HEMP_DB`). `uv run hemp show` lists past runs; `uv run hemp show 3` shows run #3 with the examples that failed.

The plan's yes/no `questions` are answered by an LLM judge, one call per question per example. A question with `compare_to: <column>` shows the judge that column as the reference answer. Pick the judge with `--judge-model` (a bigger model than the one you're testing is usually better), or skip the questions with `--no-judge`.

## Check and tune the judge

The judge's yes/no answers are only useful if they match yours. After a run with judge questions:

```
uv run hemp calibrate examples/qa/prompt.plan.yaml
```

Shows you outputs from the latest run, one at a time, with the judge's verdict hidden: answer each question yourself (`y`/`n`, `s` to skip, `q` to stop). HEMP then reports how often the judge agrees with you (accuracy and Cohen's kappa, per question). With 8 or more labels it also tunes the judge with DSPy on half of them and tests it on the other half; if the tuned judge agrees with you more, it's saved as `<plan>.judge.json` and `hemp run` uses it from then on. Labels are kept, so each session adds to them; `--label 0` just re-measures.

## Compare prompt versions

Edit your prompt, run again, then:

```
uv run hemp diff          # the two latest runs
uv run hemp diff 3 5      # run #3 (before) against run #5 (after)
```

Shows what changed between the runs (prompt, plan, dataset, models), the prompt edits, each check's pass rate before and after, and the examples that got fixed or broke. Examples are matched by their row contents, so adding or reordering rows is fine. In CI, `--fail-on-regression` exits with code 1 if any check's pass rate went down.

## Images

A column whose values are image paths (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`, `.bmp`), image URLs, or `data:image/...` URIs is an image column. Put it in your prompt like any other column:

```
Classify this photo as cat, dog or bird. Reply with the label only.
{image}
```

```
image,label
photos/cat_01.jpg,cat
photos/dog_01.jpg,dog
```

The model receives the image itself where `{image}` is, not the path. Relative paths are looked up next to the dataset file. The `zero_shot` baseline sends just the image, and the judge sees the images too. You need a model that can see images, e.g. `HEMP_MODEL=gemini/gemini-2.0-flash` (with `GEMINI_API_KEY` set) or a local vision model like `ollama_chat/gemma3:4b`. Replacing an image file counts as changing the dataset, so old baseline results aren't reused for it.

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
