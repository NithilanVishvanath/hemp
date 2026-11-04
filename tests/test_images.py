"""Image columns: detection, sending images to the model and the judge, hashing, CLI."""

import base64
import csv
from pathlib import Path
from typing import Any

import pytest
from dspy.utils import DummyLM
from fakes import FakeLM, request_text
from typer.testing import CliRunner

from hemp.calibrate import label_candidates
from hemp.cli import app
from hemp.dataset import Row
from hemp.images import ImageRef, image_columns, looks_like_image, resolve
from hemp.plan import create_plan
from hemp.plan.schema import BaselineKind, Criterion
from hemp.profile import ColumnRole, ColumnType, profile_dataset, profile_rows
from hemp.profile.profiler import dataset_hash
from hemp.runner import RunError, run_eval
from hemp.runner.baselines import zero_shot_prompt
from hemp.runner.target import build_prompt, prompt_images, prompt_text, to_request

PROMPT = "Classify this animal as cat, dog or bird. Reply with the label only.\n{image}"
PICTURES = {"cat": b"CAT-PIXELS", "dog": b"DOG-PIXELS", "bird": b"BIRD-PIXELS"}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def make_images(folder: Path, n: int = 3) -> list[Row]:
    """n copies of each animal as tiny files, plus a CSV of image paths and labels."""
    (folder / "photos").mkdir(parents=True, exist_ok=True)
    rows: list[Row] = []
    for i in range(n):
        for animal, data in PICTURES.items():
            name = f"photos/{animal}_{i}.png"
            (folder / name).write_bytes(data)
            rows.append({"image": name, "label": animal})
    with (folder / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def sees_animal(prompt: str) -> str:
    """A 'vision model' that recognises cats and dogs but calls birds cats."""
    if b64(PICTURES["dog"]) in prompt:
        return "dog"
    return "cat"


# --- detection ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("photos/cat.png", True),
        ("C:\\pics\\Dog.JPEG", True),
        ("https://example.com/a.webp?size=2", True),
        ("data:image/png;base64,AAAA", True),
        ("https://example.com/page", False),
        ("notes.txt", False),
        ("a cat.png is mentioned\nin text", False),
        (42, False),
    ],
)
def test_looks_like_image(value: object, expected: bool) -> None:
    assert looks_like_image(value) is expected


def test_image_columns() -> None:
    rows = [{"img": "a.png", "label": "cat"}, {"img": "", "label": "dog"}]
    assert image_columns(rows) == {"img"}


def test_resolve(tmp_path: Path) -> None:
    ref = resolve("photos/cat.png", tmp_path)
    assert ref.is_file and Path(ref.location) == (tmp_path / "photos/cat.png").resolve()
    url = resolve("https://x.com/a.jpg", tmp_path)
    assert not url.is_file and url.media_type == "image/jpeg"


def test_profile_marks_image_column_as_input(tmp_path: Path) -> None:
    make_images(tmp_path)
    profile = profile_dataset(tmp_path / "labels.csv", prompt=PROMPT)
    image, label = profile.column("image"), profile.column("label")
    assert image.type is ColumnType.IMAGE and image.role is ColumnRole.INPUT
    assert label.type is ColumnType.CATEGORICAL and label.role is ColumnRole.GOLD


def test_plan_for_image_classification(tmp_path: Path) -> None:
    make_images(tmp_path)
    profile = profile_dataset(tmp_path / "labels.csv", prompt=PROMPT)
    plan = create_plan(PROMPT, profile, lm=None).plan
    assert [c.metric for c in plan.checks()] == ["non_empty", "allowed_values", "exact_match"]
    assert plan.baselines == [BaselineKind.MAJORITY_CLASS, BaselineKind.ZERO_SHOT]


def test_dataset_hash_covers_image_contents(tmp_path: Path) -> None:
    rows = make_images(tmp_path)
    before = dataset_hash(rows, tmp_path)
    (tmp_path / "photos/cat_0.png").write_bytes(b"A DIFFERENT CAT")
    assert dataset_hash(rows, tmp_path) != before


# --- building the message -----------------------------------------------------------


def test_text_prompts_are_unchanged() -> None:
    assert build_prompt("Review: {review}", {"review": "ok"}) == "Review: ok"


def test_image_goes_where_its_placeholder_was(tmp_path: Path) -> None:
    make_images(tmp_path)
    template = "Look at {image} and say if it's a {guess}."
    content = build_prompt(
        template, {"image": "photos/cat_0.png", "guess": "cat"}, {"image"}, tmp_path
    )
    assert isinstance(content, tuple)
    before, image, after = content
    assert (before, after) == ("Look at ", " and say if it's a cat.")
    assert isinstance(image, ImageRef) and image.source == "photos/cat_0.png"
    assert prompt_text(content) == "Look at [image 1] and say if it's a cat."
    assert prompt_images(content) == [image]

    request = to_request("gemini/flash", content)
    assert request.model == "gemini/flash"
    (message,) = request.messages
    assert message.role == "user"
    assert message.parts[1].media_type == "image/png"
    assert message.parts[1].data == b64(PICTURES["cat"])


def test_zero_shot_sends_only_the_image(tmp_path: Path) -> None:
    rows = make_images(tmp_path)
    plan = create_plan(PROMPT, profile_rows(rows, prompt=PROMPT), lm=None).plan
    content = zero_shot_prompt(plan, rows[0], {"image"}, tmp_path)
    assert isinstance(content, tuple) and len(content) == 1
    assert isinstance(content[0], ImageRef)


# --- running ------------------------------------------------------------------------


def image_run(tmp_path: Path, **kwargs: Any) -> Any:
    rows = make_images(tmp_path)
    plan = create_plan(
        PROMPT, profile_dataset(tmp_path / "labels.csv", prompt=PROMPT), lm=None
    ).plan
    lm = FakeLM(sees_animal)
    return run_eval(PROMPT, rows, plan, lm=lm, data_dir=tmp_path, **kwargs), lm, rows, plan


def test_run_sends_images_and_scores(tmp_path: Path) -> None:
    result, lm, _, _ = image_run(tmp_path)
    first = lm.prompts[0]
    assert first.startswith("Classify this animal")
    assert f"<image:{b64(PICTURES['cat'])}>" in first
    exact = next(s for s in result.prompt.summary() if s.check_id == "exact_match_label")
    assert (exact.n_passed, exact.n_scored) == (6, 9)  # birds are called cats
    assert result.info.data_dir == str(tmp_path.resolve())


def test_missing_images_stop_the_run_before_any_call(tmp_path: Path) -> None:
    rows = make_images(tmp_path)
    (tmp_path / "photos/dog_1.png").unlink()
    plan = create_plan(PROMPT, profile_rows(rows, prompt=PROMPT), lm=None).plan
    lm = FakeLM(sees_animal)
    with pytest.raises(RunError, match=r"photos/dog_1.png"):
        run_eval(PROMPT, rows, plan, lm=lm, data_dir=tmp_path)
    assert lm.prompts == []


def test_judge_sees_the_image(tmp_path: Path) -> None:
    rows = make_images(tmp_path)
    plan = create_plan(PROMPT, profile_rows(rows, prompt=PROMPT), lm=None).plan
    plan = plan.model_copy(
        update={"criteria": [Criterion(id="q1", question="Does the label match the photo?")]}
    )
    judge = DummyLM([{"reason": "ok", "answer": "True"}] * 9)
    judge.model = "fake/judge"
    run_eval(
        PROMPT,
        rows,
        plan,
        lm=FakeLM(sees_animal),
        judge_lm=judge,
        baselines=False,
        data_dir=tmp_path,
    )
    sent = str(judge.history[0]["messages"])
    assert "[image 1]" in sent  # the task text marks where the image was
    assert f"data:image/png;base64,{b64(PICTURES['cat'])}" in sent


def test_labels_carry_the_images(tmp_path: Path) -> None:
    rows = make_images(tmp_path)
    plan = create_plan(PROMPT, profile_rows(rows, prompt=PROMPT), lm=None).plan
    plan = plan.model_copy(update={"criteria": [Criterion(id="q1", question="Is it right?")]})
    judge = DummyLM([{"reason": "ok", "answer": "True"}] * 9)
    judge.model = "fake/judge"
    result = run_eval(
        PROMPT,
        rows,
        plan,
        lm=FakeLM(sees_animal),
        judge_lm=judge,
        baselines=False,
        data_dir=tmp_path,
    ).model_copy(update={"id": 1})
    items = label_candidates(result, rows, PROMPT, plan, n=10)
    assert items and all(len(i.images) == 1 for i in items)
    assert Path(items[0].images[0]).is_file()
    assert "[image 1]" in items[0].task
    assert request_text  # imported for FakeLM's image view


# --- CLI ----------------------------------------------------------------------------


def test_cli_image_classification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hemp.cli

    data_dir = tmp_path / "data"
    make_images(data_dir)
    (tmp_path / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEMP_DB", raising=False)
    monkeypatch.setattr(hemp.cli, "make_lm", lambda model: FakeLM(sees_animal, model))
    runner = CliRunner()

    profiled = runner.invoke(app, ["profile", "data/labels.csv", "--prompt", "prompt.txt"])
    assert "image" in profiled.output

    planned = runner.invoke(app, ["plan", "prompt.txt", "data/labels.csv", "--no-llm"])
    assert planned.exit_code == 0, planned.output
    result = runner.invoke(app, ["run", "prompt.txt", "data/labels.csv"])
    assert result.exit_code == 0, result.output
    assert "67% (6/9)" in result.output  # paths resolved from the CSV's folder
    assert "has changed since the plan" not in result.output
