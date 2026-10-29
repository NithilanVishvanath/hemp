from hemp.prompt import template_fields


def test_fields_in_order_without_duplicates() -> None:
    assert template_fields("Q: {question}\nCtx: {context}\nAgain: {question}") == [
        "question",
        "context",
    ]


def test_ignores_json_and_escaped_braces() -> None:
    template = 'Return {"label": "pos"} for {text}. Literal {{not_a_field}}.'
    assert template_fields(template) == ["text"]


def test_allows_hyphens_and_underscores() -> None:
    assert template_fields("{user-input} {_x1}") == ["user-input", "_x1"]


def test_no_fields() -> None:
    assert template_fields("Just a prompt.") == []
