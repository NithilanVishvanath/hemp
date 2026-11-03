"""Run many model calls at once, failing fast if the first one fails."""

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_all(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    workers: int = 4,
    check_first: Callable[[R], None] | None = None,
) -> list[R]:
    """`[fn(item) for item in items]`, with up to `workers` calls at once, in order.

    The first item runs alone and is passed to `check_first`, which can raise to stop
    everything: when the first call fails, the model is usually unreachable, and it's
    better to say so once than to fail every item.
    """
    if not items:
        return []
    first = fn(items[0])
    if check_first is not None:
        check_first(first)
    if workers <= 1:
        return [first, *(fn(item) for item in items[1:])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [first, *pool.map(fn, items[1:])]


def short_error(error: BaseException) -> str:
    """One line describing an exception, for storing alongside a result."""
    return f"{type(error).__name__}: {error}".splitlines()[0][:500]
