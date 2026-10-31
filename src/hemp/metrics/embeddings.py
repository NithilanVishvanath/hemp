"""Text embeddings for semantic similarity, via LiteLLM (local Ollama by default)."""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager

DEFAULT_EMBED_MODEL = "ollama/nomic-embed-text"

Embedder = Callable[[list[str]], list[list[float]]]


class EmbeddingError(RuntimeError):
    """The embedding model couldn't be reached or returned something unusable."""


def litellm_embedder(model: str) -> Embedder:
    def embed(texts: list[str]) -> list[list[float]]:
        import litellm  # imported lazily: only needed when embeddings are used

        try:
            response = litellm.embedding(model=model, input=texts)
            return [list(map(float, item["embedding"])) for item in response.data]
        except Exception as e:
            hint = (
                f"Is Ollama running and the model pulled (`ollama pull {model.split('/', 1)[1]}`)? "
                if model.startswith("ollama/")
                else "Check the model name and API key. "
            )
            first_line = (str(e).splitlines() or [""])[0][:300]
            raise EmbeddingError(
                f"Couldn't get embeddings from {model}: {type(e).__name__}: {first_line}\n{hint}"
                "Or change the semantic_similarity `model` param, or use token_f1 instead."
            ) from e

    return embed


_backend: Callable[[str], Embedder] = litellm_embedder
_cache: dict[tuple[str, str], list[float]] = {}


def embed(model: str, texts: list[str]) -> list[list[float]]:
    """Embeddings for `texts`, cached per (model, text) for the life of the process."""
    missing = list(dict.fromkeys(t for t in texts if (model, t) not in _cache))
    if missing:
        vectors = _backend(model)(missing)
        if len(vectors) != len(missing):
            raise EmbeddingError(
                f"{model} returned {len(vectors)} embeddings for {len(missing)} texts."
            )
        for text, vector in zip(missing, vectors, strict=True):
            _cache[(model, text)] = vector
    return [_cache[(model, t)] for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


@contextmanager
def embedding_backend(backend: Callable[[str], Embedder]) -> Iterator[None]:
    """Swap the embedding backend, e.g. for tests. Clears the cache on entry and exit."""
    global _backend
    previous = _backend
    _backend = backend
    _cache.clear()
    try:
        yield
    finally:
        _backend = previous
        _cache.clear()
