"""Sentence-transformer embeddings, loaded once per process."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from backend.config import EMBED_CACHE, EMBED_DEVICE, EMBED_MODEL, EMBED_QUERY_PREFIX


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL, device=EMBED_DEVICE, cache_folder=EMBED_CACHE)


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 200,
    )
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    vector = get_model().encode(
        EMBED_QUERY_PREFIX + text, normalize_embeddings=True
    )
    return vector.tolist()
