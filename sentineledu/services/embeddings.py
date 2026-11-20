"""Sentence-BERT embeddings (ARCHITECTURE §5, DECISIONS D-08).

The same `all-MiniLM-L6-v2` model as the legacy `src/similarity.py`, but loaded lazily and once,
with batch encoding and unit-normalised vectors so a dot product equals cosine similarity.
Importing this module does not load the model.
"""
import logging
import threading
from typing import Sequence

import numpy as np

from sentineledu.config import Settings, get_settings

log = logging.getLogger("sentineledu.embeddings")

_lock = threading.Lock()
_model = None
_model_name: str | None = None
load_count = 0  # tests assert the model is loaded exactly once


def _get_model(settings: Settings | None = None):
    global _model, _model_name, load_count
    name = (settings or get_settings()).embedding_model
    with _lock:
        if _model is None or _model_name != name:
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            from sentence_transformers import SentenceTransformer  # slow import, kept local

            log.info("Loading embedding model %s", name)
            _model = SentenceTransformer(name)
            _model_name = name
            load_count += 1
    return _model


def dimension(settings: Settings | None = None) -> int:
    return int(_get_model(settings).get_sentence_embedding_dimension())


def encode(texts: Sequence[str] | str, settings: Settings | None = None) -> np.ndarray:
    """Unit-normalised float32 vectors, shape (n, dim). A single string gives shape (1, dim)."""
    if isinstance(texts, str):
        texts = [texts]
    texts = [t if t.strip() else " " for t in texts]  # the model dislikes empty strings
    if not texts:
        return np.zeros((0, dimension(settings)), dtype=np.float32)
    vectors = _get_model(settings).encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                                          batch_size=32, show_progress_bar=False)
    return np.asarray(vectors, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, clamped to [-1, 1]."""
    a, b = np.asarray(a, dtype=np.float32).ravel(), np.asarray(b, dtype=np.float32).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def similarity(text_a: str, text_b: str, settings: Settings | None = None) -> float:
    """Cosine similarity of two texts. Blank text gives 0."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    v = encode([text_a, text_b], settings)
    return cosine(v[0], v[1])
