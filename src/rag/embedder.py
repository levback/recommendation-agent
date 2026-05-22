"""Sentence-transformer text embedder (lazy-loaded)."""
from __future__ import annotations

from typing import Sequence

import numpy as np


class Embedder:
    """Embed text strings using a sentence-transformer model.

    The underlying model is loaded lazily on the first :meth:`embed` call to
    avoid import overhead when the class is constructed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy init

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, D) float32 array of L2-normalised embeddings."""
        self._load()
        vecs: np.ndarray = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        return vecs

    def embed_one(self, text: str) -> np.ndarray:
        """Return a (D,) float32 L2-normalised embedding for a single *text*."""
        return self.embed([text])[0]
